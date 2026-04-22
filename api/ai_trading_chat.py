"""AI Trading Chat — streaming assistant backed by OpenAI.

SSE contract
------------
Success (200, text/event-stream):
  data: {"type":"delta","text":"..."}   # append to assistant reply
  data: {"type":"done"}                 # stream finished

Errors returned *before* the stream starts use standard JSON HTTPException:
  503 — OPENAI_API_KEY not configured
  400 — bad request body / profile_path
  404 — profile not found
  502 — broker init failure
"""
from __future__ import annotations

import json
import os
import threading
import time as _cache_time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from core.profile import ProfileV1


# ---------------------------------------------------------------------------
# TTL cache for slow-moving external data (cross-asset, FRED, books)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Thread-safe in-memory cache with per-key TTL."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}  # key -> (expires_at, value)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if _cache_time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_sec: float) -> None:
        with self._lock:
            self._store[key] = (_cache_time.monotonic() + ttl_sec, value)


_ctx_cache = _TTLCache()

# LOGS_DIR mirrors api/main.py — persistent volume on Railway, else repo root.
_data_base_env = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or os.environ.get("USDJPY_DATA_DIR")
_DATA_BASE = Path(_data_base_env) if _data_base_env else Path(__file__).resolve().parent.parent
LOGS_DIR = _DATA_BASE / "logs"

# Chat models: optional UI + request body must pick from this allowlist (override with AI_CHAT_ALLOWED_MODELS).
_DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
_DEFAULT_ALLOWED_CHAT_MODELS: tuple[str, ...] = (
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4o",
)

# Trade-suggestion models: intentionally biased toward higher-reasoning models
# because the suggestion endpoint asks the model to pick an entry + exit strategy
# from a catalog and justify it. Defaults to gpt-4o (override with
# AI_SUGGEST_ALLOWED_MODELS / OPENAI_SUGGEST_MODEL).
_DEFAULT_SUGGEST_MODEL = "gpt-5.4-mini"
_DEFAULT_ALLOWED_SUGGEST_MODELS: tuple[str, ...] = (
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-4o-mini",
)


def allowed_ai_chat_models() -> list[str]:
    """Ordered list of model ids the UI may offer and the API will accept."""
    raw = os.environ.get("AI_CHAT_ALLOWED_MODELS", "").strip()
    if raw:
        out = [m.strip() for m in raw.split(",") if m.strip()]
        return out if out else list(_DEFAULT_ALLOWED_CHAT_MODELS)
    return list(_DEFAULT_ALLOWED_CHAT_MODELS)


def default_ai_chat_model() -> str:
    """Server default when the client omits chat_model."""
    allowed = allowed_ai_chat_models()
    allowed_set = frozenset(allowed)
    env = os.environ.get("OPENAI_CHAT_MODEL", _DEFAULT_CHAT_MODEL).strip() or _DEFAULT_CHAT_MODEL
    if env in allowed_set:
        return env
    return allowed[0] if allowed else _DEFAULT_CHAT_MODEL


def resolve_ai_chat_model(requested: str | None) -> str:
    """Return the model id to call OpenAI with, or raise ValueError."""
    allowed_set = frozenset(allowed_ai_chat_models())
    if not requested or not str(requested).strip():
        return default_ai_chat_model()
    rid = str(requested).strip()
    if rid not in allowed_set:
        raise ValueError(f"chat_model not allowed: {rid!r} (allowed: {', '.join(sorted(allowed_set))})")
    return rid


def allowed_ai_suggest_models() -> list[str]:
    """Ordered list of model ids the suggest endpoint will accept."""
    raw = os.environ.get("AI_SUGGEST_ALLOWED_MODELS", "").strip()
    if raw:
        out = [m.strip() for m in raw.split(",") if m.strip()]
        return out if out else list(_DEFAULT_ALLOWED_SUGGEST_MODELS)
    return list(_DEFAULT_ALLOWED_SUGGEST_MODELS)


def default_ai_suggest_model() -> str:
    """Server default model for trade-suggestion generation."""
    allowed = allowed_ai_suggest_models()
    allowed_set = frozenset(allowed)
    env = os.environ.get("OPENAI_SUGGEST_MODEL", _DEFAULT_SUGGEST_MODEL).strip() or _DEFAULT_SUGGEST_MODEL
    if env in allowed_set:
        return env
    return allowed[0] if allowed else _DEFAULT_SUGGEST_MODEL


def resolve_ai_suggest_model(requested: str | None) -> str:
    """Return the suggest model id, or raise ValueError if not allowed."""
    allowed_set = frozenset(allowed_ai_suggest_models())
    if not requested or not str(requested).strip():
        return default_ai_suggest_model()
    rid = str(requested).strip()
    if rid not in allowed_set:
        raise ValueError(
            f"suggest_model not allowed: {rid!r} (allowed: {', '.join(sorted(allowed_set))})"
        )
    return rid


BOT_ENTRY_TAGS: tuple[str, ...] = (
    "phase3",
    "v44",
    "tiered_pullback",
    "zone_entry",
    "momentum",
    "er_low",
    "er_high",
    "defended",
)


def classify_trade_source(entry_type: Any) -> str:
    """Classify trade source as manual or bot."""
    et = str(entry_type or "").strip().lower()
    if not et:
        return "manual"
    if any(tag in et for tag in BOT_ENTRY_TAGS):
        return "bot"
    return "manual"


def _plain_bot_label(entry_type: Any) -> str:
    """Human-readable bot label for prompt/tool output."""
    et = str(entry_type or "").strip().lower()
    if not et:
        return "automated strategy"
    if "momentum" in et:
        return "automated momentum entries"
    if "zone_entry" in et or "zone" in et:
        return "automated zone entries"
    if "tiered_pullback" in et:
        return "automated pullback entries"
    if "v44" in et:
        return "automated NY entries"
    if "phase3" in et:
        return "automated phase3 entries"
    if "er_low" in et or "er_high" in et:
        return "automated expansion-range entries"
    if "defended" in et:
        return "automated defended entries"
    return "automated strategy"


def _source_label_for_trade(entry_type: Any) -> str:
    src = classify_trade_source(entry_type)
    if src == "manual":
        return "manual"
    return f"bot:{_plain_bot_label(entry_type)}"


# ---------------------------------------------------------------------------
# 1. Build trading context from broker
# ---------------------------------------------------------------------------

def _fetch_cross_asset_prices(adapter: Any) -> dict[str, Any]:
    """Fetch cross-asset prices from OANDA including DXY constituents.

    Real DXY is computed from the ICE formula:
    DXY = 50.14348112 × EURUSD^(-0.576) × USDJPY^(0.136) × GBPUSD^(-0.119)
                       × USDCAD^(0.091) × USDSEK^(0.042) × USDCHF^(0.036)
    """
    aid = adapter._get_account_id()
    # DXY constituents + commodities + metals
    instruments = "EUR_USD,GBP_USD,USD_CAD,USD_SEK,USD_CHF,USD_JPY,BCO_USD,WTICO_USD,XAU_USD,XAG_USD"
    data = adapter._req("GET", f"/v3/accounts/{aid}/pricing?instruments={instruments}")
    prices = data.get("prices", [])

    result: dict[str, Any] = {}
    pair_mids: dict[str, float] = {}
    for p in prices:
        inst = p.get("instrument", "")
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        bid = float(bids[0].get("price", 0)) if bids else 0
        ask = float(asks[0].get("price", 0)) if asks else 0
        mid = (bid + ask) / 2 if (bid and ask) else 0

        if mid > 0:
            pair_mids[inst] = mid

        if inst == "EUR_USD" and mid > 0:
            result["eurusd"] = round(mid, 5)
        elif inst == "BCO_USD" and mid > 0:
            result["bco_usd"] = round(mid, 2)
        elif inst == "WTICO_USD" and mid > 0:
            result["wti_usd"] = round(mid, 2)
        elif inst == "XAU_USD" and mid > 0:
            result["xau_usd"] = round(mid, 2)
        elif inst == "XAG_USD" and mid > 0:
            result["xag_usd"] = round(mid, 2)

    # Compute real DXY from ICE formula
    eurusd = pair_mids.get("EUR_USD")
    usdjpy = pair_mids.get("USD_JPY")
    gbpusd = pair_mids.get("GBP_USD")
    usdcad = pair_mids.get("USD_CAD")
    usdsek = pair_mids.get("USD_SEK")
    usdchf = pair_mids.get("USD_CHF")
    if all(v and v > 0 for v in (eurusd, usdjpy, gbpusd, usdcad, usdsek, usdchf)):
        dxy = (50.14348112
               * (eurusd ** -0.576)
               * (usdjpy ** 0.136)
               * (gbpusd ** -0.119)
               * (usdcad ** 0.091)
               * (usdsek ** 0.042)
               * (usdchf ** 0.036))
        result["dxy"] = round(dxy, 2)

    return result


def _fetch_jpy_cross_bias(adapter: Any) -> dict[str, Any] | None:
    """Fetch EURJPY/GBPJPY/AUDJPY mids + % change over 4h & 24h via OANDA H1 bars.

    Used to disambiguate USD-strength from JPY-weakness. If all three JPY crosses
    are up alongside USDJPY, JPY weakness is the driver — BUY bias on USDJPY has
    more conviction. If USDJPY is up while JPY crosses are flat, it's a pure USD
    move and conviction is lower.
    """
    instruments = ["EUR_JPY", "GBP_JPY", "AUD_JPY"]
    out: dict[str, Any] = {}
    try:
        aid = adapter._get_account_id()
    except Exception:
        return None

    # Current mids (one pricing call, not three)
    current: dict[str, float] = {}
    try:
        inst_param = ",".join(instruments)
        data = adapter._req("GET", f"/v3/accounts/{aid}/pricing?instruments={inst_param}")
        for p in data.get("prices", []) or []:
            inst = p.get("instrument", "")
            bids = p.get("bids", [{}])
            asks = p.get("asks", [{}])
            bid = float(bids[0].get("price", 0)) if bids else 0
            ask = float(asks[0].get("price", 0)) if asks else 0
            if bid > 0 and ask > 0:
                current[inst] = (bid + ask) / 2.0
    except Exception:
        return None

    if not current:
        return None

    # H1 bars for 4h and 24h lookback change
    for inst in instruments:
        mid = current.get(inst)
        if not mid:
            continue
        entry: dict[str, Any] = {"mid": round(mid, 3)}
        try:
            # 25 H1 bars: current is incomplete — use iloc[-2] as 1h-ago reference.
            df = adapter.get_bars(inst.replace("_", ""), "H1", count=30)
            if df is not None and not df.empty and len(df) >= 5:
                # 4h change vs close of 4 completed bars ago
                if len(df) >= 5:
                    ref_4h = float(df.iloc[-5]["close"])
                    if ref_4h > 0:
                        entry["change_4h_pct"] = round((mid - ref_4h) / ref_4h * 100.0, 3)
                # 24h change vs close ~24 completed bars ago
                if len(df) >= 25:
                    ref_24h = float(df.iloc[-25]["close"])
                    if ref_24h > 0:
                        entry["change_24h_pct"] = round((mid - ref_24h) / ref_24h * 100.0, 3)
        except Exception:
            pass
        out[inst.lower()] = entry

    if not out:
        return None

    # Consensus: how many crosses are up over 4h?
    directions_4h = [v.get("change_4h_pct") for v in out.values() if isinstance(v, dict)]
    valid = [d for d in directions_4h if isinstance(d, (int, float))]
    if valid:
        ups = sum(1 for d in valid if d > 0.05)  # >5bps filter to ignore pure noise
        downs = sum(1 for d in valid if d < -0.05)
        if ups >= 2 and downs == 0:
            summary = "JPY weakness confirmed across crosses"
        elif downs >= 2 and ups == 0:
            summary = "JPY strength confirmed across crosses"
        elif ups >= 2 and downs >= 1:
            summary = "Mixed — JPY crosses diverging"
        elif ups == 0 and downs == 0:
            summary = "JPY crosses flat"
        else:
            summary = "Mixed"
        out["summary_4h"] = summary
        out["crosses_up_4h"] = ups
        out["crosses_down_4h"] = downs

    return out


def _fetch_us10y_yield() -> dict[str, Any] | None:
    """Fetch US 10-Year Treasury yield history from FRED (free, no API key).

    Uses the FRED observation endpoint for DGS10 series.
    Returns dict with value, 1d_change, 5d_change (in percentage points) or None on failure.
    """
    from urllib.request import Request, urlopen

    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2025-01-01"
        req = Request(url, headers={"User-Agent": "USDJPY-Assistant/1.0"})
        with urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
        # CSV format: DATE,DGS10\n2025-01-02,4.32\n...
        lines = raw.strip().split("\n")
        # Collect valid observations (most recent last)
        values: list[float] = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip() not in ("", ".", "DGS10"):
                try:
                    values.append(float(parts[1].strip()))
                except ValueError:
                    continue
        if not values:
            return None
        current = values[-1]
        one_day_change = round(current - values[-2], 3) if len(values) >= 2 else None
        five_day_change = round(current - values[-6], 3) if len(values) >= 6 else None
        return {
            "value": round(current, 2),
            "1d_change": one_day_change,
            "5d_change": five_day_change,
        }
    except Exception:
        pass
    return None


def _fetch_position_book_sentiment(adapter: Any, symbol: str) -> dict[str, Any] | None:
    """Fetch OANDA position book and compute net long/short sentiment.

    Position book shows % of traders with open positions at each price level.
    We aggregate to get overall long vs short ratio — OANDA's volume proxy.
    """
    try:
        book = adapter.get_position_book(symbol)
    except Exception:
        return None

    pb = book.get("positionBook", {})
    buckets = pb.get("buckets", [])
    book_price = float(pb.get("price", 0)) if pb.get("price") else None
    if not buckets:
        return None

    total_long = 0.0
    total_short = 0.0
    for b in buckets:
        total_long += float(b.get("longCountPercent", 0))
        total_short += float(b.get("shortCountPercent", 0))

    total = total_long + total_short
    if total <= 0:
        return None

    long_pct = round(total_long / total * 100, 1)
    short_pct = round(total_short / total * 100, 1)

    # Net sentiment
    if long_pct > short_pct + 5:
        bias = "net long"
    elif short_pct > long_pct + 5:
        bias = "net short"
    else:
        bias = "balanced"

    return {
        "long_pct": long_pct,
        "short_pct": short_pct,
        "bias": bias,
        "book_price": book_price,
    }


def _extract_order_book_clusters(
    buckets: list[dict],
    book_price: float,
    range_pips: float = 100,
    top_n: int = 5,
) -> dict[str, Any]:
    """Extract top buy/sell clusters from OANDA order book buckets within ±range_pips.

    OANDA buckets: {"price": "148.200", "longCountPercent": "0.0841", "shortCountPercent": "0.0312"}
    longCountPercent = pending buy orders at that level (support/demand).
    shortCountPercent = pending sell orders at that level (resistance/supply).
    """
    pip_size = 0.01  # USDJPY
    range_price = range_pips * pip_size
    lo = book_price - range_price
    hi = book_price + range_price

    nearby = []
    for b in buckets:
        try:
            price = float(b["price"])
        except (KeyError, ValueError, TypeError):
            continue
        if lo <= price <= hi:
            nearby.append({
                "price": price,
                "long_pct": float(b.get("longCountPercent", 0)),
                "short_pct": float(b.get("shortCountPercent", 0)),
            })

    buy_clusters = sorted(nearby, key=lambda x: x["long_pct"], reverse=True)[:top_n]
    sell_clusters = sorted(nearby, key=lambda x: x["short_pct"], reverse=True)[:top_n]

    # Nearest support (highest buy cluster below price) / resistance (highest sell cluster above price)
    below_buys = [b for b in buy_clusters if b["price"] < book_price]
    above_sells = [s for s in sell_clusters if s["price"] > book_price]
    nearest_support = below_buys[0]["price"] if below_buys else None
    nearest_resistance = above_sells[0]["price"] if above_sells else None

    return {
        "current_price": book_price,
        "buy_clusters": [{"price": b["price"], "pct": b["long_pct"]} for b in buy_clusters],
        "sell_clusters": [{"price": s["price"], "pct": s["short_pct"]} for s in sell_clusters],
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "nearest_support_distance_pips": round((book_price - nearest_support) / pip_size, 1) if nearest_support else None,
        "nearest_resistance_distance_pips": round((nearest_resistance - book_price) / pip_size, 1) if nearest_resistance else None,
    }


# ---------------------------------------------------------------------------
# Dashboard + runtime state (disk reads, no broker call)
# ---------------------------------------------------------------------------

def _read_dashboard_and_runtime(profile_name: str) -> dict[str, Any] | None:
    """Read dashboard_state.json and runtime_state.json from disk. Returns None if unavailable."""
    from core.dashboard_models import read_dashboard_state
    from core.execution_state import load_state

    log_dir = LOGS_DIR / profile_name
    if not log_dir.exists():
        return None

    dash = read_dashboard_state(log_dir)
    if dash is None:
        return None

    state_path = log_dir / "runtime_state.json"
    runtime = load_state(state_path)

    # Check staleness
    stale = False
    try:
        ts = datetime.fromisoformat(dash.get("timestamp_utc", "").replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        stale = age > 60
    except Exception:
        stale = True

    # Extract blocking filters (compact)
    filters_raw = dash.get("filters", [])
    blocking: list[dict[str, str]] = []
    clear_count = 0
    total_count = 0
    for f in filters_raw:
        if not f.get("enabled", True):
            continue
        total_count += 1
        if f.get("is_clear", True):
            clear_count += 1
        else:
            reason = f.get("block_reason") or f.get("explanation") or ""
            blocking.append({"name": f.get("display_name", f.get("filter_id", "?")), "reason": reason})

    # Daily summary from run loop
    daily = dash.get("daily_summary") or {}

    return {
        "preset_name": dash.get("preset_name", ""),
        "mode": dash.get("mode", ""),
        "loop_running": dash.get("loop_running", False),
        "kill_switch": getattr(runtime, "kill_switch", False),
        "exit_system_only": getattr(runtime, "exit_system_only", False),
        "entry_candidate_side": dash.get("entry_candidate_side"),
        "entry_candidate_trigger": dash.get("entry_candidate_trigger"),
        "blocking_filters": blocking[:8],
        "clear_count": clear_count,
        "total_count": total_count,
        "daily_summary": {
            "trades_today": daily.get("trades_today", 0),
            "wins": daily.get("wins", 0),
            "losses": daily.get("losses", 0),
            "total_pips": daily.get("total_pips", 0),
            "total_profit": daily.get("total_profit", 0),
            "win_rate": daily.get("win_rate", 0),
        } if daily else None,
        "stale": stale,
    }


# ---------------------------------------------------------------------------
# Technical analysis snapshot (3 key timeframes)
# ---------------------------------------------------------------------------

def _compute_adx(df: Any, period: int = 14) -> tuple[float | None, float | None]:
    """Compute ADX and ADXR from OHLC DataFrame using Wilder's smoothing.

    Returns (adx_value, adxr_value) or (None, None) if insufficient data.
    ADXR = (ADX_today + ADX_{period_bars_ago}) / 2.
    """
    import numpy as np

    if df is None or len(df) < period * 3:
        return None, None

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n = len(high)

    # True Range, +DM, -DM
    tr = np.empty(n)
    plus_dm = np.empty(n)
    minus_dm = np.empty(n)
    tr[0] = plus_dm[0] = minus_dm[0] = 0.0

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder's smoothing (period-bar EMA)
    def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.empty_like(arr)
        out[:p] = np.nan
        out[p] = np.sum(arr[1:p + 1])  # first value = sum of first p values
        for i in range(p + 1, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr_s = _wilder_smooth(tr, period)
    plus_di_s = _wilder_smooth(plus_dm, period)
    minus_di_s = _wilder_smooth(minus_dm, period)

    # DX series
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr_s[i] == 0:
            continue
        pdi = 100.0 * plus_di_s[i] / atr_s[i]
        mdi = 100.0 * minus_di_s[i] / atr_s[i]
        s = pdi + mdi
        dx[i] = 100.0 * abs(pdi - mdi) / s if s > 0 else 0.0

    # ADX = Wilder's smooth of DX
    # Find first valid DX window
    first_valid = period
    while first_valid < n and np.isnan(dx[first_valid]):
        first_valid += 1
    if first_valid + period >= n:
        return None, None

    adx = np.full(n, np.nan)
    adx[first_valid + period - 1] = np.nanmean(dx[first_valid:first_valid + period])
    for i in range(first_valid + period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    adx_val = float(adx[-1]) if not np.isnan(adx[-1]) else None

    # ADXR = (ADX_today + ADX_{period_ago}) / 2
    adxr_val = None
    if adx_val is not None and len(adx) > period and not np.isnan(adx[-1 - period]):
        adxr_val = round((adx[-1] + adx[-1 - period]) / 2.0, 1)

    if adx_val is not None:
        adx_val = round(adx_val, 1)

    return adx_val, adxr_val


def _compute_candle_streak(df: Any, lookback: int = 10) -> dict[str, Any] | None:
    """Count consecutive same-direction bars ending at the last completed bar.

    Returns {direction: 'up'|'down'|'doji', count: N, last_close: float} or None.
    """
    try:
        if df is None or df.empty or len(df) < 2:
            return None
        tail = df.tail(lookback)
        dirs: list[str] = []
        for _, row in tail.iterrows():
            o = float(row["open"])
            c = float(row["close"])
            if c > o:
                dirs.append("up")
            elif c < o:
                dirs.append("down")
            else:
                dirs.append("doji")
        if not dirs:
            return None
        last_dir = dirs[-1]
        if last_dir == "doji":
            return {"direction": "doji", "count": 1, "last_close": round(float(tail.iloc[-1]["close"]), 3)}
        streak = 0
        for d in reversed(dirs):
            if d == last_dir:
                streak += 1
            else:
                break
        return {
            "direction": last_dir,
            "count": streak,
            "last_close": round(float(tail.iloc[-1]["close"]), 3),
        }
    except Exception:
        return None


def _detect_bar_patterns(df: Any) -> list[str]:
    """Detect classic 1-3 bar reversal/continuation patterns at the last completed bar.

    Returns a list of pattern labels. Empty list if nothing clean fires.
    Patterns checked: bullish/bearish engulfing, hammer, shooting star, inside bar,
    three-bar reversal, doji.
    """
    patterns: list[str] = []
    try:
        if df is None or df.empty or len(df) < 3:
            return patterns
        tail = df.tail(5)
        rows = [(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
                for _, r in tail.iterrows()]
        if len(rows) < 3:
            return patterns
        o, h, l, c = rows[-1]
        po, ph, pl, pc = rows[-2]
        ppo, pph, ppl, ppc = rows[-3]
    except Exception:
        return patterns

    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return patterns
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    prev_body = abs(pc - po)
    is_bull = c > o
    is_bear = c < o
    prev_bull = pc > po
    prev_bear = pc < po

    # Doji (indecision) — tiny body relative to range
    if body <= 0.1 * rng:
        patterns.append("doji")

    # Bullish engulfing: prev bear, current bull, body engulfs prev body
    if prev_bear and is_bull and c >= po and o <= pc and body > prev_body:
        patterns.append("bullish_engulfing")

    # Bearish engulfing: prev bull, current bear, body engulfs prev body
    if prev_bull and is_bear and o >= pc and c <= po and body > prev_body:
        patterns.append("bearish_engulfing")

    # Hammer: small body in upper half, long lower wick (>=2x body)
    if body > 0 and lower_wick >= 2 * body and upper_wick <= body and min(o, c) > (l + 0.5 * rng):
        patterns.append("hammer")

    # Shooting star: small body in lower half, long upper wick (>=2x body)
    if body > 0 and upper_wick >= 2 * body and lower_wick <= body and max(o, c) < (l + 0.5 * rng):
        patterns.append("shooting_star")

    # Inside bar: current H<=prev H AND current L>=prev L
    if h <= ph and l >= pl:
        patterns.append("inside_bar")

    # Three-bar reversal (bull): two bears then a bull whose close exceeds prior bar's open
    if ppc < ppo and prev_bear and is_bull and c > po:
        patterns.append("three_bar_reversal_bull")
    # Three-bar reversal (bear): two bulls then a bear whose close breaks prior bar's open
    if ppc > ppo and prev_bull and is_bear and c < po:
        patterns.append("three_bar_reversal_bear")

    return patterns


def _compute_tf_consensus(ta: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    """Collapse multi-timeframe TA into alignment scores.

    Counts timeframe agreement on: regime direction, RSI zone, MACD direction.
    Returns {regime_score, rsi_score, macd_score, overall, dominant_direction,
    divergences[]} or None if no TA available.
    """
    if not isinstance(ta, dict) or not ta:
        return None
    tfs = ["H1", "M15", "M5", "M1"]
    regimes: dict[str, list[str]] = {"bull": [], "bear": [], "neutral": []}
    macds: dict[str, list[str]] = {"positive": [], "negative": [], "neutral": []}
    rsi_zones: dict[str, list[str]] = {"overbought": [], "oversold": [], "mid": []}

    for tf in tfs:
        t = ta.get(tf)
        if not isinstance(t, dict):
            continue
        regime = str(t.get("regime") or "").lower()
        if "bull" in regime:
            regimes["bull"].append(tf)
        elif "bear" in regime:
            regimes["bear"].append(tf)
        else:
            regimes["neutral"].append(tf)

        macd = str(t.get("macd_direction") or "").lower()
        if macd in macds:
            macds[macd].append(tf)

        zone = str(t.get("rsi_zone") or "").lower()
        if "overbought" in zone:
            rsi_zones["overbought"].append(tf)
        elif "oversold" in zone:
            rsi_zones["oversold"].append(tf)
        else:
            rsi_zones["mid"].append(tf)

    total = sum(len(v) for v in regimes.values())
    if total == 0:
        return None

    # Dominant direction via regime
    if len(regimes["bull"]) > len(regimes["bear"]) and len(regimes["bull"]) >= 3:
        dominant = "bullish_aligned"
    elif len(regimes["bear"]) > len(regimes["bull"]) and len(regimes["bear"]) >= 3:
        dominant = "bearish_aligned"
    elif len(regimes["bull"]) > len(regimes["bear"]):
        dominant = "bullish_leaning"
    elif len(regimes["bear"]) > len(regimes["bull"]):
        dominant = "bearish_leaning"
    else:
        dominant = "mixed"

    # Divergences: any TF disagreeing with dominant regime
    divergences: list[str] = []
    if dominant in ("bullish_aligned", "bullish_leaning"):
        for tf in regimes["bear"]:
            divergences.append(f"{tf} bear vs dominant bull")
    elif dominant in ("bearish_aligned", "bearish_leaning"):
        for tf in regimes["bull"]:
            divergences.append(f"{tf} bull vs dominant bear")

    # RSI extremes (conviction dampener)
    rsi_warn: list[str] = []
    if len(rsi_zones["overbought"]) >= 2:
        rsi_warn.append(f"{'/'.join(rsi_zones['overbought'])} overbought")
    if len(rsi_zones["oversold"]) >= 2:
        rsi_warn.append(f"{'/'.join(rsi_zones['oversold'])} oversold")

    return {
        "regime_bull_tfs": regimes["bull"],
        "regime_bear_tfs": regimes["bear"],
        "regime_neutral_tfs": regimes["neutral"],
        "macd_positive_tfs": macds["positive"],
        "macd_negative_tfs": macds["negative"],
        "dominant_direction": dominant,
        "divergences": divergences,
        "rsi_warnings": rsi_warn,
        "total_tfs": total,
    }


def _compute_ta_snapshot(adapter: Any, profile: ProfileV1) -> dict[str, dict[str, Any]] | None:
    """Compute TA for H1, M15, M5, M1. Returns {timeframe: {regime, rsi, ...}} or None."""
    from core.ta_analysis import compute_ta_for_tf

    result: dict[str, dict[str, Any]] = {}
    pip_size = float(profile.pip_size) if profile.pip_size else 0.01

    for tf in ("H1", "M15", "M5", "M1"):
        try:
            df = adapter.get_bars(profile.symbol, tf, count=700)
            if df is None or df.empty:
                continue
            ta = compute_ta_for_tf(profile, tf, df)

            # MACD direction from histogram
            macd_dir = "neutral"
            if ta.macd_hist is not None:
                macd_dir = "positive" if ta.macd_hist > 0 else "negative"

            atr_pips = round(ta.atr_value / pip_size, 1) if ta.atr_value else None

            # ADX / ADXR
            adx_val, adxr_val = _compute_adx(df, period=14)

            # Recent candle streak
            streak = _compute_candle_streak(df, lookback=10)

            # Bar patterns (only meaningful on M5/M15 — M1 too noisy, H1 too slow for scalp)
            patterns = _detect_bar_patterns(df) if tf in ("M5", "M15") else []

            result[tf] = {
                "regime": ta.regime,
                "rsi_value": round(ta.rsi_value, 1) if ta.rsi_value is not None else None,
                "rsi_zone": ta.rsi_zone,
                "macd_direction": macd_dir,
                "atr_pips": atr_pips,
                "atr_state": ta.atr_state,
                "adx": adx_val,
                "adxr": adxr_val,
                "price": ta.price,
                "summary": ta.summary,
                "streak": streak,
                "patterns": patterns,
            }
        except Exception:
            continue

    return result if result else None


# ---------------------------------------------------------------------------
# Session awareness (pure UTC clock math)
# ---------------------------------------------------------------------------

_SESSIONS = [
    ("Tokyo",  0, 9),
    ("London", 7, 16),
    ("NY",     12, 21),
]


def _compute_session_info(now_utc: datetime) -> dict[str, Any]:
    """Determine active sessions, overlaps, and proximity to low-liquidity windows."""
    h = now_utc.hour + now_utc.minute / 60.0
    active = [name for name, start, end in _SESSIONS if start <= h < end]

    # Detect overlaps
    overlap = None
    if "London" in active and "Tokyo" in active:
        overlap = "Tokyo/London overlap"
    elif "London" in active and "NY" in active:
        overlap = "London/NY overlap"

    # Time until next session close (for the latest active session)
    next_close = None
    next_close_name = None
    for name, _start, end in _SESSIONS:
        if name in active:
            remaining_h = end - h
            if remaining_h > 0 and (next_close is None or remaining_h < next_close):
                next_close = remaining_h
                next_close_name = name

    # Low-liquidity warning: 16:00-17:00 UTC (after London, before late NY) or 21:00+ (after NY)
    warnings: list[str] = []
    if 15.75 <= h < 16.0:
        warnings.append("Approaching 16:00 UTC low-liquidity window")
    elif 16.0 <= h < 17.0:
        warnings.append("In 16:00-17:00 UTC low-liquidity window")
    if 20.75 <= h < 21.0:
        warnings.append("NY close imminent")
    elif h >= 21.0 or h < 0.0:
        warnings.append("Post-NY close — thin liquidity")

    result: dict[str, Any] = {"active_sessions": active}
    if overlap:
        result["overlap"] = overlap
    if next_close is not None:
        hrs = int(next_close)
        mins = int((next_close - hrs) * 60)
        result["next_close"] = f"{next_close_name} close in {hrs}h {mins:02d}m"
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# Derived trade stats (today / this week)
# ---------------------------------------------------------------------------

def _compute_session_stats(closed_trades: list[dict], days_back: int = 14) -> dict[str, Any] | None:
    """Group closed trades by trading session (Tokyo / London / NY / Off-hours).

    Uses close_time UTC hour as a proxy for entry session — good enough for scalpers
    whose trades typically close within minutes of entry. Windows match _SESSIONS.
    """
    if not closed_trades:
        return None
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=max(1, int(days_back)))
    # Session hour-ranges (UTC). Overlaps are intentional — we bucket by *primary* session
    # using the same precedence as _SESSIONS: later-ranked wins for overlap bars.
    def _bucket(hour_utc: int) -> str:
        # Simple precedence: NY overlap wins over London, London overlap wins over Tokyo.
        if 12 <= hour_utc < 21:
            return "NY"
        if 7 <= hour_utc < 16:
            return "London"
        if 0 <= hour_utc < 9:
            return "Tokyo"
        return "Off-hours"

    buckets: dict[str, list[float]] = {"Tokyo": [], "London": [], "NY": [], "Off-hours": []}
    for t in closed_trades:
        ct_str = t.get("close_time", "")
        if not ct_str:
            continue
        try:
            ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if ct < cutoff:
            continue
        try:
            pnl = float(t.get("profit", 0) or 0)
        except Exception:
            continue
        buckets[_bucket(ct.hour)].append(pnl)

    out: dict[str, Any] = {}
    for name, pnls in buckets.items():
        if not pnls:
            continue
        wins = sum(1 for p in pnls if p > 0)
        out[name] = {
            "trades": len(pnls),
            "wins": wins,
            "losses": sum(1 for p in pnls if p < 0),
            "win_rate": round(wins / len(pnls) * 100, 1),
            "net_pl": round(sum(pnls), 2),
        }
    return out if out else None


def _compute_exit_strategy_performance(db_path: Any, days_back: int = 90) -> list[dict[str, Any]] | None:
    """Group closed AI-suggested trades by exit strategy and summarize.

    Returns list of {strategy, closed, win_rate_pct, avg_pips, avg_pnl, total_pnl}.
    None when the DB is missing or empty (no AI suggestions accrued yet).
    """
    try:
        from api import suggestion_tracker as _st
        from pathlib import Path as _P
        if not _P(str(db_path)).exists():
            return None
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days_back)))).isoformat()
        rows = _st._load_rows_since(_P(str(db_path)), cutoff_iso)
        if not rows:
            return None
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if not r.get("closed_at") or r.get("pnl") is None:
                continue
            placed = _st._json_obj(r.get("placed_order_json"))
            strat = str(placed.get("exit_strategy") or r.get("exit_strategy") or "unknown").strip() or "unknown"
            buckets.setdefault(strat, []).append(r)
        if not buckets:
            return None
        out: list[dict[str, Any]] = []
        for strat, group in buckets.items():
            pnls = [float(g.get("pnl") or 0) for g in group]
            pips = [float(g.get("pips") or 0) for g in group]
            wins = sum(1 for g in group if str(g.get("win_loss") or "") == "win")
            out.append({
                "strategy": strat,
                "closed": len(group),
                "wins": wins,
                "win_rate_pct": round(wins / len(group) * 100.0, 1) if group else 0.0,
                "avg_pips": round(sum(pips) / len(pips), 2) if pips else 0.0,
                "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
                "total_pnl": round(sum(pnls), 2),
            })
        out.sort(key=lambda r: r["closed"], reverse=True)
        return out
    except Exception:
        return None


def _get_latest_suggestion_for_chat(db_path: Any) -> dict[str, Any] | None:
    """Fetch the most recent Fillmore suggestion for injection into the chat system prompt.

    Returns a compact dict with the suggestion details + action/outcome, or None
    if no suggestions exist. This is a read-only bridge: chat can discuss what
    Fillmore suggested, but this data never flows back into the suggestion endpoint.
    """
    try:
        from api import suggestion_tracker as _st
        from pathlib import Path as _P

        p = _P(str(db_path))
        if not p.exists():
            return None
        hist = _st.get_history(p, limit=1, offset=0)
        items = hist.get("items") or []
        if not items:
            return None
        row = items[0]
        created = str(row.get("created_utc") or "")[:16]
        side = str(row.get("side") or "?").upper()
        price = row.get("limit_price")
        sl = row.get("sl")
        tp = row.get("tp")
        lots = row.get("lots")
        model = str(row.get("model") or "?")
        quality = str(row.get("quality") or row.get("confidence") or "?")
        rationale = str(row.get("rationale") or "")
        action = str(row.get("action") or "none")
        exit_strategy = str(
            (row.get("placed_order") or {}).get("exit_strategy")
            or row.get("exit_strategy")
            or ""
        )
        exit_params = row.get("exit_params") or {}

        # Outcome
        outcome = str(row.get("outcome_status") or "").strip()
        win_loss = str(row.get("win_loss") or "").strip()
        pnl = row.get("pnl")
        pips = row.get("pips")

        out: dict[str, Any] = {
            "suggestion_id": row.get("suggestion_id"),
            "created": created,
            "model": model,
            "side": side,
            "price": price,
            "sl": sl,
            "tp": tp,
            "lots": lots,
            "quality": quality,
            "rationale": rationale,
            "action": action,
            "exit_strategy": exit_strategy,
            "exit_params": exit_params,
        }
        # Edits
        edited = row.get("edited_fields") or {}
        if edited:
            out["edits"] = edited

        # Outcome details
        if outcome:
            out["outcome"] = outcome
        if win_loss:
            out["win_loss"] = win_loss
        if pnl is not None:
            out["pnl"] = pnl
        if pips is not None:
            out["pips"] = pips
        if row.get("oanda_order_id"):
            out["order_id"] = row["oanda_order_id"]
        return out
    except Exception:
        return None


def _compute_derived_stats(closed_trades: list[dict]) -> dict[str, Any]:
    """Compute today's and this week's stats from closed trade list.

    Each trade dict must have 'close_time' (ISO str) and 'profit' (float).
    """
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    # Week starts Monday 00:00 UTC
    week_start = today_start - timedelta(days=now_utc.weekday())

    month_start = today_start - timedelta(days=30)

    today_trades: list[dict] = []
    week_trades: list[dict] = []
    month_trades: list[dict] = []

    for t in closed_trades:
        ct_str = t.get("close_time", "")
        if not ct_str:
            continue
        try:
            ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if ct >= today_start:
            today_trades.append(t)
        if ct >= week_start:
            week_trades.append(t)
        if ct >= month_start:
            month_trades.append(t)

    def _summarize(trades: list[dict]) -> dict[str, Any] | None:
        if not trades:
            return None
        profits = [float(t.get("profit", 0)) for t in trades]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        total = len(profits)
        return {
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "net_pl": round(sum(profits), 2),
            "best": round(max(profits), 2) if profits else 0,
            "worst": round(min(profits), 2) if profits else 0,
        }

    # Consecutive losses (from most recent trade backward)
    consec_losses = 0
    last_loss_time = None
    for t in sorted(closed_trades, key=lambda x: x.get("close_time", ""), reverse=True):
        if float(t.get("profit", 0)) < 0:
            consec_losses += 1
            if last_loss_time is None:
                last_loss_time = t.get("close_time")
        else:
            break

    result: dict[str, Any] = {}
    today_summary = _summarize(today_trades)
    if today_summary:
        today_summary["consecutive_losses"] = consec_losses
        if last_loss_time:
            today_summary["last_loss_time"] = last_loss_time
        result["today"] = today_summary
    week_summary = _summarize(week_trades)
    if week_summary:
        result["week"] = week_summary
    month_summary = _summarize(month_trades)
    if month_summary:
        result["month"] = month_summary
    return result


# ---------------------------------------------------------------------------
# Guardrail alerts (conditional, based on derived stats + session)
# ---------------------------------------------------------------------------

def _compute_guardrail_alerts(
    derived_stats: dict[str, Any],
    session_info: dict[str, Any],
    open_positions: list[dict],
    dashboard: dict[str, Any] | None = None,
) -> list[str]:
    """Return list of active guardrail alert strings. Empty if nothing triggered."""
    alerts: list[str] = []
    now_utc = datetime.now(timezone.utc)

    # Dashboard: only surface kill switch — manual traders don't need bot ARMED/DISARMED/EXIT-ONLY noise.
    if dashboard:
        if dashboard.get("kill_switch"):
            alerts.append("KILL SWITCH ACTIVE — all trading halted")

    # Consecutive losses
    today = derived_stats.get("today", {})
    consec = today.get("consecutive_losses", 0)
    if consec >= 2:
        msg = f"{consec} consecutive losses — suggest 30 min pause"
        last_loss = today.get("last_loss_time")
        if last_loss:
            try:
                lt = datetime.fromisoformat(last_loss.replace("Z", "+00:00"))
                mins_ago = int((now_utc - lt).total_seconds() / 60)
                msg += f" (last loss {mins_ago} min ago)"
            except Exception:
                pass
        alerts.append(msg)

    # Session warnings (from session_info)
    for w in session_info.get("warnings", []):
        alerts.append(w)

    # DCA / concentration cap: count open positions by side in last 2 hours
    if open_positions:
        buys = sum(1 for p in open_positions if p.get("side") == "BUY")
        sells = sum(1 for p in open_positions if p.get("side") == "SELL")
        if buys >= 3:
            alerts.append(f"{buys} open BUY positions — check DCA/concentration limits")
        if sells >= 3:
            alerts.append(f"{sells} open SELL positions — check DCA/concentration limits")

    # Large drawdown warning
    week = derived_stats.get("week", {})
    week_pl = week.get("net_pl", 0)
    if week_pl < -500:
        alerts.append(f"Weekly P&L at ${week_pl:+.0f} — consider reducing size")

    return alerts


# ---------------------------------------------------------------------------
# Volatility indicator (M1 candle range)
# ---------------------------------------------------------------------------

def _compute_volatility(adapter: Any, symbol: str) -> dict[str, Any] | None:
    """Fetch last 100 M1 candles, compare recent 30-bar avg range to full baseline."""
    try:
        df = adapter.get_bars(symbol, "M1", count=100)
        if df is None or len(df) < 50:
            return None
        highs = df["high"].values
        lows = df["low"].values
        ranges = highs - lows

        recent_avg = float(ranges[-30:].mean())
        baseline_avg = float(ranges.mean())
        if baseline_avg <= 0:
            return None

        ratio = round(recent_avg / baseline_avg, 1)
        if ratio >= 1.5:
            label = "Elevated"
        elif ratio >= 1.2:
            label = "Above average"
        elif ratio <= 0.6:
            label = "Very low"
        elif ratio <= 0.8:
            label = "Below average"
        else:
            label = "Normal"

        return {
            "label": label,
            "ratio": ratio,
            "recent_avg_pips": round(recent_avg / 0.01, 1),  # USDJPY pip = 0.01
        }
    except Exception:
        return None


def _compute_cross_asset_bias(adapter: Any) -> dict[str, Any] | None:
    """Compute macro bias from oil and EUR/USD (DXY proxy) daily candle returns.

    Replicates CrossAssetDashboard logic using the adapter's get_bars().
    """
    def _direction(five_day_ret: float, threshold: float) -> str:
        if five_day_ret > threshold:
            return "bullish"
        elif five_day_ret < -threshold:
            return "bearish"
        return "neutral"

    result: dict[str, Any] = {}

    # Oil (Brent)
    try:
        oil_df = adapter.get_bars("BCO_USD", "D", count=25)
        if oil_df is not None and len(oil_df) >= 2:
            closes = oil_df["close"].values.astype(float)
            current = closes[-1]
            one_ago = closes[-2]
            ret_1d = (current - one_ago) / one_ago
            ret_5d = None
            ret_20d = None
            if len(closes) >= 6:
                five_ago = closes[-6]
                ret_5d = (current - five_ago) / five_ago
            if len(closes) >= 21:
                twenty_ago = closes[-21]
                ret_20d = (current - twenty_ago) / twenty_ago
            oil_dir = _direction(ret_5d or 0.0, 0.005)
            result["oil"] = {
                "direction": oil_dir,
                "1d_return": round(ret_1d * 100, 2),
                "5d_return": round(ret_5d * 100, 2) if ret_5d is not None else None,
                "20d_return": round(ret_20d * 100, 2) if ret_20d is not None else None,
                "price": round(current, 2),
            }
    except Exception:
        pass

    # Real DXY from constituent pairs
    try:
        # Fetch daily bars for all 6 DXY constituents
        dxy_pairs = {
            "EUR_USD": None, "USD_JPY": None, "GBP_USD": None,
            "USD_CAD": None, "USD_SEK": None, "USD_CHF": None,
        }
        for pair in dxy_pairs:
            try:
                pair_df = adapter.get_bars(pair, "D", count=25)
                if pair_df is not None and len(pair_df) >= 6:
                    dxy_pairs[pair] = pair_df["close"].values.astype(float)
            except Exception:
                pass

        # Compute DXY for current, 1-day-ago, and 5-day-ago if all pairs available
        if all(v is not None and len(v) >= 2 for v in dxy_pairs.values()):
            def _calc_dxy(idx: int) -> float:
                return (50.14348112
                        * (dxy_pairs["EUR_USD"][idx] ** -0.576)
                        * (dxy_pairs["USD_JPY"][idx] ** 0.136)
                        * (dxy_pairs["GBP_USD"][idx] ** -0.119)
                        * (dxy_pairs["USD_CAD"][idx] ** 0.091)
                        * (dxy_pairs["USD_SEK"][idx] ** 0.042)
                        * (dxy_pairs["USD_CHF"][idx] ** 0.036))

            dxy_now = _calc_dxy(-1)
            dxy_1ago = _calc_dxy(-2)
            dxy_ret_1d = (dxy_now - dxy_1ago) / dxy_1ago
            dxy_ret_5d = None
            dxy_ret_20d = None
            if all(len(v) >= 6 for v in dxy_pairs.values()):
                dxy_5ago = _calc_dxy(-6)
                dxy_ret_5d = (dxy_now - dxy_5ago) / dxy_5ago
            if all(len(v) >= 21 for v in dxy_pairs.values()):
                dxy_20ago = _calc_dxy(-21)
                dxy_ret_20d = (dxy_now - dxy_20ago) / dxy_20ago
            # DXY up = USD strong = bullish for USDJPY
            dxy_dir = _direction(dxy_ret_5d or 0.0, 0.003)
            result["dxy"] = {
                "direction": dxy_dir,
                "1d_return": round(dxy_ret_1d * 100, 2),
                "5d_return": round(dxy_ret_5d * 100, 2) if dxy_ret_5d is not None else None,
                "20d_return": round(dxy_ret_20d * 100, 2) if dxy_ret_20d is not None else None,
                "value": round(dxy_now, 2),
            }
    except Exception:
        pass

    # Gold (XAU/USD)
    try:
        gold_df = adapter.get_bars("XAU_USD", "D", count=25)
        if gold_df is not None and len(gold_df) >= 2:
            closes = gold_df["close"].values.astype(float)
            current = closes[-1]
            one_ago = closes[-2]
            ret_1d = (current - one_ago) / one_ago
            ret_5d = None
            if len(closes) >= 6:
                ret_5d = (current - closes[-6]) / closes[-6]
            result["gold"] = {
                "1d_return": round(ret_1d * 100, 2),
                "5d_return": round(ret_5d * 100, 2) if ret_5d is not None else None,
                "price": round(current, 2),
            }
    except Exception:
        pass

    # Combined bias
    oil_dir = result.get("oil", {}).get("direction", "neutral")
    dxy_dir = result.get("dxy", {}).get("direction", "neutral")

    if oil_dir == "bullish" and dxy_dir == "bullish":
        combined = "bullish"
        confidence = "high"
        implication = "Oil up + USD strong both support USDJPY longs"
    elif oil_dir == "bearish" and dxy_dir == "bearish":
        combined = "bearish"
        confidence = "high"
        implication = "Oil down + USD soft both support USDJPY shorts"
    elif oil_dir == "neutral" and dxy_dir == "neutral":
        combined = "neutral"
        confidence = "low"
        implication = "No clear macro direction — trade technicals only"
    elif oil_dir == "neutral" or dxy_dir == "neutral":
        active = oil_dir if oil_dir != "neutral" else dxy_dir
        combined = active
        confidence = "low"
        implication = f"Weak {active} bias — only one macro factor aligned"
    else:
        combined = "conflicting"
        confidence = "low"
        implication = f"Conflict: oil says {oil_dir}, DXY proxy says {dxy_dir} — reduce size or skip"

    result["combined_bias"] = combined
    result["confidence"] = confidence
    result["usdjpy_implication"] = implication

    return result if result else None


def _fetch_ohlc_history(adapter: Any, profile: ProfileV1) -> dict[str, list[dict[str, Any]]] | None:
    """Fetch compact OHLC history for prompt grounding.

    Keep this intentionally small for token efficiency; deeper history should use tools.
    """
    pip_size = float(profile.pip_size) if profile.pip_size else 0.01
    result: dict[str, list[dict[str, Any]]] = {}

    configs = [
        ("H1", 12),   # last 12 H1 bars
        ("M1", 30),   # last 30 M1 bars
    ]

    for tf, count in configs:
        try:
            df = adapter.get_bars(profile.symbol, tf, count=count)
            if df is None or df.empty:
                continue
            candles: list[dict[str, Any]] = []
            for idx, row in df.tail(count).iterrows():
                candle: dict[str, Any] = {
                    "o": round(float(row["open"]), 3),
                    "h": round(float(row["high"]), 3),
                    "l": round(float(row["low"]), 3),
                    "c": round(float(row["close"]), 3),
                }
                if "volume" in row.index and row["volume"] is not None:
                    try:
                        candle["v"] = int(row["volume"])
                    except (ValueError, TypeError):
                        pass
                if idx is not None:
                    candle["t"] = str(idx)
                candles.append(candle)
            if candles:
                result[tf] = candles
        except Exception:
            continue

    return result if result else None


def _fetch_price_structure_bars(adapter: Any, profile: ProfileV1) -> dict[str, Any] | None:
    """Fetch daily & weekly bars and compute intraday O/H/L, PDH/PDL, PWH/PWL.

    Round levels, in-range positioning, and pending-order distances are layered
    on in post-processing (they depend on spot_price / pending_orders in ctx).
    """
    pip_size = float(profile.pip_size) if profile.pip_size else 0.01
    out: dict[str, Any] = {}

    # Daily: pull 3 bars with incomplete=True so today's partial candle is included.
    try:
        df_d = adapter.get_bars(profile.symbol, "D", count=3, include_incomplete=True)
        if df_d is not None and not df_d.empty:
            today = df_d.iloc[-1]
            out["intraday"] = {
                "open": round(float(today["open"]), 3),
                "high": round(float(today["high"]), 3),
                "low": round(float(today["low"]), 3),
                "range_pips": round((float(today["high"]) - float(today["low"])) / pip_size, 1),
            }
            if len(df_d) >= 2:
                prev = df_d.iloc[-2]
                out["prev_day"] = {
                    "high": round(float(prev["high"]), 3),
                    "low": round(float(prev["low"]), 3),
                    "close": round(float(prev["close"]), 3),
                    "range_pips": round((float(prev["high"]) - float(prev["low"])) / pip_size, 1),
                }
    except Exception:
        pass

    # Weekly: current (incomplete) + previous.
    try:
        df_w = adapter.get_bars(profile.symbol, "W", count=2, include_incomplete=True)
        if df_w is not None and not df_w.empty:
            cur_w = df_w.iloc[-1]
            out["current_week"] = {
                "high": round(float(cur_w["high"]), 3),
                "low": round(float(cur_w["low"]), 3),
            }
            if len(df_w) >= 2:
                prev_w = df_w.iloc[-2]
                out["prev_week"] = {
                    "high": round(float(prev_w["high"]), 3),
                    "low": round(float(prev_w["low"]), 3),
                    "close": round(float(prev_w["close"]), 3),
                }
    except Exception:
        pass

    return out if out else None


def _enrich_price_structure(ps: dict[str, Any], mid: float | None, pip_size: float,
                            pending_orders: list[dict[str, Any]] | None) -> None:
    """Attach round levels, in-range position, key-level distances, and pending-order distances.

    Mutates `ps` in place.
    """
    if mid is None or mid <= 0:
        return

    def _dist(level: float) -> float:
        return round((level - mid) / pip_size, 1)

    # Position within intraday range
    intra = ps.get("intraday")
    if isinstance(intra, dict):
        hi = float(intra.get("high") or 0)
        lo = float(intra.get("low") or 0)
        rng = hi - lo
        if rng > 0:
            intra["pct_of_range"] = round((mid - lo) / rng * 100.0, 1)
        intra["pips_from_high"] = round((hi - mid) / pip_size, 1) if hi else None
        intra["pips_from_low"] = round((mid - lo) / pip_size, 1) if lo else None

    # Distances to key daily/weekly levels
    key_levels: list[tuple[str, float]] = []
    pd_blk = ps.get("prev_day")
    if isinstance(pd_blk, dict):
        key_levels.extend([("PDH", float(pd_blk["high"])), ("PDL", float(pd_blk["low"])),
                           ("PDC", float(pd_blk["close"]))])
    pw_blk = ps.get("prev_week")
    if isinstance(pw_blk, dict):
        key_levels.extend([("PWH", float(pw_blk["high"])), ("PWL", float(pw_blk["low"]))])
    cw_blk = ps.get("current_week")
    if isinstance(cw_blk, dict):
        key_levels.extend([("WH", float(cw_blk["high"])), ("WL", float(cw_blk["low"]))])
    if key_levels:
        ps["key_level_distances"] = [
            {"name": name, "price": round(px, 3), "distance_pips": _dist(px)}
            for name, px in key_levels
        ]

    # Round/psychological levels: nearest .00 and .50 above & below
    step = 0.50
    below_half = (int(mid / step)) * step
    above_half = below_half + step
    big_below = float(int(mid))
    big_above = big_below + 1.0
    candidates = [
        (big_above, "big_figure"),
        (above_half, "half_figure"),
        (below_half, "half_figure"),
        (big_below, "big_figure"),
    ]
    seen: set[str] = set()
    rounds: list[dict[str, Any]] = []
    for px, kind in candidates:
        name = f"{px:.2f}"
        if name in seen:
            continue
        seen.add(name)
        rounds.append({
            "name": name,
            "price": round(px, 3),
            "distance_pips": _dist(px),
            "type": kind,
        })
    rounds.sort(key=lambda r: abs(r["distance_pips"]))
    ps["round_levels"] = rounds[:4]

    # Pending order distances (pre-computed so Fillmore doesn't math it himself)
    if isinstance(pending_orders, list) and pending_orders:
        po_out: list[dict[str, Any]] = []
        for o in pending_orders:
            try:
                px = float(o.get("price") or 0)
                if px <= 0:
                    continue
                po_out.append({
                    "id": o.get("id"),
                    "side": o.get("side"),
                    "price": round(px, 3),
                    "distance_pips": _dist(px),
                    "lots": o.get("lots"),
                })
            except Exception:
                continue
        if po_out:
            ps["pending_order_distances"] = po_out


def build_trading_context(profile: ProfileV1, profile_name: str = "") -> dict[str, Any]:
    """Fetch live account state from the broker and return a plain dict.

    Runs synchronously (caller wraps in ThreadPoolExecutor).
    Uses internal thread pool to parallelize independent broker calls.
    Never includes raw tokens/secrets.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from adapters.broker import get_adapter

    adapter = get_adapter(profile)
    ctx: dict[str, Any] = {
        "symbol": profile.symbol,
        "broker_type": getattr(profile, "broker_type", "mt5"),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    is_oanda = getattr(profile, "broker_type", None) == "oanda"

    # --- Helper closures for parallel dispatch ---

    def _fetch_account() -> tuple[str, Any]:
        acct = adapter.get_account_info()
        return ("account", {
            "balance": float(getattr(acct, "balance", 0)),
            "equity": float(getattr(acct, "equity", 0)),
            "margin_used": float(getattr(acct, "margin", 0)),
            "margin_free": float(getattr(acct, "margin_free", 0)),
        })

    def _fetch_tick() -> tuple[str, Any]:
        tick = adapter.get_tick(profile.symbol)
        bid = float(getattr(tick, "bid", 0) or 0)
        ask = float(getattr(tick, "ask", 0) or 0)
        if bid > 0 and ask > 0:
            pip_size = float(getattr(profile, "pip_size", 0.01) or 0.01)
            spread_pips = ((ask - bid) / pip_size) if pip_size > 0 else None
            return ("spot_price", {
                "bid": round(bid, 3),
                "ask": round(ask, 3),
                "mid": round((bid + ask) / 2.0, 3),
                "spread_pips": round(spread_pips, 1) if spread_pips is not None else None,
            })
        return ("spot_price", None)

    def _fetch_positions() -> tuple[str, Any]:
        positions = adapter.get_open_positions(profile.symbol)
        open_list = []
        if positions:
            for pos in positions[:50]:
                if isinstance(pos, dict):
                    open_list.append({
                        "id": pos.get("id"),
                        "instrument": pos.get("instrument"),
                        "side": "BUY" if float(pos.get("currentUnits", pos.get("units", 0))) > 0 else "SELL",
                        "units": abs(float(pos.get("currentUnits", pos.get("units", 0)))),
                        "entry_price": pos.get("price"),
                        "unrealized_pl": pos.get("unrealizedPL"),
                    })
                else:
                    open_list.append({
                        "id": getattr(pos, "ticket", None),
                        "instrument": getattr(pos, "symbol", profile.symbol),
                        "side": "BUY" if getattr(pos, "type", 0) == 0 else "SELL",
                        "units": getattr(pos, "volume", 0),
                        "entry_price": getattr(pos, "price_open", None),
                        "unrealized_pl": getattr(pos, "profit", None),
                    })
        return ("open_positions", open_list)

    def _fetch_pending_orders() -> tuple[str, Any]:
        """Pending limit/stop orders resting at the broker (OANDA only)."""
        if not is_oanda or not hasattr(adapter, "list_pending_orders"):
            return ("pending_orders", [])
        try:
            raw = adapter.list_pending_orders(profile.symbol) or []
        except Exception:
            return ("pending_orders", [])
        out: list[dict[str, Any]] = []
        for o in raw[:20]:
            if not isinstance(o, dict):
                continue
            try:
                units_raw = float(o.get("units") or 0)
            except Exception:
                units_raw = 0.0
            side = "BUY" if units_raw > 0 else ("SELL" if units_raw < 0 else "?")
            lots = abs(units_raw) / 100_000.0 if units_raw else None
            sl_px = None
            tp_px = None
            sl = o.get("stopLossOnFill") or {}
            tp = o.get("takeProfitOnFill") or {}
            if isinstance(sl, dict):
                sl_px = sl.get("price")
            if isinstance(tp, dict):
                tp_px = tp.get("price")
            out.append({
                "id": o.get("id"),
                "type": o.get("type"),  # e.g. LIMIT, STOP
                "instrument": o.get("instrument"),
                "side": side,
                "units": abs(units_raw) if units_raw else None,
                "lots": round(lots, 2) if lots is not None else None,
                "price": o.get("price"),
                "sl": sl_px,
                "tp": tp_px,
                "time_in_force": o.get("timeInForce"),
                "gtd_time": o.get("gtdTime"),
                "create_time": o.get("createTime"),
                "state": o.get("state"),
            })
        return ("pending_orders", out)

    def _fetch_closed_trades() -> tuple[str, Any]:
        if is_oanda:
            closed = adapter.get_closed_trade_summaries(
                days_back=30, symbol=profile.symbol, pip_size=profile.pip_size,
            ) or []
            tagged_closed: list[str] = []
            for row in closed[:10]:
                if isinstance(row, dict):
                    src = _source_label_for_trade(row.get("entry_type"))
                    base = str(row.get("summary") or row.get("text") or row)
                    tagged_closed.append(f"{base} | [{src}]")
                else:
                    txt = str(row)
                    src = "bot" if classify_trade_source(txt) == "bot" else "manual"
                    tagged_closed.append(f"{txt} | [{src}]")
            result: dict[str, Any] = {"recent_closed_trades": tagged_closed}
            if closed:
                result["derived_stats"] = _compute_derived_stats(closed)
                session_stats = _compute_session_stats(closed, days_back=14)
                if session_stats:
                    result["session_stats"] = session_stats
            return ("closed_trades", result)
        else:
            report = adapter.get_mt5_report_stats(
                symbol=profile.symbol, pip_size=profile.pip_size, days_back=30,
            )
            return ("closed_trades", {"recent_trade_stats": {
                "closed_trades": getattr(report, "closed_trades", 0),
                "wins": getattr(report, "wins", 0),
                "losses": getattr(report, "losses", 0),
                "win_rate": getattr(report, "win_rate", 0),
                "total_profit": getattr(report, "total_profit", 0),
            }})

    def _fetch_cross_assets() -> tuple[str, Any]:
        cached = _ctx_cache.get("cross_assets")
        if cached is not None:
            return ("cross_assets", cached)
        result = _fetch_cross_asset_prices(adapter)
        # Also fetch US10Y from FRED (no adapter needed)
        try:
            us10y_data = _fetch_us10y_yield()
            if us10y_data is not None:
                result["us10y_yield"] = us10y_data["value"]
                result["us10y_data"] = us10y_data
        except Exception:
            pass
        _ctx_cache.set("cross_assets", result, 120)  # 2 min
        return ("cross_assets", result)

    def _fetch_macro_bias() -> tuple[str, Any]:
        cached = _ctx_cache.get("cross_asset_bias")
        if cached is not None:
            return ("cross_asset_bias", cached)
        bias = _compute_cross_asset_bias(adapter)
        if bias:
            _ctx_cache.set("cross_asset_bias", bias, 300)  # 5 min
        return ("cross_asset_bias", bias)

    def _fetch_order_book() -> tuple[str, Any]:
        cached = _ctx_cache.get("order_book")
        if cached is not None:
            return ("order_book", cached)
        book = adapter.get_order_book(profile.symbol)
        ob = book.get("orderBook", {})
        buckets = ob.get("buckets", [])
        book_price = float(ob.get("price", 0))
        if buckets and book_price > 0:
            result = _extract_order_book_clusters(
                buckets, book_price, range_pips=100, top_n=5,
            )
            _ctx_cache.set("order_book", result, 180)  # 3 min
            return ("order_book", result)
        return ("order_book", None)

    def _fetch_pos_book() -> tuple[str, Any]:
        cached = _ctx_cache.get("position_book")
        if cached is not None:
            return ("position_book", cached)
        sentiment = _fetch_position_book_sentiment(adapter, profile.symbol)
        if sentiment:
            _ctx_cache.set("position_book", sentiment, 180)  # 3 min
        return ("position_book", sentiment)

    def _fetch_vol() -> tuple[str, Any]:
        return ("volatility", _compute_volatility(adapter, profile.symbol))

    def _fetch_ta() -> tuple[str, Any]:
        return ("ta_snapshot", _compute_ta_snapshot(adapter, profile))

    def _fetch_ohlc() -> tuple[str, Any]:
        return ("ohlc_history", _fetch_ohlc_history(adapter, profile))

    def _fetch_price_struct() -> tuple[str, Any]:
        return ("price_structure", _fetch_price_structure_bars(adapter, profile))

    def _fetch_upcoming_events() -> tuple[str, Any]:
        """Next few USD/JPY high-impact events with countdowns. Cached 1 hour via _CALENDAR_CACHE."""
        try:
            events = get_economic_calendar_events(days_ahead=7, limit=3)
        except Exception:
            events = []
        return ("upcoming_events", events)

    def _fetch_jpy_crosses() -> tuple[str, Any]:
        """EURJPY/GBPJPY/AUDJPY bias for disambiguating USD vs JPY moves."""
        cached = _ctx_cache.get("jpy_crosses")
        if cached is not None:
            return ("jpy_crosses", cached)
        result = _fetch_jpy_cross_bias(adapter)
        if result:
            _ctx_cache.set("jpy_crosses", result, 120)  # 2 min
        return ("jpy_crosses", result)

    # --- Dispatch all calls in parallel ---

    try:
        adapter.initialize()
        if hasattr(adapter, "ensure_symbol"):
            adapter.ensure_symbol(profile.symbol)

        tasks = [_fetch_account, _fetch_tick, _fetch_positions, _fetch_closed_trades,
                 _fetch_vol, _fetch_ta, _fetch_ohlc, _fetch_price_struct, _fetch_upcoming_events]

        if is_oanda:
            tasks.extend([_fetch_cross_assets, _fetch_macro_bias, _fetch_order_book, _fetch_pos_book, _fetch_pending_orders, _fetch_jpy_crosses])

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): fn.__name__ for fn in tasks}
            for fut in as_completed(futures):
                try:
                    key, value = fut.result()
                    if value is not None:
                        if key == "closed_trades":
                            # Unpack multi-key result
                            ctx.update(value)
                        else:
                            ctx[key] = value
                except Exception:
                    pass

        # Post-process: position summary (needs open_positions)
        open_list = ctx.get("open_positions", [])
        if open_list:
            side_summary: dict[str, dict[str, Any]] = {}
            for side, side_key in (("BUY", "long"), ("SELL", "short")):
                side_rows = [p for p in open_list if str(p.get("side", "")).upper() == side]
                if len(side_rows) < 2:
                    continue
                total_units = 0.0
                weighted_entry = 0.0
                total_pl = 0.0
                for p in side_rows:
                    try:
                        units = abs(float(p.get("units", 0) or 0))
                    except Exception:
                        units = 0.0
                    try:
                        entry = float(p.get("entry_price", 0) or 0)
                    except Exception:
                        entry = 0.0
                    try:
                        upl = float(p.get("unrealized_pl", 0) or 0)
                    except Exception:
                        upl = 0.0
                    total_units += units
                    weighted_entry += (entry * units)
                    total_pl += upl
                if total_units <= 0:
                    continue
                avg_entry = weighted_entry / total_units
                side_summary[side_key] = {
                    "count": len(side_rows),
                    "total_units": round(total_units, 2),
                    "total_lots": round(total_units / 100_000.0, 4),
                    "avg_entry": round(avg_entry, 3),
                    "breakeven": round(avg_entry, 3),
                    "total_pl": round(total_pl, 2),
                }
            if side_summary:
                ctx["position_summary"] = side_summary

        # Post-process: enrich price structure with round levels + distances
        # (depends on spot_price + pending_orders fetched in parallel above).
        spot = ctx.get("spot_price") or {}
        mid = None
        try:
            mid = float(spot.get("mid")) if spot.get("mid") is not None else None
        except Exception:
            mid = None
        pip_size = float(profile.pip_size) if profile.pip_size else 0.01

        ps = ctx.get("price_structure")
        if isinstance(ps, dict):
            _enrich_price_structure(ps, mid, pip_size, ctx.get("pending_orders"))

        # Open P&L total — single-number drawdown/runner read.
        open_list = ctx.get("open_positions") or []
        if isinstance(open_list, list) and open_list:
            total_upl = 0.0
            count = 0
            for p in open_list:
                try:
                    total_upl += float(p.get("unrealized_pl", 0) or 0)
                    count += 1
                except Exception:
                    pass
            if count:
                ctx["open_pl_summary"] = {
                    "count": count,
                    "unrealized_pl_usd": round(total_upl, 2),
                }

        # Multi-timeframe consensus (derived from ta_snapshot, no API call).
        ta_data = ctx.get("ta_snapshot")
        if isinstance(ta_data, dict):
            consensus = _compute_tf_consensus(ta_data)
            if consensus:
                ctx["tf_consensus"] = consensus

        # Pip value in USD per lot at current price (USDJPY specific math).
        # pip_value_per_lot_usd = (pip_size * 100000) / mid   =>  1000 / mid at pip_size 0.01
        if mid and mid > 0:
            try:
                pip_value_per_lot_usd = (pip_size * 100_000.0) / mid
                ctx["pip_value_usd"] = {
                    "per_lot_usd": round(pip_value_per_lot_usd, 2),
                    "per_0_05_lot_usd": round(pip_value_per_lot_usd * 0.05, 2),
                    "per_0_10_lot_usd": round(pip_value_per_lot_usd * 0.10, 2),
                }
            except Exception:
                pass

    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass

    # Dashboard + runtime state (disk reads, no broker call)
    dashboard: dict[str, Any] | None = None
    if profile_name:
        try:
            dashboard = _read_dashboard_and_runtime(profile_name)
            if dashboard:
                ctx["dashboard"] = dashboard
        except Exception:
            pass

        # Exit-strategy performance from per-profile suggestion tracker DB (best-effort).
        try:
            suggestions_db = LOGS_DIR / profile_name / "ai_suggestions.sqlite"
            perf = _compute_exit_strategy_performance(suggestions_db, days_back=90)
            if perf:
                ctx["exit_strategy_performance"] = perf
        except Exception:
            pass

        # Latest Fillmore suggestion — injected into the chat system prompt so
        # Fillmore can discuss its own ideas (one-way: chat reads suggestions,
        # suggestions never read chat).
        try:
            suggestions_db = LOGS_DIR / profile_name / "ai_suggestions.sqlite"
            latest = _get_latest_suggestion_for_chat(suggestions_db)
            if latest:
                ctx["latest_suggestion"] = latest
        except Exception:
            pass

    # Session awareness (no API call needed)
    now_utc = datetime.now(timezone.utc)
    ctx["session"] = _compute_session_info(now_utc)

    # Guardrail alerts (conditional)
    derived = ctx.get("derived_stats", {})
    session = ctx.get("session", {})
    open_pos = ctx.get("open_positions", [])
    alerts = _compute_guardrail_alerts(derived, session, open_pos, dashboard)
    if alerts:
        ctx["guardrail_alerts"] = alerts

    return ctx


# ---------------------------------------------------------------------------
# 2. System prompt
# ---------------------------------------------------------------------------

def system_prompt_from_context(ctx: dict[str, Any], effective_model: str) -> str:
    """Build a system prompt that grounds the assistant in live trading data."""
    lines = [
        # ---- Identity & persona ----
        "You are FILLMORE — an elite USDJPY trading assistant. Think Cornelius Fillmore: sharp, streetwise, 130+ IQ, reformed operator now on the right side of the tape. Your mission is simple — help the trader Fill More Banks.",
        "You are not a chatbot printing data. You are a desk partner with a point of view. Have personality. Be direct, confident, dry-witted, never saccharine and never robotic.",
        "Voice: Cool operator in a good suit. Short sentences. Street-smart phrasing mixed with pro trading vocabulary. Occasional dry quip when it lands — never forced, never cringy, never more than one a response.",
        "Signature vibes (use sparingly, don't overdo): 'Locked in.', 'That's the read.', 'Tape's telling us...', 'Don't fight the flow.', 'We're stacking pips, not hopes.' Never sign every message with a catchphrase — personality shows up in rhythm, not repetition.",
        "Never roleplay as a cartoon. Never narrate your own persona. Never say things like 'as Fillmore' or 'my IQ tells me'. The personality IS the voice — don't describe it.",
        "",
        "# Professional scope",
        "You serve a professional manual USDJPY scalper with a proven profitable track record.",
        "Your job is to make them faster and more informed — not to tell them what to do. Do not question their edge.",
        "Be concise but not terse — match length to what the question actually needs. Lead with numbers. Never give imperative trade instructions like 'buy now' or 'sell now'.",
        "You may discuss market context, account state, position sizing, and risk management.",
        f"MODEL IDENTITY: You are '{effective_model}'. If asked what model, say exactly that. If asked who you are, you're Fillmore.",
        "CAPABILITIES: You have live broker context AND function-calling tools. Available tools:",
        "  - get_candles(timeframe, count): Fetch OHLC candles for any timeframe (M1/M5/M15/H1/H4/D)",
        "  - get_trade_history(days_back, limit): Query closed trades from database",
        "  - get_ai_suggestion_history(days_back, limit, suggestion_id, oanda_order_id, action, outcome_status): Query Fillmore's AI suggestion history, including edited fields, market context, placed orders, and outcomes",
        "  - get_fillmore_system_info(topic, include_runtime): Explain how Fillmore trade suggestions, autonomous mode, and learning/history work in this build",
        "  - analyze_trade_patterns(days_back): Win/loss stats by session, tier, entry type",
        "  - get_cross_asset_bias(): Full macro bias reading (oil, DXY, combined USDJPY implication)",
        "  - get_economic_calendar(days_ahead): Upcoming high-impact USD/JPY events (FOMC, NFP, BOJ, CPI)",
        "  - get_news_headlines(count): Recent USDJPY/forex news from RSS feeds",
        "  - web_search(query, count): Full web search via Brave Search — use for current events, analysis, central bank statements, geopolitical context, or any question needing live web data",
        "Use tools proactively when the user's question would benefit from fresh data. Prefer web_search for broad questions about markets or events. Use get_news_headlines for quick headline scans. Use get_economic_calendar for upcoming events, get_trade_history for specific past trades, and get_ai_suggestion_history for Fillmore-generated idea history.",
        "When the user asks how Fillmore's trade suggestion flow, autonomous mode, learning memory, or suggestion tracking works, call get_fillmore_system_info before answering so you describe the real implementation instead of hand-waving.",
        "Never mention training-data cutoff dates or generic model limitations.",
        "You may answer general knowledge questions using web_search if needed. You are not limited to trading topics.",
        "",
        "SCOPE: The data below is a broker snapshot only. Recent closed trades are from roughly the last 30 days (capped in the prompt).",
        "If asked about the trader's style: they trade three modes — post-impulse range fades, proven organic range bounces, and spike mean reversion.",
        "Typical targets 4-10 pips, stops 10-15 pips, high win-rate style. They size large at strong levels and DCA within ranges.",
        "TODAY / THIS WEEK / MONTH stats (if present) are computed from the full 30-day closed-trade fetch from the broker, not from the short trade sample lines below.",
        "THIS WEEK means Monday 00:00 UTC through now (UTC), not the user's local calendar week unless they ask.",
        "The Logs & Stats equity curve in the app may use a longer window (e.g. 365 days) — do not claim totals or win rates for the whole chart unless you only summarize the trades actually listed below.",
        "Do not invent weekly P/L or percentages; use the precomputed THIS WEEK / TODAY lines when present, else derive only from listed trades.",
        "",
        "DELIVERY (desk style): Scale length to the question. Quick state checks (price, P&L, 'am I flat?') → ~40-80 words. Standard reads (setup analysis, level reasoning, what's moving) → ~120-220 words. Deeper asks (multi-part, trade review, scenario walkthroughs, 'walk me through...') → go as long as the question genuinely requires — 300+ words is fine when the substance is there. Format with short paragraphs or bullets when it aids readability. Personality shows up in *how* you phrase the numbers, not in extra words — never pad, but never amputate a real answer to hit a word count either.",
        "Never say 'I can't give financial advice', 'consult a professional', 'this is not financial advice', or similar disclaimers.",
        "Never say 'based on the data provided' or 'according to my context' — answer directly.",
        "When the user mentions adding to a position, scaling in, or averaging — proactively include: margin cost of the addition, new total margin used, and updated average entry price / breakeven. Do not wait to be asked.",
        "When the user discusses directional positioning (adding longs, going short, etc.), reference the MACRO BIAS section if it confirms or contradicts their direction in one sentence.",
        "TRADE SOURCE LABELS: Trades have a 'source' field — either 'manual' or a bot strategy label. ALWAYS break down performance by source.",
        "Never blend manual and bot stats into one number without labeling which is which. Lead with manual stats, then show bot stats separately.",
        "When discussing bot/automated trades, use plain language labels (e.g., 'bot momentum trades', 'automated NY entries'), not raw internal strategy codes.",
        "Never use numbered sections (1) 2)) or Roman numerals. No essay structure.",
        "Never end with offers to recalculate, clarify, or do more — answer once and stop.",
        "Never summarize what you just said at the end of a response.",
        "When no positions are open AND the user is just asking about current state (am I flat, any positions, etc.), keep it tight — a sentence or two. Setup/analysis/market questions still get the full read regardless of position status.",
        "When positions ARE open, always lead with position status and P&L before anything else.",
        "Never hedge with 'if your definition differs', 'assumption:', or 'say so and I'll recalc' — state facts using the definitions below.",
        "",
        "SIZING DEFAULTS (USDJPY desk math; use unless broker context clearly contradicts):",
        "1 standard lot = 100,000 units. Use ~$3,000 margin per lot and ~$6.30 pip value per pip per lot at rates near 158–160.",
        "Do not ask the user to confirm lot size or margin definition.",
        "",
        "PRICE: LIVE PRICE mid in context is authoritative. If the user quotes a different price, one short sentence that price moved, then compute from LIVE PRICE — do not silently substitute.",
        "",
        "Ignore automation/bot run state (ARMED/DISARMED/EXIT-ONLY/loop/preset/filters) — the trader is manual; do not mention it unless kill switch appears in ACTIVE GUARDRAILS.",
        "",
        "=== LIVE TRADING CONTEXT ===",
        f"Symbol: {ctx.get('symbol', 'USDJPY')}",
        f"Broker: {ctx.get('broker_type', 'unknown')}",
        f"As of: {ctx.get('as_of', 'unknown')}",
    ]

    acct = ctx.get("account")
    if acct:
        lines.append("")
        lines.append("Account:")
        lines.append(f"  Balance: {acct.get('balance', '?')}")
        lines.append(f"  Equity: {acct.get('equity', '?')}")
        lines.append(f"  Margin Used: {acct.get('margin_used', '?')}")
        lines.append(f"  Margin Free: {acct.get('margin_free', '?')}")

    positions = ctx.get("open_positions")
    if positions:
        lines.append("")
        lines.append(f"Open Positions ({len(positions)}):")
        for p in positions:
            lines.append(
                f"  {p.get('side')} {p.get('units')} @ {p.get('entry_price')} "
                f"(PL: {p.get('unrealized_pl', '?')})"
            )
    elif positions is not None:
        lines.append("")
        lines.append("Open Positions: none")

    pos_summary = ctx.get("position_summary")
    if isinstance(pos_summary, dict):
        rendered = False
        for side_key, label in (("long", "longs"), ("short", "shorts")):
            s = pos_summary.get(side_key)
            if not isinstance(s, dict):
                continue
            rendered = True
            lines.append("")
            if side_key == "long":
                lines.append("POSITION SUMMARY:")
            lines.append(
                f"  {s.get('count', 0)} open {label} | Total: {s.get('total_units', 0):,} units "
                f"({s.get('total_lots', 0)} lots) | Avg entry: {s.get('avg_entry', '?')} | "
                f"Aggregate P&L: ${s.get('total_pl', 0):+,.2f} | Breakeven: {s.get('breakeven', '?')}"
            )
        if rendered:
            lines.append("  Use this summary when evaluating adds/scales and new breakeven.")

    pending = ctx.get("pending_orders")
    if isinstance(pending, list) and pending:
        lines.append("")
        lines.append(f"Pending Orders ({len(pending)}):")
        for o in pending:
            tif = o.get("time_in_force") or "?"
            gtd = o.get("gtd_time")
            tif_disp = f"{tif}" + (f" until {gtd}" if gtd else "")
            sl = o.get("sl")
            tp = o.get("tp")
            sl_tp_parts = []
            if sl is not None:
                sl_tp_parts.append(f"SL {sl}")
            if tp is not None:
                sl_tp_parts.append(f"TP {tp}")
            sl_tp = f" [{' | '.join(sl_tp_parts)}]" if sl_tp_parts else ""
            lots_disp = o.get("lots")
            lots_txt = f"{lots_disp} lots" if lots_disp is not None else f"{o.get('units', '?')} units"
            lines.append(
                f"  #{o.get('id')} {o.get('type', '?')} {o.get('side')} {lots_txt} @ {o.get('price')}"
                f"{sl_tp} ({tif_disp})"
            )
    elif isinstance(pending, list):
        lines.append("")
        lines.append("Pending Orders: none")

    closed = ctx.get("recent_closed_trades")
    if closed:
        lines.append("")
        shown = min(len(closed), 10)
        lines.append(f"Recent Closed Trades (last 30 days, showing {shown}):")
        for t in closed[:10]:
            lines.append(f"  {t}")

    stats = ctx.get("recent_trade_stats")
    if stats:
        lines.append("")
        lines.append("Recent Trade Stats (30d):")
        for k, v in stats.items():
            lines.append(f"  {k}: {v}")

    # Session info
    session = ctx.get("session")
    if session:
        parts = []
        active = session.get("active_sessions", [])
        overlap = session.get("overlap")
        if overlap:
            parts.append(overlap)
        elif active:
            parts.append(" + ".join(active))
        else:
            parts.append("No major session active")
        nc = session.get("next_close")
        if nc:
            parts.append(nc)
        lines.append("")
        line = f"SESSION: {' | '.join(parts)}"
        for w in session.get("warnings", []):
            line += f" | {w}"
        lines.append(line)

    # Upcoming high-impact economic events (with countdowns)
    events = ctx.get("upcoming_events")
    if isinstance(events, list) and events:
        # Imminent-event banner: any event <=30 min away gets a loud warning line up top.
        imminent = [e for e in events if isinstance(e.get("minutes_to_event"), int) and e["minutes_to_event"] <= 30 and e["minutes_to_event"] >= 0]
        if imminent:
            lines.append("")
            for e in imminent:
                mins = e["minutes_to_event"]
                lines.append(
                    f"*** EVENT IMMINENT: {e.get('event', '?')} ({e.get('currency', '?')}, "
                    f"{e.get('impact', '?')}) in {mins}m — spreads can widen, factor into entries ***"
                )
        lines.append("")
        lines.append("UPCOMING EVENTS (USD/JPY, high-impact):")
        for e in events[:3]:
            mins = e.get("minutes_to_event")
            if isinstance(mins, int):
                if mins < 60:
                    when = f"in {mins}m"
                elif mins < 1440:
                    when = f"in {mins // 60}h {mins % 60}m"
                else:
                    days = mins // 1440
                    rem_hours = (mins % 1440) // 60
                    when = f"in {days}d {rem_hours}h"
            else:
                when = f"{e.get('date', '?')} {e.get('time', '?')} UTC"
            lines.append(
                f"  {e.get('event', '?')} ({e.get('currency', '?')}, {e.get('impact', '?')}) — {when}"
            )

    spot = ctx.get("spot_price")
    if spot:
        lines.append("")
        spread = spot.get("spread_pips")
        spread_txt = f" | spread {spread}p" if spread is not None else ""
        lines.append(
            f"LIVE PRICE ({ctx.get('symbol', 'USDJPY')}): {spot['bid']:.3f}/{spot['ask']:.3f} "
            f"(mid {spot['mid']:.3f}){spread_txt}"
        )

    # Price structure: intraday O/H/L, PDH/PDL, weekly H/L, round levels, pending distances
    ps = ctx.get("price_structure")
    if isinstance(ps, dict) and ps:
        lines.append("")
        lines.append("PRICE STRUCTURE:")
        intra = ps.get("intraday")
        if isinstance(intra, dict):
            pct = intra.get("pct_of_range")
            pct_txt = f" | {pct:.0f}% of range" if pct is not None else ""
            from_hi = intra.get("pips_from_high")
            from_lo = intra.get("pips_from_low")
            edge_txt = ""
            if from_hi is not None and from_lo is not None:
                edge_txt = f" | {from_hi}p from H, {from_lo}p from L"
            lines.append(
                f"  Today: O {intra.get('open')} H {intra.get('high')} L {intra.get('low')} "
                f"(range {intra.get('range_pips')}p){pct_txt}{edge_txt}"
            )
        pd_blk = ps.get("prev_day")
        if isinstance(pd_blk, dict):
            lines.append(
                f"  Prev Day: H {pd_blk.get('high')} L {pd_blk.get('low')} "
                f"C {pd_blk.get('close')} (range {pd_blk.get('range_pips')}p)"
            )
        cw_blk = ps.get("current_week")
        if isinstance(cw_blk, dict):
            lines.append(f"  This Week: H {cw_blk.get('high')} L {cw_blk.get('low')}")
        pw_blk = ps.get("prev_week")
        if isinstance(pw_blk, dict):
            lines.append(
                f"  Prev Week: H {pw_blk.get('high')} L {pw_blk.get('low')} C {pw_blk.get('close')}"
            )
        kld = ps.get("key_level_distances")
        if isinstance(kld, list) and kld:
            parts = [f"{k['name']} {k['price']} ({k['distance_pips']:+}p)" for k in kld]
            lines.append("  Key levels: " + " | ".join(parts))
        rounds = ps.get("round_levels")
        if isinstance(rounds, list) and rounds:
            parts = [f"{r['name']} ({r['distance_pips']:+}p)" for r in rounds]
            lines.append("  Round levels nearby: " + " | ".join(parts))
        pod = ps.get("pending_order_distances")
        if isinstance(pod, list) and pod:
            parts = [
                f"#{p.get('id')} {p.get('side')} @ {p.get('price')} ({p.get('distance_pips'):+}p)"
                for p in pod
            ]
            lines.append("  Pending fills: " + " | ".join(parts))

    # Today's / this week's derived stats
    derived = ctx.get("derived_stats", {})
    today_s = derived.get("today")
    if today_s:
        lines.append("")
        lines.append("TODAY'S STATS:")
        lines.append(
            f"  Trades: {today_s['trades']} | Wins: {today_s['wins']} | "
            f"Losses: {today_s['losses']} | Win Rate: {today_s['win_rate']}%"
        )
        lines.append(
            f"  Net P&L: ${today_s['net_pl']:+.2f} | "
            f"Best: ${today_s['best']:+.2f} | Worst: ${today_s['worst']:+.2f}"
        )
        lines.append(f"  Consecutive losses: {today_s.get('consecutive_losses', 0)}")
    week_s = derived.get("week")
    if week_s:
        lines.append("")
        lines.append("THIS WEEK (Mon 00:00 UTC → now, all closed trades in 30d fetch):")
        lines.append(
            f"  Net P&L: ${week_s['net_pl']:+.2f} | Win Rate: {week_s['win_rate']}% "
            f"({week_s['trades']} trades)"
        )

    month_s = derived.get("month")
    if month_s:
        lines.append("")
        lines.append("THIS MONTH (30d):")
        lines.append(
            f"  Net P&L: ${month_s['net_pl']:+.2f} | Win Rate: {month_s['win_rate']}% "
            f"({month_s['trades']} trades)"
        )

    # Open P&L summary — current exposure at a glance.
    open_pl = ctx.get("open_pl_summary")
    if isinstance(open_pl, dict) and open_pl.get("count"):
        lines.append("")
        lines.append(
            f"OPEN P&L: {open_pl.get('count')} position(s) | "
            f"Unrealized: ${open_pl.get('unrealized_pl_usd', 0):+,.2f}"
        )

    # Pip value in USD (at current mid).
    pvu = ctx.get("pip_value_usd")
    if isinstance(pvu, dict) and pvu.get("per_lot_usd"):
        lines.append("")
        lines.append(
            f"PIP VALUE (at current price): ${pvu['per_lot_usd']:.2f}/lot | "
            f"${pvu.get('per_0_05_lot_usd', 0):.2f} at 0.05 lots | "
            f"${pvu.get('per_0_10_lot_usd', 0):.2f} at 0.10 lots"
        )

    # Session win rates — where his edge lives by time of day.
    sess_stats = ctx.get("session_stats")
    if isinstance(sess_stats, dict) and sess_stats:
        lines.append("")
        lines.append("SESSION PERFORMANCE (last 14d, by close-time UTC):")
        for name in ("Tokyo", "London", "NY", "Off-hours"):
            s = sess_stats.get(name)
            if not isinstance(s, dict):
                continue
            lines.append(
                f"  {name}: {s.get('trades')} trades | {s.get('win_rate')}% wins | "
                f"Net ${s.get('net_pl', 0):+,.2f}"
            )

    # Exit-strategy performance (AI-suggested trades only, from suggestion_tracker).
    exit_perf = ctx.get("exit_strategy_performance")
    if isinstance(exit_perf, list) and exit_perf:
        lines.append("")
        lines.append("EXIT STRATEGY PERFORMANCE (AI suggestions, last 90d, closed only):")
        for e in exit_perf:
            lines.append(
                f"  {e.get('strategy')}: {e.get('closed')} closed | "
                f"{e.get('win_rate_pct')}% wins | avg {e.get('avg_pips'):+.1f}p "
                f"| net ${e.get('total_pnl', 0):+,.2f}"
            )

    # Latest Fillmore suggestion — gives chat Fillmore awareness of its own ideas.
    latest_sug = ctx.get("latest_suggestion")
    if isinstance(latest_sug, dict):
        lines.append("")
        lines.append("YOUR MOST RECENT SUGGESTION (Fillmore-generated idea the trader saw):")
        side = latest_sug.get("side", "?")
        price = latest_sug.get("price")
        sl = latest_sug.get("sl")
        tp = latest_sug.get("tp")
        lots = latest_sug.get("lots")
        quality = latest_sug.get("quality") or latest_sug.get("confidence") or "?"
        lines.append(
            f"  {side} @ {price} | SL {sl} | TP {tp} | {lots} lots | quality={quality} "
            f"| model={latest_sug.get('model', '?')} | {latest_sug.get('created', '?')}"
        )
        exit_strat = latest_sug.get("exit_strategy")
        if exit_strat:
            ep = latest_sug.get("exit_params") or {}
            ep_str = ", ".join(f"{k}={v}" for k, v in ep.items()) if ep else "defaults"
            lines.append(f"  Exit strategy: {exit_strat} ({ep_str})")
        rationale = latest_sug.get("rationale", "")
        if rationale:
            lines.append(f"  Rationale: {rationale[:300]}")
        action = latest_sug.get("action", "none")
        edits = latest_sug.get("edits") or {}
        if action == "placed":
            edit_note = ""
            if edits:
                edit_parts = [f"{k}: {v.get('before')}→{v.get('after')}" for k, v in edits.items()]
                edit_note = f" (edited: {', '.join(edit_parts[:5])})"
            lines.append(f"  Action: PLACED{edit_note}")
            oid = latest_sug.get("order_id")
            if oid:
                lines.append(f"  Order ID: {oid}")
        elif action == "rejected":
            edit_note = ""
            if edits:
                edit_parts = [f"{k}: {v.get('before')}→{v.get('after')}" for k, v in edits.items()]
                edit_note = f" (fields viewed/edited before reject: {', '.join(edit_parts[:5])})"
            lines.append(f"  Action: REJECTED{edit_note}")
        else:
            lines.append("  Action: pending (trader hasn't acted yet)")
        outcome = latest_sug.get("outcome")
        wl = latest_sug.get("win_loss")
        pnl = latest_sug.get("pnl")
        pips = latest_sug.get("pips")
        if wl:
            result_str = f"{wl.upper()}"
            if pips is not None:
                result_str += f" {pips:+.1f}p"
            if pnl is not None:
                result_str += f" (${pnl:+.2f})"
            lines.append(f"  Outcome: {result_str}")
        elif outcome:
            lines.append(f"  Outcome: {outcome}")
        lines.append(
            "  ^ This is YOUR idea. If the trader asks about it, you remember the reasoning. "
            "Reference it naturally when relevant — don't volunteer it unprompted on every message."
        )

    # Guardrail alerts (only if triggered)
    alerts = ctx.get("guardrail_alerts")
    if alerts:
        lines.append("")
        lines.append("ACTIVE GUARDRAILS:")
        for a in alerts:
            lines.append(f"  * {a}")

    # Volatility
    vol = ctx.get("volatility")
    if vol:
        lines.append("")
        desc = vol["label"]
        ratio = vol["ratio"]
        pips = vol["recent_avg_pips"]
        extra = ""
        if ratio != 1.0:
            extra = f" ({ratio}x normal)"
        lines.append(f"VOLATILITY: {desc}{extra} — recent M1 avg range {pips}p")

    # Technical analysis snapshot
    ta = ctx.get("ta_snapshot")
    if ta:
        lines.append("")
        lines.append("TECHNICAL SNAPSHOT:")
        for tf in ("H1", "M15", "M5", "M1"):
            t = ta.get(tf)
            if not t:
                continue
            regime = t.get("regime", "?").capitalize()
            rsi_val = t.get("rsi_value")
            rsi_zone = t.get("rsi_zone", "")
            rsi_str = f"RSI {rsi_val:.0f} ({rsi_zone})" if rsi_val is not None else "RSI n/a"
            macd = t.get("macd_direction", "neutral")
            atr = t.get("atr_pips")
            atr_state = t.get("atr_state", "")
            atr_str = f"ATR {atr}p ({atr_state})" if atr is not None else "ATR n/a"
            adx_val = t.get("adx")
            adxr_val = t.get("adxr")
            adx_str = f"ADX {adx_val}" if adx_val is not None else ""
            if adxr_val is not None:
                adx_str += f"/ADXR {adxr_val}"
            parts_line = f"  {tf}: {regime} | {rsi_str} | MACD {macd} | {atr_str}"
            if adx_str:
                parts_line += f" | {adx_str}"
            streak = t.get("streak")
            if isinstance(streak, dict) and streak.get("count"):
                arrow = "^" if streak.get("direction") == "up" else ("v" if streak.get("direction") == "down" else "-")
                parts_line += f" | streak {streak['count']}{arrow}"
            patterns = t.get("patterns")
            if isinstance(patterns, list) and patterns:
                parts_line += f" | patterns: {', '.join(patterns)}"
            lines.append(parts_line)

        # TF consensus
        tfc = ctx.get("tf_consensus")
        if isinstance(tfc, dict):
            dom = str(tfc.get("dominant_direction", "")).replace("_", " ")
            bull_tfs = tfc.get("regime_bull_tfs") or []
            bear_tfs = tfc.get("regime_bear_tfs") or []
            neutral_tfs = tfc.get("regime_neutral_tfs") or []
            lines.append(
                f"  CONSENSUS: {dom} | bull {'/'.join(bull_tfs) if bull_tfs else '-'} "
                f"| bear {'/'.join(bear_tfs) if bear_tfs else '-'} "
                f"| neutral {'/'.join(neutral_tfs) if neutral_tfs else '-'}"
            )
            divs = tfc.get("divergences") or []
            if divs:
                lines.append(f"  DIVERGENCES: {'; '.join(divs)}")
            rsi_warn = tfc.get("rsi_warnings") or []
            if rsi_warn:
                lines.append(f"  RSI EXTREMES: {'; '.join(rsi_warn)}")

    cross = ctx.get("cross_assets")
    if cross:
        lines.append("")
        lines.append("CROSS-ASSET SNAPSHOT:")
        dxy = cross.get("dxy")
        if dxy:
            lines.append(f"  DXY (US Dollar Index): {dxy}")
        eurusd = cross.get("eurusd")
        if eurusd:
            lines.append(f"  EUR/USD: {eurusd}")
        us10y = cross.get("us10y_yield")
        if us10y is not None:
            lines.append(f"  US 10Y Treasury Yield: {us10y}%")
        brent = cross.get("bco_usd")
        wti = cross.get("wti_usd")
        if brent and wti:
            lines.append(f"  Oil — WTI: {wti} | Brent: {brent}")
        elif wti:
            lines.append(f"  Oil — WTI (WTICO/USD): {wti}")
        elif brent:
            lines.append(f"  Oil — Brent (BCO/USD): {brent}")
        gold = cross.get("xau_usd")
        silver = cross.get("xag_usd")
        if gold:
            lines.append(f"  Gold (XAU/USD): {gold}")
        if silver:
            lines.append(f"  Silver (XAG/USD): {silver}")

    # Macro bias (cross-asset daily returns)
    bias = ctx.get("cross_asset_bias")
    if bias:
        lines.append("")
        lines.append("MACRO BIAS:")
        oil_b = bias.get("oil")
        if oil_b:
            lines.append(f"  Oil: {oil_b['direction'].upper()} (5D: {oil_b['5d_return']:+.1f}%, 20D: {oil_b['20d_return']:+.1f}%)")
        dxy_b = bias.get("dxy")
        if dxy_b:
            lines.append(f"  DXY: {dxy_b['direction'].upper()} (5D: {dxy_b['5d_return']:+.1f}%, 20D: {dxy_b['20d_return']:+.1f}%, value: {dxy_b.get('value', '?')})")
        combined = bias.get("combined_bias", "")
        conf = bias.get("confidence", "")
        impl = bias.get("usdjpy_implication", "")
        if combined:
            lines.append(f"  Combined: {combined.upper()} ({conf} confidence) — {impl}")

    # JPY cross bias — disambiguates USD strength from JPY weakness.
    jc = ctx.get("jpy_crosses")
    if isinstance(jc, dict) and jc:
        lines.append("")
        lines.append("JPY CROSS BIAS (USD-strength vs JPY-weakness disambiguation):")
        for key in ("eur_jpy", "gbp_jpy", "aud_jpy"):
            entry = jc.get(key)
            if not isinstance(entry, dict):
                continue
            mid = entry.get("mid")
            c4h = entry.get("change_4h_pct")
            c24h = entry.get("change_24h_pct")
            parts = []
            if mid is not None:
                parts.append(f"{mid}")
            if c4h is not None:
                parts.append(f"4h {c4h:+.2f}%")
            if c24h is not None:
                parts.append(f"24h {c24h:+.2f}%")
            label = key.replace("_", "").upper()
            lines.append(f"  {label}: {' | '.join(parts)}")
        summary = jc.get("summary_4h")
        if summary:
            lines.append(
                f"  -> {summary} (4h: {jc.get('crosses_up_4h', 0)} up, "
                f"{jc.get('crosses_down_4h', 0)} down)"
            )
        lines.append(
            "  Read: if USDJPY is up AND crosses are up -> JPY weakness (higher BUY conviction). "
            "If USDJPY up but crosses flat/down -> pure USD move (lower conviction)."
        )

    # Position book sentiment (OANDA volume proxy)
    pb = ctx.get("position_book")
    if pb:
        lines.append("")
        lines.append("OANDA VOLUME (Position Book):")
        lines.append(
            f"  Traders long: {pb['long_pct']}% | short: {pb['short_pct']}% | Bias: {pb['bias']}"
        )
        if pb.get("book_price"):
            lines.append(f"  Snapshot price: {pb['book_price']:.3f}")

    ob = ctx.get("order_book")
    if ob:
        lines.append("")
        lines.append("KEY LEVELS (OANDA Order Book):")
        buy_cl = ob.get("buy_clusters", [])
        sell_cl = ob.get("sell_clusters", [])
        if buy_cl:
            parts = [f"{c['price']:.3f} ({c['pct']:.1%})" for c in buy_cl]
            lines.append(f"  Buy clusters: {', '.join(parts)}")
        if sell_cl:
            parts = [f"{c['price']:.3f} ({c['pct']:.1%})" for c in sell_cl]
            lines.append(f"  Sell clusters: {', '.join(parts)}")
        cp = ob.get("current_price")
        if cp:
            summary = f"  Order-book snapshot price: {cp:.3f}"
            sup = ob.get("nearest_support")
            res = ob.get("nearest_resistance")
            if sup:
                summary += f" — nearest support {ob['nearest_support_distance_pips']}p below at {sup:.3f}"
            if res:
                summary += f", nearest resistance {ob['nearest_resistance_distance_pips']}p above at {res:.3f}"
            lines.append(summary)

    # OHLC candle history
    ohlc = ctx.get("ohlc_history")
    if ohlc:
        lines.append("")
        lines.append("OHLC CANDLE HISTORY (o=open, h=high, l=low, c=close, t=time):")
        for tf in ("H1", "M1"):
            candles = ohlc.get(tf)
            if not candles:
                continue
            lines.append(f"  {tf} ({len(candles)} bars, oldest→newest):")
            for c in candles:
                t_str = f" @{c['t']}" if "t" in c else ""
                v_str = f" v={c['v']}" if "v" in c else ""
                rng = round((c["h"] - c["l"]) / 0.01, 1)  # USDJPY pip = 0.01
                rng_str = f" rng={rng}p"
                lines.append(f"    {c['o']:.3f}/{c['h']:.3f}/{c['l']:.3f}/{c['c']:.3f}{rng_str}{v_str}{t_str}")

    return "\n".join(lines)


def autonomous_system_prompt_from_context(
    ctx: dict[str, Any],
    effective_model: str,
    autonomous_config: dict[str, Any] | None = None,
    risk_regime: dict[str, Any] | None = None,
) -> str:
    """Lean system prompt for Autonomous Fillmore — strips chat persona/delivery/tool blocks.

    Keeps the LIVE TRADING CONTEXT sections intact (sessions, price structure, TA,
    macro bias, order book, etc.) but replaces the Fillmore-voice preamble with a
    disciplined-analyst framing. Autonomous mode doesn't need personality — it
    needs tight judgment, and persona tokens dilute focus.
    """
    full = system_prompt_from_context(ctx, effective_model)
    marker = "=== LIVE TRADING CONTEXT ==="
    idx = full.find(marker)
    if idx < 0:
        return full  # safety fallback — shouldn't happen in practice

    cfg = autonomous_config or {}
    multi_enabled = bool(cfg.get("multi_trade_enabled"))
    max_suggestions = int(cfg.get("max_suggestions_per_call") or 1)
    if multi_enabled and max_suggestions > 1:
        commit_line = (
            f"evaluate the live context below and return UP TO {max_suggestions} trade objects, "
            "but only when the edge is clear. Returning 0 lots is normal whenever the opportunity is not clean enough."
        )
    else:
        commit_line = (
            "evaluate the live context below and return at most ONE trade object. "
            "Passing with 0 lots is normal whenever the edge is not clean enough."
        )

    corr_pips = float(cfg.get("correlation_distance_pips") or 15.0)
    spot = (ctx.get("spot_price") or {}) if isinstance(ctx, dict) else {}
    try:
        current_mid = float(spot.get("mid") or 0.0)
    except (TypeError, ValueError):
        current_mid = 0.0
    if current_mid > 0:
        pip_value = 100000.0 * 0.01 / current_mid
        sizing_line = (
            f"SIZING: USDJPY — 1 lot = 100,000 units, ~${pip_value:.2f}/pip/lot at {current_mid:.3f}."
        )
    else:
        sizing_line = "SIZING: USDJPY — 1 lot = 100,000 units."

    regime_append = ""
    if str((risk_regime or {}).get("label") or "normal") != "normal":
        regime_append = "\n".join([
            "",
            f"RISK REGIME: {str((risk_regime or {}).get('label') or '').upper()}",
            "You are in a defensive drawdown state.",
            "Be more selective than usual and downgrade marginal setups rather than forcing action.",
        ])

    lean_preamble = "\n".join([
        "You are a neutral USDJPY autonomous trader operating in AUTONOMOUS mode.",
        "The run loop has flagged the snapshot as a possible opportunity. That is not an instruction to trade.",
        f"Your job: {commit_line}",
        "You are allowed to pass freely. Do not force a trade because a snapshot looks interesting.",
        "Policy and geopolitical context are a primary alpha source in this system, not a side note.",
        "",
        "Critical discipline:",
        "- ENTRY PRICING: You choose 'market' or 'limit' per trade. "
        "Market fills instantly at current bid/ask — use when the tape is moving and you want in now. "
        "Limit is for near-touch passive entries only. "
        "For limit orders: BUY LIMIT must be below current bid, SELL LIMIT must be above current ask. "
        "The server will clamp autonomous limits into a near-market band rather than leave them several pips away. "
        "If the structural level is far from current price, prefer market or pass.",
        "- Check the OPEN POSITIONS block below. Stacking and hedging are allowed when thesis quality is clear. "
        f"If you add exposure near an existing position (within ~{corr_pips:g} pips), explain why this is additive "
        "alpha (or why a hedge is justified) instead of a duplicate idea.",
        "- Check YOUR MOST RECENT SUGGESTION and recent closed trades. If the same side+level has been "
        "firing repeatedly without working, step back — the tape is eating that idea.",
        "- Treat trigger families differently: trend-expansion setups favor market execution; compression-breakout setups also favor market execution once the squeeze edge is actively being pressed; critical-level reactions and Tokyo tight-range mean-reversion setups can use market or near-touch limits.",
        "- POLICY + GEOPOLITICAL ALPHA (required each trade): explicitly evaluate what Japan MOF is doing, whether rate-check/intervention risk is rising, whether Japan-US Treasury/Fed coordination is active, and what Japan's finance minister is signaling. Also evaluate geopolitical war-premium channels (risk sentiment, oil shock, safe-haven flow) and their directional impact on USDJPY.",
        "- Treat policy/geopolitical factors as directional signal inputs that modify conviction, side confidence, entry style, and size. They are not automatic trade blockers.",
        "- In your analysis, state whether policy/geopolitical context confirms, contradicts, or is mixed versus the technical setup, and reflect that in lots/order_type.",
        "- News is context, not a script. If catalyst context is stale or contradictory, downgrade confidence rather than pretending certainty.",
        "- No persona, no narration, no desk voice. Just clear analysis and a clean commit.",
        "",
        f"MODEL: You are '{effective_model}'.",
        sizing_line,
        "PRICE AUTHORITY: The LIVE PRICE section below is authoritative.",
        regime_append,
        "",
    ])
    return lean_preamble + full[idx:]


# ---------------------------------------------------------------------------
# 3. Function calling tools
# ---------------------------------------------------------------------------

_AI_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_candles",
            "description": "Fetch OHLC candle data for USDJPY at a specific timeframe. Use when the user asks about price action, chart patterns, or specific timeframe analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string", "enum": ["M1", "M5", "M15", "H1", "H4", "D"], "description": "Candle timeframe"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of candles (default 30)"},
                },
                "required": ["timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_history",
            "description": "Query closed trade history from the database. Use when the user asks about past trades, P&L breakdown, or specific trade details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "minimum": 1, "maximum": 90, "description": "How many days back to look (default 7)"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max trades to return (default 20)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ai_suggestion_history",
            "description": "Query Fillmore's AI suggestion history, including generated ideas, edited fields, placed orders, market snapshot, and eventual outcomes. Use when the user asks about AI-generated trades, specific suggestion IDs, specific order IDs, or why a Fillmore idea was edited, cancelled, filled, or left open.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "minimum": 1, "maximum": 365, "description": "How many days back to search (default 30)"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max suggestion rows to return (default 10)"},
                    "suggestion_id": {"type": "string", "description": "Optional exact Fillmore suggestion_id to inspect"},
                    "oanda_order_id": {"type": "string", "description": "Optional broker order id to inspect"},
                    "action": {"type": "string", "enum": ["placed", "rejected"], "description": "Optional action filter"},
                    "outcome_status": {"type": "string", "enum": ["filled", "cancelled", "expired"], "description": "Optional outcome filter"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fillmore_system_info",
            "description": "Explain how Fillmore's manual trade suggestion flow, autonomous Fillmore engine, and learning/history tracking work in this build. Use when the user asks how Fillmore works, how autonomous mode works, how suggestion history is tracked, or what Fillmore can see about its own trades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["trade_suggestion", "autonomous_fillmore", "history_learning", "all"],
                        "description": "Which Fillmore subsystem to explain (default all)",
                    },
                    "include_runtime": {
                        "type": "boolean",
                        "description": "Whether to include the current profile's autonomous config/runtime snapshot when relevant (default true)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trade_patterns",
            "description": "Analyze win/loss patterns grouped by session (Tokyo/London/NY), entry type, and tier. Use when the user asks about which setups work best or worst.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "minimum": 1, "maximum": 90, "description": "How many days to analyze (default 30)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cross_asset_bias",
            "description": "Get full macro bias reading from oil, DXY, and their combined USDJPY implication. Use for macro/correlation context.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_economic_calendar",
            "description": "Get upcoming high-impact economic events for USD and JPY. Use when the user asks about events, catalysts, or risk ahead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "minimum": 1, "maximum": 30, "description": "How many days ahead to look (default 7)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_headlines",
            "description": "Fetch recent forex/USDJPY news headlines from free RSS feeds. Use when the user asks about news, what's moving the market, or recent developments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of headlines (default 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_orders",
            "description": "Fetch current pending (resting) limit and stop orders on the broker account. Use when the user asks about pending orders, resting orders, working orders, or what limits they have out.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Brave Search. Use when the user asks about current events, news, market analysis, economic data, or anything that needs live web information. Also useful for looking up specific topics like central bank statements, geopolitical events, or market commentary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 3a. Tool executors
# ---------------------------------------------------------------------------

def _exec_get_candles(args: dict, profile: ProfileV1, **_: Any) -> str:
    """Fetch OHLC candles via broker adapter."""
    from adapters.broker import get_adapter

    tf = args.get("timeframe", "H1")
    count = min(int(args.get("count", 30)), 100)
    adapter = get_adapter(profile)
    try:
        adapter.initialize()
        if hasattr(adapter, "ensure_symbol"):
            adapter.ensure_symbol(profile.symbol)
        df = adapter.get_bars(profile.symbol, tf, count=count)
        if df is None or df.empty:
            return f"No candle data available for {tf}."
        pip_size = float(profile.pip_size) if profile.pip_size else 0.01
        lines = [f"{tf} candles ({len(df)} bars, oldest→newest):"]
        for idx, row in df.iterrows():
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            rng = round((h - l) / pip_size, 1)
            t_str = str(idx) if idx is not None else ""
            lines.append(f"  {o:.3f}/{h:.3f}/{l:.3f}/{c:.3f} rng={rng}p @{t_str}")
        return "\n".join(lines)
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass


def _exec_get_trade_history(args: dict, profile_name: str, **_: Any) -> str:
    """Query closed trades from SQLite database."""
    from storage.sqlite_store import SqliteStore

    days_back = min(int(args.get("days_back", 7)), 90)
    limit = min(int(args.get("limit", 20)), 50)

    db_path = LOGS_DIR / profile_name / "assistant.db"
    if not db_path.exists():
        return "No trade database found for this profile."

    store = SqliteStore(db_path)
    df = store.read_trades_df(profile_name)
    if df.empty:
        return "No trades found in database."

    # Filter by date
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    if "exit_timestamp_utc" in df.columns:
        closed = df[df["exit_price"].notna() & (df["exit_timestamp_utc"] >= cutoff)].copy()
    else:
        closed = df[df["exit_price"].notna()].copy()

    if closed.empty:
        return f"No closed trades in the last {days_back} days."

    closed = closed.sort_values("exit_timestamp_utc", ascending=False).head(limit)

    lines = [f"Closed trades (last {days_back}d, showing {len(closed)}):"]
    for _, row in closed.iterrows():
        side = row.get("side", "?")
        entry = row.get("entry_price", "?")
        exit_p = row.get("exit_price", "?")
        pips = row.get("pips")
        profit = row.get("profit")
        entry_type = row.get("entry_type", "")
        exit_reason = row.get("exit_reason", "")
        session = row.get("entry_session", "")
        tier = row.get("tier_number", "")
        exit_time = row.get("exit_timestamp_utc", "")
        source = classify_trade_source(entry_type)
        source_label = _source_label_for_trade(entry_type)

        parts = [f"{side}"]
        if entry != "?":
            parts.append(f"@{entry}")
        if exit_p != "?":
            parts.append(f"→{exit_p}")
        if pips is not None:
            parts.append(f"{float(pips):+.1f}p")
        if profit is not None:
            parts.append(f"${float(profit):+.2f}")
        if source == "bot":
            parts.append(f"[{_plain_bot_label(entry_type)}]")
        parts.append(f"[source:{source_label}]")
        if tier:
            parts.append(f"tier{tier}")
        if session:
            parts.append(f"({session})")
        if exit_reason:
            parts.append(f"exit:{exit_reason}")
        if exit_time:
            parts.append(f"@{str(exit_time)[:16]}")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


def _exec_get_ai_suggestion_history(args: dict, profile_name: str, **_: Any) -> str:
    """Query Fillmore AI suggestion history from the tracker DB."""
    from api import suggestion_tracker

    days_back = min(int(args.get("days_back", 30)), 365)
    limit = min(int(args.get("limit", 10)), 50)
    suggestion_id = str(args.get("suggestion_id") or "").strip()
    order_id = str(args.get("oanda_order_id") or "").strip()
    action_filter = str(args.get("action") or "").strip().lower()
    outcome_filter = str(args.get("outcome_status") or "").strip().lower()

    db_path = LOGS_DIR / profile_name / "ai_suggestions.sqlite"
    if not db_path.exists():
        return "No Fillmore suggestion history found for this profile."

    if order_id:
        row = suggestion_tracker.get_by_order_id(db_path, order_id)
        if row is None:
            return f"No Fillmore suggestion found for order {order_id}."
        _attach_ai_suggestion_trade_results([row], profile_name)
        return _format_ai_suggestion_rows([row], heading=f"Fillmore suggestion for order {order_id}:")

    history = suggestion_tracker.get_history(db_path, limit=500, offset=0)
    items = list(history.get("items") or [])
    if not items:
        return "No Fillmore suggestion history found for this profile."

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    def _created_after(row: dict[str, Any]) -> bool:
        created = row.get("created_utc")
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except Exception:
            return False
        return dt >= cutoff

    rows = [row for row in items if _created_after(row)]
    if suggestion_id:
        rows = [row for row in rows if str(row.get("suggestion_id") or "") == suggestion_id]
    if action_filter:
        rows = [row for row in rows if str(row.get("action") or "").lower() == action_filter]
    if outcome_filter:
        rows = [row for row in rows if str(row.get("outcome_status") or "").lower() == outcome_filter]

    if not rows:
        filters: list[str] = []
        if suggestion_id:
            filters.append(f"suggestion_id={suggestion_id}")
        if action_filter:
            filters.append(f"action={action_filter}")
        if outcome_filter:
            filters.append(f"outcome_status={outcome_filter}")
        if not filters:
            filters.append(f"last {days_back}d")
        return f"No Fillmore suggestions matched ({', '.join(filters)})."

    _attach_ai_suggestion_trade_results(rows, profile_name)

    return _format_ai_suggestion_rows(
        rows[:limit],
        heading=f"Fillmore suggestion history (last {days_back}d, showing {min(limit, len(rows))} of {len(rows)}):",
    )


def _attach_ai_suggestion_trade_results(rows: list[dict[str, Any]], profile_name: str) -> None:
    """Enrich suggestion rows with actual ai_manual trade results from assistant.db."""
    if not rows:
        return

    from storage.sqlite_store import SqliteStore

    db_path = LOGS_DIR / profile_name / "assistant.db"
    if not db_path.exists():
        for row in rows:
            _set_resolved_suggestion_result(row, None)
        return

    try:
        store = SqliteStore(db_path)
        df = store.read_trades_df(profile_name)
    except Exception:
        for row in rows:
            _set_resolved_suggestion_result(row, None)
        return

    def _is_fillmore_trade(trade: dict[str, Any]) -> bool:
        entry_type = str(trade.get("entry_type") or "").strip().lower()
        policy_type = str(trade.get("policy_type") or "").strip().lower()
        opened_by = str(trade.get("opened_by") or "").strip().lower()
        trade_id = str(trade.get("trade_id") or "").strip().lower()
        notes = str(trade.get("notes") or "").strip().lower()
        if entry_type == "ai_manual":
            return True
        if policy_type == "ai_manual":
            return True
        if opened_by == "ai_manual":
            return True
        if trade_id.startswith("ai_manual:"):
            return True
        if notes.startswith("ai_manual:") or ":order_" in notes:
            return True
        return False

    trade_by_id: dict[str, dict[str, Any]] = {}
    fillmore_trade_by_order: dict[str, dict[str, Any]] = {}
    any_trade_by_order: dict[str, dict[str, Any]] = {}
    if not df.empty:
        for _, rec in df.iterrows():
            trade = rec.to_dict()
            trade_id = str(trade.get("trade_id") or "").strip()
            if trade_id:
                trade_by_id[trade_id] = trade
            order_val = trade.get("mt5_order_id")
            if order_val is not None and not pd.isna(order_val):
                try:
                    order_key = str(int(float(order_val)))
                except (TypeError, ValueError):
                    order_key = str(order_val).strip()
                if order_key:
                    any_trade_by_order[order_key] = trade
                    if _is_fillmore_trade(trade):
                        fillmore_trade_by_order[order_key] = trade

    for row in rows:
        linked_trade = None
        trade_id = str(row.get("trade_id") or "").strip()
        order_id = str(row.get("oanda_order_id") or "").strip()
        if trade_id and trade_id in trade_by_id:
            linked_trade = trade_by_id[trade_id]
        elif order_id and order_id in fillmore_trade_by_order:
            linked_trade = fillmore_trade_by_order[order_id]
        elif order_id and order_id in any_trade_by_order:
            linked_trade = any_trade_by_order[order_id]
        _set_resolved_suggestion_result(row, linked_trade)


def _set_resolved_suggestion_result(row: dict[str, Any], linked_trade: dict[str, Any] | None) -> None:
    row["linked_trade"] = linked_trade
    action = str(row.get("action") or "").strip().lower()
    outcome = str(row.get("outcome_status") or "").strip().lower()
    win_loss = str(row.get("win_loss") or "").strip().lower()

    if linked_trade:
        exit_price = linked_trade.get("exit_price")
        if exit_price is not None and not pd.isna(exit_price):
            pnl = linked_trade.get("profit")
            pips = linked_trade.get("pips")
            exit_reason = linked_trade.get("exit_reason")
            closed_at = linked_trade.get("exit_timestamp_utc")
            wl = win_loss
            if not wl:
                try:
                    pnl_f = float(pnl or 0.0)
                    wl = "win" if pnl_f > 0 else "loss" if pnl_f < 0 else "breakeven"
                except Exception:
                    wl = "closed"
            row["resolved_outcome"] = f"closed/{wl}"
            row["resolved_result"] = {
                "pnl": None if pnl is None or pd.isna(pnl) else float(pnl),
                "pips": None if pips is None or pd.isna(pips) else float(pips),
                "exit_reason": str(exit_reason or ""),
                "closed_at": str(closed_at or ""),
            }
            return
        row["resolved_outcome"] = "filled/open"
        row["resolved_result"] = {
            "entry_price": None if linked_trade.get("entry_price") is None or pd.isna(linked_trade.get("entry_price")) else float(linked_trade.get("entry_price")),
            "size_lots": None if linked_trade.get("size_lots") is None or pd.isna(linked_trade.get("size_lots")) else float(linked_trade.get("size_lots")),
            "timestamp_utc": str(linked_trade.get("timestamp_utc") or ""),
        }
        return

    if outcome:
        row["resolved_outcome"] = outcome
        row["resolved_result"] = {}
        return
    if action == "placed":
        row["resolved_outcome"] = "pending"
        row["resolved_result"] = {}
        return
    if action == "rejected":
        row["resolved_outcome"] = "rejected"
        row["resolved_result"] = {}
        return
    row["resolved_outcome"] = "generated_only"
    row["resolved_result"] = {}


def _format_ai_suggestion_rows(rows: list[dict[str, Any]], *, heading: str) -> str:
    lines = [heading]
    for row in rows:
        created = str(row.get("created_utc") or "")[:16]
        side = str(row.get("side") or "?").upper()
        requested_px = row.get("requested_price")
        limit_px = row.get("limit_price")
        sl = row.get("sl")
        tp = row.get("tp")
        lots = row.get("lots")
        model = str(row.get("model") or "?")
        quality = str(row.get("quality") or row.get("confidence") or "?")
        action = str(row.get("action") or "none")
        outcome = str(row.get("resolved_outcome") or row.get("outcome_status") or row.get("win_loss") or "open")
        order_id = row.get("oanda_order_id")
        trade_id = row.get("trade_id")
        exit_strategy = str((row.get("placed_order") or {}).get("exit_strategy") or row.get("exit_strategy") or "")

        head = f"  {created} | {model} | {side} {limit_px} SL {sl} TP {tp} lots {lots} | quality={quality} | action={action} | outcome={outcome}"
        lines.append(head)

        meta: list[str] = [f"id={row.get('suggestion_id')}"]
        try:
            if requested_px is not None and limit_px is not None and abs(float(requested_px) - float(limit_px)) >= 0.0005:
                meta.append(f"requested={float(requested_px):.3f}")
        except Exception:
            pass
        if order_id:
            meta.append(f"order={order_id}")
        if trade_id:
            meta.append(f"trade={trade_id}")
        linked_trade = row.get("linked_trade") or {}
        linked_trade_id = str(linked_trade.get("trade_id") or "").strip()
        if linked_trade_id and linked_trade_id != str(trade_id or "").strip():
            meta.append(f"linked_trade={linked_trade_id}")
        if exit_strategy:
            meta.append(f"exit={exit_strategy}")
        lines.append("    " + " | ".join(meta))

        edited_fields = row.get("edited_fields") or {}
        if isinstance(edited_fields, dict) and edited_fields:
            edits = []
            for key, change in list(edited_fields.items())[:6]:
                before = change.get("before") if isinstance(change, dict) else None
                after = change.get("after") if isinstance(change, dict) else None
                edits.append(f"{key}: {before} -> {after}")
            lines.append("    edited: " + "; ".join(edits))

        snap = row.get("market_snapshot") or {}
        session = ((snap.get("session") or {}).get("overlap") or _join_list((snap.get("session") or {}).get("active_sessions")))
        macro = (snap.get("macro_bias") or {}).get("combined_bias")
        vol = (snap.get("volatility") or {}).get("label")
        spread = snap.get("spread_pips")
        ctx_bits = []
        if session:
            ctx_bits.append(f"session={session}")
        if macro:
            ctx_bits.append(f"macro={macro}")
        if vol:
            ctx_bits.append(f"vol={vol}")
        if spread is not None:
            ctx_bits.append(f"spread={spread}p")
        if ctx_bits:
            lines.append("    context: " + ", ".join(ctx_bits))

        resolved = row.get("resolved_result") or {}
        if outcome.startswith("closed/"):
            pnl = resolved.get("pnl")
            pips = resolved.get("pips")
            exit_reason = str(resolved.get("exit_reason") or "").strip()
            closed_at = str(resolved.get("closed_at") or "").strip()
            result_bits = []
            if pips is not None:
                result_bits.append(f"pips={float(pips):+.1f}")
            if pnl is not None:
                result_bits.append(f"pnl=${float(pnl):+,.2f}")
            if exit_reason:
                result_bits.append(f"exit={exit_reason}")
            if closed_at:
                result_bits.append(f"closed={closed_at[:16]}")
            if result_bits:
                lines.append("    result: " + " | ".join(result_bits))
        elif outcome == "filled/open":
            entry_price = resolved.get("entry_price")
            size_lots = resolved.get("size_lots")
            ts = str(resolved.get("timestamp_utc") or "").strip()
            result_bits = []
            if entry_price is not None:
                result_bits.append(f"entry={float(entry_price):.3f}")
            if size_lots is not None:
                result_bits.append(f"size={float(size_lots)} lots")
            if ts:
                result_bits.append(f"filled={ts[:16]}")
            if result_bits:
                lines.append("    result: " + " | ".join(result_bits))

        rationale = str(row.get("rationale") or "").strip()
        if rationale:
            lines.append(f"    rationale: {rationale[:320]}")
    return "\n".join(lines)


def _join_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return " + ".join(cleaned)


def _exec_get_fillmore_system_info(args: dict, profile_name: str, **_: Any) -> str:
    """Explain Fillmore's own suggestion / autonomous / learning systems."""
    topic = str(args.get("topic") or "all").strip().lower()
    if topic not in {"trade_suggestion", "autonomous_fillmore", "history_learning", "all"}:
        topic = "all"
    include_runtime = args.get("include_runtime")
    if include_runtime is None:
        include_runtime = True
    include_runtime = bool(include_runtime)

    sections: list[str] = []

    if topic in {"trade_suggestion", "all"}:
        sections.append(_fillmore_trade_suggestion_help())
    if topic in {"autonomous_fillmore", "all"}:
        sections.append(_fillmore_autonomous_help(profile_name, include_runtime=include_runtime))
    if topic in {"history_learning", "all"}:
        sections.append(_fillmore_history_learning_help())

    return "\n\n".join(part for part in sections if part.strip())


def _fillmore_trade_suggestion_help() -> str:
    return "\n".join([
        "FILLMORE TRADE SUGGESTION",
        "- This is the manual/operator-assisted flow from the Trade Suggestion card in the Fillmore UI.",
        "- When you generate a suggestion, the server builds live trading context first: account state, spot price/spread, open positions, recent trades, session context, volatility, TA snapshot, price structure, upcoming events, and OANDA-specific extras like macro bias, order book, pending orders, and JPY cross bias when available.",
        "- The suggest prompt also appends a prefetched external news block and a learning-memory block built from prior Fillmore suggestion outcomes.",
        "- The model returns one structured JSON idea with side, price, stop, target, lots, time-in-force, expiration, exit_strategy, exit_params, rationale, and quality (A/B/C).",
        "- That generated idea is persisted immediately into ai_suggestions.sqlite with a suggestion_id, market snapshot, and the original suggestion fields.",
        "- In the UI, the user can edit side/price/SL/TP/lots/expiration/exit strategy before placement. Those edits are diffed and logged.",
        "- If the user places the order, the suggestion row is updated with action=placed, the placed order payload, and the broker order ID. If the user rejects it, action=rejected is stored instead.",
        "- If the broker order later fills, the system stamps filled_at, fill_price, and the linked local trade_id. If that trade later closes, the suggestion row is updated with realized exit price, P&L, pips, and win/loss result.",
        "- Net effect: Fillmore can inspect the whole lifecycle of a suggestion: generated -> edited or verbatim -> placed or rejected -> filled/pending/cancelled/expired -> closed result when available.",
    ])


def _fillmore_autonomous_help(profile_name: str, *, include_runtime: bool) -> str:
    lines = [
        "AUTONOMOUS FILLMORE",
        "- This is the unattended engine that runs inside the main trading loop each tick, not the normal chat reply path.",
        "- It first loads autonomous config/runtime state from the profile runtime_state.json block.",
        "- Then it evaluates a three-layer gate before paying for an LLM call.",
        "- Layer 1 hard filters: enabled flag, mode not off, spread cap, no-trade-zone flag, max open AI trades, daily loss cap, daily LLM budget cap, allowed trading sessions, and minimum cooldown since the last model call.",
        "- Layer 2 adaptive throttle: if Fillmore is in a cooldown from repeated no-trade replies, a loss streak, or too many consecutive errors, it blocks before calling the model.",
        "- Layer 3 signal gate: this depends on aggressiveness. Conservative wants M3 trend + M1 stack + pullback/zone + daily high/low buffer. Balanced wants aligned M3 and M1. Aggressive only needs some trend evidence. Very aggressive is close to hard-filters-only.",
        "- If the gate passes, Autonomous Fillmore calls the same suggestion pipeline as manual suggestions, including live context, news enrichment, and learning memory.",
        "- The LLM expresses conviction via lot sizing: lots=0 skips, lots>0 places at that size. No separate confidence gate.",
        "- Mode behavior: OFF disables it, SHADOW logs would-have-traded decisions without placing, PAPER places to the OANDA practice environment, and LIVE places to the real broker account.",
        "- Order type: the LLM chooses per-suggestion — MARKET fills immediately, LIMIT rests at the named price.",
        "- When a trade is placed, the order id and suggestion id are written back into autonomous runtime stats. Suggestion history and later trade results are meant to feed the same learning loop as manual suggestions.",
    ]

    if include_runtime:
        try:
            from api import autonomous_fillmore as _af

            state_path = LOGS_DIR / profile_name / "runtime_state.json"
            cfg = _af.get_config(state_path)
            stats = _af.build_stats(state_path, cfg=cfg)
            today = stats.get("today") or {}
            throttle = stats.get("throttle") or {}
            lines.extend([
                "CURRENT AUTONOMOUS SNAPSHOT",
                f"- mode={cfg.get('mode')} enabled={bool(cfg.get('enabled'))} aggressiveness={cfg.get('aggressiveness')} model={cfg.get('model')}",
                f"- budgets/caps: daily_budget=${float(cfg.get('daily_budget_usd') or 0):.2f}, max_open_ai_trades={int(cfg.get('max_open_ai_trades') or 0)}, max_daily_loss=${float(cfg.get('max_daily_loss_usd') or 0):.2f}, max_lots={float(cfg.get('max_lots_per_trade') or 0):.2f}",
                f"- activity today: llm_calls={int(today.get('llm_calls') or 0)}, trades_placed={int(today.get('trades_placed') or 0)}, spend=${float(today.get('spend_usd') or 0):.4f}, pnl=${float(today.get('pnl_usd') or 0):.2f}",
                f"- throttle: active={bool(throttle.get('active'))}, reason={throttle.get('reason') or 'none'}, no_trade_streak={int(throttle.get('consecutive_no_trade_replies') or 0)}, loss_streak={int(throttle.get('consecutive_losses') or 0)}, consecutive_errors={int(throttle.get('consecutive_errors') or 0)}",
            ])
        except Exception as e:
            lines.append(f"- Current autonomous runtime snapshot unavailable: {e}")

    return "\n".join(lines)


def _fillmore_history_learning_help() -> str:
    return "\n".join([
        "FILLMORE HISTORY AND LEARNING",
        "- Fillmore suggestion history lives in ai_suggestions.sqlite on a per-profile basis. It is separate from the closed-trade ledger in assistant.db.",
        "- The suggestion tracker stores the generated idea, market snapshot, edits, placed order payload, broker order id, fill status, linked trade id, and eventual realized result when matched.",
        "- To answer outcome questions, Fillmore can inspect both suggestion history and trade history. Suggestion rows hold the idea/order trail; trade rows hold the execution/close trail.",
        "- The learning prompt block is not a raw dump of every old suggestion. It builds a compact summary from prior rows: generated/placed/rejected counts, fill and cancel rates, closed-trade win rate, average pnl/pips/hold time, top edited fields, exit strategy results, and recent examples.",
        "- When current live context is available, the learning block tries to select a matched analog cohort by session, macro bias, and volatility before falling back to broad history.",
        "- That means Fillmore is supposed to learn from behavior in similar conditions rather than blindly replaying the whole database every time.",
        "- Suggestion history can still differ from broker trade history in shape: generated-only or rejected suggestions never become trades, placed limits can remain pending/cancel/expire, and filled trades may close later in the broker ledger before being summarized back into suggestion history.",
    ])


def _exec_analyze_trade_patterns(args: dict, profile_name: str, **_: Any) -> str:
    """Analyze win/loss patterns by session, entry type, and tier."""
    from storage.sqlite_store import SqliteStore

    days_back = min(int(args.get("days_back", 30)), 90)
    db_path = LOGS_DIR / profile_name / "assistant.db"
    if not db_path.exists():
        return "No trade database found."

    store = SqliteStore(db_path)
    df = store.read_trades_df(profile_name)
    if df.empty:
        return "No trades found."

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    closed = df[df["exit_price"].notna()].copy()
    if "exit_timestamp_utc" in closed.columns:
        closed = closed[closed["exit_timestamp_utc"] >= cutoff]
    if closed.empty:
        return f"No closed trades in the last {days_back} days."

    if "entry_type" in closed.columns:
        closed["source"] = closed["entry_type"].apply(classify_trade_source)
    else:
        closed["source"] = "manual"
    lines = [f"Trade pattern analysis ({len(closed)} trades, last {days_back}d):"]
    lines.append("")
    lines.append("  By source (manual first):")
    for src in ("manual", "bot"):
        grp = closed[closed["source"] == src]
        if grp.empty:
            lines.append(f"    {src}: 0 trades, 0.0% WR, net $+0.00")
            continue
        wins = len(grp[grp["profit"] > 0]) if "profit" in grp.columns else 0
        losses = len(grp[grp["profit"] <= 0]) if "profit" in grp.columns else 0
        total = len(grp)
        wr = round(wins / total * 100, 1) if total else 0.0
        net_pl = float(grp["profit"].sum()) if "profit" in grp.columns else 0.0
        lines.append(
            f"    {src}: {total} trades, {wins}W/{losses}L, {wr}% WR, net ${net_pl:+.2f}"
        )

    def _group_stats(group_col: str, label: str) -> None:
        if group_col not in closed.columns or closed[group_col].isna().all():
            return
        lines.append(f"\n  By {label}:")
        for name, grp in closed.groupby(group_col):
            if not name:
                continue
            total = len(grp)
            wins = len(grp[grp["profit"] > 0]) if "profit" in grp.columns else 0
            wr = round(wins / total * 100, 1) if total else 0
            avg_profit = round(grp["profit"].mean(), 2) if "profit" in grp.columns else 0
            avg_pips = round(grp["pips"].mean(), 1) if "pips" in grp.columns else 0
            display_name = str(name)
            if group_col == "entry_type":
                display_name = _plain_bot_label(name) if classify_trade_source(name) == "bot" else str(name)
            lines.append(
                f"    {display_name}: {total} trades, {wr}% WR, avg {avg_pips:+.1f}p, avg ${avg_profit:+.2f}"
            )

    _group_stats("entry_session", "Session")
    _group_stats("entry_type", "Entry Type")
    _group_stats("tier_number", "Tier")
    _group_stats("side", "Side")

    return "\n".join(lines)


def _exec_get_pending_orders(args: dict, *, profile: ProfileV1, **_: Any) -> str:
    """List resting limit/stop orders on the broker account."""
    from adapters.broker import get_adapter

    adapter = get_adapter(profile)
    if not hasattr(adapter, "list_pending_orders"):
        return "Pending orders are only available for OANDA accounts on this build."
    try:
        adapter.initialize()
        raw = adapter.list_pending_orders(profile.symbol) or []
    except Exception as e:
        return f"Unable to fetch pending orders: {e}"
    if not raw:
        return "No pending orders."
    lines = [f"Pending orders ({len(raw)}):"]
    for o in raw[:20]:
        if not isinstance(o, dict):
            continue
        try:
            units_raw = float(o.get("units") or 0)
        except Exception:
            units_raw = 0.0
        side = "BUY" if units_raw > 0 else ("SELL" if units_raw < 0 else "?")
        lots = abs(units_raw) / 100_000.0 if units_raw else None
        lots_txt = f"{round(lots, 2)} lots" if lots is not None else f"{abs(units_raw) or '?'} units"
        sl = (o.get("stopLossOnFill") or {}).get("price") if isinstance(o.get("stopLossOnFill"), dict) else None
        tp = (o.get("takeProfitOnFill") or {}).get("price") if isinstance(o.get("takeProfitOnFill"), dict) else None
        sl_tp_parts = []
        if sl is not None:
            sl_tp_parts.append(f"SL {sl}")
        if tp is not None:
            sl_tp_parts.append(f"TP {tp}")
        sl_tp = f" [{' | '.join(sl_tp_parts)}]" if sl_tp_parts else ""
        tif = o.get("timeInForce") or "?"
        gtd = o.get("gtdTime")
        tif_disp = tif + (f" until {gtd}" if gtd else "")
        lines.append(
            f"  #{o.get('id')} {o.get('type', '?')} {side} {lots_txt} @ {o.get('price')}"
            f"{sl_tp} ({tif_disp}) | created {o.get('createTime')}"
        )
    return "\n".join(lines)


def _exec_get_cross_asset_bias(profile: ProfileV1, **_: Any) -> str:
    """Get full cross-asset macro bias."""
    from adapters.broker import get_adapter

    adapter = get_adapter(profile)
    try:
        adapter.initialize()
        bias = _compute_cross_asset_bias(adapter)
        if not bias:
            return "Unable to compute cross-asset bias."
        lines = ["Cross-Asset Macro Bias:"]
        oil = bias.get("oil")
        if oil:
            parts = [f"  Oil (Brent): {oil['direction'].upper()}"]
            if oil.get("1d_return") is not None:
                parts.append(f"1D: {oil['1d_return']:+.1f}%")
            if oil.get("5d_return") is not None:
                parts.append(f"5D: {oil['5d_return']:+.1f}%")
            if oil.get("20d_return") is not None:
                parts.append(f"20D: {oil['20d_return']:+.1f}%")
            parts.append(f"price: {oil['price']}")
            lines.append(" — ".join(parts[:1]) + " — " + ", ".join(parts[1:]))
        dxy = bias.get("dxy")
        if dxy:
            parts = [f"  DXY: {dxy['direction'].upper()}"]
            if dxy.get("1d_return") is not None:
                parts.append(f"1D: {dxy['1d_return']:+.1f}%")
            if dxy.get("5d_return") is not None:
                parts.append(f"5D: {dxy['5d_return']:+.1f}%")
            if dxy.get("20d_return") is not None:
                parts.append(f"20D: {dxy['20d_return']:+.1f}%")
            parts.append(f"value: {dxy.get('value', '?')}")
            lines.append(" — ".join(parts[:1]) + " — " + ", ".join(parts[1:]))
        lines.append(f"  Combined: {bias.get('combined_bias', '?').upper()} ({bias.get('confidence', '?')} confidence)")
        lines.append(f"  Implication: {bias.get('usdjpy_implication', '?')}")
        return "\n".join(lines)
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3b. Economic calendar (free: static + ForexFactory XML)
# ---------------------------------------------------------------------------

import time as _time
import xml.etree.ElementTree as _ET
from calendar import monthrange as _monthrange
from urllib.request import Request as _Request, urlopen as _urlopen

_CALENDAR_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
_CALENDAR_CACHE_TTL = 3600  # 1 hour

# FOMC 2025-2026 meeting dates (public: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
_FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]

# BOJ 2025-2026 meeting dates (approximate — usually 2 days)
_BOJ_DATES = [
    "2025-01-24", "2025-03-14", "2025-04-25", "2025-06-13",
    "2025-07-31", "2025-09-19", "2025-10-31", "2025-12-19",
    "2026-01-23", "2026-03-13", "2026-04-28", "2026-06-16",
    "2026-07-16", "2026-09-17", "2026-10-29", "2026-12-18",
]


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> datetime:
    """Return the nth occurrence of weekday (0=Mon, 4=Fri) in the given month."""
    first_day = datetime(year, month, 1, tzinfo=timezone.utc)
    # weekday of first day
    first_wd = first_day.weekday()
    # days until first target weekday
    days_until = (weekday - first_wd) % 7
    first_occurrence = first_day + timedelta(days=days_until)
    return first_occurrence + timedelta(weeks=n - 1)


def _get_static_calendar_events(days_ahead: int) -> list[dict[str, str]]:
    """Generate static calendar events for known recurring USD/JPY events."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    events: list[dict[str, str]] = []

    # FOMC
    for d in _FOMC_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=18)
        if now <= dt <= cutoff:
            events.append({"date": d, "time": "18:00 UTC", "currency": "USD", "event": "FOMC Interest Rate Decision", "impact": "HIGH"})

    # BOJ
    for d in _BOJ_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=3)
        if now <= dt <= cutoff:
            events.append({"date": d, "time": "~03:00 UTC", "currency": "JPY", "event": "BOJ Interest Rate Decision", "impact": "HIGH"})

    # NFP: first Friday of each month, 12:30 UTC
    for month_offset in range(0, max(days_ahead // 28 + 2, 3)):
        m = now.month + month_offset
        y = now.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        nfp = _nth_weekday_of_month(y, m, 4, 1)  # Friday=4, 1st occurrence
        nfp = nfp.replace(hour=12, minute=30)
        if now <= nfp <= cutoff:
            events.append({"date": nfp.strftime("%Y-%m-%d"), "time": "12:30 UTC", "currency": "USD", "event": "Non-Farm Payrolls (NFP)", "impact": "HIGH"})

    # US CPI: typically 10th-15th of each month, 12:30 UTC
    for month_offset in range(0, max(days_ahead // 28 + 2, 3)):
        m = now.month + month_offset
        y = now.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        # Approximate: 2nd Tuesday-Thursday of month
        cpi_approx = _nth_weekday_of_month(y, m, 2, 2)  # 2nd Wednesday
        cpi_approx = cpi_approx.replace(hour=12, minute=30)
        if now <= cpi_approx <= cutoff:
            events.append({"date": cpi_approx.strftime("%Y-%m-%d"), "time": "12:30 UTC (approx)", "currency": "USD", "event": "US CPI (Consumer Price Index)", "impact": "HIGH"})

    events.sort(key=lambda e: e["date"])
    return events


def _fetch_forexfactory_calendar() -> list[dict[str, str]]:
    """Try to fetch ForexFactory XML calendar for additional events. Best-effort."""
    now = _time.time()
    cached = _CALENDAR_CACHE.get("ff")
    if cached and now - cached[0] < _CALENDAR_CACHE_TTL:
        return cached[1]

    events: list[dict[str, str]] = []
    try:
        req = _Request(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            headers={"User-Agent": "USDJPY-Assistant/1.0"},
        )
        with _urlopen(req, timeout=5) as resp:
            tree = _ET.parse(resp)
        for item in tree.findall(".//event"):
            currency = (item.findtext("country", "") or "").upper()
            if currency not in ("USD", "JPY"):
                continue
            impact = (item.findtext("impact", "") or "").upper()
            if impact not in ("HIGH", "MEDIUM"):
                continue
            title = item.findtext("title", "") or ""
            date = item.findtext("date", "") or ""
            time_str = item.findtext("time", "") or ""
            events.append({
                "date": date,
                "time": time_str,
                "currency": currency,
                "event": title,
                "impact": impact,
            })
    except Exception:
        pass

    _CALENDAR_CACHE["ff"] = (now, events)
    return events


def _get_upcoming_economic_events(days_ahead: int) -> list[dict[str, str]]:
    """Canonical merged event list used by both chat output and UI rail."""
    days_ahead = max(1, min(int(days_ahead), 30))
    events = _get_static_calendar_events(days_ahead)
    try:
        ff = _fetch_forexfactory_calendar()
        if ff:
            # Keep the same dedupe behavior used by chat to preserve parity.
            existing = {(e["date"], e["event"][:20]) for e in events}
            for f in ff:
                key = (f["date"], f["event"][:20])
                if key not in existing:
                    events.append(f)
    except Exception:
        pass
    events.sort(key=lambda e: e.get("date", ""))
    return events


def _exec_get_economic_calendar(args: dict, **_: Any) -> str:
    """Get upcoming economic events for USD/JPY."""
    days_ahead = min(int(args.get("days_ahead", 7)), 30)
    events = _get_upcoming_economic_events(days_ahead)

    if not events:
        return f"No high-impact USD/JPY events found in the next {days_ahead} days."

    lines = [f"Upcoming USD/JPY events (next {days_ahead}d):"]
    for e in events:
        lines.append(f"  {e['date']} {e.get('time', '')} [{e['currency']}] {e['event']} ({e['impact']})")
    return "\n".join(lines)


def _parse_event_datetime_utc(event: dict[str, str]) -> datetime | None:
    """Best-effort parser for calendar event date/time into UTC datetime."""
    d = str(event.get("date") or "").strip()
    if not d:
        return None
    base: datetime | None = None
    # Support static format plus common ForexFactory variants.
    date_formats = ("%Y-%m-%d", "%m-%d-%Y", "%b %d", "%a%b %d", "%a %b %d")
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(d, fmt)
            if "%Y" in fmt:
                base = parsed.replace(tzinfo=timezone.utc)
            else:
                now_utc = datetime.now(timezone.utc)
                base = parsed.replace(year=now_utc.year, tzinfo=timezone.utc)
                # If month/day already passed in current year, treat as next year.
                if base.date() < now_utc.date() - timedelta(days=2):
                    base = base.replace(year=now_utc.year + 1)
            break
        except Exception:
            continue
    if base is None:
        return None
    t = str(event.get("time") or "").strip().upper()
    # Expected formats: "18:00 UTC", "~03:00 UTC", "12:30 UTC (approx)"
    import re as _re
    m = _re.search(r"(\d{1,2}):(\d{2})", t)
    if not m:
        # Keep event on that date even when time is "All Day"/"Tentative".
        return base.replace(hour=12, minute=0, second=0, microsecond=0)
    hh = int(m.group(1))
    mm = int(m.group(2))
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


def get_economic_calendar_events(days_ahead: int = 7, limit: int = 3) -> list[dict[str, Any]]:
    """Structured upcoming USD/JPY events for UI tiles."""
    days_ahead = max(1, min(int(days_ahead), 30))
    events = _get_upcoming_economic_events(days_ahead)
    now_utc = datetime.now(timezone.utc)
    structured: list[dict[str, Any]] = []
    for e in events:
        dt = _parse_event_datetime_utc(e)
        if not dt:
            continue
        if dt < now_utc:
            continue
        mins = int((dt - now_utc).total_seconds() // 60)
        structured.append({
            "timestamp_utc": dt.isoformat(),
            "date": str(e.get("date", "")),
            "time": str(e.get("time", "")),
            "currency": str(e.get("currency", "")),
            "event": str(e.get("event", "")),
            "impact": str(e.get("impact", "")),
            "minutes_to_event": mins,
        })
    structured.sort(key=lambda x: str(x.get("timestamp_utc", "")))
    return structured[: max(1, int(limit))]


# ---------------------------------------------------------------------------
# 3c. News headlines (free RSS)
# ---------------------------------------------------------------------------

_NEWS_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
_NEWS_CACHE_TTL = 600  # 10 minutes

_RSS_FEEDS = [
    ("Google News", "https://news.google.com/rss/search?q=USDJPY+forex&hl=en-US&gl=US&ceid=US:en"),
    ("Google News JPY", "https://news.google.com/rss/search?q=japanese+yen+dollar&hl=en-US&gl=US&ceid=US:en"),
]


def _exec_get_news_headlines(args: dict, **_: Any) -> str:
    """Fetch recent forex/USDJPY news from RSS feeds."""
    count = min(int(args.get("count", 10)), 20)
    now = _time.time()

    cached = _NEWS_CACHE.get("news")
    if cached and now - cached[0] < _NEWS_CACHE_TTL:
        return _format_headlines(cached[1], count)

    headlines: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for source_name, url in _RSS_FEEDS:
        try:
            req = _Request(url, headers={"User-Agent": "USDJPY-Assistant/1.0"})
            with _urlopen(req, timeout=5) as resp:
                raw = resp.read()
            tree = _ET.fromstring(raw)
            for item in tree.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                pub_date = (item.findtext("pubDate") or "").strip()
                source = item.findtext("source") or source_name
                if hasattr(source, "text"):
                    source = source
                headlines.append({"title": title, "date": pub_date, "source": str(source)})
        except Exception:
            continue

    _NEWS_CACHE["news"] = (now, headlines)
    return _format_headlines(headlines, count)


def _format_headlines(headlines: list[dict[str, str]], count: int) -> str:
    if not headlines:
        return "No recent USDJPY/forex news headlines available."
    lines = [f"Recent forex news ({min(count, len(headlines))} headlines):"]
    for h in headlines[:count]:
        date_short = h["date"][:22] if h["date"] else ""
        lines.append(f"  [{h['source']}] {h['title']} ({date_short})")
    return "\n".join(lines)


def _headline_freshness_summary(headlines: list[dict[str, str]]) -> tuple[str, list[str]]:
    from email.utils import parsedate_to_datetime

    if not headlines:
        return "unavailable", ["No recent headlines were fetched. Treat catalyst awareness as degraded."]

    now_utc = datetime.now(timezone.utc)
    newest_age_hours: float | None = None
    notes: list[str] = []
    for h in headlines:
        raw_date = str(h.get("date") or "").strip()
        if not raw_date:
            continue
        try:
            dt = parsedate_to_datetime(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (now_utc - dt.astimezone(timezone.utc)).total_seconds() / 3600.0)
            newest_age_hours = age_hours if newest_age_hours is None else min(newest_age_hours, age_hours)
        except Exception:
            continue

    joined_titles = " | ".join(str(h.get("title") or "") for h in headlines).lower()
    if newest_age_hours is None:
        freshness = "unknown"
        notes.append("Headline timestamps were missing or unparsable. Treat catalyst timing as uncertain.")
    elif newest_age_hours <= 4:
        freshness = f"fresh (~{newest_age_hours:.1f}h)"
    elif newest_age_hours <= 24:
        freshness = f"aging (~{newest_age_hours:.1f}h)"
        notes.append("Latest headline is not fresh intraday. Check whether the catalyst is already priced.")
    else:
        freshness = f"stale (~{newest_age_hours:.1f}h)"
        notes.append("Headline flow is stale relative to current session. Be careful with narrative-driven trades.")

    if any(
        token in joined_titles
        for token in ("weekend", "strait of hormuz", "hormuz", "iran", "israel", "missile", "oil spike", "middle east")
    ):
        notes.append("Major geopolitical headline detected. Verify recency and whether this is weekend carryover rather than fresh intraday information.")
    return freshness, notes


# ---------------------------------------------------------------------------
# 3d. Web search (Brave Search API — free tier: 2,000 queries/month)
# ---------------------------------------------------------------------------

def _exec_web_search(args: dict, **_: Any) -> str:
    """Search the web via Brave Search API. Requires BRAVE_SEARCH_API_KEY env var."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return "Web search not available — BRAVE_SEARCH_API_KEY is not configured on the server."

    query = (args.get("query") or "").strip()
    if not query:
        return "No search query provided."

    count = min(int(args.get("count", 5)), 10)

    try:
        import urllib.parse
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://api.search.brave.com/res/v1/web/search?q={encoded_q}&count={count}"
        req = _Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        })
        with _urlopen(req, timeout=8) as resp:
            raw = resp.read()
            # Handle gzip
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            data = json.loads(raw)
    except Exception as e:
        return f"Web search failed: {e}"

    results = data.get("web", {}).get("results", [])
    if not results:
        return f"No web results found for: {query}"

    lines = [f"Web search results for \"{query}\" ({len(results)} results):"]
    for r in results[:count]:
        title = r.get("title", "")
        url_str = r.get("url", "")
        description = r.get("description", "")
        # Strip HTML tags from description
        import re as _re
        description = _re.sub(r"<[^>]+>", "", description)
        age = r.get("age", "")
        age_str = f" ({age})" if age else ""
        lines.append(f"\n  [{title}]{age_str}")
        lines.append(f"  {url_str}")
        if description:
            lines.append(f"  {description[:200]}")
    return "\n".join(lines)


def _search_pair_from_symbol(symbol: str) -> str:
    """Turn broker symbols like USD_JPY / USDJPY into 'USD JPY' for search queries."""
    raw = (symbol or "USD_JPY").strip().upper().replace("/", "_")
    if "_" in raw:
        a, b = raw.split("_", 1)
        return f"{a} {b}".strip()
    if len(raw) == 6 and raw.isalpha():
        return f"{raw[:3]} {raw[3:]}"
    return raw.replace("_", " ").strip() or "USD JPY"


def build_trade_suggestion_news_block(
    *,
    symbol: str = "USD_JPY",
    rss_headline_count: int = 12,
    web_result_count: int = 5,
    parallel_timeout_sec: float = 14.0,
) -> str:
    """Prefetch RSS headlines + Brave web results for ai-suggest-trade only.

    Runs RSS and web search in parallel. Web search is a no-op message if
    BRAVE_SEARCH_API_KEY is unset (same behavior as the chat tool).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pair = _search_pair_from_symbol(symbol)
    web_query = f"{pair} forex yen dollar news price movement market drivers"
    rss_n = max(1, min(int(rss_headline_count), 20))
    web_n = max(1, min(int(web_result_count), 10))

    rss_text = ""
    web_text = ""
    rss_headlines: list[dict[str, str]] = []

    def _rss() -> str:
        nonlocal rss_headlines
        now = _time.time()
        cached = _NEWS_CACHE.get("news")
        if cached and now - cached[0] < _NEWS_CACHE_TTL:
            rss_headlines = list(cached[1])
            return _format_headlines(rss_headlines, rss_n)
        _exec_get_news_headlines({"count": rss_n})
        cached = _NEWS_CACHE.get("news")
        rss_headlines = list((cached or (0, []))[1])
        return _format_headlines(rss_headlines, rss_n)

    def _web() -> str:
        return _exec_web_search({"query": web_query, "count": web_n})

    f_rss = None
    f_web = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_rss = pool.submit(_rss)
            f_web = pool.submit(_web)
            for fut in as_completed([f_rss, f_web], timeout=parallel_timeout_sec):
                try:
                    out = fut.result()
                    if fut is f_rss:
                        rss_text = out
                    else:
                        web_text = out
                except Exception as e:
                    if fut is f_rss:
                        rss_text = f"RSS headlines unavailable: {e}"
                    else:
                        web_text = f"Web search unavailable: {e}"
    except TimeoutError:
        pass
    if not rss_text:
        rss_text = "RSS headlines unavailable: timed out"
    if not web_text:
        web_text = "Web search unavailable: timed out"

    freshness, freshness_notes = _headline_freshness_summary(rss_headlines)
    freshness_lines = "\n".join(f"- {note}" for note in freshness_notes) if freshness_notes else "- No special freshness warnings."

    return (
        "=== TRADE SUGGESTION — EXTERNAL MARKET NEWS (prefetched for this request only) ===\n"
        "Use for catalysts and narratives only; live price, spread, and levels in "
        "LIVE TRADING CONTEXT above are authoritative.\n"
        f"PREFETCHED_AT_UTC: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}\n"
        f"NEWS_FRESHNESS: {freshness}\n"
        "FRESHNESS_NOTES:\n"
        f"{freshness_lines}\n\n"
        "[RSS HEADLINES]\n"
        + rss_text
        + "\n\n[WEB SEARCH]\n"
        + web_text
    )


# ---------------------------------------------------------------------------
# 3e. Tool dispatch
# ---------------------------------------------------------------------------

_TOOL_EXECUTORS: dict[str, Any] = {
    "get_candles": _exec_get_candles,
    "get_trade_history": _exec_get_trade_history,
    "get_ai_suggestion_history": _exec_get_ai_suggestion_history,
    "get_fillmore_system_info": _exec_get_fillmore_system_info,
    "analyze_trade_patterns": _exec_analyze_trade_patterns,
    "get_cross_asset_bias": lambda args, **kw: _exec_get_cross_asset_bias(**kw),
    "get_economic_calendar": _exec_get_economic_calendar,
    "get_news_headlines": _exec_get_news_headlines,
    "get_pending_orders": _exec_get_pending_orders,
    "web_search": _exec_web_search,
}


def _execute_tool(name: str, args_str: str, profile: ProfileV1, profile_name: str) -> str:
    """Execute a tool by name and return the result as a string."""
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        args = {}

    executor = _TOOL_EXECUTORS.get(name)
    if not executor:
        return f"Unknown tool: {name}"

    try:
        return executor(args, profile=profile, profile_name=profile_name)
    except Exception as e:
        return f"Tool error ({name}): {e}"


# ---------------------------------------------------------------------------
# 4. OpenAI streaming with function calling
# ---------------------------------------------------------------------------

def stream_openai_chat(
    *,
    system: str,
    user_message: str,
    history: list[dict[str, str]],
    model: str | None = None,
    profile: ProfileV1 | None = None,
    profile_name: str = "",
) -> Generator[str, None, None]:
    """Yield SSE-formatted lines with function calling support.

    SSE events:
      {"type":"delta","text":"..."}        — text content
      {"type":"tool_status","name":"..."}  — tool being executed
      {"type":"done"}                      — stream finished
    """
    import openai

    client = openai.OpenAI()

    if model is None:
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    max_tool_rounds = 3  # prevent infinite tool loops
    tools_available = profile is not None

    for _round in range(max_tool_rounds + 1):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools_available:
            create_kwargs["tools"] = _AI_CHAT_TOOLS

        stream = client.chat.completions.create(**create_kwargs)

        # Accumulate tool calls across streamed chunks
        tool_calls_acc: dict[int, dict[str, str]] = {}  # index -> {id, name, arguments}
        finish_reason = None

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            if delta and delta.content:
                yield f"data: {json.dumps({'type': 'delta', 'text': delta.content})}\n\n"

            # Accumulate tool call deltas
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_acc[idx]["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # If no tool calls, we're done
        if finish_reason != "tool_calls" or not tool_calls_acc or not tools_available:
            break

        # Execute tool calls and append results
        # First append the assistant message with tool_calls
        assistant_tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            assistant_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })

        messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})

        # Execute tool calls in parallel when multiple are requested
        # Notify frontend of all tools first
        for tc_msg in assistant_tool_calls:
            yield f"data: {json.dumps({'type': 'tool_status', 'name': tc_msg['function']['name']})}\n\n"

        if len(assistant_tool_calls) == 1:
            tc_msg = assistant_tool_calls[0]
            result = _execute_tool(tc_msg["function"]["name"], tc_msg["function"]["arguments"], profile, profile_name)  # type: ignore[arg-type]
            messages.append({"role": "tool", "tool_call_id": tc_msg["id"], "content": result})
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _run_tool(tc_msg: dict) -> tuple[str, str]:
                res = _execute_tool(tc_msg["function"]["name"], tc_msg["function"]["arguments"], profile, profile_name)  # type: ignore[arg-type]
                return (tc_msg["id"], res)

            tool_results: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=len(assistant_tool_calls)) as pool:
                futs = {pool.submit(_run_tool, tc): tc["id"] for tc in assistant_tool_calls}
                for fut in as_completed(futs):
                    try:
                        tool_id, res = fut.result()
                        tool_results[tool_id] = res
                    except Exception as e:
                        tool_results[futs[fut]] = f"Tool error: {e}"

            # Append in original order to keep message sequence deterministic
            for tc_msg in assistant_tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_msg["id"],
                    "content": tool_results.get(tc_msg["id"], "Tool execution failed."),
                })

        # Loop back to get the model's response after tool results

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
