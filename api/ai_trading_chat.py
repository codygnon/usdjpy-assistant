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


def _compute_ta_snapshot(adapter: Any, profile: ProfileV1) -> dict[str, dict[str, Any]] | None:
    """Compute TA for H1, M5, M1 only. Returns {timeframe: {regime, rsi, ...}} or None."""
    from core.ta_analysis import compute_ta_for_tf

    result: dict[str, dict[str, Any]] = {}
    pip_size = float(profile.pip_size) if profile.pip_size else 0.01

    for tf in ("H1", "M5", "M1"):
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

    # --- Dispatch all calls in parallel ---

    try:
        adapter.initialize()
        if hasattr(adapter, "ensure_symbol"):
            adapter.ensure_symbol(profile.symbol)

        tasks = [_fetch_account, _fetch_tick, _fetch_positions, _fetch_closed_trades,
                 _fetch_vol, _fetch_ta, _fetch_ohlc]

        if is_oanda:
            tasks.extend([_fetch_cross_assets, _fetch_macro_bias, _fetch_order_book, _fetch_pos_book, _fetch_pending_orders])

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
        "Be concise. Lead with numbers. Never give imperative trade instructions like 'buy now' or 'sell now'.",
        "You may discuss market context, account state, position sizing, and risk management.",
        f"MODEL IDENTITY: You are '{effective_model}'. If asked what model, say exactly that. If asked who you are, you're Fillmore.",
        "CAPABILITIES: You have live broker context AND function-calling tools. Available tools:",
        "  - get_candles(timeframe, count): Fetch OHLC candles for any timeframe (M1/M5/M15/H1/H4/D)",
        "  - get_trade_history(days_back, limit): Query closed trades from database",
        "  - analyze_trade_patterns(days_back): Win/loss stats by session, tier, entry type",
        "  - get_cross_asset_bias(): Full macro bias reading (oil, DXY, combined USDJPY implication)",
        "  - get_economic_calendar(days_ahead): Upcoming high-impact USD/JPY events (FOMC, NFP, BOJ, CPI)",
        "  - get_news_headlines(count): Recent USDJPY/forex news from RSS feeds",
        "  - web_search(query, count): Full web search via Brave Search — use for current events, analysis, central bank statements, geopolitical context, or any question needing live web data",
        "Use tools proactively when the user's question would benefit from fresh data. Prefer web_search for broad questions about markets or events. Use get_news_headlines for quick headline scans. Use get_economic_calendar for upcoming events, and get_trade_history for specific past trades.",
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
        "DELIVERY (desk style): Target ~80 words for simple questions; ~150 max for multi-part. One tight paragraph or short bullets. Personality shows up in *how* you phrase the numbers, not in extra words — never pad.",
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
        "When no positions are open, keep responses under 80 words.",
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

    spot = ctx.get("spot_price")
    if spot:
        lines.append("")
        spread = spot.get("spread_pips")
        spread_txt = f" | spread {spread}p" if spread is not None else ""
        lines.append(
            f"LIVE PRICE ({ctx.get('symbol', 'USDJPY')}): {spot['bid']:.3f}/{spot['ask']:.3f} "
            f"(mid {spot['mid']:.3f}){spread_txt}"
        )

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
        for tf in ("H1", "M5", "M1"):
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
            lines.append(parts_line)

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

    def _rss() -> str:
        return _exec_get_news_headlines({"count": rss_n})

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

    return (
        "=== TRADE SUGGESTION — EXTERNAL MARKET NEWS (prefetched for this request only) ===\n"
        "Use for catalysts and narratives only; live price, spread, and levels in "
        "LIVE TRADING CONTEXT above are authoritative.\n\n"
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
