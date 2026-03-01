#!/usr/bin/env python3
"""
Fetch OANDA USD_JPY M1 candles into a backtest-ready CSV.

Output contract (for backtest_session_momentum.py):
  required columns: time,open,high,low,close
  optional column: spread_pips (enabled with --include-spread)

Auth/env:
  OANDA_API_KEY (required unless --api-key is passed)
  OANDA_ENV=practice|live (default: practice)

Example:
  python3 scripts/fetch_oanda_m1_csv.py \
    --bars 250000 \
    --out research_out/USDJPY_M1_OANDA_250k.csv

Then run backtest:
  python3 scripts/backtest_session_momentum.py \
    --config research_out/session_momentum_v5p3_rp075_v11a_maxopen2_fixed_config.json \
    --in research_out/USDJPY_M1_OANDA_250k.csv \
    --out research_out/session_momentum_v11a_oanda_250k.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import requests
except Exception as e:  # pragma: no cover
    print(f"Error: requests is required ({e})", file=sys.stderr)
    raise SystemExit(1)


PIP_SIZE = 0.01
MAX_OANDA_CANDLES_PER_CALL = 5000
BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


@dataclass
class FetchStats:
    requests: int = 0
    complete_candles_seen: int = 0
    rows_collected: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OANDA M1 USD_JPY candles into a backtest CSV")
    p.add_argument("--bars", type=int, default=250_000, help="Target number of completed M1 bars (default: 250000)")
    p.add_argument("--instrument", type=str, default="USD_JPY", help="OANDA instrument (default: USD_JPY)")
    p.add_argument("--granularity", type=str, default="M1", help="OANDA granularity (default: M1)")
    p.add_argument("--end", type=str, default=None, help="UTC end timestamp (ISO). Default: now UTC, floored to minute")
    p.add_argument("--out", type=str, default="research_out/USDJPY_M1_OANDA_250k.csv", help="Output CSV path")
    p.add_argument("--env", choices=["practice", "live"], default=None, help="OANDA environment; defaults to OANDA_ENV or practice")
    p.add_argument("--api-key", type=str, default=None, help="OANDA API key; defaults to OANDA_API_KEY env")
    p.add_argument("--sleep-seconds", type=float, default=0.12, help="Sleep between API requests (default: 0.12)")
    p.add_argument("--request-timeout", type=float, default=30.0, help="HTTP timeout seconds (default: 30)")
    p.add_argument("--include-spread", action="store_true", help="Add spread_pips column when bid/ask are available")
    p.add_argument("--retries", type=int, default=4, help="Retry count per request (default: 4)")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()


def _resolve_auth(args: argparse.Namespace) -> tuple[str, str]:
    env = (args.env or os.getenv("OANDA_ENV") or "practice").strip().lower()
    api_key = (args.api_key or os.getenv("OANDA_API_KEY") or "").strip()
    if env not in BASE_URLS:
        raise RuntimeError("OANDA env must be practice or live")
    if not api_key:
        raise RuntimeError("Missing OANDA API key. Set OANDA_API_KEY or pass --api-key")
    return BASE_URLS[env], api_key


def _parse_end(end_raw: Optional[str]) -> pd.Timestamp:
    if not end_raw:
        return pd.Timestamp.now(tz="UTC").floor("min")
    ts = pd.Timestamp(end_raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min")


def _fmt_oanda_time(ts: pd.Timestamp) -> str:
    ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _request_candles(
    *,
    session: requests.Session,
    base_url: str,
    api_key: str,
    instrument: str,
    granularity: str,
    count: int,
    to_ts: pd.Timestamp,
    include_spread: bool,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = f"{base_url}/v3/instruments/{instrument}/candles"
    price_mode = "MBA" if include_spread else "M"
    params = {
        "granularity": granularity,
        "count": str(count),
        "price": price_mode,
        "to": _fmt_oanda_time(to_ts),
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    body = resp.json()
                    detail = body.get("errorMessage", detail)
                except Exception:
                    pass
                raise RuntimeError(f"OANDA HTTP {resp.status_code}: {detail}")
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("OANDA response JSON root is not an object")
            return data
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(1.0, 0.2 * attempt))
    raise RuntimeError(f"Failed OANDA candle request after {retries} attempts: {last_err}")


def _parse_candle(c: dict[str, Any], include_spread: bool) -> Optional[dict[str, Any]]:
    if c.get("complete") is not True:
        return None
    t = c.get("time")
    if not isinstance(t, str):
        return None

    # Prefer OANDA mid if available.
    m = c.get("mid") or {}
    b = c.get("bid") or {}
    a = c.get("ask") or {}

    def _f(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    if m:
        o = _f(m.get("o"))
        h = _f(m.get("h"))
        l = _f(m.get("l"))
        cl = _f(m.get("c"))
    elif b and a:
        # Fallback derive midpoint OHLC from bid+ask OHLC pairs.
        bo, bh, bl, bc = _f(b.get("o")), _f(b.get("h")), _f(b.get("l")), _f(b.get("c"))
        ao, ah, al, ac = _f(a.get("o")), _f(a.get("h")), _f(a.get("l")), _f(a.get("c"))
        if None in (bo, bh, bl, bc, ao, ah, al, ac):
            return None
        o = (bo + ao) / 2.0
        h = (bh + ah) / 2.0
        l = (bl + al) / 2.0
        cl = (bc + ac) / 2.0
    else:
        return None

    if None in (o, h, l, cl):
        return None

    row: dict[str, Any] = {
        "time": t,
        "open": o,
        "high": h,
        "low": l,
        "close": cl,
    }

    if include_spread and b and a:
        bc = _f(b.get("c"))
        ac = _f(a.get("c"))
        if bc is not None and ac is not None:
            row["spread_pips"] = (ac - bc) / PIP_SIZE

    return row


def fetch_bars(args: argparse.Namespace) -> tuple[pd.DataFrame, FetchStats]:
    base_url, api_key = _resolve_auth(args)
    end_ts = _parse_end(args.end)

    target = int(args.bars)
    if target <= 0:
        raise RuntimeError("--bars must be > 0")

    stats = FetchStats()
    session = requests.Session()
    to_ts = end_ts
    rows: list[dict[str, Any]] = []
    seen_time: set[str] = set()

    while len(rows) < target:
        needed = target - len(rows)
        batch = min(MAX_OANDA_CANDLES_PER_CALL, max(needed + 64, 1000))
        payload = _request_candles(
            session=session,
            base_url=base_url,
            api_key=api_key,
            instrument=str(args.instrument),
            granularity=str(args.granularity),
            count=batch,
            to_ts=to_ts,
            include_spread=bool(args.include_spread),
            timeout=float(args.request_timeout),
            retries=int(args.retries),
        )
        stats.requests += 1

        candles = payload.get("candles", [])
        if not candles:
            if args.verbose:
                print("No more candles returned; stopping.")
            break

        oldest_time: Optional[pd.Timestamp] = None
        added = 0

        for c in candles:
            row = _parse_candle(c, include_spread=bool(args.include_spread))
            if row is None:
                continue
            stats.complete_candles_seen += 1
            t = str(row["time"])
            if t in seen_time:
                continue
            seen_time.add(t)
            rows.append(row)
            added += 1

        t0 = candles[0].get("time")
        if isinstance(t0, str):
            ot = pd.Timestamp(t0)
            if ot.tzinfo is None:
                ot = ot.tz_localize("UTC")
            else:
                ot = ot.tz_convert("UTC")
            oldest_time = ot

        if args.verbose:
            print(f"request={stats.requests} added={added} total={len(rows)} to={_fmt_oanda_time(to_ts)}")

        if oldest_time is None:
            break
        if oldest_time >= to_ts:
            # guard against loops on malformed data
            to_ts = to_ts - pd.Timedelta(minutes=1)
        else:
            to_ts = oldest_time

        if len(candles) < 2 and added == 0:
            break

        time.sleep(max(0.0, float(args.sleep_seconds)))

    stats.rows_collected = len(rows)
    df = pd.DataFrame(rows)
    return df, stats


def validate_and_prepare(df: pd.DataFrame, bars: int, include_spread: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["time", "open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in fetched data: {c}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if include_spread and "spread_pips" in out.columns:
        out["spread_pips"] = pd.to_numeric(out["spread_pips"], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    dropped_nan = before - len(out)

    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    invalid_ohlc = (
        (out["high"] < out["low"]) |
        (out["high"] < out["open"]) |
        (out["high"] < out["close"]) |
        (out["low"] > out["open"]) |
        (out["low"] > out["close"])
    )
    bad_ohlc_count = int(invalid_ohlc.sum())
    if bad_ohlc_count > 0:
        out = out.loc[~invalid_ohlc].copy()

    if len(out) > bars:
        out = out.iloc[-bars:].copy()

    out = out.reset_index(drop=True)

    gaps = out["time"].diff().dt.total_seconds().div(60.0)
    gap_mask = gaps > 1.0
    gap_count = int(gap_mask.sum())
    max_gap_minutes = float(gaps.max()) if len(gaps) else 0.0

    info = {
        "rows": int(len(out)),
        "dropped_nan": int(dropped_nan),
        "bad_ohlc_removed": int(bad_ohlc_count),
        "gap_count_gt_1m": int(gap_count),
        "max_gap_minutes": round(max_gap_minutes, 3) if pd.notna(max_gap_minutes) else 0.0,
        "first_time": str(out["time"].iloc[0]) if len(out) else None,
        "last_time": str(out["time"].iloc[-1]) if len(out) else None,
    }

    if len(out) == 0:
        raise RuntimeError("No valid rows after validation")

    # Standardize time format to parse cleanly with pandas utc=True.
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace("+0000", "Z", regex=False)

    cols = required + (["spread_pips"] if include_spread and "spread_pips" in out.columns else [])
    out = out[cols]
    return out, info


def main() -> None:
    args = parse_args()

    try:
        df, stats = fetch_bars(args)
        cleaned, info = validate_and_prepare(df, bars=int(args.bars), include_spread=bool(args.include_spread))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Wrote CSV: {out_path}")
    print(f"Rows: {info['rows']} (requested: {args.bars})")
    print(f"Time range: {info['first_time']} -> {info['last_time']}")
    print(f"Requests: {stats.requests} | complete candles parsed: {stats.complete_candles_seen}")
    print(f"Dropped NaN rows: {info['dropped_nan']} | Removed bad OHLC rows: {info['bad_ohlc_removed']}")
    print(f"Gaps >1m: {info['gap_count_gt_1m']} | Max gap minutes: {info['max_gap_minutes']}")
    if int(info["rows"]) < int(args.bars):
        print("Warning: fewer rows than requested. Increase history window or verify broker history availability.")


if __name__ == "__main__":
    main()
