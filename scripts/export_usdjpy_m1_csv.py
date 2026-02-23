#!/usr/bin/env python3
"""
Usage:
  export OANDA_ENV=practice
  export OANDA_API_KEY=...
  python scripts/export_usdjpy_m1_csv.py --count 500 --out USDJPY_M1_recent.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

MAX_OANDA_CANDLES_PER_CALL = 5000


BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export recent completed USD_JPY M1 candles from OANDA to CSV.")
    p.add_argument("--count", type=int, default=500, help="Number of completed candles to export (default: 500)")
    p.add_argument("--out", default="USDJPY_M1_recent.csv", help="Output CSV path (default: USDJPY_M1_recent.csv)")
    return p.parse_args()


def _get_env_or_exit() -> tuple[str, str]:
    api_key = (os.getenv("OANDA_API_KEY") or "").strip()
    env = (os.getenv("OANDA_ENV") or "practice").strip().lower()
    _ = os.getenv("OANDA_ACCOUNT_ID")  # Optional for this endpoint; read for compatibility.

    if not api_key:
        print("Error: OANDA_API_KEY is missing. Set OANDA_API_KEY in your environment.", file=sys.stderr)
        raise SystemExit(1)
    if env not in BASE_URLS:
        print("Error: OANDA_ENV must be 'practice' or 'live'.", file=sys.stderr)
        raise SystemExit(1)
    return api_key, BASE_URLS[env]


def _request_json(base_url: str, api_key: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{base_url}/v3/instruments/USD_JPY/candles"
    headers = {"Authorization": f"Bearer {api_key}"}

    if requests is not None:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"Error: OANDA request failed: {e}", file=sys.stderr)
            raise SystemExit(1)

        if resp.status_code >= 400:
            detail = resp.text
            try:
                body: Any = resp.json()
                detail = body.get("errorMessage", detail)
            except Exception:
                pass
            print(f"Error: OANDA API returned {resp.status_code}: {detail}", file=sys.stderr)
            raise SystemExit(1)

        try:
            return resp.json()
        except ValueError:
            print("Error: OANDA API response was not valid JSON.", file=sys.stderr)
            raise SystemExit(1)

    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(url=f"{url}?{query}", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status = getattr(r, "status", 200)
            raw = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"Error: OANDA request failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    if status >= 400:
        detail = raw
        try:
            body = json.loads(raw)
            detail = body.get("errorMessage", detail)
        except Exception:
            pass
        print(f"Error: OANDA API returned {status}: {detail}", file=sys.stderr)
        raise SystemExit(1)

    try:
        return json.loads(raw)
    except ValueError:
        print("Error: OANDA API response was not valid JSON.", file=sys.stderr)
        raise SystemExit(1)


def fetch_recent_completed_candles(api_key: str, base_url: str, count: int) -> list[dict[str, str]]:
    if count < 1:
        print("Error: --count must be >= 1.", file=sys.stderr)
        raise SystemExit(1)

    completed: list[dict[str, Any]] = []
    seen_times: set[str] = set()
    to_time: str | None = None

    # Walk backward in time until we gather the requested number of completed candles.
    while len(completed) < count:
        remaining = count - len(completed)
        fetch_count = min(MAX_OANDA_CANDLES_PER_CALL, remaining + 50)
        params = {
            "granularity": "M1",
            "count": str(fetch_count),
            "price": "M",
        }
        if to_time:
            params["to"] = to_time

        payload = _request_json(base_url=base_url, api_key=api_key, params=params)
        candles = payload.get("candles", [])
        if not candles:
            break

        for c in candles:
            if c.get("complete") is not True:
                continue
            t = c.get("time")
            if not isinstance(t, str):
                continue
            if t in seen_times:
                continue
            seen_times.add(t)
            completed.append(c)

        oldest = candles[0].get("time")
        if not isinstance(oldest, str):
            break
        if to_time == oldest:
            break
        to_time = oldest

    completed = sorted(completed, key=lambda x: x.get("time", ""))[-count:]

    rows: list[dict[str, str]] = []
    for candle in completed:
        mid = candle.get("mid") or {}
        try:
            rows.append(
                {
                    "time": candle["time"],
                    "open": mid["o"],
                    "high": mid["h"],
                    "low": mid["l"],
                    "close": mid["c"],
                }
            )
        except KeyError as e:
            print(f"Error: Unexpected candle schema from OANDA (missing key: {e}).", file=sys.stderr)
            raise SystemExit(1)
    return rows


def write_csv(rows: list[dict[str, str]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "open", "high", "low", "close"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    api_key, base_url = _get_env_or_exit()
    rows = fetch_recent_completed_candles(api_key=api_key, base_url=base_url, count=args.count)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
