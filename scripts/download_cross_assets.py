"""
Download cross-asset data from OANDA API for the Cross-Asset Confluence strategy.

Instruments:
  - BCO_USD  (Brent Crude Oil)  → 1H bars  (primary signal for JPY)
  - EUR_USD  (DXY proxy)        → 1H bars  (primary signal for USD)
  - XAU_USD  (Gold)             → Daily bars (confirmation)
  - XAG_USD  (Silver)           → Daily bars (confirmation)

Date range matches USDJPY dataset: 2023-06-20 to 2026-03-04

Usage:
  python3 scripts/download_cross_assets.py --api-key YOUR_KEY_HERE --account-type practice

  Or set environment variable:
    export OANDA_API_KEY="your-key-here"
    python3 scripts/download_cross_assets.py

  CSVs default to research_out/cross_assets/ (same area as other research data).
  If your repo has a file named `data` instead of a folder, avoid data/... paths;
  use --output-dir or remove/rename the stray `data` file.

  If you see SSL: CERTIFICATE_VERIFY_FAILED (common with python.org Python on macOS):
    pip install certifi   # then re-run (this script uses certifi when available)
    # or run Python from the project venv where `requests` already pulled in certifi
    # last resort: --insecure (disables TLS verification; not recommended)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:
    certifi = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[1]


def _build_ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()

# ─── Configuration ───────────────────────────────────────────────────────────

INSTRUMENTS = {
    "BCO_USD": {
        "name": "Brent Crude Oil",
        "granularity": "H1",
        "description": "Primary JPY signal - Japan imports all oil",
    },
    "EUR_USD": {
        "name": "EUR/USD (DXY proxy)",
        "granularity": "H1",
        "description": "Primary USD signal - inverted EURUSD ≈ 0.95 corr with DXY",
    },
    "XAU_USD": {
        "name": "Gold",
        "granularity": "D",
        "description": "Confirmation signal - anti-dollar sentiment",
    },
    "XAG_USD": {
        "name": "Silver",
        "granularity": "D",
        "description": "Confirmation signal - anti-dollar + industrial",
    },
}

# Match your USDJPY dataset range
DATE_FROM = "2023-06-20T00:00:00Z"
DATE_TO = "2026-03-04T00:00:00Z"

# OANDA API limits (count ignored if both from+to supplied; we paginate with from+count only)
MAX_CANDLES_PER_REQUEST = 5000

# Default alongside other research CSVs (repo `data` may be a stray file — see --output-dir)
DEFAULT_OUTPUT_DIR = ROOT / "research_out" / "cross_assets"

BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def _parse_oanda_iso_time(raw_time: str) -> datetime:
    """Parse OANDA candle time (nanosecond ISO) to aware UTC."""
    s = raw_time.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        base, frac = s.split(".", 1)
        frac_digits = "".join(c for c in frac if c.isdigit())
        frac_digits = (frac_digits + "000000")[:6]
        s = f"{base}.{frac_digits}"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ensure_writable_output_dir(path: Path) -> Path:
    """Create output dir if needed; exit with a clear message if a parent path is a file."""
    path = path.expanduser().resolve()
    cur: Path | None = path
    while cur is not None:
        if cur.exists() and not cur.is_dir():
            print(
                f"ERROR: '{cur}' exists but is not a directory.\n"
                f"Cannot create output directory: {path}\n"
                "Remove or rename that path, or use --output-dir with a different folder.",
                file=sys.stderr,
            )
            sys.exit(1)
        if cur == cur.parent:
            break
        cur = cur.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_api_key(args: argparse.Namespace) -> str:
    """Get API key from args or environment."""
    if args.api_key:
        return args.api_key

    env_key = os.environ.get("OANDA_API_KEY")
    if env_key:
        return env_key

    env_file = ROOT / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OANDA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: No API key found.")
    print("Provide via --api-key argument, OANDA_API_KEY env var, or .env file")
    sys.exit(1)


def fetch_candles(
    base_url: str,
    api_key: str,
    instrument: str,
    granularity: str,
    date_from: str,
    date_to: str,
    ssl_context: ssl.SSLContext,
) -> list[dict] | None:
    """
    Fetch candles from OANDA API with pagination.

    OANDA: if both `from` and `to` are set, `count` is ignored — large ranges
    would not page correctly. We request `from` + `count` only and advance
    `from` after each batch until past `date_to` or a short batch.
    """
    date_to_s = date_to.strip()
    if not date_to_s.endswith("Z") and "+" not in date_to_s[-6:]:
        date_to_s = date_to_s.rstrip() + "Z"
    date_to_dt = _parse_oanda_iso_time(date_to_s)

    if not date_from.endswith("Z") and "+" not in date_from.strip()[-6:]:
        date_from = date_from.strip().rstrip() + "Z"
    else:
        date_from = date_from.strip()
    current_from = date_from

    all_candles: list[dict] = []
    request_count = 0

    while True:
        request_count += 1
        params = {
            "granularity": granularity,
            "from": current_from,
            "count": str(MAX_CANDLES_PER_REQUEST),
            "price": "M",
        }
        url = f"{base_url}/v3/instruments/{instrument}/candles?{urlencode(params)}"

        req = Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

        print(f"  Request {request_count}: from {current_from[:19]}...", end=" ", flush=True)

        try:
            response = urlopen(req, timeout=120, context=ssl_context)
            data = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            print(f"\n  HTTP ERROR {e.code}: {error_body}")
            if e.code == 401:
                print("  -> Check your API key")
            elif e.code == 400:
                print(f"  -> Bad request. Instrument '{instrument}' may not be available")
                print("     on your account type. Try the other account type.")
            return None
        except URLError as e:
            print(f"\n  CONNECTION ERROR: {e.reason}")
            err_s = str(e.reason)
            if "CERTIFICATE_VERIFY_FAILED" in err_s or "certificate verify failed" in err_s.lower():
                print("  -> TLS certificate verify failed (typical on macOS + python.org Python).")
                if certifi is None:
                    print("     Install CA bundle:  pip install certifi")
                    print("     Or use the project venv:  .venv/bin/python scripts/download_cross_assets.py ...")
                else:
                    print("     certifi is installed; if this persists, try --insecure (not recommended).")
                print("     Or run:  /Applications/Python\\ 3.*/Install\\ Certificates.command")
            else:
                print("  -> Check your internet connection / firewall / proxy")
            return None

        candles = data.get("candles", [])
        if not candles:
            print("no more data")
            break

        complete_candles = [c for c in candles if c.get("complete", False)]
        if not complete_candles:
            print("no complete candles in batch (stop)")
            break
        # Keep only candles <= end of requested window
        in_range: list[dict] = []
        past_end = False
        for c in complete_candles:
            ct = _parse_oanda_iso_time(c["time"])
            if ct > date_to_dt:
                past_end = True
                break
            in_range.append(c)

        print(f"got {len(in_range)} complete candles (in range)")

        if not in_range and complete_candles:
            # First complete candle already past date_to
            break

        all_candles.extend(in_range)

        if past_end:
            break

        if len(candles) < MAX_CANDLES_PER_REQUEST:
            break

        last_time = complete_candles[-1]["time"]
        last_dt = _parse_oanda_iso_time(last_time)
        next_from_ts = last_dt.timestamp() + 1
        current_from = datetime.fromtimestamp(next_from_ts, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        if last_dt >= date_to_dt:
            break

        time.sleep(0.5)

    return all_candles


def candles_to_rows(candles: list[dict]) -> list[dict]:
    """Convert OANDA candle JSON to flat rows for CSV."""
    rows: list[dict] = []
    for candle in candles:
        mid = candle["mid"]
        dt = _parse_oanda_iso_time(candle["time"])
        rows.append(
            {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(candle.get("volume", 0)),
            }
        )
    return rows


def save_csv(rows: list[dict], filepath: Path) -> None:
    """Save rows to CSV."""
    if not rows:
        print(f"  WARNING: No data to save to {filepath}")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {filepath}")


def save_metadata(
    filepath: Path,
    instrument: str,
    config: dict,
    row_count: int,
    date_range: tuple[str | None, str | None],
) -> None:
    """Save metadata JSON alongside CSV for documentation."""
    meta = {
        "instrument": instrument,
        "name": config["name"],
        "description": config["description"],
        "granularity": config["granularity"],
        "price_type": "midpoint (average of bid/ask)",
        "row_count": row_count,
        "date_from": date_range[0] if date_range else None,
        "date_to": date_range[1] if date_range else None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "OANDA v20 API",
        "notes": [
            "Volume is tick volume (number of price updates), not real traded volume",
            "Timestamps are UTC",
            "Only complete candles are included",
            f"Granularity: {config['granularity']} "
            f"({'1 Hour' if config['granularity'] == 'H1' else 'Daily'})",
        ],
    }
    meta_path = filepath.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download cross-asset data from OANDA for strategy backtesting"
    )
    parser.add_argument(
        "--api-key",
        help="OANDA API key (or set OANDA_API_KEY env var)",
    )
    parser.add_argument(
        "--account-type",
        choices=["practice", "live"],
        default="practice",
        help="OANDA account type (default: practice)",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        choices=list(INSTRUMENTS.keys()) + ["all"],
        default=["all"],
        help="Which instruments to download (default: all)",
    )
    parser.add_argument(
        "--date-from",
        default=DATE_FROM,
        help=f"Start date ISO format (default: {DATE_FROM})",
    )
    parser.add_argument(
        "--date-to",
        default=DATE_TO,
        help=f"End date ISO format (default: {DATE_TO})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV + metadata (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (insecure; use only if CA fixes fail)",
    )

    args = parser.parse_args()

    if args.insecure:
        print(
            "WARNING: --insecure is set; TLS certificate verification is disabled.\n",
            file=sys.stderr,
        )
    elif certifi is not None:
        pass  # use certifi bundle in _build_ssl_context
    else:
        print(
            "Note: package 'certifi' not found; using default SSL context.\n"
            "If downloads fail with CERTIFICATE_VERIFY_FAILED, run:  pip install certifi\n",
            file=sys.stderr,
        )

    ssl_context = _build_ssl_context(args.insecure)

    api_key = get_api_key(args)
    base_url = BASE_URLS[args.account_type]

    if "all" in args.instruments:
        instruments_to_fetch = list(INSTRUMENTS.keys())
    else:
        instruments_to_fetch = args.instruments

    print("=" * 60)
    print("OANDA Cross-Asset Data Download")
    print("=" * 60)
    print(f"Account type: {args.account_type}")
    print(f"Base URL:     {base_url}")
    print(f"Date range:   {args.date_from[:10]} to {args.date_to[:10]}")
    print(f"Instruments:  {', '.join(instruments_to_fetch)}")
    output_dir = _ensure_writable_output_dir(args.output_dir)
    print(f"Output dir:   {output_dir}")
    print("=" * 60)

    results: dict[str, dict] = {}

    for instrument in instruments_to_fetch:
        config = INSTRUMENTS[instrument]
        print(f"\n{'-' * 60}")
        print(f"Downloading: {instrument} ({config['name']})")
        print(f"Granularity: {config['granularity']}")
        print(f"Purpose:     {config['description']}")
        print(f"{'-' * 60}")

        candles = fetch_candles(
            base_url=base_url,
            api_key=api_key,
            instrument=instrument,
            granularity=config["granularity"],
            date_from=args.date_from,
            date_to=args.date_to,
            ssl_context=ssl_context,
        )

        if candles is None:
            print(f"  FAILED to download {instrument}")
            results[instrument] = {"status": "FAILED", "rows": 0}
            continue

        if len(candles) == 0:
            print(f"  WARNING: No candles returned for {instrument}")
            print("  This instrument may not be available on your account.")
            print("  If using practice account, try --account-type live or vice versa.")
            results[instrument] = {"status": "NO_DATA", "rows": 0}
            continue

        rows = candles_to_rows(candles)
        granularity_label = "H1" if config["granularity"] == "H1" else "D"
        safe_name = instrument.replace("/", "_")
        filename = f"{safe_name}_{granularity_label}_OANDA.csv"
        filepath = output_dir / filename

        save_csv(rows, filepath)

        date_range = (rows[0]["timestamp"], rows[-1]["timestamp"]) if rows else (None, None)
        save_metadata(filepath, instrument, config, len(rows), date_range)

        results[instrument] = {
            "status": "OK",
            "rows": len(rows),
            "file": str(filepath),
            "date_from": date_range[0],
            "date_to": date_range[1],
        }

    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")

    all_ok = True
    for instrument, result in results.items():
        status = result["status"]
        rows = result["rows"]
        icon = "OK" if status == "OK" else "!!"
        print(f"  [{icon}] {instrument:12s}  {status:10s}  {rows:,} rows")
        if status != "OK":
            all_ok = False

    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "download_date": datetime.now(timezone.utc).isoformat(),
                "account_type": args.account_type,
                "date_from": args.date_from,
                "date_to": args.date_to,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")

    if all_ok:
        print(f"\n{'=' * 60}")
        print("ALL DOWNLOADS SUCCESSFUL")
        print(f"{'=' * 60}")
        print(f"\nData files are in: {output_dir}/")
        print("\nExpected files:")
        for inst, cfg in INSTRUMENTS.items():
            g = "H1" if cfg["granularity"] == "H1" else "D"
            print(f"  {inst}_{g}_OANDA.csv")
        print("\nNext step: Build the cross_asset_confluence strategy adapter")
        print("using these files as additional data inputs.")
    else:
        print(f"\n{'=' * 60}")
        print("SOME DOWNLOADS FAILED — see details above")
        print(f"{'=' * 60}")
        print("\nCommon fixes:")
        print("  - TLS / SSL: pip install certifi  OR  .venv/bin/python ...  OR  --insecure")
        print("  - Wrong account type: try --account-type live (or practice)")
        print("  - BCO_USD not available: some accounts don't have CFDs")
        print("    -> Check OANDA instrument list for your account")
        print("  - API key expired: regenerate at OANDA account settings")


if __name__ == "__main__":
    main()
