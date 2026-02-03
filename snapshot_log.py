import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from storage.sqlite_store import SqliteStore

BASE_DIR = Path(__file__).resolve().parent

# Strategy definition (we can move these into the profile later)
EMA_FAST = 13
SMA_SLOW = 30

BARS_M1 = 2000
BARS_M15 = 2000
BARS_H4 = 800


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    if "profile_name" not in prof:
        raise ValueError("profile missing 'profile_name'")
    if "symbol" not in prof:
        raise ValueError("profile missing 'symbol'")
    prof.setdefault("pip_size", 0.01)
    return prof


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def fetch_df(symbol: str, timeframe, bars: int) -> pd.DataFrame:
    r = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if r is None or len(r) == 0:
        raise RuntimeError(f"copy_rates_from_pos returned no data for {symbol}, tf={timeframe}, err={mt5.last_error()}")
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def compute_cross_and_trend(df: pd.DataFrame) -> dict:
    df = df.copy()
    close = df["close"]

    df["ema"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["sma"] = close.rolling(SMA_SLOW).mean()
    df["diff"] = df["ema"] - df["sma"]

    df["cross_up"] = (df["diff"].shift(1) <= 0) & (df["diff"] > 0)
    df["cross_dn"] = (df["diff"].shift(1) >= 0) & (df["diff"] < 0)

    last = df.iloc[-1]
    diff_now = last["diff"]

    if pd.isna(diff_now):
        regime = "unknown"
    elif diff_now > 0:
        regime = "bull"
    elif diff_now < 0:
        regime = "bear"
    else:
        regime = "flat"

    cross_idx = df.index[df["cross_up"] | df["cross_dn"]]
    if len(cross_idx) == 0:
        return {
            "regime": regime,
            "last_cross_dir": None,
            "last_cross_time": None,
            "last_cross_price": None,
            "trend_since_cross": "unknown",
        }

    i = cross_idx[-1]
    cross_row = df.loc[i]
    last_cross_dir = "up" if bool(cross_row["cross_up"]) else "down"
    last_cross_time = cross_row["time"]
    last_cross_price = float(cross_row["close"])

    after = df.loc[i:].copy().dropna(subset=["diff"])
    if len(after) < 10:
        trend_since = "unknown"
    else:
        frac_pos = float((after["diff"] > 0).mean())
        frac_neg = float((after["diff"] < 0).mean())
        if frac_pos >= 0.65:
            trend_since = "up"
        elif frac_neg >= 0.65:
            trend_since = "down"
        else:
            trend_since = "sideways"

    return {
        "regime": regime,
        "last_cross_dir": last_cross_dir,
        "last_cross_time": last_cross_time,
        "last_cross_price": last_cross_price,
        "trend_since_cross": trend_since,
    }


def _cross_time_iso(t) -> str | None:
    """Convert last_cross_time (pd.Timestamp or None) to ISO string for SQLite."""
    if t is None:
        return None
    if hasattr(t, "isoformat"):
        return t.isoformat()
    return pd.Timestamp(t).isoformat()


def spread_pips(symbol: str, pip_size: float) -> float | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return float((tick.ask - tick.bid) / pip_size) if pip_size else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to profile JSON")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    profile_name = prof["profile_name"]
    symbol = prof["symbol"]
    pip_size = float(prof["pip_size"])

    log_dir = BASE_DIR / "logs" / profile_name
    ensure_dir(str(log_dir))
    context_path = log_dir / "context_log.csv"
    db_path = log_dir / "assistant.db"
    store = SqliteStore(db_path)
    store.init_db()

    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return

    if not mt5.symbol_select(symbol, True):
        print("symbol_select failed:", mt5.last_error())
        mt5.shutdown()
        return

    now = now_utc_iso()
    sp = spread_pips(symbol, pip_size)

    h4 = fetch_df(symbol, mt5.TIMEFRAME_H4, BARS_H4)
    m15 = fetch_df(symbol, mt5.TIMEFRAME_M15, BARS_M15)
    m1 = fetch_df(symbol, mt5.TIMEFRAME_M1, BARS_M1)

    h4s = compute_cross_and_trend(h4)
    m15s = compute_cross_and_trend(m15)
    m1s = compute_cross_and_trend(m1)

    score = 0
    for s in (h4s, m15s, m1s):
        if s["regime"] == "bull":
            score += 1
        elif s["regime"] == "bear":
            score -= 1

    # Print a one-glance summary (this is your “assistant output”)
    print("\n=== SUMMARY ===")
    print("profile:", profile_name)
    print("symbol:", symbol)
    print("spread_pips:", sp)
    print("alignment_score (H4+M15+M1):", score, "(-3 bear … +3 bull)")
    print("H4:", h4s["regime"], "last_cross:", h4s["last_cross_dir"], h4s["last_cross_time"], "since:", h4s["trend_since_cross"])
    print("M15:", m15s["regime"], "last_cross:", m15s["last_cross_dir"], m15s["last_cross_time"], "since:", m15s["trend_since_cross"])
    print("M1:", m1s["regime"], "last_cross:", m1s["last_cross_dir"], m1s["last_cross_time"], "since:", m1s["trend_since_cross"])

    row = {
        "timestamp_utc": now,
        "profile": profile_name,
        "symbol": symbol,
        "config_json": json.dumps(prof),
        "spread_pips": sp,
        "alignment_score": score,

        "h4_regime": h4s["regime"],
        "h4_cross_dir": h4s["last_cross_dir"],
        "h4_cross_time": _cross_time_iso(h4s["last_cross_time"]),
        "h4_cross_price": h4s["last_cross_price"],
        "h4_trend_since": h4s["trend_since_cross"],

        "m15_regime": m15s["regime"],
        "m15_cross_dir": m15s["last_cross_dir"],
        "m15_cross_time": _cross_time_iso(m15s["last_cross_time"]),
        "m15_cross_price": m15s["last_cross_price"],
        "m15_trend_since": m15s["trend_since_cross"],

        "m1_regime": m1s["regime"],
        "m1_cross_dir": m1s["last_cross_dir"],
        "m1_cross_time": _cross_time_iso(m1s["last_cross_time"]),
        "m1_cross_price": m1s["last_cross_price"],
        "m1_trend_since": m1s["trend_since_cross"],
    }

    # SQLite (v1)
    snapshot_id = store.insert_snapshot(row)

    # CSV (legacy/compat)
    df = pd.DataFrame([row])
    write_header = not context_path.exists()
    df.to_csv(context_path, mode="a", header=write_header, index=False)

    print("\nOK Logged snapshot to:", str(context_path))
    print("OK Logged snapshot to DB:", str(db_path), "snapshot_id:", snapshot_id)
    mt5.shutdown()


if __name__ == "__main__":
    main()