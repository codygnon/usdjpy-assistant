import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.profile import load_profile_v1
from storage.sqlite_store import SqliteStore

BASE_DIR = Path(__file__).resolve().parent


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    for k in ["profile_name", "pip_size"]:
        if k not in prof:
            raise ValueError(f"profile missing '{k}'")
    return prof


def trades_path(profile_name: str) -> str:
    return str(BASE_DIR / "logs" / profile_name / "trades_log.csv")


def load_trades(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trades log: {path}")
    df = pd.read_csv(path)

    # Ensure expected columns exist
    for col in [
        "exit_price",
        "exit_timestamp_utc",
        "exit_reason",
        "pips",
        "risk_pips",
        "r_multiple",
        "duration_minutes",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def save_trades(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def is_open_trade(row: pd.Series) -> bool:
    return pd.isna(row.get("exit_price", pd.NA))


def compute_pips(side: str, entry: float, exit_: float, pip_size: float) -> float:
    if side == "buy":
        return (exit_ - entry) / pip_size
    if side == "sell":
        return (entry - exit_) / pip_size
    raise ValueError("side must be 'buy' or 'sell'")


def compute_r_multiple(pips: float, entry: float, stop: float | None, pip_size: float) -> tuple[float | None, float | None]:
    if stop is None:
        return None, None
    risk_pips = abs(entry - stop) / pip_size
    if risk_pips == 0:
        return float(risk_pips), None
    return float(risk_pips), float(pips / risk_pips)


def parse_iso(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to profile JSON")
    args = ap.parse_args()

    prof_v1 = load_profile_v1(args.profile)
    pname = prof_v1.profile_name
    pip_size = float(prof_v1.pip_size)

    path = trades_path(pname)
    df = load_trades(path)
    store = SqliteStore((BASE_DIR / "logs" / pname / "assistant.db"))
    store.init_db()

    open_df = df[df.apply(is_open_trade, axis=1)].copy()
    if open_df.empty:
        print("No open trades found for profile:", pname)
        print("File:", path)
        return

    print("\nOpen trades (showing last 10):")
    cols = [
        "trade_id",
        "timestamp_utc",
        "symbol",
        "side",
        "entry_price",
        "stop_price",
        "target_price",
        "size_lots",
        "ctx_alignment_score",
        "ctx_spread_pips",
    ]
    cols = [c for c in cols if c in open_df.columns]
    print(open_df.tail(10)[cols].to_string(index=False))

    trade_id = input("\nEnter trade_id to close (copy/paste exactly): ").strip()
    if trade_id == "":
        print("No trade_id entered. Exiting.")
        return

    matches = df.index[df["trade_id"] == trade_id].tolist()
    if not matches:
        print("trade_id not found.")
        return
    idx = matches[0]
    row = df.loc[idx]

    if not is_open_trade(row):
        print("That trade already has an exit_price recorded.")
        return

    side = str(row["side"]).lower()
    entry = float(row["entry_price"])
    stop_val = row.get("stop_price", pd.NA)
    stop = None if pd.isna(stop_val) else float(stop_val)

    exit_price_s = input("exit_price: ").strip()
    try:
        exit_price = float(exit_price_s)
    except ValueError:
        print("Invalid exit_price.")
        return

    exit_reason = input("exit_reason (optional): ").strip()
    exit_ts = now_utc_iso()

    pips = compute_pips(side, entry, exit_price, pip_size)
    risk_pips, r_mult = compute_r_multiple(pips, entry, stop, pip_size)

    # duration
    try:
        t0 = parse_iso(str(row["timestamp_utc"]))
        t1 = parse_iso(exit_ts)
        duration_min = float((t1 - t0).total_seconds() / 60.0)
    except Exception:
        duration_min = None

    df.at[idx, "exit_price"] = exit_price
    df.at[idx, "exit_timestamp_utc"] = exit_ts
    df.at[idx, "exit_reason"] = exit_reason if exit_reason else pd.NA
    df.at[idx, "pips"] = float(pips)
    df.at[idx, "risk_pips"] = risk_pips if risk_pips is not None else pd.NA
    df.at[idx, "r_multiple"] = r_mult if r_mult is not None else pd.NA
    df.at[idx, "duration_minutes"] = duration_min if duration_min is not None else pd.NA

    save_trades(path, df)

    # SQLite update (best effort)
    store.close_trade(
        trade_id=trade_id,
        updates={
            "exit_price": float(exit_price),
            "exit_timestamp_utc": exit_ts,
            "exit_reason": exit_reason if exit_reason else None,
            "pips": float(pips),
            "risk_pips": float(risk_pips) if risk_pips is not None else None,
            "r_multiple": float(r_mult) if r_mult is not None else None,
            "duration_minutes": float(duration_min) if duration_min is not None else None,
        },
    )

    print("\n✅ Trade closed and saved.")
    print("profile:", pname)
    print("trade_id:", trade_id)
    print("entry:", entry, "exit:", exit_price)
    print("pips:", float(pips))
    print("r_multiple:", r_mult)
    print("File:", path)
    print("DB:", str(store.path))


if __name__ == "__main__":
    main()