import argparse
import json
import os
import pandas as pd
from pathlib import Path

from core.profile import load_profile_v1
from storage.sqlite_store import SqliteStore

BASE_DIR = Path(__file__).resolve().parent


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    if "profile_name" not in prof:
        raise ValueError("profile missing 'profile_name'")
    return prof


def trades_path(profile_name: str) -> str:
    return str(BASE_DIR / "logs" / profile_name / "trades_log.csv")


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.notna().any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to profile JSON")
    args = ap.parse_args()

    prof_v1 = load_profile_v1(args.profile)
    pname = prof_v1.profile_name
    path = trades_path(pname)

    print("=== REVIEW STATS ===")
    print("profile:", pname)
    print("file:", path)

    # Prefer SQLite if present (more robust), fall back to CSV.
    store = SqliteStore((BASE_DIR / "logs" / pname / "assistant.db"))
    df = pd.DataFrame()
    if store.path.exists():
        try:
            store.init_db()
            df = store.read_trades_df(pname)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        if not os.path.exists(path):
            print("No trades_log.csv found yet for this profile.")
            return
        df = pd.read_csv(path)

    if df.empty:
        print("trades_log.csv is empty.")
        return

    # closed vs open
    exit_price = pd.to_numeric(df.get("exit_price", pd.Series([pd.NA] * len(df))), errors="coerce")
    is_closed = exit_price.notna()
    closed = df[is_closed].copy()
    open_ = df[~is_closed].copy()

    print("\n=== TRADES OVERVIEW ===")
    print("total_trades:", len(df))
    print("closed_trades:", len(closed))
    print("open_trades:", len(open_))

    if closed.empty:
        print("\nNo closed trades yet. Close at least one trade to see stats.")
        return

    # pips stats
    pips = pd.to_numeric(closed.get("pips", pd.Series([pd.NA] * len(closed))), errors="coerce")
    wins = int((pips > 0).sum())
    losses = int((pips < 0).sum())
    flat = int((pips == 0).sum())

    def _r3(x):
        return round(x, 3) if x is not None else None

    print("\n=== PIPS STATS (CLOSED) ===")
    print("avg_pips:", _r3(safe_mean(pips)))
    print("win_rate_pips:", _r3(float(wins / len(closed))))
    print("wins:", wins, "losses:", losses, "flat:", flat)

    # R stats
    r = pd.to_numeric(closed.get("r_multiple", pd.Series([pd.NA] * len(closed))), errors="coerce")
    r_avail = int(r.notna().sum())
    print("\n=== R MULTIPLE STATS (CLOSED, WHERE STOP PROVIDED) ===")
    print("r_available_count:", r_avail, "out of", len(closed))
    if r_avail > 0:
        r_wins = int((r > 0).sum())
        print("avg_r:", _r3(safe_mean(r)))
        print("win_rate_r:", _r3(float(r_wins / r_avail)))
        print("avg_r_win:", _r3(safe_mean(r[r > 0])))
        print("avg_r_loss:", _r3(safe_mean(r[r < 0])))

    # breakdown by alignment score
    score = pd.to_numeric(closed.get("ctx_alignment_score", pd.Series([pd.NA] * len(closed))), errors="coerce")
    if score.notna().any():
        print("\n=== BREAKDOWN BY ctx_alignment_score ===")
        tmp = closed.copy()
        tmp["ctx_alignment_score"] = score
        tmp["pips"] = pips
        tmp["r_multiple"] = r

        table = tmp.groupby("ctx_alignment_score").agg(
            trades=("trade_id", "count"),
            avg_pips=("pips", lambda x: round(float(pd.to_numeric(x, errors="coerce").mean()), 3)),
            win_rate_pips=("pips", lambda x: round(float((pd.to_numeric(x, errors="coerce") > 0).mean()), 3)),
            avg_r=("r_multiple", lambda x: round(float(pd.to_numeric(x, errors="coerce").mean()), 3) if pd.to_numeric(x, errors="coerce").notna().any() else float("nan")),
        ).sort_index()
        print(table.to_string())

    # last 10 closed trades (round floats to 3 decimals)
    print("\n=== LAST 10 CLOSED TRADES ===")
    cols = [
        "trade_id",
        "timestamp_utc",
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "pips",
        "r_multiple",
        "ctx_alignment_score",
        "ctx_spread_pips",
        "notes",
    ]
    cols = [c for c in cols if c in closed.columns]
    out = closed.tail(10)[cols].copy()
    for c in ["entry_price", "exit_price", "pips", "r_multiple", "ctx_spread_pips"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].apply(lambda x: round(x, 3) if pd.notna(x) else x)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()