import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.models import MarketContext, TradeCandidate
from core.profile import load_profile_v1
from core.risk_engine import evaluate_trade
from storage.sqlite_store import SqliteStore

BASE_DIR = Path(__file__).resolve().parent


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    # required
    for k in ["profile_name", "symbol", "pip_size", "max_lots", "require_stop", "min_stop_pips", "max_spread_pips", "max_trades_per_day"]:
        if k not in prof:
            raise ValueError(f"profile missing '{k}'")

    return prof


def logs_dir(profile_name: str) -> str:
    return str(BASE_DIR / "logs" / profile_name)


def context_path(profile_name: str) -> str:
    return str(Path(logs_dir(profile_name)) / "context_log.csv")


def trades_path(profile_name: str) -> str:
    return str(Path(logs_dir(profile_name)) / "trades_log.csv")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def latest_context_row(ctx_path: str) -> dict:
    if not os.path.exists(ctx_path):
        raise FileNotFoundError(f"Missing context log: {ctx_path}")
    df = pd.read_csv(ctx_path)
    if df.empty:
        raise RuntimeError("context_log.csv is empty")
    return df.tail(1).to_dict(orient="records")[0]


def load_trades(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def trades_today_count(df: pd.DataFrame) -> int:
    if df.empty or "timestamp_utc" not in df.columns:
        return 0
    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    today = pd.Timestamp.now(tz="UTC").date()
    return int((ts.dt.date == today).sum())


def append_trade(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def prompt_float(name: str, allow_blank: bool = False) -> float | None:
    while True:
        s = input(f"{name}: ").strip()
        if allow_blank and s == "":
            return None
        try:
            return float(s)
        except ValueError:
            print("Please enter a number.")


def prompt_str(name: str, allow_blank: bool = True) -> str:
    s = input(f"{name}: ").strip()
    if not allow_blank and s == "":
        print("This field cannot be blank.")
        return prompt_str(name, allow_blank=allow_blank)
    return s


def pretrade_checks(prof: dict, ctx: dict, trade: dict) -> tuple[bool, list[str], list[str]]:
    hard = []
    warn = []

    # Spread filter
    sp = ctx.get("spread_pips", None)
    try:
        sp_f = float(sp) if sp is not None else None
    except Exception:
        sp_f = None

    max_spread = float(prof["max_spread_pips"])
    if sp_f is None:
        warn.append("spread_pips missing; cannot enforce max_spread_pips")
    elif sp_f > max_spread:
        hard.append(f"Spread too wide: {sp_f:.3f} pips > max {max_spread:.3f}")

    # Trades/day filter
    max_tpd = int(prof["max_trades_per_day"])
    if trade["trades_today_count"] >= max_tpd:
        hard.append(f"Max trades/day reached: {trade['trades_today_count']} >= {max_tpd}")

    # Lots filter (if provided)
    max_lots = float(prof["max_lots"])
    size = trade.get("size", None)
    if size is not None and size > max_lots:
        hard.append(f"Size too large: {size} > max_lots {max_lots}")

    # Stop filter + sanity
    require_stop = bool(prof["require_stop"])
    pip_size = float(prof["pip_size"])

    entry = float(trade["entry_price"])
    stop = trade.get("stop_price", None)
    side = trade["side"]

    if require_stop and stop is None:
        hard.append("Stop is required (require_stop=true)")

    if stop is not None:
        stop = float(stop)
        stop_dist_pips = abs(entry - stop) / pip_size
        min_stop = float(prof["min_stop_pips"])
        if stop_dist_pips < min_stop:
            hard.append(f"Stop too tight: {stop_dist_pips:.1f} pips < min_stop_pips {min_stop}")

        if side == "buy" and stop >= entry:
            hard.append("For BUY, stop must be below entry")
        if side == "sell" and stop <= entry:
            hard.append("For SELL, stop must be above entry")

    return (len(hard) == 0, hard, warn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to profile JSON")
    args = ap.parse_args()

    # v1 profile loader (migrates legacy flat profiles)
    prof_v1 = load_profile_v1(args.profile)
    pname = prof_v1.profile_name

    ensure_dir(logs_dir(pname))

    ctx = latest_context_row(context_path(pname))
    trades_df = load_trades(trades_path(pname))

    log_dir = Path(logs_dir(pname))
    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()

    print("Enter trade details (profile-driven guarded log).")
    print("profile:", pname)
    print("symbol (default):", prof_v1.symbol)
    print("latest spread_pips:", ctx.get("spread_pips"), "alignment_score:", ctx.get("alignment_score"))

    symbol = prompt_str("symbol (blank = default)") or prof_v1.symbol
    side = prompt_str("side (buy/sell)", allow_blank=False).lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    entry = prompt_float("entry_price")
    stop = prompt_float("stop_price (blank allowed)", allow_blank=True)
    target = prompt_float("target_price (blank allowed)", allow_blank=True)
    size = prompt_float("size (lots, blank allowed)", allow_blank=True)
    notes = prompt_str("notes (optional)")

    now = now_utc_iso()

    candidate = TradeCandidate(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        entry_price=float(entry),
        stop_price=None if stop is None else float(stop),
        target_price=None if target is None else float(target),
        size_lots=None if size is None else float(size),
    )

    mkt = MarketContext(
        spread_pips=float(ctx["spread_pips"]) if ctx.get("spread_pips") is not None else None,
        alignment_score=int(ctx["alignment_score"]) if ctx.get("alignment_score") is not None else None,
    )

    decision = evaluate_trade(profile=prof_v1, candidate=candidate, context=mkt, trades_df=trades_df)

    print("\n=== PRE-TRADE CHECK ===")
    print("allow:", decision.allow)
    if decision.hard_reasons:
        print("HARD REASONS:")
        for r in decision.hard_reasons:
            print("-", r)
    if decision.warnings:
        print("WARNINGS:")
        for w in decision.warnings:
            print("-", w)

    if not decision.allow:
        print("\n❌ Trade NOT logged (failed hard rules).")
        return

    row = {
        "trade_id": f"{now}_{symbol}_{side}",
        "timestamp_utc": now,
        "profile": pname,
        "symbol": symbol,
        "side": side,
        "entry_price": float(entry),
        "stop_price": None if stop is None else float(stop),
        "target_price": None if target is None else float(target),
        "size_lots": None if size is None else float(size),
        "notes": notes,

        # snapshot fields
        "ctx_timestamp_utc": ctx.get("timestamp_utc"),
        "ctx_spread_pips": ctx.get("spread_pips"),
        "ctx_alignment_score": ctx.get("alignment_score"),
        "ctx_h4_regime": ctx.get("h4_regime"),
        "ctx_m15_regime": ctx.get("m15_regime"),
        "ctx_m1_regime": ctx.get("m1_regime"),

        # profile limits (so you can audit later)
        "limit_max_lots": float(prof_v1.risk.max_lots),
        "limit_max_spread_pips": float(prof_v1.risk.max_spread_pips),
        "limit_min_stop_pips": float(prof_v1.risk.min_stop_pips),
        "limit_require_stop": bool(prof_v1.risk.require_stop),
        "limit_max_trades_per_day": int(prof_v1.risk.max_trades_per_day),
    }

    out_path = trades_path(pname)
    append_trade(out_path, row)
    print("\n✅ Logged trade to:", out_path)

    # SQLite trade insert (attach latest snapshot_id if present)
    snap = store.latest_snapshot(pname)
    store.insert_trade(
        {
            "trade_id": row["trade_id"],
            "timestamp_utc": row["timestamp_utc"],
            "profile": row["profile"],
            "symbol": row["symbol"],
            "side": row["side"],
            "config_json": json.dumps(prof_v1.model_dump()),
            "entry_price": row["entry_price"],
            "stop_price": row["stop_price"],
            "target_price": row["target_price"],
            "size_lots": row["size_lots"],
            "notes": row["notes"],
            "snapshot_id": int(snap["id"]) if snap is not None else None,
            "preset_name": prof_v1.active_preset_name or "Manual (CLI)",
        }
    )
    print("✅ Logged trade to DB:", str(store.path))


if __name__ == "__main__":
    main()