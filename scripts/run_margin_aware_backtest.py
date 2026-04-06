from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
V7_TRADES_PATH = ROOT / "research_out/phase3_v7_pfdd_defended_real/v7_enriched_trade_log.csv"
V7_IMPLIED_PATH = ROOT / "research_out/phase3_v7_pfdd_defended_real/trade_implied_size_net.csv"
V4_TRADES_PATH = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/spike_fade_v4_backtest_v2_trades.csv"
M1_PATH = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
OUT_DIR = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest"
RESULTS_CSV = OUT_DIR / "margin_backtest_results.csv"
SUMMARY_TXT = OUT_DIR / "margin_backtest_summary.txt"
CURVES_CSV = OUT_DIR / "margin_equity_curves.csv"
REJECTIONS_CSV = OUT_DIR / "margin_rejections.csv"

PIP = 0.01
BASE_V4_UNITS = 100_000.0

CONFIGS = [
    (100_000, 1, 1, 0.04, "baseline_1lot"),
    (100_000, 5, 1, 0.04, "phase1_5lot"),
    (100_000, 10, 1, 0.04, "phase2_10lot"),
    (100_000, 20, 1, 0.04, "target_20lot_100k"),
    (200_000, 20, 1, 0.04, "target_20lot_200k"),
    (100_000, 20, 1, 0.02, "target_20lot_100k_2pct"),
    (150_000, 20, 1, 0.04, "target_20lot_150k"),
    (100_000, 15, 1, 0.04, "compromise_15lot"),
    (100_000, 10, 1, 0.02, "phase2_10lot_2pct"),
]


@dataclass
class Position:
    trade_id: str
    source: str
    sub_strategy: str
    side: str
    units: float
    entry_price: float
    entry_time: pd.Timestamp
    scheduled_exit_time: pd.Timestamp
    scheduled_exit_price: float
    scheduled_pnl_usd: float
    current_price: float
    margin_rate: float

    @property
    def margin_required(self) -> float:
        return abs(self.units) * self.margin_rate

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            pips = (self.current_price - self.entry_price) / PIP
        else:
            pips = (self.entry_price - self.current_price) / PIP
        pip_value = abs(self.units) * PIP / max(self.current_price, 1e-9)
        return float(pips * pip_value)

    def mark_to_market(self, current_price: float) -> None:
        self.current_price = float(current_price)


class MarginAccount:
    def __init__(self, starting_balance: float, margin_rate: float, leverage_max: float):
        self.starting_balance = float(starting_balance)
        self.balance = float(starting_balance)
        self.margin_rate = float(margin_rate)
        self.leverage_max = float(leverage_max)
        self.open_positions: dict[str, Position] = {}

    @property
    def unrealized_pnl(self) -> float:
        return float(sum(pos.unrealized_pnl for pos in self.open_positions.values()))

    @property
    def equity(self) -> float:
        return float(self.balance + self.unrealized_pnl)

    @property
    def margin_used(self) -> float:
        return float(sum(pos.margin_required for pos in self.open_positions.values()))

    @property
    def margin_available(self) -> float:
        return float(self.equity - self.margin_used)

    @property
    def margin_level(self) -> float:
        if self.margin_used <= 0:
            return float("inf")
        return float((self.equity / self.margin_used) * 100.0)

    def margin_required_for(self, units: float) -> float:
        return float(abs(units) * self.margin_rate)

    def can_open(self, units: float) -> tuple[bool, float, float, float]:
        required_margin = self.margin_required_for(units)
        total_notional = sum(abs(p.units) for p in self.open_positions.values()) + abs(units)
        effective_leverage = total_notional / self.equity if self.equity > 0 else float("inf")
        ok = self.margin_available >= required_margin and effective_leverage <= self.leverage_max
        return bool(ok), float(required_margin), float(self.margin_available), float(effective_leverage)

    def open(self, pos: Position) -> None:
        self.open_positions[pos.trade_id] = pos

    def close(self, trade_id: str, realized_pnl_usd: float) -> Position | None:
        pos = self.open_positions.pop(trade_id, None)
        if pos is not None:
            self.balance += float(realized_pnl_usd)
        return pos


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def load_m1() -> pd.DataFrame:
    df = pd.read_csv(M1_PATH)
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("time") or cols.get("datetime") or cols.get("timestamp")
    if time_col is None:
        raise ValueError("Could not find time column in M1 data")
    rename = {time_col: "time"}
    for name in ["open", "high", "low", "close"]:
        src = cols.get(name)
        if src is None:
            raise ValueError(f"Missing {name} column in M1 data")
        rename[src] = name
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)


def load_v7_trades(v7_size_mode: str = "fixed_1lot") -> pd.DataFrame:
    df = pd.read_csv(V7_TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    for col in ["entry_price", "exit_price", "pnl_usd", "pnl_pips", "pnl_pips_calc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    replay = pd.DataFrame(
        {
            "trade_id": "v7_" + df["trade_id"].astype(int).astype(str),
            "source": "v7",
            "sub_strategy": df["sub_strategy"].astype(str),
            "side": df["direction"].astype(str).str.lower(),
            "entry_time": df["entry_time"],
            "exit_time": df["exit_time"],
            "entry_price": df["entry_price"].astype(float),
            "exit_price": df["exit_price"].astype(float),
            "recorded_pnl_usd": df["pnl_usd"].astype(float),
            "base_units": float(BASE_V4_UNITS),
        }
    )
    if v7_size_mode == "implied":
        implied = pd.read_csv(V7_IMPLIED_PATH)[["trade_id", "implied_units", "implied_standard_lots"]].copy()
        implied["trade_id"] = "v7_" + implied["trade_id"].astype(int).astype(str)
        implied["implied_units"] = pd.to_numeric(implied["implied_units"], errors="coerce")
        replay = replay.merge(implied, on="trade_id", how="left")
        replay["base_units"] = replay["implied_units"].fillna(float(BASE_V4_UNITS))
    elif v7_size_mode != "fixed_1lot":
        raise ValueError(f"Unsupported v7_size_mode: {v7_size_mode}")
    return replay.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)


def load_v4_trades() -> pd.DataFrame:
    df = pd.read_csv(V4_TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    for col in ["entry_price", "exit_price", "pnl_usd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    replay = pd.DataFrame(
        {
            "trade_id": "v4_" + df.index.astype(str),
            "source": "v4",
            "sub_strategy": "spike_fade_v4",
            "side": df["direction"].astype(str).str.lower(),
            "entry_time": df["entry_time"],
            "exit_time": df["exit_time"],
            "entry_price": df["entry_price"].astype(float),
            "exit_price": df["exit_price"].astype(float),
            "recorded_pnl_usd": df["pnl_usd"].astype(float),
            "base_units": float(BASE_V4_UNITS),
        }
    )
    return replay.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)


def build_event_maps(trades: pd.DataFrame) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    entries: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    exits: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for row in trades.to_dict(orient="records"):
        entry_key = pd.Timestamp(row["entry_time"])
        exit_key = pd.Timestamp(row["exit_time"])
        entries.setdefault(entry_key, []).append(row)
        exits.setdefault(exit_key, []).append(row)
    return entries, exits


def calc_pf(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def summarize_pnl(values: pd.Series) -> tuple[float, float, float, float]:
    if values.empty:
        return 0.0, 0.0, 0.0, 0.0
    cum = values.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = float(abs(dd.min()))
    peak_before = float(peak.loc[dd.idxmin()]) if len(dd) else 0.0
    dd_pct = (max_dd / peak_before * 100.0) if peak_before > 0 else 0.0
    return float(values.sum()), calc_pf(values), max_dd, dd_pct


def position_realized_from_market(pos: Position, current_price: float) -> float:
    if pos.side == "long":
        pips = (current_price - pos.entry_price) / PIP
    else:
        pips = (pos.entry_price - current_price) / PIP
    pip_value = abs(pos.units) * PIP / max(current_price, 1e-9)
    return float(pips * pip_value)


def replay_config(
    m1: pd.DataFrame,
    v7_base: pd.DataFrame,
    v4_base: pd.DataFrame,
    account_size: float,
    v4_lots: float,
    v71_lot_multiplier: float,
    margin_rate: float,
    label: str,
    leverage_max_override: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    leverage_max = float(leverage_max_override) if leverage_max_override is not None else (1.0 / float(margin_rate))
    account = MarginAccount(account_size, margin_rate, leverage_max)

    v7 = v7_base.copy()
    v7["units"] = v7["base_units"].astype(float) * float(v71_lot_multiplier)
    v7["pnl_usd_scaled"] = v7["recorded_pnl_usd"].astype(float) * float(v71_lot_multiplier)

    v4 = v4_base.copy()
    v4["units"] = float(v4_lots) * BASE_V4_UNITS
    v4["pnl_usd_scaled"] = v4["recorded_pnl_usd"].astype(float) * float(v4_lots)

    all_trades = pd.concat([v7, v4], ignore_index=True).sort_values(["entry_time", "trade_id"]).reset_index(drop=True)
    entries, exits = build_event_maps(all_trades)

    executed_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    margin_calls: list[dict[str, Any]] = []

    times = m1["time"].tolist()
    close_arr = m1["close"].to_numpy(np.float64)
    hourly_last: int | None = None
    concurrent_counts: list[int] = []
    overlap_counts: list[int] = []
    utilization_samples: list[float] = []
    margin_level_samples: list[float] = []
    margin_used_samples: list[float] = []
    margin_avail_samples: list[float] = []
    peak_equity = float(account.balance)
    min_equity = float(account.balance)
    running_peak_equity = float(account.balance)
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0

    for idx, ts in enumerate(times):
        ts = pd.Timestamp(ts)
        current_price = float(close_arr[idx])

        for pos in list(account.open_positions.values()):
            pos.mark_to_market(current_price)

        while account.open_positions and account.margin_level < 100.0:
            worst = min(account.open_positions.values(), key=lambda p: p.unrealized_pnl)
            realized = position_realized_from_market(worst, current_price)
            account.close(worst.trade_id, realized)
            margin_calls.append(
                {
                    "config": label,
                    "time": ts,
                    "trade_id": worst.trade_id,
                    "source": worst.source,
                    "sub_strategy": worst.sub_strategy,
                    "equity_before": float(account.equity - realized),
                    "equity_after": float(account.equity),
                    "margin_used_after": float(account.margin_used),
                    "realized_pnl_usd": float(realized),
                    "action": "closed_worst_loser",
                }
            )
            executed_records.append(
                {
                    "config": label,
                    "trade_id": worst.trade_id,
                    "source": worst.source,
                    "sub_strategy": worst.sub_strategy,
                    "entry_time": worst.entry_time,
                    "exit_time": ts,
                    "executed": True,
                    "closed_by_margin_call": True,
                    "pnl_usd": float(realized),
                }
            )

        for event in exits.get(ts, []):
            pos = account.open_positions.get(event["trade_id"])
            if pos is None:
                continue
            realized = float(event["pnl_usd_scaled"])
            account.close(pos.trade_id, realized)
            executed_records.append(
                {
                    "config": label,
                    "trade_id": pos.trade_id,
                    "source": pos.source,
                    "sub_strategy": pos.sub_strategy,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "executed": True,
                    "closed_by_margin_call": False,
                    "pnl_usd": realized,
                }
            )

        for event in sorted(entries.get(ts, []), key=lambda r: (0 if r["source"] == "v7" else 1, r["trade_id"])):
            if event["trade_id"] in account.open_positions:
                continue
            can_open, required_margin, margin_available, effective_leverage = account.can_open(float(event["units"]))
            if not can_open:
                rejection_records.append(
                    {
                        "config": label,
                        "time": ts,
                        "trade_id": event["trade_id"],
                        "source": event["source"],
                        "sub_strategy": event["sub_strategy"],
                        "units": float(event["units"]),
                        "margin_needed": float(required_margin),
                        "margin_available": float(margin_available),
                        "equity": float(account.equity),
                        "effective_leverage": float(effective_leverage),
                    }
                )
                continue
            pos = Position(
                trade_id=str(event["trade_id"]),
                source=str(event["source"]),
                sub_strategy=str(event["sub_strategy"]),
                side=str(event["side"]),
                units=float(event["units"]) if str(event["side"]) == "long" else -abs(float(event["units"])),
                entry_price=float(event["entry_price"]),
                entry_time=pd.Timestamp(event["entry_time"]),
                scheduled_exit_time=pd.Timestamp(event["exit_time"]),
                scheduled_exit_price=float(event["exit_price"]),
                scheduled_pnl_usd=float(event["pnl_usd_scaled"]),
                current_price=current_price,
                margin_rate=margin_rate,
            )
            account.open(pos)

        equity = float(account.equity)
        peak_equity = max(peak_equity, equity)
        min_equity = min(min_equity, equity)
        running_peak_equity = max(running_peak_equity, equity)
        dd_usd = max(0.0, running_peak_equity - equity)
        dd_pct = (dd_usd / running_peak_equity * 100.0) if running_peak_equity > 0 else 0.0
        max_drawdown_usd = max(max_drawdown_usd, dd_usd)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)
        concurrent_counts.append(len(account.open_positions))
        open_sources = [p.source for p in account.open_positions.values()]
        overlap_counts.append(len(account.open_positions) if ("v7" in open_sources and "v4" in open_sources) else 0)
        util = (account.margin_used / equity * 100.0) if equity > 0 else float("inf")
        utilization_samples.append(util)
        margin_level_samples.append(float(account.margin_level))
        margin_used_samples.append(float(account.margin_used))
        margin_avail_samples.append(float(account.margin_available))

        hour_key = int(ts.floor("h").value)
        if hour_key != hourly_last:
            hourly_last = hour_key
            snapshot_records.append(
                {
                    "config": label,
                    "time": ts.floor("h"),
                    "balance": float(account.balance),
                    "equity": equity,
                    "unrealized_pnl": float(account.unrealized_pnl),
                    "margin_used": float(account.margin_used),
                    "margin_available": float(account.margin_available),
                    "margin_level": float(account.margin_level),
                    "num_open_positions": int(len(account.open_positions)),
                }
            )

    executed_df = pd.DataFrame(executed_records)
    rejected_df = pd.DataFrame(rejection_records)
    snapshots_df = pd.DataFrame(snapshot_records)
    margin_calls_df = pd.DataFrame(margin_calls)

    v7_attempted = int((all_trades["source"] == "v7").sum())
    v4_attempted = int((all_trades["source"] == "v4").sum())
    if executed_df.empty:
        executed_df = pd.DataFrame(columns=["config", "trade_id", "source", "sub_strategy", "entry_time", "exit_time", "executed", "closed_by_margin_call", "pnl_usd"])

    v7_exec = executed_df[executed_df["source"] == "v7"]
    v4_exec = executed_df[executed_df["source"] == "v4"]
    v7_rej = rejected_df[rejected_df["source"] == "v7"] if not rejected_df.empty else pd.DataFrame(columns=["source"])
    v4_rej = rejected_df[rejected_df["source"] == "v4"] if not rejected_df.empty else pd.DataFrame(columns=["source"])

    v7_net, _, _, _ = summarize_pnl(v7_exec["pnl_usd"].astype(float) if not v7_exec.empty else pd.Series(dtype=float))
    v4_net, _, _, _ = summarize_pnl(v4_exec["pnl_usd"].astype(float) if not v4_exec.empty else pd.Series(dtype=float))
    combined_net, combined_pf, _, _ = summarize_pnl(executed_df["pnl_usd"].astype(float) if not executed_df.empty else pd.Series(dtype=float))

    lowest_margin_level = float(np.min(margin_level_samples)) if margin_level_samples else float("inf")
    peak_margin_used = float(np.max(margin_used_samples)) if margin_used_samples else 0.0
    peak_margin_used_pct = float(np.max(utilization_samples)) if utilization_samples else 0.0
    mean_margin_used = float(np.mean(margin_used_samples)) if margin_used_samples else 0.0
    mean_margin_used_pct = float(np.mean(utilization_samples)) if utilization_samples else 0.0
    min_margin_available = float(np.min(margin_avail_samples)) if margin_avail_samples else 0.0
    over_80 = float(np.mean(np.array(utilization_samples) > 80.0) * 100.0) if utilization_samples else 0.0
    over_90 = float(np.mean(np.array(utilization_samples) > 90.0) * 100.0) if utilization_samples else 0.0

    result = {
        "config": label,
        "account_size": float(account_size),
        "v4_lots": float(v4_lots),
        "v71_lot_multiplier": float(v71_lot_multiplier),
        "margin_rate": float(margin_rate),
        "leverage_max": float(leverage_max),
        "v7_attempted": v7_attempted,
        "v7_executed": int(len(v7_exec)),
        "v7_rejected": int(len(v7_rej)),
        "v4_attempted": v4_attempted,
        "v4_executed": int(len(v4_exec)),
        "v4_rejected": int(len(v4_rej)),
        "total_executed": int(len(executed_df)),
        "margin_calls": int(len(margin_calls)),
        "v7_net_pnl": float(v7_net),
        "v4_net_pnl": float(v4_net),
        "combined_net_pnl": float(combined_net),
        "combined_pf": float(combined_pf),
        "peak_equity": float(peak_equity),
        "min_equity": float(min_equity),
        "max_drawdown_usd": float(max_drawdown_usd),
        "max_drawdown_pct": float(max_drawdown_pct),
        "final_equity": float(account.balance),
        "mean_margin_used": mean_margin_used,
        "mean_margin_used_pct": mean_margin_used_pct,
        "peak_margin_used": peak_margin_used,
        "peak_margin_used_pct": peak_margin_used_pct,
        "min_margin_available": min_margin_available,
        "time_over_80_pct": over_80,
        "time_over_90_pct": over_90,
        "lowest_margin_level": lowest_margin_level,
        "max_simultaneous_positions": int(max(concurrent_counts) if concurrent_counts else 0),
        "mean_simultaneous_positions": float(np.mean(concurrent_counts)) if concurrent_counts else 0.0,
        "max_v4_v7_overlap_instances": int(max(overlap_counts) if overlap_counts else 0),
    }
    return result, rejection_records, snapshot_records, margin_calls


def config_summary_text(result: dict[str, Any], rejections: list[dict[str, Any]], calls: list[dict[str, Any]]) -> str:
    rej_df = pd.DataFrame(rejections)
    v7_rejs = rej_df[rej_df["source"] == "v7"] if not rej_df.empty else pd.DataFrame(columns=["time", "sub_strategy"])
    v4_rejs = rej_df[rej_df["source"] == "v4"] if not rej_df.empty else pd.DataFrame(columns=["time", "sub_strategy"])

    lines = [
        f"CONFIG: {result['config']}",
        f"  Account: {fmt_money(result['account_size'])} | V4: {result['v4_lots']:.0f} lots | V7.1: x{result['v71_lot_multiplier']:.0f} recorded size | Margin: {result['margin_rate']*100:.0f}%",
        "",
        "  TRADES:",
        f"    V7.1 attempted: {result['v7_attempted']} | executed: {result['v7_executed']} | rejected: {result['v7_rejected']} (margin)",
        f"    V4 attempted:   {result['v4_attempted']} | executed: {result['v4_executed']} | rejected: {result['v4_rejected']} (margin)",
        f"    Total executed: {result['total_executed']}",
        "",
        "  MARGIN REJECTIONS:",
        f"    V7.1 rejections: {result['v7_rejected']}",
        f"    V4 rejections:   {result['v4_rejected']}",
    ]
    if 0 < len(v7_rejs) < 20:
        for row in v7_rejs.itertuples(index=False):
            lines.append(f"      - {row.time} | {row.sub_strategy}")
    if 0 < len(v4_rejs) < 20:
        for row in v4_rejs.itertuples(index=False):
            lines.append(f"      - {row.time} | {row.sub_strategy}")
    lines.extend(
        [
            "",
            f"  MARGIN CALLS: {result['margin_calls']}",
        ]
    )
    if 0 < len(calls) < 20:
        for row in calls:
            lines.append(
                f"    - {row['time']} | {row['sub_strategy']} | equity_after {fmt_money(row['equity_after'])} | realized {fmt_money(row['realized_pnl_usd'])}"
            )
    lines.extend(
        [
            "",
            "  P&L:",
            f"    V7.1 net (after rejections): {fmt_money(result['v7_net_pnl'])}",
            f"    V4 net (after rejections):   {fmt_money(result['v4_net_pnl'])}",
            f"    Combined net:                {fmt_money(result['combined_net_pnl'])}",
            "",
            "  EQUITY:",
            f"    Peak equity:     {fmt_money(result['peak_equity'])}",
            f"    Min equity:      {fmt_money(result['min_equity'])}",
            f"    Max drawdown:    {fmt_money(result['max_drawdown_usd'])} ({fmt_pct(result['max_drawdown_pct'])})",
            f"    Final equity:    {fmt_money(result['final_equity'])}",
            "",
            "  MARGIN UTILIZATION:",
            f"    Mean margin used:     {fmt_money(result['mean_margin_used'])} ({fmt_pct(result['mean_margin_used_pct'])} of equity)",
            f"    Peak margin used:     {fmt_money(result['peak_margin_used'])} ({fmt_pct(result['peak_margin_used_pct'])} of equity)",
            f"    Min margin available: {fmt_money(result['min_margin_available'])}",
            f"    Time spent > 80% margin utilization: {fmt_pct(result['time_over_80_pct'])}",
            f"    Time spent > 90% margin utilization: {fmt_pct(result['time_over_90_pct'])}",
            f"    Lowest margin level:  {fmt_pct(result['lowest_margin_level'])}",
            "",
            "  CONCURRENT POSITIONS:",
            f"    Max simultaneous positions: {result['max_simultaneous_positions']}",
            f"    Mean simultaneous positions: {result['mean_simultaneous_positions']:.2f}",
            f"    Max V4+V7.1 overlap instances: {result['max_v4_v7_overlap_instances']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_decision_table(results: pd.DataFrame) -> str:
    lines = [
        "MARGIN-AWARE SIZING DECISION TABLE",
        "═══════════════════════════════════════════════════════════════════════════════",
        "Config              Acct    V4Lot  Marg%  V4Rej  V71Rej  MargCalls  NetPnL    MaxDD   MaxMarg%",
        "─────────────────── ─────── ────── ────── ────── ─────── ───────── ────────── ─────── ────────",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{str(row.config):19s} ${int(row.account_size/1000):>3d}k {int(row.v4_lots):>6d} {int(row.margin_rate*100):>6d}% "
            f"{int(row.v4_rejected):>6d} {int(row.v7_rejected):>7d} {int(row.margin_calls):>9d} "
            f"{fmt_money(row.combined_net_pnl):>10s} {row.max_drawdown_pct:>7.2f}% {row.peak_margin_used_pct:>8.2f}%"
        )

    feasible_20 = results[
        (results["v4_lots"] == 20)
        & (results["v4_rejected"] == 0)
        & (results["v7_rejected"] == 0)
        & (results["margin_calls"] == 0)
    ].sort_values("account_size")
    best_100k_4 = results[(results["account_size"] == 100_000) & (results["margin_rate"] == 0.04) & (results["v7_rejected"] == 0) & (results["margin_calls"] == 0)].sort_values("v4_lots", ascending=False)
    best_100k_2 = results[(results["account_size"] == 100_000) & (results["margin_rate"] == 0.02) & (results["v7_rejected"] == 0) & (results["margin_calls"] == 0)].sort_values("v4_lots", ascending=False)

    smallest_20 = f"${int(feasible_20.iloc[0]['account_size']):,}" if not feasible_20.empty else "none in tested grid"
    best_4 = f"{int(best_100k_4.iloc[0]['v4_lots'])} lots" if not best_100k_4.empty else "none in tested grid"
    best_2 = f"{int(best_100k_2.iloc[0]['v4_lots'])} lots" if not best_100k_2.empty else "none in tested grid"

    lines.extend(
        [
            "═══════════════════════════════════════════════════════════════════════════════",
            "",
            "RECOMMENDATION:",
            f"  Smallest account that supports 20-lot V4 with 0 rejections: {smallest_20}",
            f"  Best lot size for $100k account at 4% margin:               {best_4}",
            f"  Best lot size for $100k account at 2% margin:               {best_2}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading M1 data from {M1_PATH}")
    m1 = load_m1()
    print(f"Loading V7.1 trade log from {V7_TRADES_PATH}")
    v7 = load_v7_trades()
    print(f"Loading V4 trade log from {V4_TRADES_PATH}")
    v4 = load_v4_trades()

    all_results: list[dict[str, Any]] = []
    all_rejections: list[dict[str, Any]] = []
    all_snapshots: list[dict[str, Any]] = []
    full_summary_sections: list[str] = []

    for account_size, v4_lots, v71_mult, margin_rate, label in CONFIGS:
        print(f"Running {label} ...")
        result, rejections, snapshots, calls = replay_config(
            m1=m1,
            v7_base=v7,
            v4_base=v4,
            account_size=account_size,
            v4_lots=v4_lots,
            v71_lot_multiplier=v71_mult,
            margin_rate=margin_rate,
            label=label,
        )
        all_results.append(result)
        all_rejections.extend(rejections)
        all_snapshots.extend(snapshots)
        full_summary_sections.append(config_summary_text(result, rejections, calls))

    results_df = pd.DataFrame(all_results).sort_values(["account_size", "margin_rate", "v4_lots", "config"]).reset_index(drop=True)
    rejections_df = pd.DataFrame(all_rejections)
    curves_df = pd.DataFrame(all_snapshots).sort_values(["config", "time"]).reset_index(drop=True) if all_snapshots else pd.DataFrame()

    results_df.to_csv(RESULTS_CSV, index=False)
    if rejections_df.empty:
        pd.DataFrame(columns=["config", "time", "trade_id", "source", "sub_strategy", "units", "margin_needed", "margin_available", "equity", "effective_leverage"]).to_csv(REJECTIONS_CSV, index=False)
    else:
        rejections_df.to_csv(REJECTIONS_CSV, index=False)
    curves_df.to_csv(CURVES_CSV, index=False)

    decision_table = build_decision_table(results_df)
    summary_text = "\n\n".join(full_summary_sections + ["", decision_table, ""])
    SUMMARY_TXT.write_text(summary_text, encoding="utf-8")

    print(decision_table)
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_TXT}")
    print(f"Wrote {CURVES_CSV}")
    print(f"Wrote {REJECTIONS_CSV}")


if __name__ == "__main__":
    main()
