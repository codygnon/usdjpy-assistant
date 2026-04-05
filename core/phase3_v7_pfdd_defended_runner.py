"""
Phase 3B/3C: Unified bar-by-bar backtest for v7_pfdd defended variant (pipeline vs realistic L1 spread).

Core implementation shared by scripts/backtest_v7_defended_bar_by_bar.py and
core/regime_backtest_engine/phase3_v7_pfdd_defended_engine.py.

- Tokyo V14: core.v14_entry_evaluator + backtest_tokyo_meanrev helpers (add_indicators, gates, exits subset).
- London V2: core.london_v2_entry_evaluator + incremental L1 exit (diagnostic simulate_trade semantics) /
  incremental Setup A exit (backtest_v2_multisetup_london semantics).
- V44: oracle from phase1_v44_baseline_*_report.json closed_trades (no run_backtest_v5 reimplementation).

Spread: --spread-mode pipeline (L1 0.3 pip) | realistic (L1 uses v2 execution_model realistic profile).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Tokyo shadow path (scripts/backtest_tokyo_meanrev.py): M1 load, indicators, spread, session columns, sizing helpers
from scripts import backtest_tokyo_meanrev as tokyo_bt
from scripts import backtest_v2_multisetup_london as london_bt
from scripts import backtest_v44_conservative_router as v44_router
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import run_offensive_slice_discovery as discovery

# core/ownership_table.py — ER/ΔER bucket strings for cells (aligned with variant_k)
from core.ownership_table import cell_key_from_floats

# core/phase3_variant_k_baseline.py — M1+M5 context for admission (classifier + dynamic ER); used with sliced M1 at entry
from core.phase3_variant_k_baseline import build_variant_k_baseline_context

# scripts/backtest_variant_k_london_cluster.py — London cluster block set
from scripts.backtest_variant_k_london_cluster import LONDON_BLOCK_CLUSTER

# scripts/backtest_variant_i_pbt_standdown.py — global standdown regime + ΔER lookup
from scripts.backtest_variant_i_pbt_standdown import _is_variant_i_blocked
from scripts.backtest_merged_integrated_tokyo_london_v2_ny import TradeRow
from scripts.v7_defended_london_unified import (
    LondonUnifiedDayState,
    advance_london_unified_bar,
    init_london_day_state,
)
from scripts.v7_defended_tokyo_unified import (
    TokyoState,
    advance_tokyo_bar,
    compute_tokyo_indicators,
    init_tokyo_config,
    tokyo_margin_used_open_positions,
    _get_tokyo_spread,
)

PIP_SIZE = 0.01
RESEARCH_OUT = ROOT / "research_out"
TOKYO_CFG = RESEARCH_OUT / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = RESEARCH_OUT / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = RESEARCH_OUT / "session_momentum_v44_base_config.json"
DATASETS = discovery.DATASETS
TARGET_CELL_BLOCK = "ambiguous/er_low/der_neg"
DROP_WEEKDAYS = {"Monday", "Tuesday"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["500k", "1000k"], default="1000k")
    p.add_argument("--dry-run", action="store_true", help="Process first 10_000 bars only")
    p.add_argument(
        "--spread-mode",
        choices=["pipeline", "realistic"],
        default="pipeline",
        help="pipeline: L1 0.3 pip (parity with L1 replay); realistic: L1 uses v2_exp4 execution_model spread profile",
    )
    return p.parse_args()


def _ts(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def completed_tf_slice(tf_df: pd.DataFrame, current_m1_time: pd.Timestamp) -> pd.DataFrame:
    """Return TF rows whose bar close time <= current M1 time (right-aligned bars)."""
    return tf_df[tf_df["time"] <= pd.Timestamp(current_m1_time)].copy()


def assert_no_lookahead(data_slice: pd.DataFrame, current_time: pd.Timestamp, ctx: str) -> None:
    if data_slice.empty:
        return
    last_t = pd.Timestamp(data_slice["time"].iloc[-1])
    assert last_t <= pd.Timestamp(current_time), f"Lookahead [{ctx}]: last={last_t} current={current_time}"


def _m5_times_sorted_ns(m5: pd.DataFrame) -> np.ndarray:
    """Monotonic M5 bar close times as datetime64[ns] for binary search (no copies in hot loop)."""
    t = pd.to_datetime(m5["time"], utc=True)
    return t.values.astype("datetime64[ns]")


def assert_m5_no_lookahead_fast(m5_times_ns: np.ndarray, current_time: pd.Timestamp, ctx: str) -> None:
    """Same semantics as completed_tf_slice(m5,ts)+assert_no_lookahead — O(log n), no DataFrame alloc."""
    if m5_times_ns.size == 0:
        return
    cur = pd.Timestamp(current_time)
    if cur.tzinfo is None:
        cur = cur.tz_localize("UTC")
    else:
        cur = cur.tz_convert("UTC")
    ts64 = np.datetime64(cur.asm8.astype("datetime64[ns]"))
    pos = int(np.searchsorted(m5_times_ns, ts64, side="right"))
    if pos == 0:
        return
    last = m5_times_ns[pos - 1]
    assert last <= ts64, f"Lookahead [{ctx}]: last={last} current={ts64}"


def load_v44_oracle(dataset_path: str) -> dict[int, dict[str, Any]]:
    """Map entry_time (pd.Timestamp.value) -> trade row from phase1 V44 report (shadow list)."""
    dk = "500k" if "500k" in Path(dataset_path).name else "1000k"
    report = RESEARCH_OUT / f"phase1_v44_baseline_{dk}_report.json"
    if not report.exists():
        return {}
    data = json.loads(report.read_text(encoding="utf-8"))
    closed = data.get("results", {}).get("closed_trades", [])
    out: dict[int, dict[str, Any]] = {}
    for row in closed:
        et = _ts(row["entry_time"])
        out[et.value] = row
    return out


def london_spread_pips(i: int, ts: pd.Timestamp, london_cfg: dict[str, Any]) -> float:
    """scripts/backtest_v2_multisetup_london.compute_spread_pips — session profile for Setup A."""
    return float(london_bt.compute_spread_pips(i, ts, london_cfg))


def ny_open_hard_close_utc(day: pd.Timestamp) -> pd.Timestamp:
    """diagnostic_london_setupd_trade_outcomes._ny_open_utc_hour — US DST 12 vs 13 UTC."""
    from scripts import diagnostic_london_setupd_trade_outcomes as l1diag

    h = int(l1diag._ny_open_utc_hour(pd.Timestamp(day)))
    return pd.Timestamp(day.normalize().replace(tzinfo=day.tz) if day.tzinfo else day.normalize().tz_localize("UTC")) + pd.Timedelta(hours=h)


@dataclass
class L1Incremental:
    """Incremental version of diagnostic_london_setupd_trade_outcomes.simulate_trade exit loop."""

    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    tp1_price: float
    tp2_price: float
    be_price: float
    tp1_close_fraction: float
    half_spread: float
    ny_open: pd.Timestamp
    is_long: bool
    tp1_hit: bool = False
    leg1_exit: Optional[float] = None
    leg2_exit: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    risk_pips: float = 0.0

    def on_bar(
        self,
        ts: pd.Timestamp,
        o: float,
        h: float,
        l: float,
        c: float,
        *,
        half_spread: Optional[float] = None,
    ) -> bool:
        """Return True if position closed this bar. ``half_spread`` overrides entry half-spread for this bar (realistic L1)."""
        if self.exit_time is not None:
            return True
        hs = self.half_spread if half_spread is None else float(half_spread)
        is_long = self.is_long
        if ts >= self.ny_open:
            close_px = o - hs if is_long else o + hs
            if self.tp1_hit:
                self.leg2_exit = close_px
                self.exit_reason = "TP1_PARTIAL_THEN_HARD_CLOSE"
            else:
                self.leg1_exit = close_px
                self.exit_reason = "HARD_CLOSE"
            self.exit_time = ts
            return True
        if is_long:
            self.mfe_pips = max(self.mfe_pips, (h - self.entry_price) / PIP_SIZE)
            self.mae_pips = max(self.mae_pips, (self.entry_price - l) / PIP_SIZE)
        else:
            self.mfe_pips = max(self.mfe_pips, (self.entry_price - l) / PIP_SIZE)
            self.mae_pips = max(self.mae_pips, (h - self.entry_price) / PIP_SIZE)
        current_sl = self.be_price if self.tp1_hit else self.sl_price
        sl_hit = (l <= current_sl) if is_long else (h >= current_sl)
        tp_hit = False
        tp_level = None
        if not self.tp1_hit:
            if is_long and h >= self.tp1_price:
                tp_hit, tp_level = True, "TP1"
            elif not is_long and l <= self.tp1_price:
                tp_hit, tp_level = True, "TP1"
        else:
            if is_long and h >= self.tp2_price:
                tp_hit, tp_level = True, "TP2"
            elif not is_long and l <= self.tp2_price:
                tp_hit, tp_level = True, "TP2"
        if sl_hit and tp_hit:
            if is_long:
                sd = abs(o - current_sl)
                td = abs(self.tp1_price - o) if tp_level == "TP1" else abs(self.tp2_price - o)
            else:
                sd = abs(current_sl - o)
                td = abs(o - self.tp1_price) if tp_level == "TP1" else abs(o - self.tp2_price)
            if sd <= td:
                tp_hit = False
            else:
                sl_hit = False
        if tp_hit and tp_level == "TP1":
            self.tp1_hit = True
            self.leg1_exit = self.tp1_price
            return False
        if tp_hit and tp_level == "TP2":
            self.leg2_exit = self.tp2_price
            self.exit_time = ts
            self.exit_reason = "TP1_PARTIAL_THEN_TP2"
            return True
        if sl_hit:
            if self.tp1_hit:
                self.leg2_exit = self.be_price
                self.exit_reason = "TP1_PARTIAL_THEN_BE"
            else:
                self.leg1_exit = self.sl_price
                self.exit_reason = "SL_FULL"
            self.exit_time = ts
            return True
        return False

    def finalize_eod(self, ts: pd.Timestamp, c: float, *, half_spread: Optional[float] = None) -> None:
        if self.exit_time is not None:
            return
        hs = self.half_spread if half_spread is None else float(half_spread)
        exit_px = c - hs if self.is_long else c + hs
        self.exit_time = ts
        if self.tp1_hit:
            self.leg2_exit = exit_px
            self.exit_reason = "TP1_PARTIAL_THEN_HARD_CLOSE"
        else:
            self.leg1_exit = exit_px
            self.exit_reason = "HARD_CLOSE"

    def pnl_pips(self) -> float:
        is_long = self.is_long
        if self.tp1_hit:
            if is_long:
                p1 = (self.leg1_exit - self.entry_price) / PIP_SIZE if self.leg1_exit is not None else 0.0
                p2 = (self.leg2_exit - self.entry_price) / PIP_SIZE if self.leg2_exit is not None else 0.0
            else:
                p1 = (self.entry_price - self.leg1_exit) / PIP_SIZE if self.leg1_exit is not None else 0.0
                p2 = (self.entry_price - self.leg2_exit) / PIP_SIZE if self.leg2_exit is not None else 0.0
            return self.tp1_close_fraction * p1 + (1.0 - self.tp1_close_fraction) * p2
        if is_long:
            return (self.leg1_exit - self.entry_price) / PIP_SIZE if self.leg1_exit is not None else 0.0
        return (self.entry_price - self.leg1_exit) / PIP_SIZE if self.leg1_exit is not None else 0.0


def build_l1_incremental_from_entry(
    *,
    direction: str,
    entry_bar: pd.Series,
    lor_high: float,
    lor_low: float,
    ny_open: pd.Timestamp,
    spread_pips: float,
    tp1_r: float,
    tp2_r: float,
    be_offset: float,
    tp1_close_fraction: float,
) -> Optional[L1Incremental]:
    """Open at current bar mid open ± half spread; SL from LOR (same as simulate_trade)."""
    half = spread_pips * PIP_SIZE / 2.0
    is_long = direction == "long"
    entry_open = float(entry_bar["open"])
    entry_price = entry_open + half if is_long else entry_open - half
    sl_buffer = 3.0
    if is_long:
        raw_sl = lor_low - sl_buffer * PIP_SIZE
        sl_dist = (entry_price - raw_sl) / PIP_SIZE
    else:
        raw_sl = lor_high + sl_buffer * PIP_SIZE
        sl_dist = (raw_sl - entry_price) / PIP_SIZE
    sl_dist = max(5.0, min(20.0, sl_dist))
    sl_price = entry_price - sl_dist * PIP_SIZE if is_long else entry_price + sl_dist * PIP_SIZE
    risk = sl_dist
    if is_long:
        tp1 = entry_price + tp1_r * risk * PIP_SIZE
        tp2 = entry_price + tp2_r * risk * PIP_SIZE
        be_px = entry_price + be_offset * PIP_SIZE
    else:
        tp1 = entry_price - tp1_r * risk * PIP_SIZE
        tp2 = entry_price - tp2_r * risk * PIP_SIZE
        be_px = entry_price - be_offset * PIP_SIZE
    return L1Incremental(
        direction=direction,
        entry_price=entry_price,
        entry_time=_ts(entry_bar["time"]),
        sl_price=sl_price,
        tp1_price=tp1,
        tp2_price=tp2,
        be_price=be_px,
        tp1_close_fraction=tp1_close_fraction,
        half_spread=half,
        ny_open=ny_open,
        is_long=is_long,
        risk_pips=risk,
    )


@dataclass
class LondonAPosition:
    setup_type: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_close_fraction: float
    units: int
    tp1_hit: bool = False
    remaining: int = 0
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    mfe: float = 0.0
    mae: float = 0.0

    def __post_init__(self) -> None:
        self.remaining = self.units

    def on_bar(
        self,
        ts: pd.Timestamp,
        bid_o: float,
        ask_o: float,
        bid_h: float,
        ask_h: float,
        bid_l: float,
        ask_l: float,
        bid_c: float,
        ask_c: float,
        ny_open: pd.Timestamp,
    ) -> bool:
        if self.exit_time is not None:
            return True
        p = self
        if ts >= ny_open:
            px = bid_o if p.direction == "long" else ask_o
            p.exit_time = ts
            p.exit_price = px
            p.exit_reason = "HARD_CLOSE"
            return True
        if p.direction == "long":
            px_high, px_low, px_open = bid_h, bid_l, bid_o
            p.mfe = max(p.mfe, (px_high - p.entry_price) / PIP_SIZE)
            p.mae = max(p.mae, (p.entry_price - px_low) / PIP_SIZE)
            tp1_hit = (not p.tp1_hit) and (px_high >= p.tp1_price)
            sl_hit = px_low <= p.sl_price
        else:
            px_high, px_low, px_open = ask_h, ask_l, ask_o
            p.mfe = max(p.mfe, (p.entry_price - px_low) / PIP_SIZE)
            p.mae = max(p.mae, (px_high - p.entry_price) / PIP_SIZE)
            tp1_hit = (not p.tp1_hit) and (px_low <= p.tp1_price)
            sl_hit = px_high >= p.sl_price
        if not p.tp1_hit:
            if tp1_hit and sl_hit:
                da = abs(px_open - p.tp1_price)
                db = abs(px_open - p.sl_price)
                if db <= da:
                    tp1_hit = False
                else:
                    sl_hit = False
            if sl_hit:
                p.exit_time = ts
                p.exit_price = p.sl_price
                p.exit_reason = "SL_FULL"
                return True
            if tp1_hit:
                u_close = max(1, min(p.remaining, int(math.floor(p.units * p.tp1_close_fraction))))
                p.remaining -= u_close
                p.tp1_hit = True
                be_off = 1.0 * PIP_SIZE
                if p.direction == "long":
                    p.sl_price = max(p.sl_price, p.entry_price + be_off)
                else:
                    p.sl_price = min(p.sl_price, p.entry_price - be_off)
                if p.remaining <= 0:
                    p.exit_time = ts
                    p.exit_price = p.tp1_price
                    p.exit_reason = "TP"
                    return True
        if p.tp1_hit and p.remaining > 0:
            if p.direction == "long":
                tp2_hit = px_high >= p.tp2_price
                sl2 = px_low <= p.sl_price
            else:
                tp2_hit = px_low <= p.tp2_price
                sl2 = px_high >= p.sl_price
            if tp2_hit and sl2:
                da = abs(px_open - p.tp2_price)
                db = abs(px_open - p.sl_price)
                if db <= da:
                    tp2_hit = False
                else:
                    sl2 = False
            if tp2_hit:
                p.exit_time = ts
                p.exit_price = p.tp2_price
                p.exit_reason = "TP2_FULL"
                return True
            if sl2:
                p.exit_time = ts
                p.exit_price = p.sl_price
                p.exit_reason = "BE_STOP"
                return True
        return False

    def pnl_usd(self, exit_px: float) -> tuple[float, float]:
        """Return (pips, usd) for full position (simplified full close)."""
        if self.direction == "long":
            pips = (exit_px - self.entry_price) / PIP_SIZE
        else:
            pips = (self.entry_price - exit_px) / PIP_SIZE
        usd = pips * self.units * (PIP_SIZE / max(1e-9, exit_px))
        return float(pips), float(usd)


def pip_value_usd_per_lot(price: float) -> float:
    """scripts/backtest_session_momentum.pip_value_usd_per_lot"""
    return 1000.0 / max(1e-6, float(price))


def admission_checks(
    *,
    strategy: str,
    entry_ts: pd.Timestamp,
    cell_str: str,
    regime: str,
    delta_er: float,
    setup_d: bool,
    weekday_name: str,
    variant_f_allows: Optional[Callable[[], bool]] = None,
) -> tuple[bool, str]:
    """Variant I → (K + weekday for London) → F → defensive veto (V44 only)."""
    if _is_variant_i_blocked(regime, delta_er):
        return False, "variant_i_standdown"
    if strategy == "tokyo_v14":
        return True, "ok"
    if strategy == "london_v2":
        parts = cell_str.split("/")
        if len(parts) == 3:
            cl = (parts[0], parts[1], parts[2])
            if cl in LONDON_BLOCK_CLUSTER:
                return False, "variant_k_cluster"
        if setup_d and weekday_name in DROP_WEEKDAYS:
            return False, "weekday_block"
    if strategy == "v44_ny":
        if variant_f_allows is not None and not variant_f_allows():
            return False, "variant_f"
        if cell_str == TARGET_CELL_BLOCK:
            return False, "defensive_veto"
    return True, "ok"


@dataclass(frozen=True)
class Phase3V7PfddParams:
    """Parameters for the defended v7_pfdd bar-by-bar runner (CLI or engine)."""

    data_path: str
    spread_mode: str = "pipeline"
    dry_run: bool = False
    max_bars: Optional[int] = None
    quiet: bool = False


def execute_phase3_v7_pfdd_defended(params: Phase3V7PfddParams) -> dict[str, Any]:
    """Run the full Phase3 defended loop; return logs and summary (no file I/O)."""
    dataset_path = str(params.data_path)
    spread_pipeline = params.spread_mode == "pipeline"

    t0 = time.time()
    if not params.quiet:
        print(f"Loading {dataset_path} spread_mode={params.spread_mode}", flush=True)
    m1_raw = tokyo_bt.load_m1(dataset_path)
    if params.dry_run:
        m1_raw = m1_raw.iloc[:10_000].copy()
    if params.max_bars is not None:
        m1_raw = m1_raw.iloc[: int(params.max_bars)].copy()
    if not params.quiet:
        print(f"  M1 rows={len(m1_raw)} (after load/trim) in {time.time() - t0:.1f}s", flush=True)

    tokyo_cfg = init_tokyo_config(TOKYO_CFG)
    with LONDON_CFG_PATH.open() as f:
        london_cfg = json.load(f)
    with V44_CFG_PATH.open() as f:
        v44_cfg = json.load(f)

    t_stage = time.time()
    m5 = tokyo_bt.resample_ohlc_continuous(m1_raw, "5min")
    m15 = tokyo_bt.resample_ohlc_continuous(m1_raw, "15min")
    df, m5, m15 = compute_tokyo_indicators(m1_raw, m5, m15, tokyo_cfg)
    if not params.quiet:
        print(f"  Resample M5/M15 + compute_tokyo_indicators: {time.time() - t_stage:.1f}s (df rows={len(df)})", flush=True)
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["time_jst"] = df["time"].dt.tz_convert(tokyo_bt.TOKYO_TZ)
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]

    sf = tokyo_cfg.get("session_filter", {})
    session_start_utc = str(sf.get("session_start_utc", "15:00"))
    session_end_utc = str(sf.get("session_end_utc", "00:00"))

    def hhmm_to_minutes(s: str) -> int:
        hh, mm = s.strip().split(":")
        return int(hh) * 60 + int(mm)

    start_min = hhmm_to_minutes(session_start_utc)
    end_min = hhmm_to_minutes(session_end_utc)
    allowed_days = set(sf.get("allowed_trading_days", []))

    def in_window(m: int, ws: int, we: int) -> bool:
        if ws < we:
            return ws <= m < we
        return m >= ws or m < we

    if start_min < end_min:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) & (df["minute_of_day_utc"] < end_min)
    else:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) | (df["minute_of_day_utc"] < end_min)
    df["allowed_trading_day"] = df["utc_day_name"].isin(allowed_days)

    day_rng = (
        df.assign(utc_date=df["time_utc"].dt.date)
        .groupby("utc_date")
        .agg(day_high=("high", "max"), day_low=("low", "min"))
        .reset_index()
    )
    day_rng["range_pips"] = (day_rng["day_high"] - day_rng["day_low"]) / PIP_SIZE
    day_rng["prior_day_range_pips"] = day_rng["range_pips"].shift(1)
    df["prior_day_range_pips"] = df["time_utc"].dt.date.map(dict(zip(day_rng["utc_date"], day_rng["prior_day_range_pips"])))
    df["news_blocked"] = False

    h1 = tokyo_bt.resample_ohlc_continuous(m1_raw, "1h")
    h4 = tokyo_bt.resample_ohlc_continuous(m1_raw, "4h")
    d1 = tokyo_bt.resample_ohlc_continuous(m1_raw, "1D")

    bar_frame = auth_loop._load_bar_frame(dataset_path)
    if "ownership_cell" not in bar_frame.columns:
        bar_frame = bar_frame.copy()
        bar_frame["ownership_cell"] = [
            cell_key_from_floats(str(rg), float(er), float(de))
            for rg, er, de in zip(
                bar_frame["regime_hysteresis"],
                bar_frame["sf_er"],
                bar_frame["delta_er"],
                strict=False,
            )
        ]
    bf_times = pd.to_datetime(bar_frame["time"], utc=True).values.astype("datetime64[ns]")
    bf_cells = bar_frame["ownership_cell"].astype(str).tolist()
    bf_reg = bar_frame["regime_hysteresis"].astype(str).tolist()
    bf_er = bar_frame["sf_er"].astype(float).tolist()
    bf_der = bar_frame["delta_er"].astype(float).tolist()

    def lookup_cell(ts: pd.Timestamp) -> tuple[str, str, float, float]:
        key = np.datetime64(pd.Timestamp(ts).asm8.astype("datetime64[ns]"))
        idx = int(np.searchsorted(bf_times, key, side="right") - 1)
        if idx < 0:
            return "ambiguous/er_mid/der_pos", "ambiguous", 0.5, 0.0
        return bf_cells[idx], bf_reg[idx], bf_er[idx], bf_der[idx]

    v44_oracle = load_v44_oracle(dataset_path)
    v44_risk_pct = float(v44_cfg.get("v5_risk_per_trade_pct", 0.5)) / 100.0

    # Pre-build variant F context once (full M1 — causal lookups use entry time only)
    t_stage = time.time()
    vk_ctx = build_variant_k_baseline_context({"M1": m1_raw, "M5": m5})
    if not params.quiet:
        print(f"  build_variant_k_baseline_context: {time.time() - t_stage:.1f}s", flush=True)

    m5_times_ns = _m5_times_sorted_ns(m5)

    time_ns_all = pd.to_datetime(df["time"], utc=True).values.astype("datetime64[ns]")
    high_all = np.asarray(df["high"], dtype=np.float64)
    low_all = np.asarray(df["low"], dtype=np.float64)
    day_ns_bounds: dict[str, np.datetime64] = {}

    def _set_day_bounds(i0: int) -> None:
        day_start = pd.Timestamp(df.iloc[i0]["time"]).normalize()
        london_h = london_bt.uk_london_open_utc(day_start)
        london_open = day_start + pd.Timedelta(hours=london_h)
        lor_end = london_open + pd.Timedelta(minutes=15)

        def _utc_ns(ts: pd.Timestamp) -> np.datetime64:
            p = pd.Timestamp(ts)
            if p.tzinfo is None:
                p = p.tz_localize("UTC")
            else:
                p = p.tz_convert("UTC")
            return np.datetime64(p.asm8.astype("datetime64[ns]"))

        day_ns_bounds["asian_lo"] = _utc_ns(day_start)
        day_ns_bounds["asian_hi"] = _utc_ns(london_open)
        day_ns_bounds["lor_hi"] = _utc_ns(lor_end)

    def variant_f_allows_v44(entry_ts: pd.Timestamp) -> bool:
        tr = TradeRow(
            strategy="v44_ny",
            entry_time=_ts(entry_ts),
            exit_time=_ts(entry_ts),
            entry_session="v44_ny",
            side="buy",
            pips=0.0,
            usd=0.0,
            exit_reason="x",
            standalone_entry_equity=100000.0,
            raw={},
            size_scale=1.0,
        )
        r = v44_router._filter_v44_trade(
            tr,
            vk_ctx.classified_basic,
            vk_ctx.m5_basic,
            block_breakout=True,
            block_post_breakout=True,
            block_ambiguous_non_momentum=True,
            momentum_only=False,
            exhaustion_gate=False,
            soft_exhaustion=False,
            er_threshold=0.35,
            decay_threshold=0.40,
        )
        return not r.blocked

    asian_min = float(london_cfg["levels"]["asian_range_min_pips"])
    asian_max = float(london_cfg["levels"]["asian_range_max_pips"])
    lor_min = float(london_cfg["levels"]["lor_range_min_pips"])
    lor_max = float(london_cfg["levels"]["lor_range_max_pips"])
    l1_tp1_r = 3.25
    l1_tp2_r = 2.0
    l1_be_off = 1.0
    l1_tp1_close_fraction = float(london_cfg["setups"]["D"]["tp1_close_fraction"])

    # State
    equity = 100_000.0
    peak_eq = equity
    realized_cum = 0.0
    trade_id_box = [0]
    trades_log: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    blocked = Counter()
    margin_lev = 33.3
    max_lot = 20.0

    open_positions: list[dict[str, Any]] = []

    ld: Optional[LondonUnifiedDayState] = None
    tokyo_state = TokyoState()
    cur_day: Optional[pd.Timestamp] = None
    day_start_idx = 0

    def margin_london_v44() -> float:
        s = 0.0
        if ld is not None:
            s += sum(float(p.margin_required_usd) for p in ld.open_positions)
        for p in open_positions:
            if p["kind"] == "l1" and p.get("margin_usd") is not None:
                s += float(p["margin_usd"])
            else:
                lots = float(p.get("lots", 0.0))
                s += lots * 100_000.0 / margin_lev
        return s

    def margin_used() -> float:
        return margin_london_v44() + tokyo_margin_used_open_positions(tokyo_state.open_positions)

    def margin_avail() -> float:
        # Margin available for opening new positions = full equity minus margin in use.
        # OANDA's 50% margin closeout is a liquidation trigger (when NAV/margin_used < 0.50),
        # NOT a restriction on opening new trades.
        # See: https://www.oanda.com/us-en/trading/margin-rules/
        return equity - margin_used()

    def causal_asian_lor(i0: int, i: int, t: pd.Timestamp) -> tuple[float, float, float, bool, float, float, float, bool]:
        """Same ranges as the prior df.iloc slice + boolean masks; uses numpy on pre-extracted columns."""
        asian_lo = day_ns_bounds["asian_lo"]
        asian_hi = day_ns_bounds["asian_hi"]
        lor_lo = asian_hi
        lor_hi = day_ns_bounds["lor_hi"]
        ts_ns = time_ns_all[i]
        tseg = time_ns_all[i0 : i + 1]
        hseg = high_all[i0 : i + 1]
        lseg = low_all[i0 : i + 1]
        ma = (tseg >= asian_lo) & (tseg < asian_hi)
        if not ma.any():
            return (0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, False)
        ah = float(np.max(hseg[ma]))
        al = float(np.min(lseg[ma]))
        ar = (ah - al) / PIP_SIZE
        av = asian_min <= ar <= asian_max
        ml = (tseg >= lor_lo) & (tseg < lor_hi) & (tseg <= ts_ns)
        if not ml.any():
            return (ah, al, ar, av, 0.0, 0.0, 0.0, False)
        lh = float(np.max(hseg[ml]))
        ll = float(np.min(lseg[ml]))
        lr = (lh - ll) / PIP_SIZE
        lv = lor_min <= lr <= lor_max
        return (ah, al, ar, av, lh, ll, lr, lv)

    def london_exec_gate(pe: dict[str, Any], t: pd.Timestamp) -> tuple[bool, str]:
        setup = str(pe["setup_type"])
        cell_s, reg, _er, der_v = lookup_cell(t)
        ok, reason = admission_checks(
            strategy="london_v2",
            entry_ts=t,
            cell_str=cell_s,
            regime=reg,
            delta_er=der_v,
            setup_d=(setup == "D"),
            weekday_name=str(t.day_name()),
        )
        if not ok:
            blocked[f"london_{reason}"] += 1
        return ok, reason

    warmup = 200
    n = len(df)
    v44_oracle_placed = 0
    v44_oracle_blocked = 0
    v44_oracle_total = len(v44_oracle)
    london_closed_count = 0
    max_open_peak = 0

    loop_t0 = time.time()
    if not params.quiet:
        print(
            f"  Starting main M1 loop: bars {warmup}..{n - 1} ({n - warmup} iterations) — "
            f"progress every 10k bars + ETA",
            flush=True,
        )

    for i in range(warmup, n):
        row = df.iloc[i]
        ts = pd.Timestamp(row["time"])
        dnorm = ts.normalize()
        if cur_day is None or dnorm != cur_day:
            cur_day = dnorm
            day_start_idx = i
            ld = init_london_day_state(dnorm, london_cfg)
            _set_day_bounds(day_start_idx)

        cell_s, reg, er_v, der_v = lookup_cell(ts)

        def tokyo_exec_gate(_pe: dict[str, Any]) -> tuple[bool, str]:
            ok, reason = admission_checks(
                strategy="tokyo_v14",
                entry_ts=ts,
                cell_str=cell_s,
                regime=reg,
                delta_er=der_v,
                setup_d=False,
                weekday_name=str(ts.day_name()),
            )
            if not ok:
                blocked[f"tokyo_{reason}"] += 1
            return ok, reason

        tokyo_res = advance_tokyo_bar(
            bar_idx=i,
            m1_row=row,
            m5_indicators=m5,
            m15_indicators=m15,
            pivots={},
            cfg=tokyo_cfg,
            state=tokyo_state,
            equity=equity,
            exec_gate=tokyo_exec_gate,
            margin_avail=None,
            spread_pips=_get_tokyo_spread(ts, tokyo_cfg),
            pip_size=PIP_SIZE,
            pip_value=pip_value_usd_per_lot(float(row["close"])),
            m1_full=df,
            peak_equity=peak_eq,
            margin_used_other=margin_london_v44(),
            trade_id_counter=trade_id_box,
            ownership_cell=cell_s,
            regime_label=reg,
        )
        equity += tokyo_res.equity_delta
        realized_cum += tokyo_res.equity_delta
        peak_eq = max(peak_eq, equity)
        for tr in tokyo_res.exits:
            trades_log.append(tr)

        spread_pips_tokyo = tokyo_bt.compute_spread_pips(
            i, ts, str(tokyo_cfg["execution_model"].get("spread_mode", "fixed")),
            float(tokyo_cfg["execution_model"].get("spread_pips", 1.5)),
            float(tokyo_cfg["execution_model"].get("spread_min_pips", 1.0)),
            float(tokyo_cfg["execution_model"].get("spread_max_pips", 3.0)),
        )
        bid_c, ask_c = tokyo_bt.get_bid_ask(float(row["close"]), spread_pips_tokyo)
        bid_h, ask_h = tokyo_bt.get_bid_ask(float(row["high"]), spread_pips_tokyo)
        bid_l, ask_l = tokyo_bt.get_bid_ask(float(row["low"]), spread_pips_tokyo)
        bid_o, ask_o = tokyo_bt.get_bid_ask(float(row["open"]), spread_pips_tokyo)

        assert_m5_no_lookahead_fast(m5_times_ns, ts, "m5")

        # --- exits: V44 oracle + L1 (Tokyo still TODO) ---
        still: list[dict[str, Any]] = []
        for p in open_positions:
            if p["kind"] == "v44_oracle":
                if ts >= _ts(p["oracle_exit_time"]):
                    scale = float(p["lots"]) / max(1e-9, float(p["oracle_lots"]))
                    pnl_usd = float(p["oracle_pnl_usd"]) * scale
                    equity += pnl_usd
                    realized_cum += pnl_usd
                    peak_eq = max(peak_eq, equity)
                    trade_id_box[0] += 1
                    trades_log.append(
                        {
                            "trade_id": trade_id_box[0],
                            "session": "v44_ny",
                            "setup_type": "oracle",
                            "entry_bar_index": p["entry_i"],
                            "entry_time": str(p["entry_time"]),
                            "entry_price": p["entry_price"],
                            "side": p["side"],
                            "lots": p["lots"],
                            "sl_price": "",
                            "sl_pips": p.get("oracle_sl_pips", ""),
                            "tp1_price": "",
                            "tp2_price": "",
                            "exit_bar_index": i,
                            "exit_time": str(ts),
                            "exit_price": p.get("oracle_exit_price", ""),
                            "exit_reason": "oracle",
                            "pnl_pips": "",
                            "pnl_usd": pnl_usd,
                            "ownership_cell": p.get("cell", ""),
                            "regime_label": "",
                            "er": "",
                            "delta_er": "",
                            "admission_filters_applied": p.get("adm", ""),
                            "admission_filters_passed": True,
                            "was_v44_oracle": True,
                            "oracle_pnl_usd": p["oracle_pnl_usd"],
                            "sizing_scale_vs_oracle": scale,
                            "partial_close_1_time": "",
                            "partial_close_1_price": "",
                            "partial_close_1_pct": "",
                            "be_triggered": "",
                            "be_trigger_time": "",
                            "peak_favorable_pips": "",
                            "peak_adverse_pips": "",
                            "equity_at_entry": p.get("eq_entry", ""),
                            "margin_used_at_entry": p.get("marg_entry", ""),
                        }
                    )
                else:
                    still.append(p)
                continue
            if p["kind"] == "l1":
                sim: L1Incremental = p["sim"]
                hs_bar: Optional[float] = None
                if p.get("l1_realistic_spread"):
                    spb = london_bt.compute_spread_pips(i, ts, london_cfg)
                    hs_bar = spb * PIP_SIZE / 2.0
                if not sim.on_bar(
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    half_spread=hs_bar,
                ):
                    still.append(p)
                    continue
                pips = sim.pnl_pips()
                units = int(p["units"])
                exit_px = float(row["close"])
                _, usd = london_bt.calc_leg_pnl(
                    "long" if sim.is_long else "short", sim.entry_price, exit_px, units
                )
                equity += usd
                realized_cum += usd
                peak_eq = max(peak_eq, equity)
                trade_id_box[0] += 1
                meta = dict(p.get("meta", {}))
                meta.update(
                    {
                        "pnl_usd": usd,
                        "pnl_pips": pips,
                        "exit_time": str(sim.exit_time),
                        "trade_id": trade_id_box[0],
                        "exit_bar_index": i,
                        "exit_price": exit_px,
                    }
                )
                trades_log.append(meta)
                continue
            still.append(p)
        open_positions = still

        # --- London V2 (native A + Setup D → L1 incremental) ---
        if ld is not None:
            ah, al, ar, av, lh, ll, lr, lv = causal_asian_lor(day_start_idx, i, ts)
            i_day = i - day_start_idx
            nxt_ts = pd.Timestamp(df.iloc[i + 1]["time"]) if i + 1 < n else None
            equity, ldn_closed, l1_payloads, _ = advance_london_unified_bar(
                ld,
                row=row,
                ts=ts,
                nxt_ts=nxt_ts,
                i_day=i_day,
                i_global=i,
                asian_high=ah,
                asian_low=al,
                asian_range_pips=ar,
                asian_valid=av,
                lor_high=lh,
                lor_low=ll,
                lor_range_pips=lr,
                lor_valid=lv,
                equity=equity,
                margin_avail_unified=margin_avail(),
                extra_open_positions=len(open_positions) + len(tokyo_state.open_positions),
                trade_id_counter=trade_id_box,
                spread_mode_pipeline=spread_pipeline,
                l1_tp1_r=l1_tp1_r,
                l1_tp2_r=l1_tp2_r,
                l1_be_offset=l1_be_off,
                l1_tp1_close_fraction=l1_tp1_close_fraction,
                exec_gate=london_exec_gate,
            )
            peak_eq = max(peak_eq, equity)
            for tr in ldn_closed:
                london_closed_count += 1
                tr_out = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in tr.items()}
                trades_log.append(tr_out)
            for pay in l1_payloads:
                sim = build_l1_incremental_from_entry(
                    direction=str(pay["direction"]),
                    entry_bar=pay["entry_bar_row"],
                    lor_high=float(pay["lor_high"]),
                    lor_low=float(pay["lor_low"]),
                    ny_open=pay["ny_open"],
                    spread_pips=float(pay["spread_pips_exec"]),
                    tp1_r=float(l1_tp1_r),
                    tp2_r=float(l1_tp2_r),
                    be_offset=float(l1_be_off),
                    tp1_close_fraction=float(l1_tp1_close_fraction),
                )
                if sim is None:
                    continue
                req_m = float(pay["margin_required_usd"])
                if req_m > margin_avail():
                    blocked["margin"] += 1
                    continue
                cell_s, reg, er_v, der_v = lookup_cell(ts)
                open_positions.append(
                    {
                        "kind": "l1",
                        "sim": sim,
                        "units": int(pay["units"]),
                        "margin_usd": req_m,
                        "entry_i": i,
                        "l1_realistic_spread": params.spread_mode == "realistic",
                        "meta": {
                            "trade_id": int(pay["trade_id"]),
                            "session": "london_v2",
                            "setup_type": "D",
                            "entry_bar_index": i,
                            "entry_time": str(ts),
                            "entry_price": sim.entry_price,
                            "side": "buy" if sim.is_long else "sell",
                            "lots": "",
                            "sl_price": sim.sl_price,
                            "sl_pips": sim.risk_pips,
                            "tp1_price": sim.tp1_price,
                            "tp2_price": sim.tp2_price,
                            "ownership_cell": cell_s,
                            "regime_label": reg,
                            "er": er_v,
                            "delta_er": der_v,
                            "admission_filters_applied": "variant_pipeline",
                            "was_v44_oracle": False,
                            "mfe_pips": sim.mfe_pips,
                            "mae_pips": sim.mae_pips,
                            "equity_at_entry": equity,
                            "margin_used_at_entry": margin_used(),
                        },
                    }
                )

        # --- V44 oracle (after London; Tokyo TODO before London) ---
        oracle_row = v44_oracle.get(ts.value)
        if oracle_row is not None:
            side = str(oracle_row["side"]).lower()
            ok, reason = admission_checks(
                strategy="v44_ny",
                entry_ts=ts,
                cell_str=cell_s,
                regime=reg,
                delta_er=der_v,
                setup_d=False,
                weekday_name=str(ts.day_name()),
                variant_f_allows=lambda: variant_f_allows_v44(ts),
            )
            if not ok:
                blocked[reason] += 1
                v44_oracle_blocked += 1
            else:
                raw = dict(oracle_row)
                sl_pips = float(raw.get("sl_pips", 0.0) or raw.get("sl_dist", 5.0) or 5.0)
                ep0 = float(raw.get("entry_price", 150.0))
                pip_val = pip_value_usd_per_lot(ep0)
                or_pips = float(raw.get("pips", 0.0))
                or_usd = float(raw.get("usd", 0.0))
                if abs(or_pips) > 1e-9:
                    oracle_lots = abs(or_usd) / max(1e-9, abs(or_pips) * pip_val)
                else:
                    oracle_lots = 1.0
                risk_usd = equity * v44_risk_pct
                lots = risk_usd / max(1e-9, sl_pips * pip_val)
                lots = max(0.01, min(max_lot, lots))
                req_m = lots * 100_000.0 / margin_lev
                if req_m > margin_avail():
                    blocked["margin"] += 1
                    v44_oracle_blocked += 1
                else:
                    v44_oracle_placed += 1
                    open_positions.append(
                        {
                            "kind": "v44_oracle",
                            "entry_i": i,
                            "entry_time": ts,
                            "entry_price": float(raw["entry_price"]),
                            "side": "buy" if side == "buy" else "sell",
                            "lots": lots,
                            "oracle_lots": float(oracle_lots),
                            "oracle_pnl_usd": float(raw["usd"]),
                            "oracle_exit_time": _ts(raw["exit_time"]),
                            "oracle_exit_price": float(raw.get("exit_price", raw.get("exit_price_last", 0)) or 0),
                            "oracle_sl_pips": sl_pips,
                            "cell": cell_s,
                            "adm": "variant_pipeline",
                            "eq_entry": equity,
                            "marg_entry": margin_used(),
                        }
                    )

        realized_cum = equity - 100_000.0
        unreal = 0.0
        for p in open_positions:
            if p["kind"] != "v44_oracle":
                continue
            ep = float(p["entry_price"])
            lot = float(p["lots"])
            if p["side"] == "buy":
                u = (bid_c - ep) / PIP_SIZE * lot * pip_value_usd_per_lot(bid_c)
            else:
                u = (ep - ask_c) / PIP_SIZE * lot * pip_value_usd_per_lot(ask_c)
            unreal += u
        nav = equity + unreal
        if i % 1000 == 0:
            assert abs(equity - (100_000.0 + realized_cum)) < 0.05, "equity accounting drift"

        if not params.quiet:
            if i == warmup:
                print(f"  Phase3: first main-loop bar i={i} (warmup done)", flush=True)
            elif (i - warmup) % 10_000 == 0:
                elapsed = time.time() - loop_t0
                done = i - warmup
                total = n - warmup
                rate = done / max(elapsed, 1e-9)
                rem_s = (total - done) / max(rate, 1e-9)
                pct = 100.0 * done / max(total, 1)
                print(
                    f"  Phase3 progress: bar {i}/{n - 1} ({pct:.1f}%) "
                    f"loop_elapsed={elapsed:.0f}s ETA~{rem_s / 60:.1f}min equity={equity:.2f}",
                    flush=True,
                )

        open_ct = (
            len(open_positions)
            + (len(ld.open_positions) if ld is not None else 0)
            + len(tokyo_state.open_positions)
        )
        max_open_peak = max(max_open_peak, open_ct)
        equity_rows.append(
            {
                "bar_index": i,
                "timestamp": str(ts),
                "equity": equity,
                "nav": nav,
                "open_position_count": open_ct,
                "realized_pnl_cumulative": realized_cum,
                "unrealized_pnl": unreal,
                "drawdown_from_peak": peak_eq - equity,
                "drawdown_pct": (peak_eq - equity) / peak_eq * 100.0 if peak_eq > 0 else 0.0,
            }
        )

    def _row_pnl_usd(r: dict[str, Any]) -> Optional[float]:
        v = r.get("pnl_usd")
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    closed_with_pnl = [r for r in trades_log if _row_pnl_usd(r) is not None]
    pnls = [float(_row_pnl_usd(r)) for r in closed_with_pnl]
    n_tr = len(pnls)
    n_win = sum(1 for p in pnls if p > 0)
    n_loss = sum(1 for p in pnls if p < 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else (999.0 if gross_win > 0 else 0.0)
    win_rate_pct = (100.0 * n_win / n_tr) if n_tr else 0.0
    max_dd_usd = max((float(r.get("drawdown_from_peak", 0.0)) for r in equity_rows), default=0.0)

    trade_counts_by_type: dict[str, int] = {
        "london_setup_a": 0,
        "london_setup_d_l1": 0,
        "v44_ny": 0,
        "tokyo_v14": 0,
        "other": 0,
    }
    net_usd_by_type: dict[str, float] = {k: 0.0 for k in trade_counts_by_type}
    for r in closed_with_pnl:
        p = float(_row_pnl_usd(r))
        sess = str(r.get("session", ""))
        st = str(r.get("setup_type", ""))
        if sess == "v44_ny":
            key = "v44_ny"
        elif sess == "london_v2" and st == "A":
            key = "london_setup_a"
        elif sess == "london_v2" and st == "D":
            key = "london_setup_d_l1"
        elif sess in ("tokyo_v14", "v14"):
            key = "tokyo_v14"
        else:
            key = "other"
        trade_counts_by_type[key] += 1
        net_usd_by_type[key] += p

    summary = {
        "phase": "3B",
        "variant": "v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg",
        "spread_model": params.spread_mode,
        "data_file": Path(dataset_path).name,
        "total_bars": int(n),
        "starting_equity": 100_000.0,
        "ending_equity": float(equity),
        "net_pnl_usd": float(equity - 100_000.0),
        "closed_trades_total": n_tr,
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_concurrent_positions": int(max_open_peak),
        "trade_counts_by_type": trade_counts_by_type,
        "net_pnl_usd_by_type": {k: round(v, 2) for k, v in net_usd_by_type.items()},
        "v44_oracle_stats": {
            "oracle_entries_available": v44_oracle_total,
            "placed": v44_oracle_placed,
            "blocked": v44_oracle_blocked,
        },
        "trades_blocked_by": dict(blocked),
        "london_native_trades_closed": london_closed_count,
        "tokyo_diagnostics": {k: int(v) for k, v in dict(tokyo_state.diag).items()},
        "note": "Tokyo V14 (shadow spread from config, not --spread-mode) → London V2 → V44 oracle per bar; unified margin is equity − margin_used (London+V44+L1+Tokyo).",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    return {
        "trades_log": trades_log,
        "equity_rows": equity_rows,
        "summary": summary,
        "dataset_path": dataset_path,
        "total_bars_processed": int(n),
        "warmup_bars": int(warmup),
    }


def write_phase3_bar_by_bar_csv_outputs(
    result: dict[str, Any],
    *,
    trades_path: Path,
    equity_path: Path,
    summary_path: Path,
) -> None:
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["trades_log"]).to_csv(trades_path, index=False)
    pd.DataFrame(result["equity_rows"]).to_csv(equity_path, index=False)
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")


def params_from_argparse(args: argparse.Namespace) -> Phase3V7PfddParams:
    return Phase3V7PfddParams(
        data_path=str(DATASETS[args.dataset]),
        spread_mode=str(args.spread_mode),
        dry_run=bool(args.dry_run),
        quiet=False,
    )


def run() -> int:
    args = parse_args()
    out_suffix = "_realistic_spread" if args.spread_mode == "realistic" else ""
    trades_path = RESEARCH_OUT / f"v7_defended_bar_by_bar_trades{out_suffix}.csv"
    equity_path = RESEARCH_OUT / f"v7_defended_bar_by_bar_equity{out_suffix}.csv"
    summary_path = RESEARCH_OUT / f"v7_defended_bar_by_bar_summary{out_suffix}.json"
    result = execute_phase3_v7_pfdd_defended(params_from_argparse(args))
    write_phase3_bar_by_bar_csv_outputs(result, trades_path=trades_path, equity_path=equity_path, summary_path=summary_path)
    print(f"Wrote {trades_path} {equity_path} {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
