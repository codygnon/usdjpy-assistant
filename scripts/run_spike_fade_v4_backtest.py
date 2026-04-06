#!/usr/bin/env python3
"""
Bar-by-bar Spike Fade V4 backtest on full M1 sample (no core/ changes).

Spec: user-provided detection → exhaustion → trigger touch entry → SL/TP/prove-it/trail.
Spread: 1.6 pips adverse on entry. Prove-it: wall-clock >=15 min and unrealized <= -5 pips (from fill).
pip_value_usd = 100_000 * 0.01 / rate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M1_DEFAULT = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
OUT_DIR = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest"
TRADES_CSV = OUT_DIR / "spike_fade_v4_backtest_trades.csv"
SUMMARY_TXT = OUT_DIR / "spike_fade_v4_backtest_summary.txt"
V71_LOG = ROOT / "research_out/phase3_v7_pfdd_defended_real/v7_enriched_trade_log.csv"

LOT_UNITS = 100_000
PIP = 0.01
ENTRY_SPREAD_PIPS = 1.6
ENTRY_SPREAD_PRICE = ENTRY_SPREAD_PIPS * PIP
# Validated base (tighter 10/-3 caused excess prove-it with 1.6 pip entry spread)
PROVE_IT_MINUTES = 15.0
PROVE_IT_PIPS = -5.0
DEBUG_REPORT = OUT_DIR / "SPIKE_FADE_V4_DEBUG_REPORT.txt"


def pip_value_usd(mid: float) -> float:
    return LOT_UNITS * PIP / max(mid, 1e-9)


def pnl_usd_from_pips(pips: float, mid: float, units: float = LOT_UNITS) -> float:
    return pips * (units / LOT_UNITS) * pip_value_usd(mid)


class Phase(Enum):
    IDLE = auto()
    PENDING_FADE = auto()
    ARMED = auto()
    IN_TRADE = auto()


@dataclass
class M5Bar:
    t_end: pd.Timestamp
    end_i: int
    o: float
    h: float
    l: float
    c: float
    atr: Optional[float] = None


@dataclass
class M15Bar:
    t_end: pd.Timestamp
    end_i: int
    o: float
    h: float
    l: float
    c: float
    ema50: Optional[float] = None


@dataclass
class AtrState:
    n: int = 0
    tr_sum: float = 0.0
    atr: Optional[float] = None


@dataclass
class Funnel:
    m1_bars: int = 0
    m5_bars: int = 0
    stage1_range_atr: int = 0
    stage2_plus_body: int = 0
    stage3_plus_break: int = 0
    stage4_plus_cluster: int = 0
    spike_candidates: int = 0
    stage5_opposite_color: int = 0
    stage6_reclaim_mid: int = 0
    stage7_reclaim_prior: int = 0
    stage8_extension_score: int = 0
    exhaustion_confirmed: int = 0
    stage9_no_position: int = 0
    armed: int = 0
    filled: int = 0
    expired: int = 0
    exit_tp: int = 0
    exit_sl: int = 0
    exit_pif: int = 0
    exit_trail: int = 0


def wilder_push(st: AtrState, tr: float) -> Optional[float]:
    if st.n < 14:
        st.tr_sum += tr
        st.n += 1
        if st.n == 14:
            st.atr = st.tr_sum / 14.0
        return st.atr
    assert st.atr is not None
    st.atr = (st.atr * 13.0 + tr) / 14.0
    return st.atr


def true_range(h: float, l: float, prev_close: float) -> float:
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


def cluster_count_prior24(m5: list[M5Bar], j_spike: int) -> int:
    lo = max(0, j_spike - 24)
    cnt = 0
    for k in range(lo, j_spike):
        b = m5[k]
        if b.atr is None or b.atr <= 0:
            continue
        rp = (b.h - b.l) * 100.0
        atrp = b.atr * 100.0
        if rp / atrp >= 1.50:
            cnt += 1
    return cnt


def spike_candidate(m5: list[M5Bar], j: int) -> Optional[dict[str, Any]]:
    if j < 24:
        return None
    b = m5[j]
    if b.atr is None or b.atr <= 0:
        return None
    range_pips = (b.h - b.l) * 100.0
    body_pips = abs(b.c - b.o) * 100.0
    atrp = b.atr * 100.0
    range_atr = range_pips / atrp
    body_atr = body_pips / atrp
    if range_atr < 2.25 or body_atr < 1.50:
        return None
    if b.c > b.o:
        direction = "bullish"
    elif b.c < b.o:
        direction = "bearish"
    else:
        return None
    prev_h = max(m5[k].h for k in range(j - 12, j))
    prev_l = min(m5[k].l for k in range(j - 12, j))
    if direction == "bullish":
        if (b.h - prev_h) * 100.0 < 5.0:
            return None
    else:
        if (prev_l - b.l) * 100.0 < 5.0:
            return None
    if cluster_count_prior24(m5, j) > 2:
        return None
    return {
        "j": j,
        "o": b.o,
        "h": b.h,
        "l": b.l,
        "c": b.c,
        "t_end": b.t_end,
        "end_i": b.end_i,
        "direction": direction,
        "range_atr_ratio": range_atr,
        "body_atr_ratio": body_atr,
        "prior_12_high": prev_h,
        "prior_12_low": prev_l,
        "spike_range_pips": range_pips,
    }


def last_m15_ema50(m15: list[M15Bar]) -> Optional[float]:
    for b in reversed(m15):
        if b.ema50 is not None:
            return b.ema50
    return None


def exhaustion_ok(
    fade: M5Bar,
    spike: dict[str, Any],
    m5_atr: float,
    ema50: Optional[float],
) -> bool:
    if ema50 is None or m5_atr <= 0:
        return False
    if spike["direction"] == "bullish":
        if not (fade.c < fade.o):
            return False
    else:
        if not (fade.c > fade.o):
            return False
    mid = (spike["h"] + spike["l"]) * 0.5
    ph = spike["prior_12_high"]
    pl = spike["prior_12_low"]
    A = abs(fade.c - ema50) / m5_atr >= 1.25
    B = abs(fade.c - ema50) * 100.0 >= 20.0
    if spike["direction"] == "bullish":
        C = fade.c < mid
        D = fade.c < ph
    else:
        C = fade.c > mid
        D = fade.c > pl
    if not C or not D:
        return False
    score = int(A) + int(B) + int(C) + int(D)
    return score >= 2


def update_spike_detection_funnel(m5: list[M5Bar], j: int, funnel: Funnel) -> None:
    """On each completed M5 bar j, count cumulative spike-detection stages."""
    funnel.m5_bars += 1
    if j < 24:
        return
    b = m5[j]
    if b.atr is None or b.atr <= 0:
        return
    range_pips = (b.h - b.l) * 100.0
    body_pips = abs(b.c - b.o) * 100.0
    atrp = b.atr * 100.0
    range_atr = range_pips / atrp
    body_atr = body_pips / atrp
    if range_atr >= 2.25:
        funnel.stage1_range_atr += 1
    if range_atr >= 2.25 and body_atr >= 1.50:
        funnel.stage2_plus_body += 1
    if b.c > b.o:
        direction = "bullish"
    elif b.c < b.o:
        direction = "bearish"
    else:
        return
    prev_h = max(m5[k].h for k in range(j - 12, j))
    prev_l = min(m5[k].l for k in range(j - 12, j))
    if direction == "bullish":
        brk_ok = (b.h - prev_h) * 100.0 >= 5.0
    else:
        brk_ok = (prev_l - b.l) * 100.0 >= 5.0
    if range_atr >= 2.25 and body_atr >= 1.50 and brk_ok:
        funnel.stage3_plus_break += 1
    cl_ok = cluster_count_prior24(m5, j) <= 2
    if range_atr >= 2.25 and body_atr >= 1.50 and brk_ok and cl_ok:
        funnel.stage4_plus_cluster += 1
    if spike_candidate(m5, j) is not None:
        funnel.spike_candidates += 1


def exhaustion_funnel_counts(
    fade: M5Bar,
    spike: dict[str, Any],
    m5_atr: float,
    ema50: Optional[float],
    funnel: Funnel,
) -> None:
    """Count exhaustion sub-stages on spike+1 (fade) bar close."""
    if ema50 is None or m5_atr <= 0:
        return
    if spike["direction"] == "bullish":
        opp = fade.c < fade.o
    else:
        opp = fade.c > fade.o
    if not opp:
        return
    funnel.stage5_opposite_color += 1
    mid = (spike["h"] + spike["l"]) * 0.5
    ph = spike["prior_12_high"]
    pl = spike["prior_12_low"]
    if spike["direction"] == "bullish":
        C = fade.c < mid
        D = fade.c < ph
    else:
        C = fade.c > mid
        D = fade.c > pl
    if not C:
        return
    funnel.stage6_reclaim_mid += 1
    if not D:
        return
    funnel.stage7_reclaim_prior += 1
    A = abs(fade.c - ema50) / m5_atr >= 1.25
    B = abs(fade.c - ema50) * 100.0 >= 20.0
    score = int(A) + int(B) + int(C) + int(D)
    if score >= 2:
        funnel.stage8_extension_score += 1
    if exhaustion_ok(fade, spike, m5_atr, ema50):
        funnel.exhaustion_confirmed += 1


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


STANDALONE_TRADES = ROOT / "research_out/trade_analysis/spike_fade_v4/spike_v4_trades.csv"


def aggregate_m5_m15_from_df(df: pd.DataFrame) -> tuple[list[M5Bar], list[M15Bar]]:
    """Replay M1 → M5/M15 aggregation + Wilder ATR-14 + M15 EMA50 (same as backtest)."""
    df = df.sort_values("time").reset_index(drop=True)
    ts = df["time"].to_numpy()
    op = df["open"].to_numpy(np.float64)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    n = len(df)
    m5_key = pd.to_datetime(df["time"]).dt.floor("5min").to_numpy()
    m15_key = pd.to_datetime(df["time"]).dt.floor("15min").to_numpy()

    m5: list[M5Bar] = []
    m15: list[M15Bar] = []
    atr_st = AtrState()
    prev_close_m5 = float(cl[0])

    m5_o = m5_h = m5_l = m5_c = 0.0
    cur_m5_k: Optional[np.datetime64] = None
    m15_o = m15_h = m15_l = m15_c = 0.0
    cur_m15_k: Optional[np.datetime64] = None
    ema15: Optional[float] = None

    def finalize_m5(end_i: int) -> None:
        nonlocal prev_close_m5, m5_o, m5_h, m5_l, m5_c
        tr = true_range(m5_h, m5_l, prev_close_m5)
        atr_v = wilder_push(atr_st, tr)
        m5.append(
            M5Bar(
                t_end=pd.Timestamp(ts[end_i]),
                end_i=end_i,
                o=m5_o,
                h=m5_h,
                l=m5_l,
                c=m5_c,
                atr=atr_v,
            )
        )
        prev_close_m5 = m5_c

    def finalize_m15(end_i: int) -> None:
        nonlocal ema15, m15_o, m15_h, m15_l, m15_c
        c = m15_c
        if ema15 is None:
            ema15 = c
        else:
            k = 2.0 / 51.0
            ema15 = c * k + ema15 * (1.0 - k)
        m15.append(
            M15Bar(
                t_end=pd.Timestamp(ts[end_i]),
                end_i=end_i,
                o=m15_o,
                h=m15_h,
                l=m15_l,
                c=m15_c,
                ema50=ema15,
            )
        )

    for i in range(n):
        pk5 = m5_key[i]
        pk15 = m15_key[i]
        if cur_m5_k is None:
            cur_m5_k = pk5
            m5_o, m5_h, m5_l, m5_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
        elif pk5 != cur_m5_k:
            finalize_m5(i - 1)
            cur_m5_k = pk5
            m5_o, m5_h, m5_l, m5_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
        else:
            m5_h = max(m5_h, float(hi[i]))
            m5_l = min(m5_l, float(lo[i]))
            m5_c = float(cl[i])

        if cur_m15_k is None:
            cur_m15_k = pk15
            m15_o, m15_h, m15_l, m15_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
        elif pk15 != cur_m15_k:
            finalize_m15(i - 1)
            cur_m15_k = pk15
            m15_o, m15_h, m15_l, m15_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
        else:
            m15_h = max(m15_h, float(hi[i]))
            m15_l = min(m15_l, float(lo[i]))
            m15_c = float(cl[i])

    if n > 0 and cur_m5_k is not None:
        finalize_m5(n - 1)
    if n > 0 and cur_m15_k is not None:
        finalize_m15(n - 1)
    return m5, m15


def find_m5_index_for_t_end(m5: list[M5Bar], t_ref: pd.Timestamp) -> int:
    """Match standalone spike_time (bucket label) to our M5 bar (t_end = last M1 in bucket)."""
    t_ref = pd.to_datetime(t_ref, utc=True)
    key = t_ref.floor("5min")
    for j, b in enumerate(m5):
        bt = pd.Timestamp(b.t_end)
        if bt.tzinfo is None:
            bt = bt.tz_localize("UTC")
        else:
            bt = bt.tz_convert("UTC")
        if bt.floor("5min") == key:
            return j
    return -1


def spike_filter_checklist(m5: list[M5Bar], j: int) -> list[str]:
    lines: list[str] = []
    if j < 24:
        lines.append(f"FAIL: need j>=24 (have j={j})")
        return lines
    b = m5[j]
    if b.atr is None or b.atr <= 0:
        lines.append("FAIL: ATR missing or <=0 on spike M5 bar")
        return lines
    range_pips = (b.h - b.l) * 100.0
    body_pips = abs(b.c - b.o) * 100.0
    atrp = b.atr * 100.0
    range_atr = range_pips / atrp
    body_atr = body_pips / atrp
    lines.append(
        f"Spike M5 t_end={b.t_end} O={b.o:.5f} H={b.h:.5f} L={b.l:.5f} C={b.c:.5f} "
        f"ATR={b.atr:.6f} range_atr={range_atr:.4f} body_atr={body_atr:.4f}"
    )
    lines.append(f"  range_atr>=2.25 -> {'PASS' if range_atr >= 2.25 else 'FAIL'}")
    lines.append(f"  body_atr>=1.50 -> {'PASS' if body_atr >= 1.50 else 'FAIL'}")
    if b.c > b.o:
        direction = "bullish"
    elif b.c < b.o:
        direction = "bearish"
    else:
        lines.append("  direction (body color) -> FAIL (doji)")
        return lines
    lines.append(f"  direction -> {direction} (PASS)")
    prev_h = max(m5[k].h for k in range(j - 12, j))
    prev_l = min(m5[k].l for k in range(j - 12, j))
    if direction == "bullish":
        brk = (b.h - prev_h) * 100.0
        lines.append(f"  break prior 12 high by >=5 pips: {brk:.2f} pips -> {'PASS' if brk >= 5 else 'FAIL'}")
    else:
        brk = (prev_l - b.l) * 100.0
        lines.append(f"  break prior 12 low by >=5 pips: {brk:.2f} pips -> {'PASS' if brk >= 5 else 'FAIL'}")
    cc = cluster_count_prior24(m5, j)
    lines.append(f"  cluster_count_prior24 <=2 -> count={cc} {'PASS' if cc <= 2 else 'FAIL'}")
    cand = spike_candidate(m5, j)
    lines.append(f"  spike_candidate() -> {'PASS (candidate)' if cand is not None else 'FAIL'}")
    return lines


def exhaustion_checklist(fade: M5Bar, spike: dict[str, Any], m5_atr: float, ema50: Optional[float]) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"Fade M5 t_end={fade.t_end} O={fade.o:.5f} H={fade.h:.5f} L={fade.l:.5f} C={fade.c:.5f} fade_ATR={fade.atr}"
    )
    if ema50 is None:
        lines.append("FAIL: M15 EMA50 not available")
        return lines
    if m5_atr <= 0:
        lines.append("FAIL: fade ATR <= 0")
        return lines
    lines.append(f"M15 EMA50 (last completed)={ema50:.5f}")
    if spike["direction"] == "bullish":
        opp = fade.c < fade.o
        lines.append(f"  opposite color (bearish fade) -> {'PASS' if opp else 'FAIL'}")
    else:
        opp = fade.c > fade.o
        lines.append(f"  opposite color (bullish fade) -> {'PASS' if opp else 'FAIL'}")
    if not opp:
        return lines
    mid = (spike["h"] + spike["l"]) * 0.5
    ph = spike["prior_12_high"]
    pl = spike["prior_12_low"]
    if spike["direction"] == "bullish":
        C = fade.c < mid
        D = fade.c < ph
        lines.append(f"  reclaim spike mid ({mid:.5f}) -> {'PASS' if C else 'FAIL'}")
        lines.append(f"  reclaim prior high ({ph:.5f}) -> {'PASS' if D else 'FAIL'}")
    else:
        C = fade.c > mid
        D = fade.c > pl
        lines.append(f"  reclaim spike mid ({mid:.5f}) -> {'PASS' if C else 'FAIL'}")
        lines.append(f"  reclaim prior low ({pl:.5f}) -> {'PASS' if D else 'FAIL'}")
    A = abs(fade.c - ema50) / m5_atr >= 1.25
    B = abs(fade.c - ema50) * 100.0 >= 20.0
    score = int(A) + int(B) + int(C) + int(D)
    lines.append(f"  extension A(>=1.25*ATR dist EMA)={A} B(>=20pip dist EMA)={B} score={score} -> {'PASS' if score >= 2 else 'FAIL'}")
    lines.append(f"  exhaustion_ok -> {'PASS' if exhaustion_ok(fade, spike, m5_atr, ema50) else 'FAIL'}")
    return lines


def m15_bars_completed_through(m15: list[M15Bar], m1_end_i: int) -> list[M15Bar]:
    return [b for b in m15 if b.end_i <= m1_end_i]


def format_funnel_report(fd: dict[str, Any]) -> str:
    return "\n".join(
        [
            "SPIKE FADE V4 — DETECTION FUNNEL",
            "═══════════════════════════════════════════",
            f"Total M5 bars processed:              {fd.get('m5_bars', 0)}",
            f"Total M1 bars processed:              {fd.get('m1_bars', 0)}",
            "",
            f"Stage 1: range_atr >= 2.25:           {fd.get('stage1_range_atr', 0)}",
            f"Stage 2: + body_atr >= 1.50:          {fd.get('stage2_plus_body', 0)}",
            f"Stage 3: + breaks prior 12-bar range: {fd.get('stage3_plus_break', 0)}",
            f"Stage 4: + cluster count <= 2:        {fd.get('stage4_plus_cluster', 0)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Spike candidates passed:              {fd.get('spike_candidates', 0)}",
            "",
            f"Stage 5: spike+1 opposite color:      {fd.get('stage5_opposite_color', 0)}",
            f"Stage 6: + reclaims spike mid:        {fd.get('stage6_reclaim_mid', 0)}",
            f"Stage 7: + reclaims prior range:      {fd.get('stage7_reclaim_prior', 0)}",
            f"Stage 8: + extension score >= 2:      {fd.get('stage8_extension_score', 0)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Exhaustion confirmed:                 {fd.get('exhaustion_confirmed', 0)}",
            "",
            f"Stage 9: no existing V4 position:     {fd.get('stage9_no_position', 0)}",
            f"Stage 10: armed (limit order placed): {fd.get('armed', 0)}",
            f"Stage 11: filled (trigger touched):   {fd.get('filled', 0)}",
            f"Stage 12: expired (no fill in 15min): {fd.get('expired', 0)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Trades opened:                        {fd.get('filled', 0)}",
            f"Trades closed:                        {fd.get('trades_closed', 0)}",
            "",
            "Exit breakdown:",
            f"  TP:              {fd.get('exit_tp', 0)}",
            f"  SL:              {fd.get('exit_sl', 0)}",
            f"  prove_it_fast:   {fd.get('exit_pif', 0)}",
            f"  trailing:        {fd.get('exit_trail', 0)}",
            "",
            "EXPECTED (from standalone Family C / Model 4 — different signal pipeline):",
            "  Spike candidates: ~500+",
            "  Exhaustion passed: ~250+",
            "  Trades filled:     ~221",
        ]
    )


def trace_missing_standalone_rows(
    df: pd.DataFrame,
    m5: list[M5Bar],
    m15: list[M15Bar],
    missing: pd.DataFrame,
) -> str:
    chunks: list[str] = []
    for _, row in missing.iterrows():
        chunks.append("=" * 72)
        chunks.append("MISSING STANDALONE TRADE (not in bar-by-bar log)")
        chunks.append(str(row.to_dict()))
        st = pd.to_datetime(row["spike_time"], utc=True)
        j = find_m5_index_for_t_end(m5, st)
        if j < 0:
            chunks.append(f"Could not locate M5 bar with t_end={st}")
            continue
        chunks.extend(spike_filter_checklist(m5, j))
        cand = spike_candidate(m5, j)
        if cand is None:
            chunks.append("REJECTED AT: spike detection (see checklist above)")
            continue
        if j + 1 >= len(m5):
            chunks.append("REJECTED AT: no fade (spike+1) bar in series")
            continue
        fade = m5[j + 1]
        atr_f = fade.atr if fade.atr is not None else 0.0
        ema50 = last_m15_ema50(m15_bars_completed_through(m15, fade.end_i))
        chunks.append("--- Exhaustion (spike+1 M5) ---")
        if fade.atr is None or fade.atr <= 0:
            chunks.append("REJECTED AT: exhaustion (fade ATR missing or <=0)")
            continue
        chunks.extend(exhaustion_checklist(fade, cand, float(fade.atr), ema50))
        if not exhaustion_ok(fade, cand, float(fade.atr), ema50):
            chunks.append("REJECTED AT: exhaustion (M15 EMA / reclaim / extension)")
            continue
        side = "short" if cand["direction"] == "bullish" else "long"
        el = float(cand["prior_12_high"] if side == "short" else cand["prior_12_low"])
        fill = el - ENTRY_SPREAD_PRICE if side == "short" else el + ENTRY_SPREAD_PRICE
        chunks.append(
            f"Bar-by-bar would arm: side={side} entry_level={el:.5f} entry_fill={fill:.5f} "
            f"(touch uses level; spread adverse on fill)."
        )
        chunks.append(
            "NOTE: Standalone CSV is Family C Model 4 trigger_touch — not this ATR/cluster exhaustion "
            "pipeline; passing checks here does not imply the standalone engine fired the same rule."
        )
    return "\n".join(chunks)


def trace_prove_it_trades(df: pd.DataFrame, bt: pd.DataFrame, n: int = 5) -> str:
    pif = bt[bt["exit_reason"].astype(str) == "prove_it_fast"].head(n)
    chunks: list[str] = []
    df = df.sort_values("time").reset_index(drop=True)
    tcol = df["time"]
    for _, row in pif.iterrows():
        chunks.append("=" * 72)
        chunks.append("BAR-BY-BAR prove_it_fast TRADE TRACE")
        chunks.append(str(row.to_dict()))
        et = pd.to_datetime(row["entry_time"], utc=True)
        xt = pd.to_datetime(row["exit_time"], utc=True)
        side = str(row["side"])
        entry_fill = float(row["entry_price"])
        entry_level = float(row["entry_level"])
        mask = (tcol >= et) & (tcol <= xt)
        sub = df.loc[mask, ["time", "open", "high", "low", "close"]].copy()
        chunks.append(f"M1 bars from entry to exit: {len(sub)} rows")
        for _, r in sub.iterrows():
            mid = float(r["close"])
            if side == "short":
                u_fill = (entry_fill - mid) * 100.0
                u_lvl = (entry_level - mid) * 100.0
            else:
                u_fill = (mid - entry_fill) * 100.0
                u_lvl = (mid - entry_level) * 100.0
            elap = (pd.to_datetime(r["time"], utc=True) - et).total_seconds() / 60.0
            trig = ""
            if elap >= PROVE_IT_MINUTES and u_fill <= PROVE_IT_PIPS:
                trig = " <<< PIF (from fill)"
            chunks.append(
                f"  {r['time']} close={mid:.5f} unreal_pips_fill={u_fill:.2f} "
                f"unreal_pips_level={u_lvl:.2f} elapsed_min={elap:.2f}{trig}"
            )
        spread_pips = abs(entry_fill - entry_level) * 100.0
        chunks.append(f"Spread pips (|fill-level|): {spread_pips:.2f}")
        chunks.append(
            f"PIF rule: elapsed_min>={PROVE_IT_MINUTES} and unreal(from fill)<={PROVE_IT_PIPS} "
            f"(measuring from trigger would shift unreal by ~±{spread_pips:.2f} pips vs fill)."
        )
    return "\n".join(chunks)


def run_backtest(
    csv_path: Path,
    *,
    max_bars: int = -1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.sort_values("time", inplace=True)
    if max_bars > 0:
        df = df.iloc[:max_bars].copy()

    ts = df["time"].to_numpy()
    op = df["open"].to_numpy(np.float64)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    n = len(df)

    m5_key = pd.to_datetime(df["time"]).dt.floor("5min").to_numpy()
    m15_key = pd.to_datetime(df["time"]).dt.floor("15min").to_numpy()

    m5: list[M5Bar] = []
    m15: list[M15Bar] = []
    atr_st = AtrState()
    prev_close_m1 = float(cl[0])
    prev_close_m5 = float(cl[0])

    m5_o = m5_h = m5_l = m5_c = 0.0
    m5_start_i = 0
    cur_m5_k: Optional[np.datetime64] = None

    m15_o = m15_h = m15_l = m15_c = 0.0
    m15_start_i = 0
    cur_m15_k: Optional[np.datetime64] = None
    ema15: Optional[float] = None

    phase = Phase.IDLE
    pending_spike: Optional[dict[str, Any]] = None
    armed: Optional[dict[str, Any]] = None
    pos: Optional[dict[str, Any]] = None
    funnel = Funnel()

    trades: list[dict[str, Any]] = []

    def finalize_m5(end_i: int) -> None:
        nonlocal prev_close_m5, m5_o, m5_h, m5_l, m5_c, m5_start_i
        tr = true_range(m5_h, m5_l, prev_close_m5)
        atr_v = wilder_push(atr_st, tr)
        bar = M5Bar(
            t_end=pd.Timestamp(ts[end_i]),
            end_i=end_i,
            o=m5_o,
            h=m5_h,
            l=m5_l,
            c=m5_c,
            atr=atr_v,
        )
        m5.append(bar)
        prev_close_m5 = m5_c
        j = len(m5) - 1
        update_spike_detection_funnel(m5, j, funnel)

        nonlocal phase, pending_spike, armed, pos

        if pos is not None:
            return
        if armed is not None:
            return

        if phase == Phase.PENDING_FADE and pending_spike is not None:
            sp_j = pending_spike["j"]
            if j == sp_j + 1:
                fade = bar
                sp = pending_spike
                atr_f = fade.atr
                if atr_f is None or atr_f <= 0:
                    phase = Phase.IDLE
                    pending_spike = None
                    return
                ema50 = last_m15_ema50(m15)
                exhaustion_funnel_counts(fade, sp, atr_f, ema50, funnel)
                if exhaustion_ok(fade, sp, atr_f, ema50):
                    funnel.stage9_no_position += 1
                    if sp["direction"] == "bullish":
                        side = "short"
                        entry_level = sp["prior_12_high"]
                    else:
                        side = "long"
                        entry_level = sp["prior_12_low"]
                    raw_stop = sp["h"] + 0.02 if sp["direction"] == "bullish" else sp["l"] - 0.02
                    if side == "short":
                        stop_pips = (raw_stop - entry_level) * 100.0
                    else:
                        stop_pips = (entry_level - raw_stop) * 100.0
                    stop_pips = clamp(stop_pips, 15.0, 35.0)
                    if side == "short":
                        stop_price = entry_level + stop_pips / 100.0
                        tp_price = entry_level - sp["spike_range_pips"] * 0.50 / 100.0
                    else:
                        stop_price = entry_level - stop_pips / 100.0
                        tp_price = entry_level + sp["spike_range_pips"] * 0.50 / 100.0
                    armed = {
                        "side": side,
                        "entry_level": entry_level,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "fade_close_i": end_i,
                        "expiry_until_i": end_i + 15,
                        "spike": sp,
                    }
                    funnel.armed += 1
                    phase = Phase.ARMED
                    pending_spike = None
                else:
                    phase = Phase.IDLE
                    pending_spike = None
            elif j > sp_j + 1:
                phase = Phase.IDLE
                pending_spike = None
            return

        if phase == Phase.IDLE:
            cand = spike_candidate(m5, j)
            if cand is not None:
                pending_spike = cand
                phase = Phase.PENDING_FADE

    def finalize_m15(end_i: int) -> None:
        nonlocal ema15, m15_o, m15_h, m15_l, m15_c, m15_start_i
        c = m15_c
        if ema15 is None:
            ema15 = c
        else:
            k = 2.0 / 51.0
            ema15 = c * k + ema15 * (1.0 - k)
        bar = M15Bar(
            t_end=pd.Timestamp(ts[end_i]),
            end_i=end_i,
            o=m15_o,
            h=m15_h,
            l=m15_l,
            c=m15_c,
            ema50=ema15,
        )
        m15.append(bar)

    for i in range(n):
        t = pd.Timestamp(ts[i])
        pk5 = m5_key[i]
        pk15 = m15_key[i]

        if cur_m5_k is None:
            cur_m5_k = pk5
            m5_o, m5_h, m5_l, m5_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
            m5_start_i = i
        elif pk5 != cur_m5_k:
            finalize_m5(i - 1)
            cur_m5_k = pk5
            m5_o, m5_h, m5_l, m5_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
            m5_start_i = i
        else:
            m5_h = max(m5_h, float(hi[i]))
            m5_l = min(m5_l, float(lo[i]))
            m5_c = float(cl[i])

        if cur_m15_k is None:
            cur_m15_k = pk15
            m15_o, m15_h, m15_l, m15_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
            m15_start_i = i
        elif pk15 != cur_m15_k:
            finalize_m15(i - 1)
            cur_m15_k = pk15
            m15_o, m15_h, m15_l, m15_c = float(op[i]), float(hi[i]), float(lo[i]), float(cl[i])
            m15_start_i = i
        else:
            m15_h = max(m15_h, float(hi[i]))
            m15_l = min(m15_l, float(lo[i]))
            m15_c = float(cl[i])

        mid = float(cl[i])
        h_i = float(hi[i])
        l_i = float(lo[i])

        if pos is not None:
            side = pos["side"]
            entry = pos["entry_fill"]
            stop_px = pos["stop_price"]
            tp_px = pos["tp_price"]
            exit_reason = None
            exit_px = None

            if side == "short":
                fav = (entry - l_i) * 100.0
                adv = (h_i - entry) * 100.0
                cur_unreal = (entry - mid) * 100.0
            else:
                fav = (h_i - entry) * 100.0
                adv = (entry - l_i) * 100.0
                cur_unreal = (mid - entry) * 100.0

            pos["mfe"] = max(pos["mfe"], fav)
            pos["mae"] = max(pos["mae"], adv)

            if side == "short":
                if h_i >= stop_px:
                    exit_reason = "stop_loss"
                    exit_px = stop_px
                elif l_i <= tp_px:
                    exit_reason = "take_profit"
                    exit_px = tp_px
            else:
                if l_i <= stop_px:
                    exit_reason = "stop_loss"
                    exit_px = stop_px
                elif h_i >= tp_px:
                    exit_reason = "take_profit"
                    exit_px = tp_px

            elapsed_min = (t - pos["entry_time"]).total_seconds() / 60.0
            if exit_reason is None and elapsed_min >= PROVE_IT_MINUTES:
                if cur_unreal <= PROVE_IT_PIPS:
                    exit_reason = "prove_it_fast"
                    exit_px = mid

            if exit_reason is None:
                best = pos["mfe"]
                if best >= 10.0:
                    trail_lvl = best - 5.0
                    if cur_unreal <= trail_lvl:
                        exit_reason = "trailing_stop"
                        exit_px = mid

            if exit_reason is not None:
                pips = (entry - exit_px) * 100.0 if side == "short" else (exit_px - entry) * 100.0
                pnl_usd = pnl_usd_from_pips(pips, exit_px, LOT_UNITS)
                sp = pos["spike"]
                er = str(exit_reason)
                if er == "take_profit":
                    funnel.exit_tp += 1
                elif er == "stop_loss":
                    funnel.exit_sl += 1
                elif er == "prove_it_fast":
                    funnel.exit_pif += 1
                elif er == "trailing_stop":
                    funnel.exit_trail += 1
                hold_min = (t - pos["entry_time"]).total_seconds() / 60.0
                trades.append(
                    {
                        "entry_time": str(pos["entry_time"]),
                        "exit_time": str(t),
                        "side": side,
                        "entry_price": round(entry, 5),
                        "exit_price": round(exit_px, 5),
                        "stop_price": round(stop_px, 5),
                        "tp_price": round(tp_px, 5),
                        "pnl_pips": round(pips, 2),
                        "pnl_usd": round(pnl_usd, 2),
                        "exit_reason": exit_reason,
                        "sub_strategy": "spike_fade_v4",
                        "spike_time": str(sp["t_end"]),
                        "spike_high": sp["h"],
                        "spike_low": sp["l"],
                        "spike_range_pips": round(sp["spike_range_pips"], 2),
                        "range_atr_ratio": round(sp["range_atr_ratio"], 4),
                        "body_atr_ratio": round(sp["body_atr_ratio"], 4),
                        "prior_range_high": sp["prior_12_high"],
                        "prior_range_low": sp["prior_12_low"],
                        "entry_level": round(pos["entry_level"], 5),
                        "duration_minutes": round(hold_min, 2),
                        "mfe_pips": round(pos["mfe"], 2),
                        "mae_pips": round(pos["mae"], 2),
                    }
                )
                pos = None
                phase = Phase.IDLE

        if pos is None and armed is not None:
            fc = armed["fade_close_i"]
            if i > armed["expiry_until_i"]:
                funnel.expired += 1
                armed = None
                phase = Phase.IDLE
                pending_spike = None
            elif i > fc:
                side = armed["side"]
                el = armed["entry_level"]
                touched = (h_i >= el) if side == "short" else (l_i <= el)
                if touched:
                    if side == "short":
                        entry_fill = el - ENTRY_SPREAD_PRICE
                    else:
                        entry_fill = el + ENTRY_SPREAD_PRICE
                    pos = {
                        "side": side,
                        "entry_fill": entry_fill,
                        "entry_level": el,
                        "stop_price": armed["stop_price"],
                        "tp_price": armed["tp_price"],
                        "fill_i": i,
                        "entry_time": t,
                        "spike": armed["spike"],
                        "mfe": 0.0,
                        "mae": 0.0,
                    }
                    funnel.filled += 1
                    armed = None
                    phase = Phase.IN_TRADE

        prev_close_m1 = float(cl[i])

    if n > 0 and cur_m5_k is not None:
        finalize_m5(n - 1)
    if n > 0 and cur_m15_k is not None:
        finalize_m15(n - 1)

    funnel.m1_bars = n
    stats = summarize_trades(trades)
    stats["funnel"] = asdict(funnel)
    return trades, stats


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"n": 0}
    pnls = np.array([float(t["pnl_usd"]) for t in trades])
    pips = np.array([float(t["pnl_pips"]) for t in trades])
    wins = pnls > 0
    gp = float(pnls[wins].sum()) if wins.any() else 0.0
    gl = float(abs(pnls[~wins].sum())) if (~wins).any() else 0.0
    pf = gp / gl if gl > 1e-9 else float("inf")

    eq = 100_000.0
    peak = eq
    max_dd = 0.0
    for x in pnls:
        eq += x
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    max_dd_pct = 100.0 * max_dd / 100_000.0

    reasons: dict[str, int] = {}
    for t in trades:
        r = str(t["exit_reason"])
        reasons[r] = reasons.get(r, 0) + 1

    et = pd.to_datetime([t["entry_time"] for t in trades], utc=True)
    years_span = max((et.max() - et.min()).days / 365.25, 1e-9)

    return {
        "n": len(trades),
        "win_rate": float(wins.mean()),
        "profit_factor": float(min(pf, 999.0)),
        "net_pnl_usd": float(pnls.sum()),
        "max_drawdown_usd": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "avg_win_pips": float(pips[wins].mean()) if wins.any() else 0.0,
        "avg_loss_pips": float(pips[~wins].mean()) if (~wins).any() else 0.0,
        "avg_win_usd": float(pnls[wins].mean()) if wins.any() else 0.0,
        "avg_loss_usd": float(pnls[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold_min": float(np.mean([float(t["duration_minutes"]) for t in trades])),
        "trades_per_year": len(trades) / years_span,
        "exit_reasons": reasons,
        "et": et,
        "pnls": pnls,
    }


def yearly_quarterly(trades: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}, {}
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["pnl_usd"] = df["pnl_usd"].astype(float)
    ex = df["exit_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    y = df.groupby(ex.dt.to_period("Y"))["pnl_usd"].sum().to_dict()
    q = df.groupby(ex.dt.to_period("Q"))["pnl_usd"].sum().to_dict()
    y_str = {str(k): float(v) for k, v in y.items()}
    q_str = {str(k): float(v) for k, v in q.items()}
    return y_str, q_str


def combine_with_v71(v4_trades: list[dict], v71_path: Path) -> dict[str, Any]:
    if not v71_path.is_file():
        return {"error": f"missing {v71_path}"}
    v71 = pd.read_csv(v71_path)
    v71["exit_time"] = pd.to_datetime(v71["exit_time"], utc=True)
    v71["pnl_usd"] = v71["pnl_usd"].astype(float)
    v4_df = pd.DataFrame(v4_trades) if v4_trades else pd.DataFrame()
    if not v4_df.empty:
        v4_df["exit_time"] = pd.to_datetime(v4_df["exit_time"], utc=True)
        v4_df["pnl_usd"] = v4_df["pnl_usd"].astype(float)

    def curve(df: pd.DataFrame) -> tuple[float, float, float]:
        if df.empty:
            return 0.0, 0.0, 0.0
        s = df.sort_values("exit_time")["pnl_usd"].cumsum() + 100_000.0
        peak = s.cummax()
        dd = (peak - s).max()
        dd_pct = 100.0 * dd / 100_000.0
        gp = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
        gl = abs(df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum())
        pf = gp / gl if gl > 1e-9 else 999.0
        return float(df["pnl_usd"].sum()), float(pf), float(dd_pct)

    v71_net, v71_pf, v71_dd = curve(v71)

    v4_net, v4_pf, v4_dd = curve(v4_df)

    merged = pd.concat(
        [
            v71[["exit_time", "pnl_usd"]].assign(src="v71"),
            v4_df[["exit_time", "pnl_usd"]].assign(src="v4") if not v4_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    comb_net, comb_pf, comb_dd = curve(merged[["exit_time", "pnl_usd"]]) if not merged.empty else (0.0, 0.0, 0.0)

    return {
        "v71_trades": len(v71),
        "v71_net_pnl_usd": v71_net,
        "v71_pf": v71_pf,
        "v71_max_dd_pct": v71_dd,
        "v4_trades": len(v4_df),
        "v4_net_pnl_usd": v4_net,
        "v4_pf": v4_pf,
        "v4_max_dd_pct": v4_dd,
        "combined_trades": len(v71) + len(v4_df),
        "combined_net_pnl_usd": comb_net,
        "combined_pf": comb_pf,
        "combined_max_dd_pct": comb_dd,
    }


def write_summary(
    path: Path,
    trades: list[dict],
    st: dict[str, Any],
    combo: dict[str, Any],
) -> None:
    yd, qd = yearly_quarterly(trades)
    lines = [
        "Spike Fade V4 — bar-by-bar backtest summary",
        "",
        f"Total trades: {st.get('n', 0)}",
        f"Win rate: {100*st.get('win_rate', 0):.2f}%",
        f"Profit factor: {st.get('profit_factor', 0):.4f}",
        f"Net P&L ($): {st.get('net_pnl_usd', 0):,.2f}",
        f"Max drawdown ($): {st.get('max_drawdown_usd', 0):,.2f}",
        f"Max drawdown (%): {st.get('max_drawdown_pct', 0):.2f}",
        f"Average winner (pips): {st.get('avg_win_pips', 0):.2f}",
        f"Average loser (pips): {st.get('avg_loss_pips', 0):.2f}",
        f"Average winner ($): {st.get('avg_win_usd', 0):.2f}",
        f"Average loser ($): {st.get('avg_loss_usd', 0):.2f}",
        f"Avg hold time (minutes): {st.get('avg_hold_min', 0):.2f}",
        f"Trades per year: {st.get('trades_per_year', 0):.2f}",
        "",
        "Exit reason breakdown:",
    ]
    for k, v in sorted(st.get("exit_reasons", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")
    lines.extend(["", "Yearly P&L ($):"])
    for k, v in sorted(yd.items()):
        lines.append(f"  {k}: {v:,.2f}")
    lines.extend(["", "Quarterly P&L ($):"])
    for k, v in sorted(qd.items()):
        lines.append(f"  {k}: {v:,.2f}")
    lines.extend(["", "=== Combined with V7.1 enriched log ===", ""])
    if "error" in combo:
        lines.append(combo["error"])
    else:
        lines.extend(
            [
                f"V7.1 alone: trades={combo['v71_trades']} PF={combo['v71_pf']:.4f} net=${combo['v71_net_pnl_usd']:,.2f} maxDD%≈{combo['v71_max_dd_pct']:.2f}",
                f"V4 alone:   trades={combo['v4_trades']} PF={combo['v4_pf']:.4f} net=${combo['v4_net_pnl_usd']:,.2f} maxDD%≈{combo['v4_max_dd_pct']:.2f}",
                f"Combined (merged exit times): trades={combo['combined_trades']} PF={combo['combined_pf']:.4f} net=${combo['combined_net_pnl_usd']:,.2f} maxDD%≈{combo['combined_max_dd_pct']:.2f}",
                "",
                "Note: combined PF/DD from sorted exit_time merge is approximate vs true bar-by-bar overlap.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=M1_DEFAULT)
    ap.add_argument("--max-bars", type=int, default=-1, help="Debug trim")
    ap.add_argument("--v71-log", type=Path, default=V71_LOG)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades, st = run_backtest(args.csv, max_bars=args.max_bars)
    fd = st.get("funnel", {})
    if isinstance(fd, dict):
        fd["trades_closed"] = len(trades)

    pd.DataFrame(trades).to_csv(TRADES_CSV, index=False)
    combo = combine_with_v71(trades, args.v71_log)
    write_summary(SUMMARY_TXT, trades, st, combo)

    dbg_parts = [
        "SPIKE FADE V4 — BAR-BY-BAR BACKTEST (current run)",
        "═══════════════════════════════════════════════",
        f"Trades: {st.get('n', 0)} | WR: {100*st.get('win_rate', 0):.1f}% | "
        f"PF: {st.get('profit_factor', 0):.3f} | net: ${st.get('net_pnl_usd', 0):,.2f}",
        "",
        format_funnel_report(fd if isinstance(fd, dict) else {}),
    ]

    if STANDALONE_TRADES.is_file() and args.max_bars <= 0:
        sa = pd.read_csv(STANDALONE_TRADES)
        sa["entry_time"] = pd.to_datetime(sa["entry_time"], utc=True)
        bt_et = set(pd.to_datetime([t["entry_time"] for t in trades], utc=True)) if trades else set()
        miss = sa[~sa["entry_time"].isin(bt_et)].head(5)
        dfg = pd.read_csv(args.csv)
        dfg["time"] = pd.to_datetime(dfg["time"], utc=True)
        m5a, m15a = aggregate_m5_m15_from_df(dfg)
        dbg_parts.extend(["", trace_missing_standalone_rows(dfg, m5a, m15a, miss)])

    if trades:
        dfg2 = pd.read_csv(args.csv)
        dfg2["time"] = pd.to_datetime(dfg2["time"], utc=True)
        if args.max_bars > 0:
            dfg2 = dfg2.iloc[: args.max_bars].copy()
        dbg_parts.extend(["", trace_prove_it_trades(dfg2, pd.DataFrame(trades), 5)])

    DEBUG_REPORT.write_text("\n".join(dbg_parts), encoding="utf-8")

    print(f"Wrote {TRADES_CSV} ({len(trades)} trades)")
    print(f"Wrote {SUMMARY_TXT}")
    print(f"Wrote {DEBUG_REPORT}")
    print()
    print(format_funnel_report(fd if isinstance(fd, dict) else {}))
    print()
    print(
        f"V4: trades={st.get('n')} WR={100*st.get('win_rate', 0):.1f}% PF={st.get('profit_factor', 0):.3f} "
        f"net=${st.get('net_pnl_usd', 0):,.2f}"
    )
    if "error" not in combo:
        print(
            f"Combined: trades={combo['combined_trades']} PF={combo['combined_pf']:.3f} net=${combo['combined_net_pnl_usd']:,.2f}"
        )


if __name__ == "__main__":
    main()
