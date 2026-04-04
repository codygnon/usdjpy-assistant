#!/usr/bin/env python3
"""Compare baseline vs wide-stop Swing-Macro summary.json outputs side by side."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DIRS = {
    "Baseline (real)": ROOT / "research_out/swing_macro_real",
    "Baseline (zero)": ROOT / "research_out/swing_macro_zero",
    "Wide (real)": ROOT / "research_out/swing_macro_wide_real",
    "Wide (zero)": ROOT / "research_out/swing_macro_wide_zero",
}


def _load_summary(dir_path: Path) -> dict[str, Any] | None:
    p = dir_path / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _exit_mix(summary: dict[str, Any] | None) -> dict[str, float]:
    if not summary:
        return {}
    reasons = summary.get("exit_reasons")
    if not isinstance(reasons, dict):
        return {}
    tc = float(summary.get("trade_count") or 0)
    if tc <= 0:
        return {}
    out: dict[str, float] = {}
    for k, v in reasons.items():
        try:
            out[str(k)] = 100.0 * float(v) / tc
        except (TypeError, ValueError):
            continue
    return out


def _fmt_num(x: Any, *, pct: bool = False, money: bool = False) -> str:
    if x is None:
        return "N/A"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "N/A"
    if money:
        return f"${f:,.0f}"
    if pct:
        return f"{f:.1f}%"
    return f"{f:.2f}" if abs(f) < 1000 else f"{f:,.0f}"


def _wl_ratio(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "N/A"
    aw = float(summary.get("avg_win") or 0.0)
    al = float(summary.get("avg_loss") or 0.0)
    if al == 0:
        return "N/A"
    return f"{abs(aw / al):.2f}"


def main() -> None:
    cols = list(DIRS.keys())
    data = {c: _load_summary(DIRS[c]) for c in cols}

    def cell(label: str, key: str, *, pct: bool = False, money: bool = False) -> list[str]:
        return [_fmt_num((data[c] or {}).get(key), pct=pct, money=money) for c in cols]

    print()
    print(
        f"{'Metric':<28}"
        + "".join(f"{c:>18}" for c in cols)
    )
    print("─" * (28 + 18 * len(cols)))

    trades = cell("trade_count", "trade_count")
    print(f"{'Total Trades':<28}" + "".join(f"{t:>18}" for t in trades))

    wr = cell("win_rate", "win_rate", pct=True)
    print(f"{'Win Rate':<28}" + "".join(f"{w:>18}" for w in wr))

    pf = cell("profit_factor", "profit_factor")
    print(f"{'Profit Factor':<28}" + "".join(f"{p:>18}" for p in pf))

    pnl = cell("net_pnl", "net_pnl", money=True)
    print(f"{'Net PnL':<28}" + "".join(f"{p:>18}" for p in pnl))

    dd = cell("max_drawdown_pct", "max_drawdown_pct", pct=True)
    print(f"{'Max Drawdown':<28}" + "".join(f"{d:>18}" for d in dd))

    aw = cell("avg_win", "avg_win", money=True)
    print(f"{'Avg Win':<28}" + "".join(f"{a:>18}" for a in aw))

    al = cell("avg_loss", "avg_loss", money=True)
    print(f"{'Avg Loss':<28}" + "".join(f"{a:>18}" for a in al))

    wl = [_wl_ratio(data[c]) for c in cols]
    print(f"{'W/L Ratio (|avg|)':<28}" + "".join(f"{w:>18}" for w in wl))

    stop_pct = []
    for c in cols:
        mix = _exit_mix(data[c])
        stop_pct.append(_fmt_num(mix.get("stop_loss", 0.0), pct=True))
    print(f"{'Stop-out rate':<28}" + "".join(f"{s:>18}" for s in stop_pct))

    tr_pct = []
    for c in cols:
        mix = _exit_mix(data[c])
        tr_pct.append(_fmt_num(mix.get("trend_reversal", 0.0), pct=True))
    print(f"{'Trend reversal exit %':<28}" + "".join(f"{t:>18}" for t in tr_pct))

    mf_pct = []
    for c in cols:
        mix = _exit_mix(data[c])
        mf_pct.append(_fmt_num(mix.get("macro_flip", 0.0), pct=True))
    print(f"{'Macro flip exit %':<28}" + "".join(f"{m:>18}" for m in mf_pct))

    print()
    wide_zero = data["Wide (zero)"] or {}
    pf_zero = float(wide_zero.get("profit_factor") or 0.0)
    tc = int(wide_zero.get("trade_count") or 0)
    dd_w = float(wide_zero.get("max_drawdown_pct") or 0.0)
    mix_w = _exit_mix(wide_zero)
    stop_w = mix_w.get("stop_loss", 999.0) if mix_w else 999.0

    pf_pass = pf_zero >= 1.30
    tc_pass = tc >= 100
    dd_pass = dd_w <= 20.0
    so_pass = stop_w < 50.0

    print("DECISION CRITERIA (wide zero-spread run):")
    print(f"  PF(zero) ≥ 1.30?        {'YES' if pf_pass else 'NO'}  (PF={pf_zero:.3f})")
    print(f"  Trades ≥ 100?           {'YES' if tc_pass else 'NO'}  (n={tc})")
    print(f"  Max DD ≤ 20%?           {'YES' if dd_pass else 'NO'}  (DD={dd_w:.1f}%)")
    print(f"  Stop-out rate < 50%?    {'YES' if so_pass else 'NO'}  ({stop_w:.1f}%)")
    print()
    if pf_pass:
        verdict = "PASS — proceed to robustness grid (see run_swing_macro_grid.py)"
    else:
        verdict = "FAIL — archive strategy, move to Path 2"
    print(f"  VERDICT: {verdict}")
    print()


if __name__ == "__main__":
    main()
