#!/usr/bin/env python3
"""Compare Swing-Macro (zero) vs Daily Trend real/zero summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DIRS = {
    "Swing-Macro (zero)": ROOT / "research_out/swing_macro_zero",
    "DailyTrend (real)": ROOT / "research_out/daily_trend_real",
    "DailyTrend (zero)": ROOT / "research_out/daily_trend_zero",
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


def _fmt(x: Any, *, pct: bool = False, money: bool = False) -> str:
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


def _wl(summary: dict[str, Any] | None) -> str:
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

    def cell(key: str, *, pct: bool = False, money: bool = False) -> list[str]:
        return [_fmt((data[c] or {}).get(key), pct=pct, money=money) for c in cols]

    print()
    print("DAILY TREND vs SWING-MACRO COMPARISON")
    print("═" * 70)
    print()
    print(f"{'Metric':<28}" + "".join(f"{c:>22}" for c in cols))
    print("─" * (28 + 22 * len(cols)))

    tc = cell("trade_count")
    tc_fmt = [t.replace(".00", "") if isinstance(t, str) and t.endswith(".00") else t for t in tc]
    print(f"{'Total Trades':<28}" + "".join(f"{t:>22}" for t in tc_fmt))

    wr = cell("win_rate", pct=True)
    print(f"{'Win Rate':<28}" + "".join(f"{w:>22}" for w in wr))

    pf = cell("profit_factor")
    print(f"{'Profit Factor':<28}" + "".join(f"{p:>22}" for p in pf))

    pnl = cell("net_pnl", money=True)
    print(f"{'Net PnL':<28}" + "".join(f"{p:>22}" for p in pnl))

    dd = cell("max_drawdown_pct", pct=True)
    print(f"{'Max Drawdown':<28}" + "".join(f"{d:>22}" for d in dd))

    aw = cell("avg_win", money=True)
    print(f"{'Avg Win':<28}" + "".join(f"{a:>22}" for a in aw))

    al = cell("avg_loss", money=True)
    print(f"{'Avg Loss':<28}" + "".join(f"{a:>22}" for a in al))

    wl = [_wl(data[c]) for c in cols]
    print(f"{'W/L Ratio (|avg|)':<28}" + "".join(f"{w:>22}" for w in wl))

    stop_pct = [_fmt(_exit_mix(data[c]).get("stop_loss", 0.0), pct=True) for c in cols]
    print(f"{'Stop-out rate':<28}" + "".join(f"{s:>22}" for s in stop_pct))

    tr_pct = [_fmt(_exit_mix(data[c]).get("trend_reversal", 0.0), pct=True) for c in cols]
    print(f"{'Trend Reversal Exit %':<28}" + "".join(f"{t:>22}" for t in tr_pct))

    ah = [_fmt((data[c] or {}).get("avg_bars_held")) for c in cols]
    print(f"{'Avg Hold (bars)':<28}" + "".join(f"{a:>22}" for a in ah))

    asp = [_fmt((data[c] or {}).get("avg_initial_stop_pips")) for c in cols]
    print(f"{'Avg Stop Width (pips)':<28}" + "".join(f"{a:>22}" for a in asp))

    print()
    print("STRUCTURAL (indicative; cost drag uses 2.0 pip real spread on DailyTrend real):")
    smz = data["Swing-Macro (zero)"] or {}
    dtz = data["DailyTrend (zero)"] or {}
    dtr = data["DailyTrend (real)"] or {}
    print(
        f"  Avg stop width (pips):  Swing-Macro ~{_fmt(smz.get('avg_initial_stop_pips'))} "
        f"vs DailyTrend ~{_fmt(dtz.get('avg_initial_stop_pips'))}"
    )
    print("  (Cost as % of stop: approximate; see run methodology for spread model.)")
    print()

    dz = data["DailyTrend (zero)"] or {}
    pf0 = float(dz.get("profit_factor") or 0.0)
    n0 = int(float(dz.get("trade_count") or 0))
    dd0 = float(dz.get("max_drawdown_pct") or 0.0)
    pf_ok = pf0 >= 1.30
    n_ok = n0 >= 80
    dd_ok = dd0 <= 20.0
    all_ok = pf_ok and n_ok and dd_ok

    print("DECISION CRITERIA (daily trend zero-spread run):")
    print(f"  PF(zero) ≥ 1.30?          {'YES' if pf_ok else 'NO'}  (PF={pf0:.3f})")
    print(f"  Trades ≥ 80?              {'YES' if n_ok else 'NO'}  (n={n0})")
    print(f"  Max DD ≤ 20%?             {'YES' if dd_ok else 'NO'}  (DD={dd0:.1f}%)")
    print()
    if all_ok:
        verdict = "PASS — proceed to walk-forward validation"
    else:
        verdict = "FAIL — archive forever, move to Path 2"
    print(f"  VERDICT: {verdict}")
    print()


if __name__ == "__main__":
    main()
