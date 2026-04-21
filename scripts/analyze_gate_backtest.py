#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "gate_backtest_wakes.csv"
DEFAULT_JSON = ROOT / "gate_backtest_analysis.json"
DEFAULT_TXT = ROOT / "gate_backtest_analysis.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze gate_backtest_wakes.csv and emit the highest-signal cuts fast.")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to gate_backtest_wakes.csv")
    p.add_argument("--output-json", default=str(DEFAULT_JSON), help="Path to write structured analysis JSON")
    p.add_argument("--output-txt", default=str(DEFAULT_TXT), help="Path to write human-readable summary")
    return p.parse_args()


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def _fmt_pct(v: Any) -> str:
    return f"{float(v):.1%}" if isinstance(v, (int, float)) and math.isfinite(float(v)) else "n/a"


def _fmt_num(v: Any, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}" if isinstance(v, (int, float)) and math.isfinite(float(v)) else "n/a"


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for col in [
        "strategy_a_filled",
        "false_breakout",
        "expansion_followed",
        "retest_of_breakout_level",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    return df


def _family_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "trigger_family" not in df.columns:
        return []
    counts = df["trigger_family"].fillna("unknown").value_counts(dropna=False)
    total = int(counts.sum())
    return [
        {"trigger_family": str(fam), "count": int(count), "pct": float(count / total) if total else None}
        for fam, count in counts.items()
    ]


def _session_family_pivot(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if df.empty or "session" not in df.columns or "trigger_family" not in df.columns:
        return {}
    pivot = pd.pivot_table(
        df,
        index="session",
        columns="trigger_family",
        values="timestamp_utc" if "timestamp_utc" in df.columns else df.columns[0],
        aggfunc="count",
        fill_value=0,
    )
    out: dict[str, dict[str, int]] = {}
    for session, row in pivot.iterrows():
        out[str(session)] = {str(col): int(row[col]) for col in pivot.columns}
    return out


def _strategy_a_by_family(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "trigger_family" not in df.columns:
        return []
    out: list[dict[str, Any]] = []
    for family, g in df.groupby("trigger_family", dropna=False):
        filled = g.loc[g.get("strategy_a_filled") == True].copy()
        wins = int((filled.get("strategy_a_outcome") == "win").sum()) if "strategy_a_outcome" in filled.columns else 0
        out.append(
            {
                "trigger_family": str(family),
                "trades": int(len(filled)),
                "win_rate": _rate(wins, int(len(filled))),
                "avg_pnl_pips": float(pd.to_numeric(filled.get("strategy_a_pnl_pips"), errors="coerce").dropna().mean()) if "strategy_a_pnl_pips" in filled.columns and not filled.empty else None,
                "total_pnl_pips": float(pd.to_numeric(filled.get("strategy_a_pnl_pips"), errors="coerce").dropna().sum()) if "strategy_a_pnl_pips" in filled.columns and not filled.empty else None,
            }
        )
    out.sort(key=lambda r: (r["win_rate"] is None, -(r["win_rate"] or -1)))
    return out


def _strategy_a_by_family_session(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "trigger_family" not in df.columns or "session" not in df.columns:
        return []
    out: list[dict[str, Any]] = []
    for (family, session), g in df.groupby(["trigger_family", "session"], dropna=False):
        filled = g.loc[g.get("strategy_a_filled") == True].copy()
        wins = int((filled.get("strategy_a_outcome") == "win").sum()) if "strategy_a_outcome" in filled.columns else 0
        out.append(
            {
                "trigger_family": str(family),
                "session": str(session),
                "trades": int(len(filled)),
                "win_rate": _rate(wins, int(len(filled))),
                "avg_pnl_pips": float(pd.to_numeric(filled.get("strategy_a_pnl_pips"), errors="coerce").dropna().mean()) if "strategy_a_pnl_pips" in filled.columns and not filled.empty else None,
            }
        )
    out.sort(key=lambda r: (r["trigger_family"], r["session"]))
    return out


def _mfe15_by_family(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "trigger_family" not in df.columns or "mfe_15m_pips" not in df.columns:
        return []
    out: list[dict[str, Any]] = []
    for family, g in df.groupby("trigger_family", dropna=False):
        vals = pd.to_numeric(g["mfe_15m_pips"], errors="coerce").dropna()
        out.append(
            {
                "trigger_family": str(family),
                "count": int(len(vals)),
                "mean": float(vals.mean()) if not vals.empty else None,
                "median": float(vals.median()) if not vals.empty else None,
                "p25": float(vals.quantile(0.25)) if not vals.empty else None,
            }
        )
    out.sort(key=lambda r: (r["mean"] is None, -(r["mean"] or -1)))
    return out


def _gap_analysis(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "timestamp_utc" not in df.columns:
        return {}
    ordered = df.sort_values("timestamp_utc").copy()
    gaps = ordered["timestamp_utc"].diff().dt.total_seconds() / 60.0
    desc = gaps.describe().to_dict()
    return {
        "summary_minutes": {k: (float(v) if pd.notna(v) else None) for k, v in desc.items()},
        "wakes_within_5m_of_previous": int((gaps <= 5).sum()),
        "wakes_within_15m_of_previous": int((gaps <= 15).sum()),
        "pct_within_5m": _rate(int((gaps <= 5).sum()), int(gaps.notna().sum())),
        "pct_within_15m": _rate(int((gaps <= 15).sum()), int(gaps.notna().sum())),
    }


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "five_key_cuts": {
            "trigger_family_counts_and_percentages": _family_counts(df),
            "session_x_family_wake_counts": _session_family_pivot(df),
            "strategy_a_win_rate_by_family": _strategy_a_by_family(df),
            "strategy_a_win_rate_by_family_x_session": _strategy_a_by_family_session(df),
            "mfe_15m_distribution_by_family": _mfe15_by_family(df),
        },
        "clustering_diagnostics": _gap_analysis(df),
    }


def render_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Gate Backtest Analysis")
    lines.append("======================")
    lines.append(f"Rows: {result.get('rows', 0):,}")
    lines.append("")

    cuts = result.get("five_key_cuts") or {}

    lines.append("1. Trigger Family Counts And Percentages")
    for row in cuts.get("trigger_family_counts_and_percentages") or []:
        lines.append(f"- {row['trigger_family']}: {row['count']:,} ({_fmt_pct(row.get('pct'))})")
    lines.append("")

    lines.append("2. Session x Family Wake Counts")
    pivot = cuts.get("session_x_family_wake_counts") or {}
    for session, row in pivot.items():
        joined = ", ".join(f"{fam}={count:,}" for fam, count in sorted(row.items()))
        lines.append(f"- {session}: {joined}")
    lines.append("")

    lines.append("3. Strategy A Win Rate By Family")
    for row in cuts.get("strategy_a_win_rate_by_family") or []:
        lines.append(
            f"- {row['trigger_family']}: trades={row['trades']:,} "
            f"win_rate={_fmt_pct(row.get('win_rate'))} "
            f"avg_pnl={_fmt_num(row.get('avg_pnl_pips'))}p "
            f"total_pnl={_fmt_num(row.get('total_pnl_pips'))}p"
        )
    lines.append("")

    lines.append("4. Strategy A Win Rate By Family x Session")
    for row in cuts.get("strategy_a_win_rate_by_family_x_session") or []:
        lines.append(
            f"- {row['trigger_family']} @ {row['session']}: trades={row['trades']:,} "
            f"win_rate={_fmt_pct(row.get('win_rate'))} avg_pnl={_fmt_num(row.get('avg_pnl_pips'))}p"
        )
    lines.append("")

    lines.append("5. MFE-15m Distribution By Family")
    for row in cuts.get("mfe_15m_distribution_by_family") or []:
        lines.append(
            f"- {row['trigger_family']}: count={row['count']:,} "
            f"mean={_fmt_num(row.get('mean'))}p median={_fmt_num(row.get('median'))}p "
            f"p25={_fmt_num(row.get('p25'))}p"
        )
    lines.append("")

    lines.append("Clustering Diagnostics")
    cluster = result.get("clustering_diagnostics") or {}
    if cluster:
        lines.append(
            f"- Wakes within 5 min of previous: {cluster.get('wakes_within_5m_of_previous', 0):,} "
            f"({_fmt_pct(cluster.get('pct_within_5m'))})"
        )
        lines.append(
            f"- Wakes within 15 min of previous: {cluster.get('wakes_within_15m_of_previous', 0):,} "
            f"({_fmt_pct(cluster.get('pct_within_15m'))})"
        )
        desc = cluster.get("summary_minutes") or {}
        lines.append(
            f"- Gap minutes: mean={_fmt_num(desc.get('mean'))} "
            f"median={_fmt_num(desc.get('50%'))} p25={_fmt_num(desc.get('25%'))} p75={_fmt_num(desc.get('75%'))}"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = _load(csv_path)
    result = analyze(df)
    text = render_text(result)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.output_txt).write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[write] {args.output_json}")
    print(f"[write] {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
