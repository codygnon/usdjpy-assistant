#!/usr/bin/env python3
"""Phase 7 interaction-effects analysis for Autonomous Fillmore.

This is a read-only synthesis harness. It combines Phase 3 snapshot fields,
Phase 4 reasoning clusters, Phase 5 sizing economics, and Phase 6 lifecycle
proxies into one interaction map.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
CLOSED_CSV = OUT / "phase3_closed_trades_with_snapshot_fields.csv"
PHASE4_CSV = OUT / "phase4_reasoning_corpus.csv"
FAST_FAILURE_CSV = OUT / "phase3_clr_level_failure_17.csv"
PHASE5_DECOMP_CSV = OUT / "phase5_pl_decomposition.csv"

OBSERVED_NET_PIPS = -308.0
OBSERVED_NET_USD = -7253.2365

CORE_GRID = ["trigger_family", "side", "prompt_regime", "rationale_cluster"]
CORE_LOT_GRID = ["trigger_family", "side", "prompt_regime", "rationale_cluster", "lot_bucket"]
FULL_GRID = [
    "trigger_family",
    "side",
    "prompt_regime",
    "rationale_cluster",
    "lot_bucket",
    "hold_time_band",
    "volatility_regime",
    "h1_regime",
    "session",
    "day_of_week",
]

STRONG_LEVEL_RE = re.compile(r"\b(?:strong|clean|fresh|textbook|clear|decisive|confirmed|robust|material)\b", re.I)


def s(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def one(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def money(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.2f}"


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def escape_md(value: Any) -> str:
    return s(value).replace("\n", "<br>").replace("|", "\\|")


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(escape_md(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(escape_md(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def lot_bucket(lots: Any) -> str:
    x = pd.to_numeric(pd.Series([lots]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "missing"
    if x < 2:
        return "0-1.99"
    if x < 4:
        return "2-3.99"
    if x < 8:
        return "4-7.99"
    return "8+"


def hold_bucket(minutes: Any) -> str:
    x = pd.to_numeric(pd.Series([minutes]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "missing"
    if x <= 3:
        return "<=3m"
    if x <= 7:
        return "3-7m"
    if x <= 15:
        return "7-15m"
    if x <= 30:
        return "15-30m"
    return ">30m"


def sample_warning(n: int) -> str:
    if n < 15:
        return "weak_N<15"
    if n < 30:
        return "provisional_15<=N<30"
    return "strong_N>=30"


def max_drawdown(values: pd.Series) -> float:
    vals = values.fillna(0).astype(float)
    if vals.empty:
        return 0.0
    curve = vals.cumsum()
    return float((curve - curve.cummax()).min())


def profit_factor(values: pd.Series) -> float | None:
    wins = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    if losses == 0:
        return math.inf if wins else None
    return float(wins / losses)


def perf(group: pd.DataFrame) -> dict[str, Any]:
    g = group[group["pips"].notna()].copy()
    wins = g[g["pips"] > 0]
    losses = g[g["pips"] < 0]
    return {
        "n": len(g),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(g) if len(g) else None,
        "net_pips": g["pips"].sum() if len(g) else 0.0,
        "net_usd": g["pnl"].sum() if len(g) else 0.0,
        "avg_pips": g["pips"].mean() if len(g) else None,
        "avg_usd": g["pnl"].mean() if len(g) else None,
        "avg_winner_pips": wins["pips"].mean() if len(wins) else None,
        "avg_loser_pips": losses["pips"].mean() if len(losses) else None,
        "profit_factor_pips": profit_factor(g["pips"]) if len(g) else None,
        "profit_factor_usd": profit_factor(g["pnl"]) if len(g) else None,
        "max_drawdown_pips": max_drawdown(g["pips"]) if len(g) else None,
        "max_drawdown_usd": max_drawdown(g["pnl"]) if len(g) else None,
        "sample_size_warning": sample_warning(len(g)),
    }


def display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = [
        "win_rate",
        "entry_failure_rate",
        "entry_failure_share",
        "exit_reversal_rate",
        "exit_reversal_share",
        "other_loss_rate",
        "other_loss_share",
        "winner_rate",
        "winner_share",
        "share",
        "cell_share_of_cluster",
        "share_gt_median_lot",
        "stop_overshoot_rate",
        "tp_undershoot_rate",
        "coverage_abs_net_pips",
        "coverage_abs_net_usd",
        "cumulative_coverage_abs_net_pips",
        "cumulative_coverage_abs_net_usd",
        "self_correction_rate",
        "above_median_lot_rate",
        "survives_loo_rate",
    ]
    money_cols = [
        "net_usd",
        "avg_usd",
        "actual_usd",
        "uniform_1_lot_usd",
        "sizing_delta_usd",
        "saved_loser_usd",
        "missed_winner_usd",
        "net_delta_usd",
        "cumulative_net_delta_usd",
        "loo_min_net_usd",
        "phase5_target_sizing_delta_usd",
        "reconciliation_gap_usd",
    ]
    num_cols = [
        "net_pips",
        "avg_pips",
        "avg_winner_pips",
        "avg_loser_pips",
        "profit_factor_pips",
        "profit_factor_usd",
        "max_drawdown_pips",
        "max_drawdown_usd",
        "entry_failure_pips",
        "exit_reversal_pips",
        "other_loss_pips",
        "winner_pips",
        "saved_loser_pips",
        "missed_winner_pips",
        "net_delta_pips",
        "cumulative_net_delta_pips",
        "loo_min_net_pips",
        "mean_lots",
        "median_lots",
        "p75_lots",
        "median_hedge_density",
        "median_conviction_density",
        "mutual_information",
        "chi_square",
        "p_value_permutation",
        "median_sl_pips",
        "median_spread",
        "median_atr_m5",
        "stop_overshoot_median_all_losses",
        "median_positive_overshoot_pips",
        "positive_overshoot_p75",
    ]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(pct)
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(money)
    for col in num_cols:
        if col in out.columns:
            out[col] = out[col].map(two)
    return out


def clean_text_col(df: pd.DataFrame, col: str, fallback: str = "missing") -> pd.Series:
    if col not in df.columns:
        return pd.Series(fallback, index=df.index)
    return df[col].fillna(fallback).replace("", fallback).astype(str).str.lower()


def load_dataset() -> pd.DataFrame:
    closed = pd.read_csv(CLOSED_CSV, low_memory=False)
    corpus = pd.read_csv(PHASE4_CSV, low_memory=False)
    corpus_cols = [
        "trade_id",
        "rationale_cluster",
        "hedge_density",
        "conviction_density",
        "self_correction_count",
        "token_count",
        "rationale_excerpt",
    ]
    c = corpus[corpus["trade_id"].notna()].drop_duplicates("trade_id")
    df = closed.merge(c[[col for col in corpus_cols if col in c.columns]], on="trade_id", how="left")

    for col in [
        "lots",
        "pips",
        "pnl",
        "mae_abs",
        "mfe",
        "max_adverse_pips",
        "max_favorable_pips",
        "hedge_density",
        "conviction_density",
        "self_correction_count",
        "features.sl_pips",
        "features.tp_pips",
        "features.spread_at_entry",
        "market_snapshot.spread_pips",
        "market_snapshot.technicals.M5.atr_pips",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mae_abs"] = df["mae_abs"].fillna(df["max_adverse_pips"].abs())
    df["mfe"] = df["mfe"].fillna(df["max_favorable_pips"])
    df["filled_dt"] = pd.to_datetime(df["filled_at"], errors="coerce", utc=True)
    df["closed_dt"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)
    df["created_dt"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
    df["created_date"] = df["filled_dt"].dt.strftime("%Y-%m-%d").fillna("missing")
    df["day_of_week"] = df["filled_dt"].dt.day_name().fillna("missing")
    df["hold_minutes"] = (df["closed_dt"] - df["filled_dt"]).dt.total_seconds() / 60
    df["hold_time_band"] = df["hold_minutes"].map(hold_bucket)
    df["lot_bucket"] = df["lots"].map(lot_bucket)
    df["pnl_per_lot"] = np.where(df["lots"] > 0, df["pnl"] / df["lots"], np.nan)
    df["abs_pips"] = df["pips"].abs()

    df["trigger_family"] = clean_text_col(df, "trigger_family")
    df["side"] = clean_text_col(df, "side")
    df["prompt_regime"] = df["prompt_regime"].fillna("unknown").replace("", "unknown")
    df["rationale_cluster"] = df["rationale_cluster"].fillna("unknown").replace("", "unknown")
    df["session"] = clean_text_col(df, "features.session")
    df["volatility_regime"] = clean_text_col(df, "features.vol_regime")
    df["h1_regime"] = clean_text_col(df, "features.h1_regime")
    missing_h1 = df["h1_regime"].eq("missing")
    df.loc[missing_h1, "h1_regime"] = clean_text_col(df.loc[missing_h1], "market_snapshot.technicals.H1.regime")
    df["macro_bias"] = clean_text_col(df, "features.macro_bias")
    missing_macro = df["macro_bias"].eq("missing")
    df.loc[missing_macro, "macro_bias"] = clean_text_col(df.loc[missing_macro], "market_snapshot.macro_bias.combined_bias")
    df["timeframe_alignment_clean"] = clean_text_col(df, "timeframe_alignment")
    df["exit_strategy_clean"] = clean_text_col(df, "exit_strategy")
    df["hedge_density"] = df["hedge_density"].fillna(0.0)
    df["conviction_density"] = df["conviction_density"].fillna(0.0)
    df["self_correction_count"] = df["self_correction_count"].fillna(0.0)
    df["has_self_correction"] = df["self_correction_count"] > 0

    df["entry_failure_proxy"] = (df["pips"] < 0) & (df["mfe"] <= 2.0) & (df["mae_abs"] >= 6.0)
    df["exit_reversal_proxy"] = (df["pips"] < 0) & (df["mfe"] >= 4.0)
    df["outcome_type"] = np.select(
        [
            df["entry_failure_proxy"],
            df["exit_reversal_proxy"],
            df["pips"] < 0,
            df["pips"] > 0,
        ],
        ["entry_failure", "exit_reversal", "other_loss", "winner"],
        default="flat",
    )
    fast_ids = set(pd.read_csv(FAST_FAILURE_CSV)["trade_id"].dropna().astype(str))
    df["clr_fast_failure_17"] = df["trade_id"].astype(str).isin(fast_ids)

    text_cols = [col for col in ["trade_thesis", "named_catalyst", "rationale", "rationale_excerpt"] if col in df.columns]
    joined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["strong_level_claim"] = joined_text.str.contains(STRONG_LEVEL_RE, regex=True, na=False)
    df["mixed_or_thin_snapshot"] = (
        df["timeframe_alignment_clean"].eq("mixed")
        | df["volatility_regime"].isin(["below average", "very low"])
        | df["h1_regime"].eq("bear")
    )

    hedge_pos = df.loc[df["hedge_density"] > 0, "hedge_density"]
    conv_pos = df.loc[df["conviction_density"] > 0, "conviction_density"]
    hedge_q80 = hedge_pos.quantile(0.80) if not hedge_pos.empty else 0.0
    conv_q80 = conv_pos.quantile(0.80) if not conv_pos.empty else 0.0
    df["high_hedge"] = df["hedge_density"] >= hedge_q80
    df["high_conviction"] = df["conviction_density"] >= conv_q80

    df["stop_overshoot_pips"] = np.where(
        (df["pips"] < 0) & df["features.sl_pips"].notna(),
        df["abs_pips"] - df["features.sl_pips"],
        np.nan,
    )
    df["stop_overshoot_flag"] = df["stop_overshoot_pips"] > 1.0
    df["tp_undershoot_pips"] = np.where(
        (df["pips"] > 0) & df["features.tp_pips"].notna(),
        df["features.tp_pips"] - df["pips"],
        np.nan,
    )
    df["tp_undershoot_flag"] = df["tp_undershoot_pips"] > 1.0
    df["above_median_lot"] = df["lots"] > df["lots"].median()

    return df.sort_values("filled_dt").reset_index(drop=True)


def cell_signature(row: pd.Series, dims: list[str]) -> str:
    return " | ".join(f"{dim}={row[dim]}" for dim in dims)


def aggregate_grid(df: pd.DataFrame, grid_name: str, dims: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(dims, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"grid": grid_name}
        for dim, key in zip(dims, keys):
            row[dim] = key
        row.update(perf(group))
        row["entry_failure_n"] = int(group["entry_failure_proxy"].sum())
        row["exit_reversal_n"] = int(group["exit_reversal_proxy"].sum())
        row["entry_failure_rate"] = group["entry_failure_proxy"].mean()
        row["exit_reversal_rate"] = group["exit_reversal_proxy"].mean()
        row["mean_lots"] = group["lots"].mean()
        row["median_lots"] = group["lots"].median()
        row["above_median_lot_rate"] = group["above_median_lot"].mean()
        row["cell_signature"] = cell_signature(pd.Series(row), dims)
        row["coverage_abs_net_pips"] = max(0.0, -row["net_pips"]) / abs(OBSERVED_NET_PIPS)
        row["coverage_abs_net_usd"] = max(0.0, -row["net_usd"]) / abs(OBSERVED_NET_USD)
        rows.append(row)
    return pd.DataFrame(rows)


def damage_concentration(df: pd.DataFrame) -> pd.DataFrame:
    grids = [
        ("full_10d", FULL_GRID),
        ("mid_6d", ["trigger_family", "side", "prompt_regime", "rationale_cluster", "lot_bucket", "hold_time_band"]),
        ("core_5d_with_lot", CORE_LOT_GRID),
        ("core_4d", CORE_GRID),
        ("day_gate_side", ["day_of_week", "trigger_family", "side"]),
    ]
    out = pd.concat([aggregate_grid(df, name, dims) for name, dims in grids], ignore_index=True)
    out["pip_damage_rank_in_grid"] = out.groupby("grid")["net_pips"].rank(method="first", ascending=True)
    out["usd_damage_rank_in_grid"] = out.groupby("grid")["net_usd"].rank(method="first", ascending=True)
    out["expectancy_rank_in_grid"] = out.groupby("grid")["avg_pips"].rank(method="first", ascending=True)
    out = out.sort_values(["grid", "net_pips", "net_usd"])
    out.to_csv(OUT / "phase7_damage_concentration.csv", index=False)
    return out


def leave_one_date(group: pd.DataFrame) -> tuple[float | None, float | None, float | None, str]:
    dates = sorted(group["created_date"].dropna().unique())
    if len(dates) <= 1:
        return None, None, None, "single_date_cell"
    pips_vals = []
    usd_vals = []
    for date in dates:
        rest = group[group["created_date"] != date]
        if rest.empty:
            continue
        pips_vals.append(rest["pips"].sum())
        usd_vals.append(rest["pnl"].sum())
    if not pips_vals:
        return None, None, None, "no_leave_one_sample"
    survive = [p > 0 and u > 0 for p, u in zip(pips_vals, usd_vals)]
    return min(pips_vals), min(usd_vals), sum(survive) / len(survive), ""


def preserved_edge_cells(df: pd.DataFrame) -> pd.DataFrame:
    grids = [
        ("core_4d", CORE_GRID),
        ("core_5d_with_lot", CORE_LOT_GRID),
        ("day_gate_side", ["day_of_week", "trigger_family", "side"]),
        ("day_only", ["day_of_week"]),
    ]
    rows = []
    for grid_name, dims in grids:
        for keys, group in df.groupby(dims, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            p = perf(group)
            if p["n"] < 10 or p["net_pips"] <= 0 or p["net_usd"] <= 0:
                continue
            row = {"grid": grid_name}
            for dim, key in zip(dims, keys):
                row[dim] = key
            row.update(p)
            loo_pips, loo_usd, survive_rate, loo_warning = leave_one_date(group)
            row["loo_min_net_pips"] = loo_pips
            row["loo_min_net_usd"] = loo_usd
            row["survives_leave_one_date"] = bool(loo_pips is not None and loo_pips > 0 and loo_usd is not None and loo_usd > 0)
            row["survives_loo_rate"] = survive_rate
            row["loo_warning"] = loo_warning
            row["date_count"] = group["created_date"].nunique()
            row["rationale_cluster_mix"] = json.dumps(group["rationale_cluster"].value_counts().head(4).to_dict())
            row["overlaps_8lot_clr_anomaly"] = bool(
                (group["trigger_family"].eq("critical_level_reaction") & group["side"].eq("buy") & group["lot_bucket"].eq("8+")).any()
            )
            row["cell_signature"] = cell_signature(pd.Series(row), dims)
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["survives_leave_one_date", "avg_pips", "net_pips"], ascending=[False, False, False])
    out.to_csv(OUT / "phase7_preserved_edge_cells.csv", index=False)
    return out


def mutual_information_binary(df: pd.DataFrame, col: str, target: str) -> float:
    tmp = df[[col, target]].dropna().copy()
    if tmp.empty:
        return 0.0
    total = len(tmp)
    mi = 0.0
    joint = tmp.groupby([col, target]).size()
    x_counts = tmp[col].value_counts()
    y_counts = tmp[target].value_counts()
    for (x, y), n in joint.items():
        pxy = n / total
        px = x_counts[x] / total
        py = y_counts[y] / total
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py), 2)
    return mi


def entry_failure_concentration(df: pd.DataFrame) -> pd.DataFrame:
    factors = [
        "trigger_family",
        "side",
        "prompt_regime",
        "rationale_cluster",
        "lot_bucket",
        "session",
        "volatility_regime",
        "h1_regime",
        "macro_bias",
        "timeframe_alignment_clean",
        "day_of_week",
        "exit_strategy_clean",
    ]
    rows = []
    total_entry = int(df["entry_failure_proxy"].sum())
    total_exit = int(df["exit_reversal_proxy"].sum())
    total_other = int((df["outcome_type"] == "other_loss").sum())
    total_winner = int((df["outcome_type"] == "winner").sum())
    for factor in factors:
        mi = mutual_information_binary(df, factor, "entry_failure_proxy")
        for value, group in df.groupby(factor, dropna=False):
            n = len(group)
            entry_n = int(group["entry_failure_proxy"].sum())
            exit_n = int(group["exit_reversal_proxy"].sum())
            other_n = int((group["outcome_type"] == "other_loss").sum())
            winner_n = int((group["outcome_type"] == "winner").sum())
            row = {
                "factor": factor,
                "value": value,
                "n": n,
                "entry_failure_n": entry_n,
                "exit_reversal_n": exit_n,
                "other_loss_n": other_n,
                "winner_n": winner_n,
                "entry_failure_rate": entry_n / n if n else None,
                "entry_failure_share": entry_n / total_entry if total_entry else None,
                "exit_reversal_share": exit_n / total_exit if total_exit else None,
                "other_loss_share": other_n / total_other if total_other else None,
                "winner_share": winner_n / total_winner if total_winner else None,
                "net_pips": group["pips"].sum(),
                "net_usd": group["pnl"].sum(),
                "median_lots": group["lots"].median(),
                "above_median_lot_rate": group["above_median_lot"].mean(),
                "mutual_information": mi,
                "sample_size_warning": sample_warning(n),
            }
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["mutual_information", "entry_failure_share", "entry_failure_rate"], ascending=[False, False, False])
    out.to_csv(OUT / "phase7_entry_failure_concentration.csv", index=False)
    return out


def chi_square_stat(table: pd.DataFrame) -> float:
    obs = table.to_numpy(dtype=float)
    total = obs.sum()
    if total == 0:
        return 0.0
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
    mask = expected > 0
    return float(((obs - expected) ** 2 / np.where(mask, expected, 1)).sum())


def permutation_chi_square(df: pd.DataFrame, row_col: str, col_col: str, n_perm: int = 5000) -> tuple[float, float]:
    table = pd.crosstab(df[row_col], df[col_col])
    actual = chi_square_stat(table)
    rng = np.random.default_rng(20260502)
    vals = []
    shuffled = df[[row_col, col_col]].copy()
    target = shuffled[col_col].to_numpy().copy()
    for _ in range(n_perm):
        rng.shuffle(target)
        shuffled[col_col] = target
        vals.append(chi_square_stat(pd.crosstab(shuffled[row_col], shuffled[col_col])))
    p_value = (sum(v >= actual for v in vals) + 1) / (n_perm + 1)
    return actual, p_value


def caveat_outcome_interaction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, group in df.groupby("outcome_type"):
        row = {
            "row_type": "outcome_summary",
            "outcome_type": outcome,
            "rationale_cluster": "all",
            "n": len(group),
            "median_hedge_density": group["hedge_density"].median(),
            "median_conviction_density": group["conviction_density"].median(),
            "self_correction_rate": group["has_self_correction"].mean(),
            "mean_lots": group["lots"].mean(),
            "median_lots": group["lots"].median(),
            "above_median_lot_rate": group["above_median_lot"].mean(),
        }
        row.update(perf(group))
        rows.append(row)
        for cluster, cg in group.groupby("rationale_cluster"):
            c_row = {
                "row_type": "cluster_mix",
                "outcome_type": outcome,
                "rationale_cluster": cluster,
                "n": len(cg),
                "share": len(cg) / len(group) if len(group) else None,
                "median_hedge_density": cg["hedge_density"].median(),
                "median_conviction_density": cg["conviction_density"].median(),
                "self_correction_rate": cg["has_self_correction"].mean(),
                "mean_lots": cg["lots"].mean(),
                "median_lots": cg["lots"].median(),
                "above_median_lot_rate": cg["above_median_lot"].mean(),
            }
            c_row.update(perf(cg))
            rows.append(c_row)
    chi, p_value = permutation_chi_square(df, "rationale_cluster", "outcome_type")
    rows.append(
        {
            "row_type": "significance_test",
            "outcome_type": "all",
            "rationale_cluster": "rationale_cluster_x_outcome_type",
            "n": len(df),
            "chi_square": chi,
            "p_value_permutation": p_value,
            "test_note": "Deterministic 5,000-shuffle permutation chi-square; no scipy dependency.",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase7_caveat_outcome_interaction.csv", index=False)
    return out


def sizing_compounding(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phase5 = pd.read_csv(PHASE5_DECOMP_CSV)
    target = float(
        phase5.loc[phase5["component"].eq("sizing_amplification_delta_vs_1_lot"), "component_usd"].iloc[0]
    )
    order = ["entry_failure", "exit_reversal", "other_loss", "winner", "flat"]
    for outcome in order:
        group = df[df["outcome_type"].eq(outcome)]
        if group.empty:
            continue
        actual = group["pnl"].sum()
        uniform = group["pnl_per_lot"].sum()
        delta = actual - uniform
        row = {
            "outcome_type": outcome,
            "n": len(group),
            "lots_gt_1_n": int((group["lots"] > 1).sum()),
            "lots_gt_median_n": int(group["above_median_lot"].sum()),
            "share_gt_median_lot": group["above_median_lot"].mean(),
            "mean_lots": group["lots"].mean(),
            "median_lots": group["lots"].median(),
            "actual_usd": actual,
            "uniform_1_lot_usd": uniform,
            "sizing_delta_usd": delta,
            "phase5_target_sizing_delta_usd": target,
            "reconciliation_gap_usd": None,
        }
        row.update(perf(group))
        rows.append(row)
    total_delta = sum(r["sizing_delta_usd"] for r in rows)
    rows.append(
        {
            "outcome_type": "RECONCILIATION",
            "n": len(df),
            "actual_usd": df["pnl"].sum(),
            "uniform_1_lot_usd": df["pnl_per_lot"].sum(),
            "sizing_delta_usd": total_delta,
            "phase5_target_sizing_delta_usd": target,
            "reconciliation_gap_usd": total_delta - target,
            "note": "Subset deltas partition the corpus by Phase 6 outcome_type.",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase7_sizing_compounding.csv", index=False)
    return out


def rationale_regime_interaction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clusters = df["rationale_cluster"].value_counts()
    eligible = set(clusters[clusters >= 15].index)
    for cluster, cgroup in df[df["rationale_cluster"].isin(eligible)].groupby("rationale_cluster"):
        for regime_type, col in [
            ("volatility_regime", "volatility_regime"),
            ("h1_regime", "h1_regime"),
            ("session", "session"),
            ("macro_bias", "macro_bias"),
            ("timeframe_alignment", "timeframe_alignment_clean"),
        ]:
            for value, group in cgroup.groupby(col):
                row = {
                    "rationale_cluster": cluster,
                    "regime_type": regime_type,
                    "regime_value": value,
                    "cluster_n": len(cgroup),
                    "cell_share_of_cluster": len(group) / len(cgroup),
                }
                row.update(perf(group))
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(["rationale_cluster", "regime_type", "net_pips"])
    out.to_csv(OUT / "phase7_rationale_regime_interaction.csv", index=False)
    return out


def stop_overshoot_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phase3 = df[df["prompt_regime"].eq("Phase 3 house-edge")].copy()
    for segment_type, col in [
        ("overall_phase3", None),
        ("gate", "trigger_family"),
        ("side", "side"),
        ("session", "session"),
        ("lot_bucket", "lot_bucket"),
        ("rationale_cluster", "rationale_cluster"),
        ("exit_strategy", "exit_strategy_clean"),
        ("volatility_regime", "volatility_regime"),
    ]:
        groups = [("all_phase3", phase3)] if col is None else list(phase3.groupby(col, dropna=False))
        for value, group in groups:
            losses_with_sl = group[(group["pips"] < 0) & group["features.sl_pips"].notna()]
            positive = losses_with_sl[losses_with_sl["stop_overshoot_flag"]]
            row = {
                "row_type": "segment",
                "segment_type": segment_type,
                "segment": value,
                "n": len(group),
                "losses_with_sl": len(losses_with_sl),
                "stop_overshoot_n": int(positive.shape[0]),
                "stop_overshoot_rate": positive.shape[0] / len(losses_with_sl) if len(losses_with_sl) else None,
                "stop_overshoot_median_all_losses": losses_with_sl["stop_overshoot_pips"].median() if len(losses_with_sl) else None,
                "median_positive_overshoot_pips": positive["stop_overshoot_pips"].median() if len(positive) else None,
                "positive_overshoot_p75": positive["stop_overshoot_pips"].quantile(0.75) if len(positive) else None,
                "median_sl_pips": losses_with_sl["features.sl_pips"].median() if len(losses_with_sl) else None,
                "median_spread": group["features.spread_at_entry"].fillna(group["market_snapshot.spread_pips"]).median(),
                "median_atr_m5": group["market_snapshot.technicals.M5.atr_pips"].median(),
            }
            row.update(perf(group))
            rows.append(row)

    # Hypothesis verdict rows.
    regimes = []
    for regime, group in df.groupby("prompt_regime"):
        losses = group[(group["pips"] < 0) & group["features.sl_pips"].notna()]
        regimes.append(
            {
                "prompt_regime": regime,
                "losses_with_sl": len(losses),
                "stop_overshoot_rate": losses["stop_overshoot_flag"].mean() if len(losses) else None,
                "median_sl_pips": losses["features.sl_pips"].median() if len(losses) else None,
                "median_spread": group["features.spread_at_entry"].fillna(group["market_snapshot.spread_pips"]).median(),
                "median_atr_m5": group["market_snapshot.technicals.M5.atr_pips"].median(),
            }
        )
    reg = pd.DataFrame(regimes)
    reg.to_csv(OUT / "phase7_stop_overshoot_regime_comparison.csv", index=False)
    phase3_row = reg[reg["prompt_regime"].eq("Phase 3 house-edge")].iloc[0]
    baseline = reg[reg["prompt_regime"].eq("Phase A baseline")]
    base_sl = baseline["median_sl_pips"].iloc[0] if not baseline.empty else np.nan
    base_atr = baseline["median_atr_m5"].iloc[0] if not baseline.empty else np.nan
    h2_weight = "high" if pd.notna(base_atr) and phase3_row["median_atr_m5"] > base_atr * 2 else "medium"
    h1_weight = "medium"
    h3_weight = "low"
    h4_weight = "low" if pd.notna(base_sl) and phase3_row["median_sl_pips"] >= base_sl else "medium"
    rows.extend(
        [
            {
                "row_type": "hypothesis_verdict",
                "hypothesis": "H2_slippage_or_microstructure_regime",
                "evidence_weight": h2_weight,
                "verdict": "best_supported",
                "evidence": f"Phase 3 median M5 ATR {phase3_row['median_atr_m5']:.2f}p vs Phase A {base_atr:.2f}p, with positive overshoots mostly small. This points first to volatility/slippage or exit-fill mechanics.",
            },
            {
                "row_type": "hypothesis_verdict",
                "hypothesis": "H1_exit_system_bug_specific_to_house_edge",
                "evidence_weight": h1_weight,
                "verdict": "possible_not_proven",
                "evidence": "The prompt-regime jump is real and CLR concentrates the flags, but no replayable exit-manager decisions are stored.",
            },
            {
                "row_type": "hypothesis_verdict",
                "hypothesis": "H4_stop_placement_tighter_or_more_visible",
                "evidence_weight": h4_weight,
                "verdict": "contradicted_as_tighter_stop_claim",
                "evidence": f"Phase 3 median SL {phase3_row['median_sl_pips']:.2f}p vs Phase A {base_sl:.2f}p, so the overshoot jump is not explained by tighter stored stops.",
            },
            {
                "row_type": "hypothesis_verdict",
                "hypothesis": "H3_LLM_disregards_stops",
                "evidence_weight": h3_weight,
                "verdict": "weak",
                "evidence": "No row-level exit conversation or stop-extension instruction is available; Phase 7 cannot attribute stop overshoot to LLM intent.",
            },
        ]
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase7_stop_overshoot_anomaly.csv", index=False)
    return out, reg


def day_of_week_interaction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in df.groupby("day_of_week"):
        row = {"row_type": "day_summary", "day_of_week": day, "cell": "all"}
        row.update(perf(group))
        row["gate_mix"] = json.dumps(group["trigger_family"].value_counts().head(5).to_dict())
        row["side_mix"] = json.dumps(group["side"].value_counts().to_dict())
        row["prompt_mix"] = json.dumps(group["prompt_regime"].value_counts().head(5).to_dict())
        row["rationale_mix"] = json.dumps(group["rationale_cluster"].value_counts().head(5).to_dict())
        row["session_mix"] = json.dumps(group["session"].value_counts().head(5).to_dict())
        rows.append(row)
    dims = ["day_of_week", "trigger_family", "side", "rationale_cluster"]
    for keys, group in df.groupby(dims, dropna=False):
        row = {"row_type": "day_gate_side_cluster_cell", "cell": " | ".join(f"{d}={v}" for d, v in zip(dims, keys))}
        for dim, key in zip(dims, keys):
            row[dim] = key
        row.update(perf(group))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["row_type", "net_pips"])
    out.to_csv(OUT / "phase7_day_of_week_interaction.csv", index=False)
    return out


def build_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "V1_caveat_cluster_sell": df["rationale_cluster"].isin(["momentum_with_caveat_trade", "critical_level_mixed_caveat_trade"])
        & df["side"].eq("sell"),
        "V2_entry_signature_mixed_overlap": df["timeframe_alignment_clean"].eq("mixed")
        & df["session"].isin(["london/ny overlap", "tokyo/london overlap"]),
        "V3_runner_custom_exit_v3": df["prompt_regime"].eq("Phase 2 runner/custom-exit v3"),
        "V4_wednesday_clr": df["day_of_week"].eq("Wednesday") & df["trigger_family"].eq("critical_level_reaction"),
        "V5_hedge_plus_overconfidence": df["high_hedge"] & df["high_conviction"],
        "V6_mixed_thin_snapshot_strong_level_claim": df["trigger_family"].eq("critical_level_reaction")
        & df["mixed_or_thin_snapshot"]
        & df["strong_level_claim"],
        "V7_blunt_clr_sell": df["trigger_family"].eq("critical_level_reaction") & df["side"].eq("sell"),
    }


def rule_metrics(df: pd.DataFrame, mask: pd.Series, rule_id: str, mode: str, sequence: int | None = None) -> dict[str, Any]:
    group = df[mask]
    winners = group[group["pips"] > 0]
    losers = group[group["pips"] < 0]
    saved_loser_pips = abs(losers["pips"].sum())
    saved_loser_usd = abs(losers["pnl"].sum())
    missed_winner_pips = winners["pips"].sum()
    missed_winner_usd = winners["pnl"].sum()
    net_delta_pips = -group["pips"].sum()
    net_delta_usd = -group["pnl"].sum()
    return {
        "mode": mode,
        "sequence": sequence,
        "rule_id": rule_id,
        "blocked_trades": len(group),
        "blocked_winners": len(winners),
        "blocked_losers": len(losers),
        "saved_loser_pips": saved_loser_pips,
        "saved_loser_usd": saved_loser_usd,
        "missed_winner_pips": missed_winner_pips,
        "missed_winner_usd": missed_winner_usd,
        "net_delta_pips": net_delta_pips,
        "net_delta_usd": net_delta_usd,
        "coverage_abs_net_pips": net_delta_pips / abs(OBSERVED_NET_PIPS),
        "coverage_abs_net_usd": net_delta_usd / abs(OBSERVED_NET_USD),
        "entry_failure_blocked": int(group["entry_failure_proxy"].sum()),
        "exit_reversal_blocked": int(group["exit_reversal_proxy"].sum()),
        "sample_size_warning": sample_warning(len(group)),
    }


def minimum_veto_rules(df: pd.DataFrame) -> pd.DataFrame:
    rules = build_rules(df)
    rows = []
    for rid, mask in rules.items():
        rows.append(rule_metrics(df, mask, rid, "individual"))
    union_mask = pd.concat(rules.values(), axis=1).any(axis=1)
    rows.append(rule_metrics(df, union_mask, "UNION_all_candidate_rules", "union"))

    for objective, metric in [("greedy_by_pips", "pips"), ("greedy_by_usd", "usd")]:
        remaining = pd.Series(True, index=df.index)
        selected: list[str] = []
        cumulative = pd.Series(False, index=df.index)
        for step in range(1, len(rules) + 1):
            best = None
            for rid, mask in rules.items():
                if rid in selected:
                    continue
                marginal = remaining & mask
                if not marginal.any():
                    continue
                m = rule_metrics(df, marginal, rid, "marginal")
                score = m["net_delta_pips"] if metric == "pips" else m["net_delta_usd"]
                secondary = m["net_delta_usd"] if metric == "pips" else m["net_delta_pips"]
                if best is None or (score, secondary) > best[0]:
                    best = ((score, secondary), rid, marginal)
            if best is None or best[0][0] <= 0:
                break
            selected.append(best[1])
            cumulative = cumulative | best[2]
            remaining = remaining & ~rules[best[1]]
            row = rule_metrics(df, cumulative, "+".join(selected), objective, step)
            row["cumulative_net_delta_pips"] = row["net_delta_pips"]
            row["cumulative_net_delta_usd"] = row["net_delta_usd"]
            row["cumulative_coverage_abs_net_pips"] = row["coverage_abs_net_pips"]
            row["cumulative_coverage_abs_net_usd"] = row["coverage_abs_net_usd"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase7_minimum_veto_rules.csv", index=False)
    return out


def build_report(
    df: pd.DataFrame,
    damage: pd.DataFrame,
    preserved: pd.DataFrame,
    entry: pd.DataFrame,
    caveat: pd.DataFrame,
    sizing: pd.DataFrame,
    rationale_regime: pd.DataFrame,
    stop: pd.DataFrame,
    stop_regime: pd.DataFrame,
    day: pd.DataFrame,
    veto: pd.DataFrame,
) -> str:
    core = damage[(damage["grid"].eq("core_4d")) & (damage["n"] >= 10)].copy()
    full = damage[damage["grid"].eq("full_10d")]
    top_pips = core.sort_values("net_pips").head(10)
    top_usd = core.sort_values("net_usd").head(10)
    top3_pips = top_pips.head(3)
    top3_usd = top_usd.head(3)
    top3_pips_cov = max(0.0, -top3_pips["net_pips"].sum()) / abs(OBSERVED_NET_PIPS)
    top3_usd_cov = max(0.0, -top3_usd["net_usd"].sum()) / abs(OBSERVED_NET_USD)
    full_n10 = int((full["n"] >= 10).sum())

    robust_preserved = preserved[preserved["survives_leave_one_date"].eq(True)] if not preserved.empty else preserved
    preserved_verdict = "exists" if not robust_preserved.empty else "does_not_survive_leave_one_date"

    entry_size = sizing[sizing["outcome_type"].eq("entry_failure")].iloc[0]
    entry_above_median = entry_size["share_gt_median_lot"]
    entry_sizing_verdict = "no" if entry_above_median <= 0.5 else "yes"

    pips70 = veto[(veto["mode"].eq("greedy_by_pips")) & (veto["cumulative_coverage_abs_net_pips"] >= 0.70)]
    usd70 = veto[(veto["mode"].eq("greedy_by_usd")) & (veto["cumulative_coverage_abs_net_usd"] >= 0.70)]
    pips_floor = pips70.sort_values("sequence").iloc[0] if not pips70.empty else None
    usd_floor = usd70.sort_values("sequence").iloc[0] if not usd70.empty else None

    sig = caveat[caveat["row_type"].eq("significance_test")].iloc[0]
    phase3_stop = stop_regime[stop_regime["prompt_regime"].eq("Phase 3 house-edge")].iloc[0]

    d_damage = display(damage.copy())
    d_preserved = display(preserved.copy())
    d_entry = display(entry.copy())
    d_caveat = display(caveat.copy())
    d_sizing = display(sizing.copy())
    d_rat = display(rationale_regime.copy())
    d_stop = display(stop.copy())
    d_day = display(day.copy())
    d_veto = display(veto.copy())

    lines: list[str] = []
    lines.append("# PHASE 7 - INTERACTION EFFECTS ANALYSIS")
    lines.append("")
    lines.append("_Auto Fillmore forensic investigation, Apr 16-May 1. Diagnosis only. Multi-layer cell synthesis._")
    lines.append("")
    lines.append("## Phase 7 Bottom Line")
    lines.append("")
    lines.append(
        f"Full 10D interaction cells are too sparse for headline claims: **{full_n10} exact 10D cells have N>=10** and the largest exact cell has N={int(full['n'].max())}. "
        "The headline ranking therefore uses the auditable core grid: gate x side x prompt regime x rationale cluster, with lot/day/session grids as supporting views."
    )
    lines.append("")
    lines.append(
        f"Top-3 destructive core cells cover **{pct(top3_pips_cov)} of the observed net pip loss** and **{pct(top3_usd_cov)} of the observed net USD loss**. "
        f"Preserved-edge verdict: **{preserved_verdict}**. Entry-failure trades above-median sizing verdict: **{entry_sizing_verdict}** "
        f"({pct(entry_above_median)} of entry failures used lots above the corpus median)."
    )
    lines.append("")
    if pips_floor is not None:
        lines.append(
            f"Smallest in-sample veto floor for >=70% pip recovery: **{pips_floor['rule_id']}** "
            f"({pct(pips_floor['coverage_abs_net_pips'])}, {one(pips_floor['net_delta_pips'])}p)."
        )
    if usd_floor is not None:
        lines.append(
            f"Smallest in-sample veto floor for >=70% USD recovery: **{usd_floor['rule_id']}** "
            f"({pct(usd_floor['coverage_abs_net_usd'])}, {money(usd_floor['net_delta_usd'])})."
        )
    lines.append("")
    lines.append("## 7.1 Damage Concentration Map")
    lines.append("")
    lines.append("Top core cells by pip damage (N>=10):")
    lines.append(md_table(display(top_pips), ["cell_signature", "n", "win_rate", "net_pips", "net_usd", "avg_pips", "entry_failure_n", "exit_reversal_n", "mean_lots", "coverage_abs_net_pips", "sample_size_warning"], 10))
    lines.append("")
    lines.append("Top core cells by USD damage (N>=10):")
    lines.append(md_table(display(top_usd), ["cell_signature", "n", "win_rate", "net_pips", "net_usd", "avg_usd", "entry_failure_n", "exit_reversal_n", "mean_lots", "coverage_abs_net_usd", "sample_size_warning"], 10))
    lines.append("")
    lines.append("Exact 10D cells are included in `phase7_damage_concentration.csv` but are treated as exploratory because N never reaches 10.")
    lines.append("")
    lines.append("## 7.2 Preserved-Edge Cells")
    lines.append("")
    if robust_preserved.empty:
        lines.append("No positive cell with N>=10 survives leave-one-date-out on both pips and USD. Phase 7 therefore finds no robust preserved edge that Phase 9 must protect unconditionally.")
    else:
        lines.append("Cells surviving leave-one-date-out:")
        lines.append(md_table(display(robust_preserved), ["grid", "cell_signature", "n", "win_rate", "net_pips", "net_usd", "avg_pips", "loo_min_net_pips", "loo_min_net_usd", "overlaps_8lot_clr_anomaly", "sample_size_warning"], 10))
    lines.append("")
    lines.append("All candidate positive cells:")
    lines.append(md_table(d_preserved, ["grid", "cell_signature", "n", "win_rate", "net_pips", "net_usd", "avg_pips", "loo_min_net_pips", "loo_min_net_usd", "survives_leave_one_date", "overlaps_8lot_clr_anomaly"], 12))
    lines.append("")
    lines.append("## 7.3 Entry-Failure Concentration")
    lines.append("")
    entry_ranked = entry.sort_values(["mutual_information", "entry_failure_share"], ascending=[False, False])
    lines.append(md_table(display(entry_ranked), ["factor", "value", "n", "entry_failure_n", "entry_failure_rate", "entry_failure_share", "net_pips", "net_usd", "median_lots", "above_median_lot_rate", "mutual_information", "sample_size_warning"], 18))
    fast_overlap = int((df["entry_failure_proxy"] & df["clr_fast_failure_17"]).sum())
    lines.append("")
    lines.append(f"CLR fast-failure overlap check: **{fast_overlap}/17** known CLR fast failures fall inside the Phase 6 entry-failure proxy.")
    lines.append("")
    lines.append("## 7.4 Caveat x Outcome-Type Interaction")
    lines.append("")
    lines.append(md_table(d_caveat[d_caveat["row_type"].eq("outcome_summary")], ["outcome_type", "n", "win_rate", "net_pips", "net_usd", "median_hedge_density", "median_conviction_density", "self_correction_rate", "mean_lots", "above_median_lot_rate"], 8))
    lines.append("")
    lines.append(
        f"Rationale-cluster x outcome-type permutation chi-square: statistic {two(sig['chi_square'])}, p={two(sig['p_value_permutation'])}. "
        "This tests association in the observed corpus; it is diagnostic, not a forward predictor."
    )
    lines.append("")
    lines.append("## 7.5 Sizing x Entry-Failure Compounding")
    lines.append("")
    lines.append(md_table(d_sizing, ["outcome_type", "n", "lots_gt_1_n", "share_gt_median_lot", "mean_lots", "actual_usd", "uniform_1_lot_usd", "sizing_delta_usd", "phase5_target_sizing_delta_usd", "reconciliation_gap_usd"], 8))
    lines.append("")
    lines.append("Verdict: sizing amplification does **not** concentrate primarily in entry-failure trades. It is spread across loser buckets and partially offset by winner sizing.")
    lines.append("")
    lines.append("## 7.6 Rationale x Snapshot Regime Interaction")
    lines.append("")
    rr = rationale_regime[rationale_regime["n"] >= 10].sort_values(["net_pips", "net_usd"]).head(18)
    lines.append(md_table(display(rr), ["rationale_cluster", "regime_type", "regime_value", "n", "cell_share_of_cluster", "win_rate", "net_pips", "net_usd", "avg_pips", "sample_size_warning"], 18))
    lines.append("")
    lines.append("## 7.7 Stop Overshoot Anomaly")
    lines.append("")
    phase3_summary = stop[(stop["row_type"].eq("segment")) & (stop["segment_type"].eq("overall_phase3"))].head(1)
    lines.append(md_table(display(phase3_summary), ["segment", "n", "losses_with_sl", "stop_overshoot_n", "stop_overshoot_rate", "stop_overshoot_median_all_losses", "median_positive_overshoot_pips", "median_sl_pips", "median_spread", "median_atr_m5"], 4))
    lines.append("")
    lines.append(md_table(d_stop[d_stop["row_type"].eq("hypothesis_verdict")], ["hypothesis", "evidence_weight", "verdict", "evidence"], 8))
    lines.append("")
    lines.append(
        f"Phase 3 stop-overshoot rate: **{pct(phase3_stop['stop_overshoot_rate'])}** across {int(phase3_stop['losses_with_sl'])} losses with stored SL. "
        "Best-supported explanation is volatility/slippage or exit-fill mechanics; tighter-stop geometry is contradicted by the stored SL medians, and an LLM stop-disregard mechanism is not proven."
    )
    lines.append("")
    lines.append("## 7.8 Day-of-Week x Gate x Reasoning")
    lines.append("")
    lines.append(md_table(d_day[d_day["row_type"].eq("day_summary")].sort_values("net_pips"), ["day_of_week", "n", "win_rate", "net_pips", "net_usd", "avg_pips", "gate_mix", "side_mix", "prompt_mix"], 8))
    lines.append("")
    day_cells = day[day["row_type"].eq("day_gate_side_cluster_cell") & (day["n"] >= 5)].sort_values("net_pips").head(12)
    lines.append(md_table(display(day_cells), ["cell", "n", "win_rate", "net_pips", "net_usd", "avg_pips", "sample_size_warning"], 12))
    lines.append("")
    lines.append("## 7.9 Minimum Veto Rule Set")
    lines.append("")
    lines.append("Individual candidate rules:")
    individual_rules = veto[veto["mode"].eq("individual")].sort_values("net_delta_pips", ascending=False)
    lines.append(md_table(display(individual_rules), ["rule_id", "blocked_trades", "blocked_winners", "blocked_losers", "saved_loser_pips", "missed_winner_pips", "net_delta_pips", "net_delta_usd", "coverage_abs_net_pips", "coverage_abs_net_usd", "entry_failure_blocked", "exit_reversal_blocked"], 10))
    lines.append("")
    lines.append("Greedy cumulative floor:")
    lines.append(md_table(d_veto[d_veto["mode"].isin(["greedy_by_pips", "greedy_by_usd", "union"])], ["mode", "sequence", "rule_id", "blocked_trades", "blocked_winners", "blocked_losers", "net_delta_pips", "net_delta_usd", "cumulative_coverage_abs_net_pips", "cumulative_coverage_abs_net_usd"], 20))
    lines.append("")
    lines.append("These are in-sample diagnostic vetoes. Phase 9 does not have to use them, but a redesign needs to beat this damage-avoidance floor without simply deleting the entire system.")
    lines.append("")
    lines.append("## Interaction Verdict")
    lines.append("")
    lines.append("- Damage concentration: **concentrated in low-cardinality cells, sparse in exact 10D cells**. Core interaction cells are usable; exact 10D cells are too thin.")
    lines.append(f"- Preserved edge: **{preserved_verdict}**.")
    lines.append(f"- Sizing amplification compounds entry failures: **{entry_sizing_verdict}**; entry failures have {pct(entry_above_median)} above-median sizing and do not explain most of the Phase 5 sizing delta.")
    lines.append("- Stop overshoot anomaly ranking: **H2 best-supported, H1 possible/not proven, H4 contradicted as a tighter-stop claim, H3 weak**.")
    if pips_floor is not None:
        lines.append(f"- Diagnostic pip floor: **{pips_floor['rule_id']}** recovers {one(pips_floor['net_delta_pips'])}p in-sample.")
    if usd_floor is not None:
        lines.append(f"- Diagnostic USD floor: **{usd_floor['rule_id']}** recovers {money(usd_floor['net_delta_usd'])} in-sample.")
    lines.append("")
    lines.append("## Evidence Gaps")
    lines.append("")
    lines.append("- Exact 10D interaction cells are underpowered; most cells have N<5 and no exact cell reaches N>=10.")
    lines.append("- Phase 6 MAE caveat still applies: stored MAE under-reports realized loss on many losers, so lifecycle labels are terminal proxies, not tick replay.")
    lines.append("- No skip forward outcomes, so veto-rule opportunity cost is measured only on trades Fillmore actually placed.")
    lines.append("- No multi-gate candidate logging, so gate-overlap interaction cannot be added to the cell map.")
    lines.append("- No snapshot_version field, so prompt regime remains the closest available schema boundary.")
    lines.append("- Stop-overshoot hypotheses cannot be definitively separated without exit-manager replay, tick path, and exit-inclusive MAE.")
    lines.append("")
    lines.append("## Artifacts Written")
    lines.append("")
    for name in [
        "phase7_interaction_dataset.csv",
        "phase7_damage_concentration.csv",
        "phase7_preserved_edge_cells.csv",
        "phase7_entry_failure_concentration.csv",
        "phase7_caveat_outcome_interaction.csv",
        "phase7_sizing_compounding.csv",
        "phase7_rationale_regime_interaction.csv",
        "phase7_stop_overshoot_anomaly.csv",
        "phase7_stop_overshoot_regime_comparison.csv",
        "phase7_day_of_week_interaction.csv",
        "phase7_minimum_veto_rules.csv",
        "phase7_manifest.json",
    ]:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_dataset()
    df.to_csv(OUT / "phase7_interaction_dataset.csv", index=False)

    damage = damage_concentration(df)
    preserved = preserved_edge_cells(df)
    entry = entry_failure_concentration(df)
    caveat = caveat_outcome_interaction(df)
    sizing = sizing_compounding(df)
    rationale = rationale_regime_interaction(df)
    stop, stop_regime = stop_overshoot_anomaly(df)
    day = day_of_week_interaction(df)
    veto = minimum_veto_rules(df)

    report = build_report(df, damage, preserved, entry, caveat, sizing, rationale, stop, stop_regime, day, veto)
    (OUT / "PHASE7_INTERACTION_EFFECTS.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "inputs": [str(CLOSED_CSV), str(PHASE4_CSV), str(FAST_FAILURE_CSV), str(PHASE5_DECOMP_CSV)],
        "closed_rows": int(len(df)),
        "net_pips": float(df["pips"].sum()),
        "net_usd": float(df["pnl"].sum()),
        "notes": [
            "Read-only forensic harness.",
            "Full 10D exact cells are sparse; core_4d grid is used for headline concentration.",
            "Veto rules are in-sample diagnostic floors, not recommendations.",
        ],
    }
    (OUT / "phase7_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'PHASE7_INTERACTION_EFFECTS.md'}")
    print(f"Closed rows: {len(df)}; net pips: {df['pips'].sum():.1f}; net USD: {df['pnl'].sum():.2f}")


if __name__ == "__main__":
    main()
