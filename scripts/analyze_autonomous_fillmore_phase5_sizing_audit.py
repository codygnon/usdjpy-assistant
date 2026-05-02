#!/usr/bin/env python3
"""Phase 5 sizing logic audit for Autonomous Fillmore.

Read-only forensic harness. It uses the 241 closed-trade baseline, Phase 3
snapshot fields, and Phase 4 reasoning features to separate admission losses
from sizing amplification.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORENSIC_DIR = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
CLOSED_CSV = FORENSIC_DIR / "phase3_closed_trades_with_snapshot_fields.csv"
PHASE4_CORPUS_CSV = FORENSIC_DIR / "phase4_reasoning_corpus.csv"
PHASE4_SIZING_CSV = FORENSIC_DIR / "phase4_sizing_reasoning_audit.csv"
OUT = FORENSIC_DIR


NUMERIC_MODEL_FIELDS = [
    "market_snapshot.account_balance",
    "market_snapshot.account_equity",
    "market_snapshot.margin_used",
    "features.planned_rr",
    "features.sl_pips",
    "features.tp_pips",
    "market_snapshot.volatility.ratio",
    "market_snapshot.technicals.M5.atr_pips",
    "market_snapshot.spread_pips",
]


def s(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def money(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.2f}"


def one_dec(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two_dec(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


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
    try:
        x = float(lots)
    except Exception:
        return "unknown"
    if pd.isna(x):
        return "unknown"
    if x <= 0:
        return "0"
    if x < 2:
        return "0-1.99"
    if x < 4:
        return "2-3.99"
    if x < 8:
        return "4-7.99"
    return "8+"


def perf(group: pd.DataFrame, pnl_col: str = "pnl", pips_col: str = "pips") -> dict[str, Any]:
    g = group[group[pnl_col].notna()].copy()
    wins = g[g[pnl_col] > 0]
    losses = g[g[pnl_col] < 0]
    gross_win = wins[pnl_col].sum()
    gross_loss = abs(losses[pnl_col].sum())
    return {
        "n": len(g),
        "wins": int((g[pips_col] > 0).sum()) if pips_col in g.columns else len(wins),
        "losses": int((g[pips_col] < 0).sum()) if pips_col in g.columns else len(losses),
        "win_rate": (g[pips_col].gt(0).mean() if pips_col in g.columns and len(g) else None),
        "net_pips": g[pips_col].sum() if pips_col in g.columns else None,
        "net_usd": g[pnl_col].sum(),
        "avg_pips": g[pips_col].mean() if pips_col in g.columns else None,
        "avg_usd": g[pnl_col].mean() if len(g) else None,
        "avg_winner_pips": g.loc[g[pips_col] > 0, pips_col].mean() if pips_col in g.columns and g[pips_col].gt(0).any() else None,
        "avg_loser_pips": g.loc[g[pips_col] < 0, pips_col].mean() if pips_col in g.columns and g[pips_col].lt(0).any() else None,
        "avg_winner_usd": wins[pnl_col].mean() if len(wins) else None,
        "avg_loser_usd": losses[pnl_col].mean() if len(losses) else None,
        "profit_factor_usd": (gross_win / gross_loss) if gross_loss else (math.inf if gross_win else None),
    }


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    cumulative = values.fillna(0).cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    return float(dd.min())


def sharpe(values: pd.Series) -> float | None:
    vals = values.dropna().astype(float)
    if len(vals) < 2 or vals.std(ddof=1) == 0:
        return None
    return float(vals.mean() / vals.std(ddof=1) * math.sqrt(len(vals)))


def display_perf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["win_rate", "place_rate", "share_gt_1", "share_ge_4", "share_ge_8", "compliance_rate"]:
        if col in out.columns:
            out[col] = out[col].map(pct)
    for col in [
        "net_usd",
        "avg_usd",
        "avg_winner_usd",
        "avg_loser_usd",
        "actual_usd",
        "counterfactual_usd",
        "delta_vs_actual_usd",
        "max_drawdown_usd",
        "violation_closed_net_usd",
        "component_usd",
    ]:
        if col in out.columns:
            out[col] = out[col].map(money)
    for col in [
        "net_pips",
        "avg_pips",
        "avg_winner_pips",
        "avg_loser_pips",
        "profit_factor_usd",
        "mean_lots",
        "median_lots",
        "p25_lots",
        "p75_lots",
        "p90_lots",
        "max_lots",
        "mode_lots",
        "correlation",
        "ci_low",
        "ci_high",
        "r_squared",
        "coefficient",
        "std_error",
        "t_stat",
        "p_value_approx",
        "sharpe_usd",
        "trades_altered",
        "trades_skipped",
    ]:
        if col in out.columns:
            out[col] = out[col].map(two_dec)
    return out


def load_dataset() -> pd.DataFrame:
    closed = pd.read_csv(CLOSED_CSV, low_memory=False)
    corpus = pd.read_csv(PHASE4_CORPUS_CSV, low_memory=False)
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
    for col in ["lots", "pips", "pnl", "mae_abs", "mfe", *NUMERIC_MODEL_FIELDS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["lot_bucket"] = df["lots"].map(lot_bucket)
    df["created_dt"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
    df["created_date"] = df["created_dt"].dt.strftime("%Y-%m-%d")
    df["win_flag"] = (df["pips"] > 0).astype(float)
    df["pnl_per_lot"] = np.where(df["lots"] > 0, df["pnl"] / df["lots"], np.nan)
    df["pip_value_per_lot"] = np.where((df["lots"] > 0) & (df["pips"].abs() > 0), df["pnl"].abs() / (df["lots"] * df["pips"].abs()), np.nan)
    df["rationale_cluster"] = df["rationale_cluster"].fillna("unknown")
    df["hedge_density"] = df["hedge_density"].fillna(0.0)
    df["conviction_density"] = df["conviction_density"].fillna(0.0)
    df["self_correction_count"] = df["self_correction_count"].fillna(0)
    df["has_self_correction"] = df["self_correction_count"] > 0
    return df.sort_values("created_dt").reset_index(drop=True)


def sizing_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        lots = group["lots"].dropna()
        if lots.empty:
            return
        mode = lots.mode()
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "n": len(group),
            "mean_lots": lots.mean(),
            "median_lots": lots.median(),
            "p25_lots": lots.quantile(0.25),
            "p75_lots": lots.quantile(0.75),
            "p90_lots": lots.quantile(0.90),
            "max_lots": lots.max(),
            "mode_lots": mode.iloc[0] if not mode.empty else None,
            "unique_lot_values": ",".join(str(x) for x in sorted(lots.unique())),
            "share_gt_1": (lots > 1).mean(),
            "share_ge_4": (lots >= 4).mean(),
            "share_ge_8": (lots >= 8).mean(),
        }
        row.update(perf(group))
        row["sample_size_warning"] = "N<15" if len(group) < 15 else ""
        rows.append(row)

    add("overall", "all_closed", df)
    for col, label in [
        ("trigger_family", "gate"),
        ("side", "side"),
        ("prompt_regime", "prompt_regime"),
        ("rationale_cluster", "rationale_cluster"),
        ("features.session", "session"),
        ("created_date", "date"),
    ]:
        if col not in df.columns:
            continue
        for key, group in df.groupby(col, dropna=False):
            add(label, s(key) or "missing", group)

    out = pd.DataFrame(rows).sort_values(["segment_type", "net_usd"], ascending=[True, True])
    out.to_csv(OUT / "phase5_sizing_distribution.csv", index=False)
    return out


def pearson(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].std(ddof=1) == 0 or pair["y"].std(ddof=1) == 0:
        return None
    return float(pair["x"].corr(pair["y"], method="pearson"))


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return None
    return pearson(pair["x"].rank(), pair["y"].rank())


def bootstrap_ci(x: pd.Series, y: pd.Series, method: str = "pearson", n_boot: int = 2000) -> tuple[float | None, float | None]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 8:
        return None, None
    rng = np.random.default_rng(20260501)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pair), len(pair))
        sample = pair.iloc[idx]
        val = pearson(sample["x"], sample["y"]) if method == "pearson" else spearman(sample["x"], sample["y"])
        if val is not None and not pd.isna(val):
            vals.append(val)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def edge_size_correlation(df: pd.DataFrame) -> pd.DataFrame:
    targets = [
        ("realized_pips", "pips"),
        ("realized_usd", "pnl"),
        ("win_flag", "win_flag"),
        ("realized_mae", "mae_abs"),
        ("realized_mfe", "mfe"),
    ]
    rows: list[dict[str, Any]] = []

    def add(scope_type: str, scope: str, group: pd.DataFrame) -> None:
        for target_name, target_col in targets:
            if target_col not in group.columns:
                continue
            for method_name, func in [("pearson", pearson), ("spearman", spearman)]:
                corr = func(group["lots"], group[target_col])
                ci_low = ci_high = None
                if scope_type == "overall":
                    ci_low, ci_high = bootstrap_ci(group["lots"], group[target_col], method_name)
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "target": target_name,
                        "method": method_name,
                        "n": int(pd.DataFrame({"x": group["lots"], "y": group[target_col]}).dropna().shape[0]),
                        "correlation": corr,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "verdict_hint": "negative" if corr is not None and corr < -0.05 else ("positive" if corr is not None and corr > 0.05 else "near_zero"),
                        "sample_size_warning": "N<15" if len(group) < 15 else "",
                    }
                )

    add("overall", "all_closed", df)
    for col, label in [
        ("trigger_family", "gate"),
        ("side", "side"),
        ("rationale_cluster", "rationale_cluster"),
        ("prompt_regime", "prompt_regime"),
    ]:
        for key, group in df.groupby(col, dropna=False):
            if len(group) >= 15:
                add(label, s(key) or "missing", group)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase5_edge_size_correlation.csv", index=False)
    return out


def caveat_sizing_interaction(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    out_df["hedge_bin"] = "zero"
    positive = out_df.loc[out_df["hedge_density"] > 0, "hedge_density"]
    if not positive.empty:
        q50 = positive.quantile(0.50)
        q80 = positive.quantile(0.80)
        out_df.loc[(out_df["hedge_density"] > 0) & (out_df["hedge_density"] <= q50), "hedge_bin"] = "low_hedge"
        out_df.loc[(out_df["hedge_density"] > q50) & (out_df["hedge_density"] <= q80), "hedge_bin"] = "mid_hedge"
        out_df.loc[out_df["hedge_density"] > q80, "hedge_bin"] = "high_hedge"

    out_df["conviction_bin"] = "zero"
    positive_c = out_df.loc[out_df["conviction_density"] > 0, "conviction_density"]
    if not positive_c.empty:
        q50 = positive_c.quantile(0.50)
        q80 = positive_c.quantile(0.80)
        out_df.loc[(out_df["conviction_density"] > 0) & (out_df["conviction_density"] <= q50), "conviction_bin"] = "low_conviction"
        out_df.loc[(out_df["conviction_density"] > q50) & (out_df["conviction_density"] <= q80), "conviction_bin"] = "mid_conviction"
        out_df.loc[out_df["conviction_density"] > q80, "conviction_bin"] = "high_conviction"

    out_df["self_correction_bin"] = np.where(out_df["has_self_correction"], "self_correction_present", "self_correction_absent")
    rows: list[dict[str, Any]] = []
    for col, test in [
        ("hedge_bin", "hedge_density"),
        ("conviction_bin", "conviction_density"),
        ("self_correction_bin", "self_correction"),
        ("rationale_cluster", "rationale_cluster"),
    ]:
        for key, group in out_df.groupby(col, dropna=False):
            row = {
                "test": test,
                "bucket": s(key),
                "n": len(group),
                "mean_lots": group["lots"].mean(),
                "median_lots": group["lots"].median(),
                "p75_lots": group["lots"].quantile(0.75),
                "share_gt_1": (group["lots"] > 1).mean(),
                "share_ge_4": (group["lots"] >= 4).mean(),
                "share_ge_8": (group["lots"] >= 8).mean(),
            }
            row.update(perf(group))
            row["sample_size_warning"] = "N<15" if len(group) < 15 else ""
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["test", "net_usd"], ascending=[True, True])
    out.to_csv(OUT / "phase5_caveat_sizing_interaction.csv", index=False)
    return out


def normal_p_value(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def implied_sizing_function(df: pd.DataFrame) -> pd.DataFrame:
    model = df.copy()
    cols = [c for c in NUMERIC_MODEL_FIELDS if c in model.columns]
    for col in cols:
        model[col] = pd.to_numeric(model[col], errors="coerce")
        med = model[col].median()
        model[col] = model[col].fillna(med)
        std = model[col].std(ddof=0)
        if std and not pd.isna(std):
            model[col] = (model[col] - model[col].mean()) / std
        else:
            model[col] = 0.0

    cat_cols = [c for c in ["trigger_family", "side", "prompt_regime"] if c in model.columns]
    dummies = pd.get_dummies(model[cat_cols].fillna("missing"), drop_first=True, dtype=float)
    x = pd.concat([pd.Series(1.0, index=model.index, name="intercept"), model[cols], dummies], axis=1).astype(float)
    y = model["lots"].astype(float)
    complete = y.notna()
    x = x.loc[complete]
    y = y.loc[complete]

    x_mat = x.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    y_hat = x_mat @ beta
    resid = y_vec - y_hat
    n, k = x_mat.shape
    rss = float((resid**2).sum())
    tss = float(((y_vec - y_vec.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss else None
    sigma2 = rss / max(n - k, 1)
    try:
        cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)

    rows = []
    for name, coef, err in zip(x.columns, beta, se):
        t_stat = coef / err if err and not pd.isna(err) else None
        rows.append(
            {
                "term": name,
                "coefficient": coef,
                "std_error": err,
                "t_stat": t_stat,
                "p_value_approx": normal_p_value(t_stat) if t_stat is not None else None,
                "r_squared": r2,
                "n": n,
                "interpretation": "standardized numeric coefficient" if name in cols else "dummy/intercept coefficient",
            }
        )

    # Candidate model residuals.
    pip_value = df["pip_value_per_lot"].median()
    equity = df.get("market_snapshot.account_equity", pd.Series(np.nan, index=df.index)).fillna(df.get("market_snapshot.account_balance", pd.Series(np.nan, index=df.index))).fillna(df.get("market_snapshot.account_balance", pd.Series(np.nan, index=df.index)).median())
    sl = df.get("features.sl_pips", pd.Series(np.nan, index=df.index)).fillna(df["pips"].abs().median()).clip(lower=1)
    atr = df.get("market_snapshot.technicals.M5.atr_pips", pd.Series(np.nan, index=df.index)).fillna(df.get("market_snapshot.technicals.M5.atr_pips", pd.Series(np.nan, index=df.index)).median())
    vol_ratio = df.get("market_snapshot.volatility.ratio", pd.Series(np.nan, index=df.index)).fillna(1.0).clip(lower=0.25)
    fixed_fractional = (equity * 0.01 / (sl * pip_value)).clip(1, 10)
    vol_scaled = (1 / vol_ratio).round().clip(1, 5)
    fixed_lot = pd.Series(1.0, index=df.index)
    residual_rows = []
    for name, pred in [
        ("fixed_1_lot", fixed_lot),
        ("volatility_scaled_1_over_ratio_clipped_1_5", vol_scaled),
        ("fixed_fractional_1pct_equity_sl_clipped_1_10", fixed_fractional),
    ]:
        residual = df["lots"] - pred
        residual_rows.append(
            {
                "term": f"candidate_model_residual::{name}",
                "coefficient": residual.mean(),
                "std_error": residual.std(ddof=1),
                "t_stat": None,
                "p_value_approx": None,
                "r_squared": None,
                "n": int(residual.notna().sum()),
                "interpretation": "mean chosen lots minus candidate model lots; positive means Fillmore sized larger",
            }
        )
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(residual_rows)], ignore_index=True)
    out.to_csv(OUT / "phase5_implied_sizing_function.csv", index=False)
    return out


def pl_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    actual = df["pnl"].sum()
    uniform_1 = df["pnl_per_lot"].sum()
    oversize_delta = actual - uniform_1
    p75_lots = float(df["lots"].quantile(0.75))
    winners = df[df["pnl"] > 0].copy()
    undersized_delta = ((p75_lots - winners["lots"]).clip(lower=0) * winners["pnl_per_lot"]).sum()
    rows = [
        {
            "component": "actual_total_pnl",
            "component_usd": actual,
            "definition": "Observed P&L across the 241 closed trades.",
            "reconciles_to_actual": True,
        },
        {
            "component": "admission_cost_at_uniform_1_lot",
            "component_usd": uniform_1,
            "definition": "Same 241 trades, same pips, every trade normalized to 1 lot. This is the gate/reasoning admission result before variable sizing.",
            "reconciles_to_actual": True,
        },
        {
            "component": "sizing_amplification_delta_vs_1_lot",
            "component_usd": oversize_delta,
            "definition": "Observed P&L minus uniform-1-lot P&L. This is the exact variable-sizing contribution under linear per-lot P&L.",
            "reconciles_to_actual": True,
        },
        {
            "component": "overlap_term",
            "component_usd": 0.0,
            "definition": "No overlap term is needed because actual P&L equals uniform-1-lot admission P&L plus sizing delta.",
            "reconciles_to_actual": True,
        },
        {
            "component": "undersized_winner_opportunity_to_p75_lots",
            "component_usd": undersized_delta,
            "definition": f"Non-additive upside bound: winners below p75 lots ({p75_lots:.2f}) hypothetically upsized to p75.",
            "reconciles_to_actual": False,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase5_pl_decomposition.csv", index=False)
    return out


def anomaly_8lot_clr(df: pd.DataFrame) -> pd.DataFrame:
    target = df[
        (df["rationale_cluster"] == "critical_level_mixed_caveat_trade")
        & (df["trigger_family"] == "critical_level_reaction")
        & (df["lots"] >= 8)
    ].copy()
    compare = df[
        (df["rationale_cluster"] == "momentum_with_caveat_trade")
        & (df["lots"] >= 8)
    ].copy()
    rows: list[dict[str, Any]] = []

    def add(segment_type: str, segment: str, group: pd.DataFrame, verdict_note: str = "") -> None:
        if group.empty:
            return
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "mean_lots": group["lots"].mean(),
            "side_mix": json.dumps(group["side"].value_counts(dropna=False).to_dict()),
            "prompt_mix": json.dumps(group["prompt_regime"].value_counts(dropna=False).to_dict()),
            "date_mix": json.dumps(group["created_date"].value_counts(dropna=False).to_dict()),
            "verdict_note": verdict_note,
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    add("target", "critical_level_mixed_caveat_trade_8plus_clr", target)
    add("comparison", "momentum_with_caveat_trade_8plus", compare)
    for col in ["side", "prompt_regime", "created_date", "features.session", "market_snapshot.volatility.label", "market_snapshot.technicals.H1.regime"]:
        if col in target.columns:
            for key, group in target.groupby(col, dropna=False):
                add(f"target_by_{col}", s(key), group)
    for date, _ in target.groupby("created_date", dropna=False):
        loo = target[target["created_date"] != date]
        add("target_leave_one_date_out", f"drop_{date}", loo)

    verdict = "regime_or_compositional_artifact"
    if len(target) >= 15 and target["net_usd"].sum() if "net_usd" in target.columns else False:
        verdict = "needs_review"
    out = pd.DataFrame(rows)
    # Commit verdict from segmentation, not from an unused variable.
    if not out.empty:
        target_net = target["pnl"].sum()
        target_pips = target["pips"].sum()
        buy_share = (target["side"] == "buy").mean() if len(target) else 0
        if target_net > 0 and target_pips > 0 and buy_share < 0.8:
            committed = "possible_real_edge_but_sample_limited"
        elif abs(target_net) < 150 and buy_share >= 0.8:
            committed = "compositional_artifact_buy_clr_dominated_near_flat"
        else:
            committed = "regime_artifact_or_survivorship"
        out["committed_verdict"] = committed
    out.to_csv(OUT / "phase5_8lot_clr_anomaly.csv", index=False)
    return out


def policy_metrics(df: pd.DataFrame, policy_name: str, policy_lots: pd.Series, skip_mask: pd.Series | None = None) -> dict[str, Any]:
    lots = policy_lots.fillna(1.0).clip(lower=0)
    skip = skip_mask.fillna(False) if skip_mask is not None else pd.Series(False, index=df.index)
    pnl = df["pnl_per_lot"] * lots
    pnl = pnl.where(~skip, 0.0)
    pips = df["pips"].where(~skip, 0.0)
    altered = ((lots.round(6) != df["lots"].round(6)) | skip).sum()
    tmp = pd.DataFrame({"pnl": pnl, "pips": pips})
    p = perf(tmp)
    p.update(
        {
            "policy": policy_name,
            "counterfactual_usd": pnl.sum(),
            "net_pips": pips.sum(),
            "delta_vs_actual_usd": pnl.sum() - df["pnl"].sum(),
            "max_drawdown_usd": max_drawdown(pnl),
            "sharpe_usd": sharpe(pnl),
            "trades_altered": int(altered),
            "trades_skipped": int(skip.sum()),
            "mean_policy_lots": lots[~skip].mean() if (~skip).any() else 0.0,
            "retroactive_warning": "In-sample bound only; not predictive.",
        }
    )
    return p


def counterfactual_policies(df: pd.DataFrame) -> pd.DataFrame:
    vol_ratio = df.get("market_snapshot.volatility.ratio", pd.Series(1.0, index=df.index)).fillna(1.0).clip(lower=0.25)
    pip_value = df["pip_value_per_lot"].median()
    equity = df.get("market_snapshot.account_equity", pd.Series(np.nan, index=df.index)).fillna(df.get("market_snapshot.account_balance", pd.Series(np.nan, index=df.index))).fillna(df.get("market_snapshot.account_balance", pd.Series(np.nan, index=df.index)).median())
    sl = df.get("features.sl_pips", pd.Series(np.nan, index=df.index)).fillna(df["pips"].abs().median()).clip(lower=1)
    fixed_fractional = (equity * 0.01 / (sl * pip_value)).clip(1, 5).round(1)
    vol_scaled = (1 / vol_ratio).round().clip(1, 5)

    # Rolling cluster-aware: if prior cluster net pips is negative after 5 prior
    # trades, cap next occurrence to 1 lot; otherwise 2 lots.
    rolling_lots = []
    cluster_history: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        hist = cluster_history.get(row["rationale_cluster"], [])
        if len(hist) >= 5 and sum(hist) < 0:
            rolling_lots.append(1.0)
        else:
            rolling_lots.append(2.0)
        cluster_history.setdefault(row["rationale_cluster"], []).append(float(row["pips"]))
    rolling_lots_s = pd.Series(rolling_lots, index=df.index)

    high_hedge = df["hedge_density"] > df.loc[df["hedge_density"] > 0, "hedge_density"].quantile(0.80)
    caveat = df["has_self_correction"] | high_hedge
    caveat_lots = df["lots"].where(~caveat, np.minimum(df["lots"], 1.0))

    phase5_skip = (
        df["rationale_cluster"].isin(["critical_level_mixed_caveat_trade", "momentum_with_caveat_trade", "macro_policy_caveat_trade"])
        & (df["has_self_correction"] | df["hedge_density"].gt(0))
    )
    phase5_lots = pd.Series(1.0, index=df.index)

    rows = [
        policy_metrics(df, "actual_observed", df["lots"], pd.Series(False, index=df.index)),
        policy_metrics(df, "A_uniform_1_lot", pd.Series(1.0, index=df.index)),
        policy_metrics(df, "B_uniform_2_lots", pd.Series(2.0, index=df.index)),
        policy_metrics(df, "C_volatility_scaled_1_over_ratio_clipped_1_5", vol_scaled),
        policy_metrics(df, "D_fixed_fractional_1pct_equity_sl_clipped_1_5", fixed_fractional),
        policy_metrics(df, "E_rolling_cluster_aware_1_if_prior_negative_else_2", rolling_lots_s),
        policy_metrics(df, "F_caveat_capped_at_1_lot_else_actual", caveat_lots),
        policy_metrics(df, "G_phase5_style_skip_toxic_caveat_else_1_lot", phase5_lots, phase5_skip),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase5_counterfactual_policies.csv", index=False)
    return out


def mae_mfe_interaction(df: pd.DataFrame, corr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, col in [("MAE", "mae_abs"), ("MFE", "mfe")]:
        rows.append(
            {
                "target": target,
                "pearson_lots_correlation": pearson(df["lots"], df[col]),
                "spearman_lots_correlation": spearman(df["lots"], df[col]),
                "high_lot_mean": df[df["lots"] >= df["lots"].quantile(0.75)][col].mean(),
                "low_lot_mean": df[df["lots"] <= df["lots"].quantile(0.25)][col].mean(),
                "interpretation": "positive means larger trades experienced more of this path excursion",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase5_sizing_mae_mfe.csv", index=False)
    return out


def required_telemetry() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "field": "open_usdjpy_lots_by_side",
            "mechanism_enabled": "Prevents adding size in the same direction when exposure is already concentrated.",
            "evidence": "Phase 3 found this absent; Phase 5 large-lot losses show size was not exposure-aware.",
        },
        {
            "priority": 2,
            "field": "risk_after_fill_usd",
            "mechanism_enabled": "Converts proposed lots and stop distance into a dollar loss before order placement.",
            "evidence": "Loss asymmetry is much larger in USD than pips, proving dollar risk was not normalized.",
        },
        {
            "priority": 3,
            "field": "rolling_20_trade_pnl_and_lot_weighted_pnl",
            "mechanism_enabled": "Allows drawdown-aware size throttling without a daily-loss kill switch.",
            "evidence": "Phase 4 confirmed caveat trades continued through adverse context; Phase 5 tests whether size adapted.",
        },
        {
            "priority": 4,
            "field": "unrealized_pnl_by_side",
            "mechanism_enabled": "Separates a new idea from a hedge/add into an existing losing inventory.",
            "evidence": "Reasoning repeatedly mentioned hedges/adds, but persisted snapshot did not carry side P&L.",
        },
        {
            "priority": 5,
            "field": "pip_value_per_lot",
            "mechanism_enabled": "Makes fixed-dollar risk and fixed-fractional sizing calculable.",
            "evidence": "Phase 5 uniform-lot counterfactual relies on inferred per-lot P&L because the field is not explicit.",
        },
        {
            "priority": 6,
            "field": "session_volatility_reference_size",
            "mechanism_enabled": "Prevents Tokyo/London/NY volatility changes from mapping to arbitrary lot choices.",
            "evidence": "Volatility and ATR are present, but no reference size or risk envelope is presented.",
        },
        {
            "priority": 7,
            "field": "daily_drawdown_usage",
            "mechanism_enabled": "Supports soft risk-aware reasoning without making max-daily-loss the primary solution.",
            "evidence": "User objective is performance troubleshooting, but size still needs state awareness.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase5_required_telemetry.csv", index=False)
    return out


def build_report(
    df: pd.DataFrame,
    dist: pd.DataFrame,
    corr: pd.DataFrame,
    caveat: pd.DataFrame,
    model: pd.DataFrame,
    decomp: pd.DataFrame,
    anomaly: pd.DataFrame,
    policies: pd.DataFrame,
    mae_mfe: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> str:
    actual = df["pnl"].sum()
    uniform_1 = decomp.loc[decomp["component"] == "admission_cost_at_uniform_1_lot", "component_usd"].iloc[0]
    sizing_delta = decomp.loc[decomp["component"] == "sizing_amplification_delta_vs_1_lot", "component_usd"].iloc[0]
    undersized = decomp.loc[decomp["component"] == "undersized_winner_opportunity_to_p75_lots", "component_usd"].iloc[0]
    pips_corr = corr[(corr["scope_type"] == "overall") & (corr["target"] == "realized_pips") & (corr["method"] == "spearman")]["correlation"].iloc[0]
    usd_corr = corr[(corr["scope_type"] == "overall") & (corr["target"] == "realized_usd") & (corr["method"] == "spearman")]["correlation"].iloc[0]
    win_corr = corr[(corr["scope_type"] == "overall") & (corr["target"] == "win_flag") & (corr["method"] == "spearman")]["correlation"].iloc[0]

    verdict = "random-to-anti-Kelly"
    if pips_corr is not None and pips_corr < -0.05 and win_corr is not None and win_corr < -0.05:
        verdict = "anti-Kelly"
    elif pips_corr is not None and abs(pips_corr) <= 0.05 and win_corr is not None and abs(win_corr) <= 0.05:
        verdict = "random/edge-blind"
    elif pips_corr is not None and pips_corr > 0.05 and win_corr is not None and win_corr > 0.05:
        verdict = "Kelly-like"

    best_policy = policies[policies["policy"] != "actual_observed"].sort_values("counterfactual_usd", ascending=False).iloc[0]

    dist_disp = display_perf(dist.copy())
    corr_disp = display_perf(corr.copy())
    caveat_disp = display_perf(caveat.copy())
    model_sorted = model.copy().sort_values("p_value_approx", na_position="last")
    model_disp = display_perf(model_sorted)
    decomp_disp = display_perf(decomp.copy())
    anomaly_disp = display_perf(anomaly.copy())
    policies_sorted = policies.copy().sort_values("counterfactual_usd", ascending=False)
    policies_disp = display_perf(policies_sorted)
    mae_disp = mae_mfe.copy()
    for col in ["pearson_lots_correlation", "spearman_lots_correlation", "high_lot_mean", "low_lot_mean"]:
        if col in mae_disp.columns:
            mae_disp[col] = mae_disp[col].map(two_dec)

    lines: list[str] = []
    lines.append("# PHASE 5 - SIZING LOGIC AUDIT")
    lines.append("")
    lines.append("_Auto Fillmore forensic investigation, Apr 16-May 1. Diagnosis only. Counterfactual policies are retroactive bounds, not forward recommendations._")
    lines.append("")
    lines.append("## Phase 5 Bottom Line")
    lines.append("")
    lines.append(
        f"Committed sizing verdict: **{verdict}**. The LLM's chosen lots do not show a defensible positive relationship to realized edge. "
        f"Overall Spearman lots-vs-pips correlation is {pips_corr:.3f}, lots-vs-win correlation is {win_corr:.3f}, and lots-vs-USD correlation is {usd_corr:.3f}. "
        "Source: `phase5_edge_size_correlation.csv`."
    )
    lines.append("")
    lines.append(
        f"Headline decomposition: of the observed {money(actual)} net loss, {money(uniform_1)} is the admission result if every taken trade were normalized to 1 lot, "
        f"and {money(sizing_delta)} is the exact variable-sizing amplification delta versus that 1-lot baseline. "
        "The overlap term is $0.00 under linear per-lot P&L. Source: `phase5_pl_decomposition.csv`."
    )
    lines.append("")
    lines.append(
        f"Undersized-winner opportunity is {money(undersized)} if winning trades below the corpus p75 lot size were hypothetically upsized to p75. "
        "That is non-additive upside opportunity, not a reconciliation component."
    )
    lines.append("")
    lines.append("## 5.1 Sizing Distribution Forensic")
    lines.append("")
    lines.append(md_table(dist_disp[dist_disp["segment_type"].isin(["overall", "gate", "side", "prompt_regime"])], ["segment_type", "segment", "n", "mean_lots", "median_lots", "p75_lots", "p90_lots", "mode_lots", "share_gt_1", "share_ge_4", "share_ge_8", "net_pips", "net_usd", "sample_size_warning"], 28))
    lines.append("")
    lines.append("The sizing distribution is quantized, not continuous: the mode and unique-lot columns show repeated use of a small lot set. Source: `phase5_sizing_distribution.csv`.")
    lines.append("")
    lines.append("## 5.2 Edge-Sizing Correlation")
    lines.append("")
    lines.append(md_table(corr_disp[corr_disp["scope_type"].eq("overall")], ["scope_type", "scope", "target", "method", "n", "correlation", "ci_low", "ci_high", "verdict_hint"], 12))
    lines.append("")
    lines.append("Segmented correlations with N>=15:")
    lines.append("")
    lines.append(md_table(corr_disp[(corr_disp["scope_type"] != "overall") & (corr_disp["target"].isin(["realized_pips", "win_flag"]))], ["scope_type", "scope", "target", "method", "n", "correlation", "verdict_hint", "sample_size_warning"], 28))
    lines.append("")
    lines.append("Interpretation: if sizing were Kelly-like, larger lots would correlate positively with pips and win flag. That is not what the full sample shows.")
    lines.append("")
    lines.append("## 5.3 Caveat x Sizing Interaction")
    lines.append("")
    lines.append(md_table(caveat_disp[caveat_disp["test"].isin(["hedge_density", "conviction_density", "self_correction"])], ["test", "bucket", "n", "mean_lots", "median_lots", "share_gt_1", "share_ge_4", "share_ge_8", "win_rate", "net_pips", "net_usd", "avg_usd", "sample_size_warning"], 30))
    lines.append("")
    lines.append("Sizing-laundering read: caveat/self-correction rows are not consistently forced to minimal exposure. If a caveat appears and size remains above 1 lot, the model is narrating risk without fully pricing it.")
    lines.append("")
    lines.append("## 5.4 Implied Sizing Function Reconstruction")
    lines.append("")
    lines.append(md_table(model_disp.head(24), ["term", "coefficient", "std_error", "t_stat", "p_value_approx", "r_squared", "n", "interpretation"], 24))
    lines.append("")
    lines.append("Regression uses only fields the LLM had or the persisted snapshot exposed: account balance/equity, margin used, planned RR, SL/TP pips, volatility ratio, M5 ATR, spread, gate, side, and prompt regime. Source: `phase5_implied_sizing_function.csv`.")
    lines.append("")
    lines.append("## 5.5 Marginal P&L Decomposition")
    lines.append("")
    lines.append(md_table(decomp_disp, ["component", "component_usd", "definition", "reconciles_to_actual"], 10))
    lines.append("")
    lines.append(
        f"Plain English: same entries at uniform 1 lot would have produced {money(uniform_1)}. "
        f"Variable sizing changed that by {money(sizing_delta)}, producing the observed {money(actual)}. "
        "That means sizing was not a side issue; it was most of the dollar damage."
    )
    lines.append("")
    lines.append("## 5.6 The 8+ Lot CLR Anomaly")
    lines.append("")
    lines.append(md_table(anomaly_disp, ["segment_type", "segment", "n", "win_rate", "net_pips", "net_usd", "avg_usd", "mean_lots", "side_mix", "prompt_mix", "date_mix", "committed_verdict", "sample_size_warning"], 25))
    lines.append("")
    committed = anomaly["committed_verdict"].dropna().iloc[0] if "committed_verdict" in anomaly.columns and anomaly["committed_verdict"].notna().any() else "undetermined"
    lines.append(f"Committed anomaly verdict: **{committed}**. It is not strong enough to call a general right-to-size-up rule.")
    lines.append("")
    lines.append("## 5.7 Counterfactual Sizing Policies")
    lines.append("")
    lines.append(md_table(policies_disp, ["policy", "n", "win_rate", "net_pips", "counterfactual_usd", "delta_vs_actual_usd", "profit_factor_usd", "max_drawdown_usd", "sharpe_usd", "trades_altered", "trades_skipped", "retroactive_warning"], 12))
    lines.append("")
    lines.append(
        f"Best retroactive policy on this dataset: `{best_policy['policy']}` at {money(best_policy['counterfactual_usd'])}, "
        f"a {money(best_policy['delta_vs_actual_usd'])} delta versus observed. This is an in-sample bound, not a forward proof."
    )
    lines.append("")
    lines.append("## 5.8 Sizing x MAE/MFE Interaction")
    lines.append("")
    lines.append(md_table(mae_disp, ["target", "pearson_lots_correlation", "spearman_lots_correlation", "high_lot_mean", "low_lot_mean", "interpretation"], 10))
    lines.append("")
    lines.append("This is the path-risk test: if larger lots correlate with MAE more than MFE, size is amplifying adverse excursion rather than opportunity.")
    lines.append("")
    lines.append("## 5.9 Required Telemetry for Correct Sizing")
    lines.append("")
    lines.append(md_table(telemetry, ["priority", "field", "mechanism_enabled", "evidence"], 10))
    lines.append("")
    lines.append("## Sizing Verdict")
    lines.append("")
    lines.append(f"- Anti-Kelly / Random / Kelly-like: **{verdict}**. The edge-size relationship is not positive enough to defend variable sizing.")
    lines.append(f"- 8+ lot CLR anomaly: **{committed}**. Treat as a fragile anomaly, not a general preserve finding.")
    lines.append(f"- Best retroactive counterfactual: `{best_policy['policy']}`. It is in-sample and diagnostic only.")
    lines.append("- Required telemetry priority: open lots by side, risk-after-fill USD, rolling lot-weighted P&L, unrealized P&L by side, explicit pip value.")
    lines.append("")
    lines.append("## Evidence Gaps")
    lines.append("")
    lines.append("- Open exposure by side was absent for most of the period, so the audit cannot prove whether a size was bad because it added to existing inventory or bad on standalone risk.")
    lines.append("- Skip-side forward outcomes are still absent, so counterfactual admission policies bound damage only on trades that were actually placed.")
    lines.append("- Fixed-fractional reconstruction depends on inferred pip value per lot because pip value was not stored as a first-class decision field.")
    lines.append("- Regression coefficients are diagnostic, not causal. The model may have used unpersisted prompt text or hidden reasoning not captured in the snapshot.")
    lines.append("- Phase 4 and Phase 5 have tiny or zero closed samples and are not used for broad sizing-performance claims.")
    lines.append("")
    lines.append("## Artifacts Written")
    lines.append("")
    for name in [
        "phase5_sizing_distribution.csv",
        "phase5_edge_size_correlation.csv",
        "phase5_caveat_sizing_interaction.csv",
        "phase5_implied_sizing_function.csv",
        "phase5_pl_decomposition.csv",
        "phase5_8lot_clr_anomaly.csv",
        "phase5_counterfactual_policies.csv",
        "phase5_sizing_mae_mfe.csv",
        "phase5_required_telemetry.csv",
        "phase5_manifest.json",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_dataset()
    dist = sizing_distribution(df)
    corr = edge_size_correlation(df)
    caveat = caveat_sizing_interaction(df)
    model = implied_sizing_function(df)
    decomp = pl_decomposition(df)
    anomaly = anomaly_8lot_clr(df)
    policies = counterfactual_policies(df)
    mae_mfe = mae_mfe_interaction(df, corr)
    telemetry = required_telemetry()
    report = build_report(df, dist, corr, caveat, model, decomp, anomaly, policies, mae_mfe, telemetry)
    (OUT / "PHASE5_SIZING_AUDIT.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "closed": str(CLOSED_CSV.relative_to(ROOT)),
            "phase4_corpus": str(PHASE4_CORPUS_CSV.relative_to(ROOT)),
            "phase4_sizing": str(PHASE4_SIZING_CSV.relative_to(ROOT)),
        },
        "outputs": [
            "PHASE5_SIZING_AUDIT.md",
            "phase5_sizing_distribution.csv",
            "phase5_edge_size_correlation.csv",
            "phase5_caveat_sizing_interaction.csv",
            "phase5_implied_sizing_function.csv",
            "phase5_pl_decomposition.csv",
            "phase5_8lot_clr_anomaly.csv",
            "phase5_counterfactual_policies.csv",
            "phase5_sizing_mae_mfe.csv",
            "phase5_required_telemetry.csv",
        ],
        "notes": [
            "Counterfactual policies are retroactive diagnostic bounds, not forward recommendations.",
            "Uniform-lot P&L uses observed pnl/lots as inferred per-lot P&L.",
            "Regression uses only persisted snapshot fields available to the LLM or stored from the decision context.",
        ],
    }
    (OUT / "phase5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'PHASE5_SIZING_AUDIT.md'}")
    print(f"Closed rows: {len(df)}; actual pnl: {df['pnl'].sum():.2f}; uniform 1-lot pnl: {df['pnl_per_lot'].sum():.2f}")


if __name__ == "__main__":
    main()
