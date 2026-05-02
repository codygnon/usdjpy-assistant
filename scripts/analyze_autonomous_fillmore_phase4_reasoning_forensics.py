#!/usr/bin/env python3
"""Phase 4 reasoning forensics for Autonomous Fillmore.

This harness is intentionally read-only over source/evidence inputs. It builds
diagnostic CSVs and a markdown report from the existing forensic sidecar,
closed-trade snapshot table, and prior reasoning dossier.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORENSIC_DIR = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
REASONING_DIR = ROOT / "research_out" / "autonomous_fillmore_reasoning_deep_dive_20260501"

SIDECAR_CSV = FORENSIC_DIR / "phase3_snapshot_flattened_sidecar.csv"
CLOSED_CSV = FORENSIC_DIR / "phase3_closed_trades_with_snapshot_fields.csv"
FAST_FAILURE_CSV = FORENSIC_DIR / "phase3_clr_level_failure_17.csv"
PRIOR_DOSSIER_CSV = REASONING_DIR / "reasoning_trade_dossier.csv"
PRIOR_COUNTERFACTUALS_CSV = REASONING_DIR / "reasoning_counterfactuals.csv"
PRIOR_PHRASES_CSV = REASONING_DIR / "reasoning_phrase_impact.csv"
PROMPT_PERF_CSV = ROOT / "research_out" / "autonomous_fillmore_change_timeline_20260430" / "prompt_version_performance.csv"

OUT = FORENSIC_DIR


PROMPT_REGIME = {
    "autonomous_phase_a_v1": "Phase A baseline",
    "autonomous_phase2_zone_memory_custom_exit_v1": "Phase 2 zone-memory",
    "autonomous_phase2_runner_custom_exit_v3": "Phase 2 runner/custom-exit v3",
    "autonomous_phase3_house_edge_v1": "Phase 3 house-edge",
    "autonomous_phase4_selectivity_sizing_v1": "Phase 4 selectivity/sizing",
    "autonomous_phase5_reasoning_quality_v1": "Phase 5 reasoning-quality",
}

HEDGE_WORDS = {
    "might",
    "may",
    "could",
    "possibly",
    "seems",
    "appears",
    "likely",
    "maybe",
    "somewhat",
    "marginal",
    "thin",
    "tentative",
    "modest",
    "barely",
    "mixed",
    "conflicting",
}
CONVICTION_WORDS = {
    "clear",
    "strong",
    "obvious",
    "decisive",
    "textbook",
    "clean",
    "confirmed",
    "fresh",
    "valid",
    "material",
    "robust",
    "aligned",
    "high-conviction",
    "direct",
}
SELF_CORRECTION = {
    "however",
    "but",
    "although",
    "despite",
    "even though",
}
INDICATOR_TERMS = {
    "rsi",
    "macd",
    "ema",
    "adx",
    "adxr",
    "atr",
    "vwap",
    "order book",
    "cluster",
    "dxy",
    "oil",
    "jpy",
    "h1",
    "m15",
    "m5",
    "m1",
}
LEVEL_TERMS = {
    "support",
    "resistance",
    "reclaim",
    "reclaimed",
    "reject",
    "rejected",
    "level",
    "half-yen",
    "whole-yen",
    "cluster",
    "round",
}
MACRO_TERMS = {
    "boj",
    "mof",
    "intervention",
    "policy",
    "macro",
    "headline",
    "yield",
    "treasury",
    "dxy",
    "oil",
    "war",
    "risk-off",
    "risk on",
    "risk-off",
    "jpy strength",
    "jpy weakness",
}
MICRO_TERMS = {
    "m1",
    "m3",
    "m5",
    "micro",
    "retest",
    "wick",
    "close",
    "higher low",
    "lower high",
    "impulse",
    "acceptance",
    "failure",
}

NUMERIC_RE = re.compile(r"\b(?:1[4-7]\d\.\d{2,3}|\d+(?:\.\d+)?p)\b", re.I)
PRICE_RE = re.compile(r"\b1[4-7]\d\.\d{2,3}\b")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


def s(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def lower_text(*parts: Any) -> str:
    return " ".join(s(p) for p in parts if s(p).strip()).lower()


def text_has(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def count_terms(text: str, terms: set[str]) -> int:
    return sum(text.count(term) for term in terms)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.2f}"


def one_dec(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two_dec(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def compact_text(text: Any, n: int = 180) -> str:
    t = re.sub(r"\s+", " ", s(text)).strip()
    if len(t) <= n:
        return t
    return t[: n - 3].rstrip() + "..."


def escape_md(value: Any) -> str:
    t = s(value)
    t = t.replace("\n", "<br>")
    t = t.replace("|", "\\|")
    return t


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


def perf(group: pd.DataFrame) -> dict[str, Any]:
    closed = group[group["pips"].notna()].copy()
    wins = closed[closed["pips"] > 0]
    losses = closed[closed["pips"] < 0]
    gross_win_usd = wins["pnl"].clip(lower=0).sum()
    gross_loss_usd = abs(losses["pnl"].clip(upper=0).sum())
    return {
        "calls": len(group),
        "placed": int((group["decision_bucket"] == "placed").sum()),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if len(closed) else None,
        "net_pips": closed["pips"].sum() if len(closed) else 0.0,
        "net_usd": closed["pnl"].sum() if len(closed) else 0.0,
        "usd_profit_factor": (gross_win_usd / gross_loss_usd) if gross_loss_usd else (math.inf if gross_win_usd else None),
        "avg_winner_pips": wins["pips"].mean() if len(wins) else None,
        "avg_loser_pips": losses["pips"].mean() if len(losses) else None,
        "avg_winner_usd": wins["pnl"].mean() if len(wins) else None,
        "avg_loser_usd": losses["pnl"].mean() if len(losses) else None,
        "expectancy_pips": closed["pips"].mean() if len(closed) else None,
        "expectancy_usd": closed["pnl"].mean() if len(closed) else None,
        "mae_p75": closed["mae_abs"].quantile(0.75) if closed["mae_abs"].notna().any() else None,
        "mfe_p75": closed["mfe"].quantile(0.75) if closed["mfe"].notna().any() else None,
        "avg_lots": closed["lots"].mean() if closed["lots"].notna().any() else None,
    }


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(by, keys)}
        row.update(perf(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["net_usd", "net_pips"], ascending=[True, True])


def display_perf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["win_rate"]:
        if col in out.columns:
            out[col] = out[col].map(pct)
    for col in ["net_usd", "avg_winner_usd", "avg_loser_usd", "expectancy_usd"]:
        if col in out.columns:
            out[col] = out[col].map(money)
    for col in ["net_pips", "avg_winner_pips", "avg_loser_pips", "expectancy_pips", "mae_p75", "mfe_p75", "avg_lots", "usd_profit_factor"]:
        if col in out.columns:
            out[col] = out[col].map(two_dec)
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sidecar = pd.read_csv(SIDECAR_CSV, low_memory=False)
    closed = pd.read_csv(CLOSED_CSV, low_memory=False)
    prior = pd.read_csv(PRIOR_DOSSIER_CSV, low_memory=False)
    fast = pd.read_csv(FAST_FAILURE_CSV, low_memory=False)
    return sidecar, closed, prior, fast


def add_suffix(df: pd.DataFrame, cols: list[str], suffix: str, keys: list[str]) -> pd.DataFrame:
    keep = [c for c in keys + cols if c in df.columns]
    out = df[keep].copy()
    rename = {c: f"{c}{suffix}" for c in cols if c in out.columns}
    return out.rename(columns=rename)


def choose_first(df: pd.DataFrame, cols: list[str], default: Any = "") -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([default] * len(df), index=df.index)
    result = df[existing[0]]
    for col in existing[1:]:
        result = result.where(result.notna() & (result.astype(str).str.strip() != ""), df[col])
    return result.fillna(default)


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


def build_corpus(sidecar: pd.DataFrame, closed: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    base = sidecar.copy()
    if "trade_id" in sidecar.columns and "trade_id" in closed.columns:
        present_trade_ids = set(sidecar["trade_id"].dropna().astype(str))
        missing_closed = closed[
            closed["trade_id"].notna()
            & ~closed["trade_id"].astype(str).isin(present_trade_ids)
        ].copy()
        if not missing_closed.empty:
            additions = pd.DataFrame(index=missing_closed.index, columns=base.columns)
            direct_cols = [
                "profile",
                "suggestion_id",
                "created_utc",
                "prompt_version",
                "prompt_regime",
                "side",
                "lots",
                "trigger_family",
                "gate_group",
                "trigger_reason",
                "trade_id",
                "filled_at",
                "closed_at",
            ]
            for col in direct_cols:
                if col in additions.columns and col in missing_closed.columns:
                    additions[col] = missing_closed[col]
            if "action" in additions.columns:
                additions["action"] = "placed"
            if "decision" in additions.columns:
                additions["decision"] = "trade"
            if "pips_raw" in additions.columns and "pips" in missing_closed.columns:
                additions["pips_raw"] = missing_closed["pips"]
            if "pnl_raw" in additions.columns and "pnl" in missing_closed.columns:
                additions["pnl_raw"] = missing_closed["pnl"]
            for src, dest in [
                ("trade_thesis", "features.edge_reason"),
                ("named_catalyst", "features.named_catalyst"),
                ("exit_strategy", "features.exit_strategy"),
                ("timeframe_alignment", "features.confidence"),
            ]:
                if src in missing_closed.columns and dest in additions.columns:
                    additions[dest] = missing_closed[src]
            base = pd.concat([base, additions], ignore_index=True)

    prior_cols = [
        "rationale",
        "analysis_text",
        "trade_thesis",
        "named_catalyst",
        "side_bias_check",
        "why_trade_despite_weakness",
        "pips",
        "pnl",
        "win_loss",
        "catalyst_score",
        "green_match_count",
        "weakness_signals",
        "reasoning_risk_score",
        "reasoning_flags",
    ]
    prior_by_suggestion = add_suffix(prior.drop_duplicates("suggestion_id"), prior_cols, "_prior", ["suggestion_id"])
    base = base.merge(prior_by_suggestion, on="suggestion_id", how="left")

    closed_cols = [
        "pips",
        "pnl",
        "win_loss",
        "rationale",
        "trade_thesis",
        "named_catalyst",
        "prompt_version",
        "side",
        "lots",
        "trigger_family",
        "trigger_reason",
        "timeframe_alignment",
        "trigger_fit",
        "conviction_rung",
        "exit_strategy",
        "max_adverse_pips",
        "max_favorable_pips",
        "mae_abs",
        "mfe",
        "prompt_regime",
    ]
    closed_keyed = closed[closed["trade_id"].notna()].drop_duplicates("trade_id")
    closed_by_trade = add_suffix(closed_keyed, closed_cols, "_closed", ["trade_id"])
    base = base.merge(closed_by_trade, on="trade_id", how="left")

    base["pips"] = to_num(choose_first(base, ["pips_raw", "pips_closed", "pips_prior"], default=math.nan))
    base["pnl"] = to_num(choose_first(base, ["pnl_raw", "pnl_closed", "pnl_prior"], default=math.nan))
    # The forensic baseline is the closed-trade table from Phase 1/3. A sidecar
    # row can contain stale P&L for a trade that is not in that authoritative
    # 241-row population; keep it as a placed/call row, but do not let it alter
    # closed-trade math.
    valid_closed_ids = set(closed["trade_id"].dropna().astype(str)) if "trade_id" in closed.columns else set()
    if valid_closed_ids and "trade_id" in base.columns:
        outside_closed_baseline = base["pips"].notna() & ~base["trade_id"].astype(str).isin(valid_closed_ids)
        base.loc[outside_closed_baseline, ["pips", "pnl"]] = math.nan
    base["lots"] = to_num(base["lots"]) if "lots" in base.columns else math.nan
    base["mae_abs"] = to_num(choose_first(base, ["mae_abs_closed", "max_adverse_pips_closed"], default=math.nan)).abs()
    base["mfe"] = to_num(choose_first(base, ["mfe_closed", "max_favorable_pips_closed"], default=math.nan))
    base["win_loss"] = choose_first(base, ["win_loss_closed", "win_loss_prior"], default="")
    base["prompt_version"] = choose_first(base, ["prompt_version", "prompt_version_closed"], default="")
    base["prompt_regime"] = choose_first(base, ["prompt_regime", "prompt_regime_closed"], default="")
    base["prompt_regime"] = base.apply(
        lambda r: s(r["prompt_regime"]).strip() or PROMPT_REGIME.get(s(r.get("prompt_version")), s(r.get("prompt_version")) or "unversioned/unknown"),
        axis=1,
    )
    base["gate_group"] = choose_first(base, ["gate_group", "trigger_family"], default="")
    base["trigger_family"] = choose_first(base, ["trigger_family", "trigger_family_closed", "features.trigger_family"], default="")
    base["trigger_reason"] = choose_first(base, ["trigger_reason", "trigger_reason_closed", "features.trigger_reason"], default="")
    base["timeframe_alignment"] = choose_first(base, ["timeframe_alignment_closed"], default="")
    base["side"] = choose_first(base, ["side", "side_closed", "features.side"], default="").str.lower()
    base["lots"] = to_num(choose_first(base, ["lots", "lots_closed"], default=math.nan))

    text_cols = [
        "rationale_closed",
        "rationale_prior",
        "analysis_text_prior",
        "trade_thesis_closed",
        "trade_thesis_prior",
        "named_catalyst_closed",
        "named_catalyst_prior",
        "features.named_catalyst",
        "side_bias_check_prior",
        "features.side_bias_check",
        "why_trade_despite_weakness_prior",
        "features.edge_reason",
        "features.setup_location",
        "features.adverse_context",
        "features.caveat_resolution",
        "features.micro_confirmation_event",
        "features.reasoning_quality_gate",
        "features.phase5_reasoning_flags",
    ]
    for col in text_cols:
        if col not in base.columns:
            base[col] = ""
    base["reasoning_text"] = base[text_cols].apply(lambda r: " ".join(s(v) for v in r.values if s(v).strip()), axis=1)
    base["rationale_excerpt"] = base["reasoning_text"].map(lambda x: compact_text(x, 260))

    base["decision_bucket"] = "other_call"
    base.loc[(base.get("action", "") == "placed") | (base.get("decision", "") == "trade") | base["pips"].notna(), "decision_bucket"] = "placed"
    base.loc[base.get("decision", "") == "skip", "decision_bucket"] = "skipped"
    base["is_closed"] = base["pips"].notna()
    base["outcome"] = ""
    base.loc[base["pips"] > 0, "outcome"] = "win"
    base.loc[base["pips"] < 0, "outcome"] = "loss"
    base.loc[base["pips"] == 0, "outcome"] = "flat"
    base["lot_bucket"] = base["lots"].map(lot_bucket)

    return base


def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    texts = out["reasoning_text"].fillna("").astype(str)
    lower = texts.str.lower()
    out["token_count"] = texts.map(lambda x: len(WORD_RE.findall(x)))
    out["sentence_count"] = texts.map(lambda x: max(1, len([p for p in re.split(r"[.!?]+", x) if p.strip()])) if x.strip() else 0)
    out["hedge_word_count"] = lower.map(lambda x: count_terms(x, HEDGE_WORDS))
    out["conviction_word_count"] = lower.map(lambda x: count_terms(x, CONVICTION_WORDS))
    out["self_correction_count"] = lower.map(lambda x: count_terms(x, SELF_CORRECTION))
    out["numeric_claim_count"] = texts.map(lambda x: len(NUMERIC_RE.findall(x)))
    out["indicator_name_count"] = lower.map(lambda x: count_terms(x, INDICATOR_TERMS))
    out["level_term_count"] = lower.map(lambda x: count_terms(x, LEVEL_TERMS))
    out["macro_term_count"] = lower.map(lambda x: count_terms(x, MACRO_TERMS))
    out["micro_term_count"] = lower.map(lambda x: count_terms(x, MICRO_TERMS))
    denom = out["token_count"].replace(0, pd.NA)
    out["hedge_density"] = (out["hedge_word_count"] / denom).fillna(0.0)
    out["conviction_density"] = (out["conviction_word_count"] / denom).fillna(0.0)
    out["self_correction_density"] = (out["self_correction_count"] / denom).fillna(0.0)
    out["has_hedge"] = out["hedge_word_count"] > 0
    out["has_conviction"] = out["conviction_word_count"] > 0
    out["has_self_correction"] = out["self_correction_count"] > 0
    return out


def cluster_reasoning(row: pd.Series) -> str:
    text = lower_text(row.get("reasoning_text"))
    family = s(row.get("trigger_family")).lower()
    tf = s(row.get("features.h1_regime")).lower() + " " + s(row.get("features.m5_regime")).lower() + " " + s(row.get("features.m1_regime")).lower()
    flags = lower_text(row.get("reasoning_flags_prior"), row.get("features.phase5_reasoning_flags"), row.get("weakness_signals_prior"))
    has_caveat = ("contradiction_admitted" in flags) or text_has(text, SELF_CORRECTION)
    is_mixed = "mixed" in text or "mixed" in flags or "mixed" in s(row.get("timeframe_alignment")).lower()

    if not text.strip():
        return "no_reasoning_text"
    if "critical_level_reaction" in family and is_mixed and has_caveat and row.get("decision_bucket") == "placed":
        return "critical_level_mixed_caveat_trade"
    if "critical_level_reaction" in family and text_has(text, LEVEL_TERMS):
        if text_has(text, MICRO_TERMS):
            return "critical_level_reclaim_reject_micro"
        return "critical_level_generic_reclaim_reject"
    if "momentum" in family or any(k in text for k in ["momentum", "ema", "impulse", "continuation", "higher low", "lower high", "pullback"]):
        if has_caveat and row.get("decision_bucket") == "placed":
            return "momentum_with_caveat_trade"
        return "momentum_ema_retest_continuation"
    if text_has(text, MACRO_TERMS):
        if has_caveat and row.get("decision_bucket") == "placed":
            return "macro_policy_caveat_trade"
        return "macro_policy_context_override"
    if text_has(text, LEVEL_TERMS):
        if text_has(text, MICRO_TERMS):
            return "level_reclaim_reject_with_micro_claim"
        return "generic_level_reclaim_reject"
    if any(k in text for k in ["base rate", "sell-side", "buy-side", "not generic"]):
        return "base_rate_side_bias_argument"
    if any(k in text for k in ["runner", "trail", "custom exit", "tp1", "partial"]):
        return "runner_custom_exit_argument"
    if "mean_reversion" in family or any(k in text for k in ["mean reversion", "range", "fade", "compressed"]):
        return "mean_reversion_range_fade"
    if any(k in text for k in ["fresh", "zone", "retry", "fingerprint", "memory"]):
        return "fresh_zone_memory_argument"
    if has_caveat and row.get("decision_bucket") == "placed":
        return "caveat_laundering_despite_trade"
    if any(k in tf + " " + text for k in ["aligned", "trend", "bull", "bear"]):
        return "plain_trend_alignment"
    return "other_reasoning"


def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rationale_cluster"] = out.apply(cluster_reasoning, axis=1)
    return out


def bin_density(series: pd.Series, label: str) -> pd.Series:
    out = pd.Series(["zero"] * len(series), index=series.index)
    positive = series[series > 0]
    if positive.empty:
        return out
    q1 = positive.quantile(0.5)
    q2 = positive.quantile(0.8)
    out.loc[(series > 0) & (series <= q1)] = f"low_{label}"
    out.loc[(series > q1) & (series <= q2)] = f"mid_{label}"
    out.loc[series > q2] = f"high_{label}"
    return out


def write_corpus_outputs(corpus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    corpus_cols = [
        "created_utc",
        "profile",
        "prompt_version",
        "prompt_regime",
        "suggestion_id",
        "trade_id",
        "decision_bucket",
        "side",
        "lots",
        "lot_bucket",
        "pips",
        "pnl",
        "outcome",
        "trigger_family",
        "gate_group",
        "trigger_reason",
        "features.h1_regime",
        "features.m5_regime",
        "features.m1_regime",
        "features.macro_bias",
        "features.planned_rr",
        "features.reasoning_quality_gate",
        "features.phase5_reasoning_flags",
        "rationale_cluster",
        "token_count",
        "sentence_count",
        "hedge_word_count",
        "conviction_word_count",
        "self_correction_count",
        "numeric_claim_count",
        "indicator_name_count",
        "level_term_count",
        "macro_term_count",
        "micro_term_count",
        "hedge_density",
        "conviction_density",
        "self_correction_density",
        "mae_abs",
        "mfe",
        "rationale_excerpt",
    ]
    corpus_out = corpus[[c for c in corpus_cols if c in corpus.columns]].copy()
    corpus_out.to_csv(OUT / "phase4_reasoning_corpus.csv", index=False)

    clusters = summarize(corpus, ["rationale_cluster"])
    clusters["sample_size_warning"] = clusters["closed"].map(lambda n: "N<15" if n < 15 else "")
    clusters.to_csv(OUT / "phase4_rationale_clusters.csv", index=False)

    sell_buy = corpus[
        (corpus["is_closed"])
        & (corpus["trigger_family"].fillna("").str.lower() == "critical_level_reaction")
        & (corpus["side"].isin(["buy", "sell"]))
    ].copy()
    comp = summarize(sell_buy, ["side"])
    ling = sell_buy.groupby("side", dropna=False).agg(
        token_count_mean=("token_count", "mean"),
        hedge_density_mean=("hedge_density", "mean"),
        conviction_density_mean=("conviction_density", "mean"),
        self_correction_rate=("has_self_correction", "mean"),
        numeric_claim_mean=("numeric_claim_count", "mean"),
        level_term_mean=("level_term_count", "mean"),
        macro_term_mean=("macro_term_count", "mean"),
        avg_lots_all=("lots", "mean"),
    ).reset_index()
    comp = comp.merge(ling, on="side", how="left")
    comp.to_csv(OUT / "phase4_sell_clr_vs_buy_clr.csv", index=False)

    cluster_side = summarize(sell_buy, ["side", "rationale_cluster"])
    cluster_side.to_csv(OUT / "phase4_sell_buy_clr_cluster_mix.csv", index=False)

    place_skip = corpus[corpus["decision_bucket"].isin(["placed", "skipped"])].copy()
    ps_summary = place_skip.groupby(["decision_bucket", "trigger_family"], dropna=False).agg(
        calls=("suggestion_id", "count"),
        avg_token_count=("token_count", "mean"),
        avg_hedge_density=("hedge_density", "mean"),
        avg_conviction_density=("conviction_density", "mean"),
        self_correction_rate=("has_self_correction", "mean"),
        avg_numeric_claims=("numeric_claim_count", "mean"),
        avg_level_terms=("level_term_count", "mean"),
    ).reset_index()
    ps_summary.to_csv(OUT / "phase4_place_skip_reasoning_comparison.csv", index=False)

    sizing = summarize(corpus[corpus["is_closed"]], ["lot_bucket", "rationale_cluster"])
    sizing_ling = corpus[corpus["is_closed"]].groupby(["lot_bucket", "rationale_cluster"], dropna=False).agg(
        avg_conviction_density=("conviction_density", "mean"),
        avg_hedge_density=("hedge_density", "mean"),
        explicit_size_language_rate=("reasoning_text", lambda x: sum(bool(re.search(r"\b(lot|size|risk|reduced|small|max|cap|exposure|margin)\b", s(v).lower())) for v in x) / len(x) if len(x) else 0),
    ).reset_index()
    sizing = sizing.merge(sizing_ling, on=["lot_bucket", "rationale_cluster"], how="left")
    sizing.to_csv(OUT / "phase4_sizing_reasoning_audit.csv", index=False)

    destructive = summarize(
        corpus[corpus["is_closed"]],
        ["rationale_cluster", "trigger_family", "side", "prompt_regime", "lot_bucket"],
    )
    destructive = destructive[
        (destructive["closed"] >= 10)
        & ((destructive["net_usd"] < 0) | (destructive["net_pips"] < 0))
    ].sort_values("net_usd")
    destructive.to_csv(OUT / "phase4_destructive_cells.csv", index=False)

    corpus["created_date"] = pd.to_datetime(corpus["created_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    drift = corpus.groupby(["created_date", "prompt_regime"], dropna=False).agg(
        calls=("suggestion_id", "count"),
        placed=("decision_bucket", lambda x: int((x == "placed").sum())),
        skipped=("decision_bucket", lambda x: int((x == "skipped").sum())),
        closed=("pips", lambda x: int(x.notna().sum())),
        net_pips=("pips", "sum"),
        net_usd=("pnl", "sum"),
        avg_hedge_density=("hedge_density", "mean"),
        avg_conviction_density=("conviction_density", "mean"),
        self_correction_rate=("has_self_correction", "mean"),
    ).reset_index()
    drift.to_csv(OUT / "phase4_reasoning_drift_by_date.csv", index=False)

    return {
        "corpus": corpus_out,
        "clusters": clusters,
        "sell_buy": comp,
        "cluster_side": cluster_side,
        "place_skip": ps_summary,
        "sizing": sizing,
        "destructive": destructive,
        "drift": drift,
    }


def fast_failure_audit(corpus: pd.DataFrame, fast: pd.DataFrame) -> pd.DataFrame:
    keyed = corpus[corpus["trade_id"].notna()].drop_duplicates("trade_id")
    cols = [
        "trade_id",
        "prompt_regime",
        "rationale_cluster",
        "reasoning_text",
        "token_count",
        "hedge_word_count",
        "conviction_word_count",
        "self_correction_count",
        "numeric_claim_count",
    ]
    merged = fast.merge(keyed[[c for c in cols if c in keyed.columns]], on="trade_id", how="left")

    def overconf(row: pd.Series) -> str:
        evidence = s(row.get("snapshot_level_evidence_read")).lower()
        text = s(row.get("reasoning_text")).lower()
        strong_words = any(w in text for w in ["strong", "clean", "textbook", "confirmed", "fresh", "material", "direct"])
        weak_evidence = any(w in evidence for w in ["thin", "weak", "mixed", "ambiguous"])
        if strong_words and weak_evidence:
            return "overconfident_vs_weak_snapshot"
        if strong_words:
            return "strong_claim"
        if weak_evidence:
            return "weak_snapshot_no_strong_claim"
        return "neutral"

    merged["rationale_evidence_claim"] = merged.apply(overconf, axis=1)
    merged["rationale_excerpt"] = merged["reasoning_text"].map(lambda x: compact_text(x, 240))
    out_cols = [
        "trade_id",
        "created_utc",
        "side",
        "lots",
        "pips",
        "pnl",
        "mae_abs",
        "mfe",
        "snapshot_level_evidence_read",
        "prompt_regime",
        "rationale_cluster",
        "rationale_evidence_claim",
        "hedge_word_count",
        "conviction_word_count",
        "self_correction_count",
        "numeric_claim_count",
        "rationale_excerpt",
    ]
    out = merged[[c for c in out_cols if c in merged.columns]].copy()
    out.to_csv(OUT / "phase4_clr_fast_failure_rationale_audit.csv", index=False)
    return out


def cognitive_tests(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []

    ambiguous = (
        corpus["reasoning_text"].fillna("").str.lower().str.contains("mixed|conflicting|contradict|despite|however|but|although", regex=True)
        | corpus.get("features.phase5_reasoning_flags", pd.Series("", index=corpus.index)).fillna("").str.lower().str.contains("contradiction|critical_level_mixed", regex=True)
        | corpus.get("features.phase4_weakness_signals", pd.Series("", index=corpus.index)).fillna("").astype(str).str.len().gt(2)
    )
    for label, mask in [("ambiguous_or_contradictory_snapshot_reasoning", ambiguous), ("clean_or_unflagged_snapshot_reasoning", ~ambiguous)]:
        g = corpus[mask]
        p = perf(g)
        p.update(
            test="confirmation_bias",
            bucket=label,
            place_rate=(g["decision_bucket"].eq("placed").mean() if len(g) else None),
        )
        rows.append(p)

    tmp = corpus.copy()
    tmp["hedge_bin"] = bin_density(tmp["hedge_density"], "hedge")
    for bucket, g in tmp.groupby("hedge_bin", dropna=False):
        p = perf(g)
        p.update(test="hedging_as_tell", bucket=bucket, place_rate=g["decision_bucket"].eq("placed").mean())
        rows.append(p)

    tmp["conviction_bin"] = bin_density(tmp["conviction_density"], "conviction")
    for bucket, g in tmp.groupby("conviction_bin", dropna=False):
        p = perf(g)
        p.update(test="overconfidence_as_tell", bucket=bucket, place_rate=g["decision_bucket"].eq("placed").mean())
        rows.append(p)

    for term_name, terms in [
        ("support_resistance_level_anchor", LEVEL_TERMS),
        ("macro_policy_anchor", MACRO_TERMS),
        ("micro_confirmation_anchor", MICRO_TERMS),
        ("indicator_anchor", INDICATOR_TERMS),
    ]:
        mask = corpus["reasoning_text"].fillna("").str.lower().map(lambda x: text_has(x, terms))
        g = corpus[mask]
        p = perf(g)
        p.update(test="anchoring", bucket=term_name, place_rate=g["decision_bucket"].eq("placed").mean() if len(g) else None)
        rows.append(p)

    correction = corpus["has_self_correction"]
    for label, mask in [("self_correction_present", correction), ("self_correction_absent", ~correction)]:
        g = corpus[mask]
        p = perf(g)
        p.update(test="self_correction_followthrough", bucket=label, place_rate=g["decision_bucket"].eq("placed").mean() if len(g) else None)
        rows.append(p)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase4_cognitive_failure_tests.csv", index=False)
    return out


def snapshot_text_for_row(row: pd.Series) -> str:
    pieces = []
    for col, value in row.items():
        if col.startswith("market_snapshot.") or col.startswith("features."):
            pieces.append(s(value))
    return " ".join(pieces)


def extract_snapshot_prices(text: str) -> list[float]:
    values = []
    for m in PRICE_RE.findall(text):
        try:
            values.append(float(m))
        except Exception:
            pass
    return values


def hallucination_audit(corpus: pd.DataFrame) -> pd.DataFrame:
    placed = corpus[(corpus["decision_bucket"] == "placed") & corpus["reasoning_text"].fillna("").str.strip().ne("")].copy()
    placed = placed.sort_values("created_utc")
    sample_parts = []
    for _, group in placed.groupby(["prompt_regime", "trigger_family", "side"], dropna=False):
        if len(group):
            sample_parts.append(group.sample(n=1, random_state=42))
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if len(sample) < 40 and len(placed) > len(sample):
        remaining = placed.drop(index=sample.index, errors="ignore")
        sample = pd.concat([sample, remaining.sample(n=min(40 - len(sample), len(remaining)), random_state=7)], ignore_index=True)
    sample = sample.head(40).copy()

    rows = []
    for _, row in sample.iterrows():
        reasoning = s(row.get("reasoning_text"))
        snapshot_text = snapshot_text_for_row(row)
        reason_prices = []
        for m in PRICE_RE.findall(reasoning):
            try:
                reason_prices.append(float(m))
            except Exception:
                pass
        snap_prices = extract_snapshot_prices(snapshot_text)
        unsupported_prices = []
        supported_prices = []
        for price in reason_prices:
            if any(abs(price - sp) <= 0.035 for sp in snap_prices):
                supported_prices.append(price)
            else:
                unsupported_prices.append(price)
        lower = reasoning.lower()
        m15_claim = "m15" in lower
        candle_claim = any(k in lower for k in ["candle", "wick", "sweep"])
        indicator_claim = any(k in lower for k in INDICATOR_TERMS)
        level_claim = any(k in lower for k in LEVEL_TERMS)

        if unsupported_prices:
            verdict = "unsupported_in_persisted_snapshot"
        elif m15_claim or candle_claim:
            verdict = "unverifiable_from_persisted_snapshot"
        elif reason_prices or indicator_claim or level_claim:
            verdict = "supported_or_snapshot-consistent"
        else:
            verdict = "no_specific_claim"

        rows.append(
            {
                "created_utc": row.get("created_utc"),
                "trade_id": row.get("trade_id"),
                "prompt_regime": row.get("prompt_regime"),
                "trigger_family": row.get("trigger_family"),
                "side": row.get("side"),
                "pips": row.get("pips"),
                "pnl": row.get("pnl"),
                "rationale_cluster": row.get("rationale_cluster"),
                "numeric_prices_claimed": json.dumps(reason_prices),
                "numeric_prices_supported": json.dumps(supported_prices),
                "numeric_prices_unsupported": json.dumps(unsupported_prices),
                "indicator_or_level_claim": indicator_claim or level_claim,
                "m15_or_candle_unverifiable": bool(m15_claim or candle_claim),
                "hallucination_verdict": verdict,
                "rationale_excerpt": compact_text(reasoning, 240),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase4_hallucination_audit_40.csv", index=False)
    return out


def prompt_audit(corpus: pd.DataFrame) -> pd.DataFrame:
    perf_by_version = summarize(corpus, ["prompt_version", "prompt_regime"])
    perf_lookup = {
        (s(r.get("prompt_version")), s(r.get("prompt_regime"))): r
        for _, r in perf_by_version.iterrows()
    }

    rows = []
    entries = [
        {
            "prompt_version": "autonomous_phase_a_v1",
            "source": "phase4_reasoning_corpus.csv:prompt_version/action/decision",
            "stated_objective": "Legacy Phase A calls are only partially structured in the persisted sidecar.",
            "clause_flagged": "Prompt text not fully recoverable from persisted rows.",
            "diagnostic_read": "Prompt-causality is weak here because 881 Phase A rows have no explicit decision, but 79 closed/placed trade IDs exist in the sidecar.",
            "risk_sizing_read": "Sizing behavior can be audited from lots, not from a reliable prompt clause.",
        },
        {
            "prompt_version": "autonomous_phase2_zone_memory_custom_exit_v1",
            "source": "prompt_version_performance.csv plus phase4 corpus columns reasoning_text/reasoning_flags",
            "stated_objective": "Add zone memory and custom-exit context to autonomous decisions.",
            "clause_flagged": "Zone-memory terms such as fresh/working/failing/retry became admissive language rather than veto language.",
            "diagnostic_read": "The corpus shows caveat/self-correction language was often followed by placement; this is prompt-model interaction, not a pure data-input issue.",
            "risk_sizing_read": "No reliable account-exposure packet was persisted; size was governed by lots output plus downstream caps.",
        },
        {
            "prompt_version": "autonomous_phase2_runner_custom_exit_v3",
            "source": "api/autonomous_fillmore.py:4201-4202,4473,4693-4694 and prompt_version_performance.csv",
            "stated_objective": "Preserve wider runners and leave size on for continuation.",
            "clause_flagged": "Clauses like 'Preserve wider runners' and 'leave a runner' plausibly biased exit/sizing language toward holding continuation ideas.",
            "diagnostic_read": "This regime has 44 calls / 25 trades in the sidecar; Phase 2 already identified the CLR damage cluster for this period.",
            "risk_sizing_read": "Runner framing is directionally inconsistent with the realized loss asymmetry when no open-exposure field is shown.",
        },
        {
            "prompt_version": "autonomous_phase3_house_edge_v1",
            "source": "api/ai_trading_chat.py:2679-2761 and phase4 corpus columns features.phase4_*",
            "stated_objective": "Inject historical house-edge patterns, sell-side burden, mixed-alignment discipline, and binding skip rules.",
            "clause_flagged": "The prompt names red patterns, but the model often treated naming the weakness plus a thin catalyst as permission to trade.",
            "diagnostic_read": "High win rate with negative expectancy indicates reasoning selected many small winners while still admitting larger losers.",
            "risk_sizing_read": "The prompt raised selectivity but did not yet make loss-asymmetry and large-lot eligibility fully mechanical.",
        },
        {
            "prompt_version": "autonomous_phase4_selectivity_sizing_v1",
            "source": "api/autonomous_fillmore.py:_phase4_apply_selectivity_sizing and phase4 corpus",
            "stated_objective": "Add selectivity and sizing caps for weak sells, large lots, rolling pressure, and mixed CLR.",
            "clause_flagged": "Closed-trade sample in this forensic cut is N=1, so prompt performance cannot be inferred.",
            "diagnostic_read": "Only reasoning-shape and telemetry presence can be evaluated; outcome claims would be sample-size abuse.",
            "risk_sizing_read": "Structured fields phase4_catalyst_score/green_matches/weakness_signals begin appearing in the sidecar.",
        },
        {
            "prompt_version": "autonomous_phase5_reasoning_quality_v1",
            "source": "api/autonomous_fillmore.py:4620-4636,4696 and phase3_snapshot_flattened_sidecar.csv",
            "stated_objective": "Force explicit adverse-context resolution and reasoning-quality gate.",
            "clause_flagged": "The regime is skip-heavy: local sidecar has 69 calls and zero placed trades.",
            "diagnostic_read": "This may be the first prompt that stops caveat laundering, but it may also create analysis paralysis; no closed outcomes exist in this cut.",
            "risk_sizing_read": "Reasoning-quality gate is persisted, but with no placed trades it cannot yet be tied to P&L.",
        },
    ]
    for e in entries:
        pv = e["prompt_version"]
        rows_matching = [r for (v, _), r in perf_lookup.items() if v == pv]
        p = rows_matching[0].to_dict() if rows_matching else {}
        e.update(
            {
                "calls": p.get("calls", 0),
                "placed": p.get("placed", 0),
                "closed": p.get("closed", 0),
                "win_rate": p.get("win_rate", None),
                "net_pips": p.get("net_pips", 0.0),
                "net_usd": p.get("net_usd", 0.0),
            }
        )
        rows.append(e)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase4_prompt_regime_audit.csv", index=False)
    return out


def prompt_rule_compliance(corpus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rule(rule: str, prompt_version: str, universe: pd.Series, compliant: pd.Series, source_cols: str) -> None:
        u = corpus[universe].copy()
        if u.empty:
            rows.append(
                {
                    "rule": rule,
                    "prompt_version": prompt_version,
                    "universe_n": 0,
                    "compliant_n": 0,
                    "violation_n": 0,
                    "compliance_rate": None,
                    "closed_n": 0,
                    "violation_closed_net_pips": 0.0,
                    "violation_closed_net_usd": 0.0,
                    "source_cols": source_cols,
                }
            )
            return
        c = compliant.loc[u.index].fillna(False)
        viol = u[~c]
        viol_closed = viol[viol["pips"].notna()]
        rows.append(
            {
                "rule": rule,
                "prompt_version": prompt_version,
                "universe_n": len(u),
                "compliant_n": int(c.sum()),
                "violation_n": int((~c).sum()),
                "compliance_rate": float(c.mean()),
                "closed_n": int(u["pips"].notna().sum()),
                "violation_closed_net_pips": float(viol_closed["pips"].sum()) if len(viol_closed) else 0.0,
                "violation_closed_net_usd": float(viol_closed["pnl"].sum()) if len(viol_closed) else 0.0,
                "source_cols": source_cols,
            }
        )

    text = corpus["reasoning_text"].fillna("").astype(str)
    named = choose_first(corpus, ["features.named_catalyst", "named_catalyst_closed", "named_catalyst_prior"], default="").astype(str)
    side_bias = choose_first(corpus, ["features.side_bias_check", "side_bias_check_prior"], default="").astype(str)
    phase3 = corpus["prompt_version"].eq("autonomous_phase3_house_edge_v1")
    placed = corpus["decision_bucket"].eq("placed")
    mixed = corpus["timeframe_alignment"].fillna("").str.lower().eq("mixed") | text.str.lower().str.contains("mixed", regex=False)
    generic_catalyst = named.str.lower().str.contains(r"^(?:reclaimed_support|rejected_resistance|support|resistance|level)\b", regex=True) | named.str.len().lt(18)

    add_rule(
        "mixed_alignment_should_have_specific_named_catalyst",
        "autonomous_phase3_house_edge_v1",
        phase3 & placed & mixed,
        named.str.len().ge(18) & ~generic_catalyst,
        "prompt_version,decision_bucket,timeframe_alignment,features.named_catalyst,reasoning_text",
    )
    add_rule(
        "sell_side_burden_should_be_explicit",
        "autonomous_phase3_house_edge_v1",
        phase3 & placed & corpus["side"].eq("sell"),
        side_bias.str.len().ge(24),
        "prompt_version,decision_bucket,side,features.side_bias_check",
    )
    add_rule(
        "reasoning_quality_gate_skip_should_skip",
        "autonomous_phase5_reasoning_quality_v1",
        corpus["prompt_version"].eq("autonomous_phase5_reasoning_quality_v1")
        & corpus.get("features.reasoning_quality_gate", pd.Series("", index=corpus.index)).fillna("").eq("skip"),
        corpus["decision_bucket"].eq("skipped"),
        "prompt_version,decision_bucket,features.reasoning_quality_gate",
    )
    add_rule(
        "reasoning_quality_gate_cap_should_not_place_above_1_lot",
        "autonomous_phase5_reasoning_quality_v1",
        corpus["prompt_version"].eq("autonomous_phase5_reasoning_quality_v1")
        & corpus.get("features.reasoning_quality_gate", pd.Series("", index=corpus.index)).fillna("").eq("cap_to_1_lot"),
        corpus["decision_bucket"].ne("placed") | corpus["lots"].le(1),
        "prompt_version,decision_bucket,lots,features.reasoning_quality_gate",
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase4_prompt_rule_compliance.csv", index=False)
    return out


def sell_clr_matched_examples(corpus: pd.DataFrame) -> pd.DataFrame:
    clr = corpus[
        corpus["is_closed"]
        & corpus["trigger_family"].eq("critical_level_reaction")
        & corpus["side"].isin(["buy", "sell"])
    ].copy()
    sells = clr[(clr["side"] == "sell") & (clr["pips"] < 0)].sort_values("pips").head(8)
    buys = clr[(clr["side"] == "buy") & (clr["pips"] > 0)].copy()
    rows = []
    for _, sell in sells.iterrows():
        candidates = buys.copy()
        if candidates.empty:
            break
        candidates["score"] = 0.0
        candidates["score"] += (candidates["prompt_regime"] != sell.get("prompt_regime")).astype(float) * 5
        candidates["score"] += (candidates["lot_bucket"] != sell.get("lot_bucket")).astype(float) * 2
        candidates["score"] += (candidates["features.h1_regime"].fillna("") != s(sell.get("features.h1_regime"))).astype(float)
        candidates["score"] += (candidates["lots"].fillna(0) - float(sell.get("lots") or 0)).abs() / 10
        buy = candidates.sort_values(["score", "created_utc"]).iloc[0]
        rows.append(
            {
                "sell_trade_id": sell.get("trade_id"),
                "sell_prompt_regime": sell.get("prompt_regime"),
                "sell_lots": sell.get("lots"),
                "sell_pips": sell.get("pips"),
                "sell_pnl": sell.get("pnl"),
                "sell_cluster": sell.get("rationale_cluster"),
                "sell_excerpt": compact_text(sell.get("reasoning_text"), 220),
                "matched_buy_trade_id": buy.get("trade_id"),
                "buy_prompt_regime": buy.get("prompt_regime"),
                "buy_lots": buy.get("lots"),
                "buy_pips": buy.get("pips"),
                "buy_pnl": buy.get("pnl"),
                "buy_cluster": buy.get("rationale_cluster"),
                "buy_excerpt": compact_text(buy.get("reasoning_text"), 220),
                "divergence_read": "Both sides use level/caveat language; sell framing fails when rejection language is treated as enough to overcome mixed/bullish lower-timeframe context.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase4_sell_clr_matched_examples.csv", index=False)
    return out


def top_terms(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    closed = corpus[corpus["is_closed"]].copy()
    for name, terms in [
        ("hedge", HEDGE_WORDS),
        ("conviction", CONVICTION_WORDS),
        ("self_correction", SELF_CORRECTION),
        ("level", LEVEL_TERMS),
        ("macro", MACRO_TERMS),
        ("micro", MICRO_TERMS),
        ("indicator", INDICATOR_TERMS),
    ]:
        for term in sorted(terms):
            mask = closed["reasoning_text"].fillna("").str.lower().str.contains(re.escape(term), regex=True)
            if mask.sum() < 5:
                continue
            p = perf(closed[mask])
            p.update({"term_group": name, "term": term})
            rows.append(p)
    out = pd.DataFrame(rows).sort_values(["net_usd", "closed"], ascending=[True, False])
    out.to_csv(OUT / "phase4_reasoning_term_impact.csv", index=False)
    return out


def get_counterfactuals() -> pd.DataFrame:
    if PRIOR_COUNTERFACTUALS_CSV.exists():
        return pd.read_csv(PRIOR_COUNTERFACTUALS_CSV)
    return pd.DataFrame()


def build_report(
    corpus: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    fast_audit: pd.DataFrame,
    cognitive: pd.DataFrame,
    hallucination: pd.DataFrame,
    prompt_df: pd.DataFrame,
    compliance_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    term_df: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> str:
    closed = corpus[corpus["is_closed"]].copy()
    placed = corpus[corpus["decision_bucket"] == "placed"].copy()
    skips = corpus[corpus["decision_bucket"] == "skipped"].copy()
    wins = closed[closed["pips"] > 0]
    losses = closed[closed["pips"] < 0]
    sell_clr = closed[(closed["trigger_family"] == "critical_level_reaction") & (closed["side"] == "sell")]
    buy_clr = closed[(closed["trigger_family"] == "critical_level_reaction") & (closed["side"] == "buy")]
    phase5 = corpus[corpus["prompt_version"] == "autonomous_phase5_reasoning_quality_v1"]

    cluster_disp = display_perf(outputs["clusters"]).head(14)
    sell_buy_disp = display_perf(outputs["sell_buy"])
    destructive_disp = display_perf(outputs["destructive"]).head(8)
    cognitive_disp = display_perf(cognitive).copy()
    if "place_rate" in cognitive_disp.columns:
        cognitive_disp["place_rate"] = cognitive["place_rate"].map(pct)
    prompt_disp = display_perf(prompt_df.copy())
    if "win_rate" in prompt_disp.columns:
        prompt_disp["win_rate"] = prompt_df["win_rate"].map(pct)
    compliance_disp = compliance_df.copy()
    if "compliance_rate" in compliance_disp.columns:
        compliance_disp["compliance_rate"] = compliance_disp["compliance_rate"].map(pct)
    for c in ["violation_closed_net_pips"]:
        if c in compliance_disp.columns:
            compliance_disp[c] = compliance_disp[c].map(two_dec)
    if "violation_closed_net_usd" in compliance_disp.columns:
        compliance_disp["violation_closed_net_usd"] = compliance_disp["violation_closed_net_usd"].map(money)
    matched_disp = matched_df.copy()
    for c in ["sell_pips", "buy_pips"]:
        if c in matched_disp.columns:
            matched_disp[c] = matched_disp[c].map(two_dec)
    for c in ["sell_pnl", "buy_pnl"]:
        if c in matched_disp.columns:
            matched_disp[c] = matched_disp[c].map(money)
    fast_disp = fast_audit.copy()
    for c in ["pips", "pnl", "mae_abs", "mfe"]:
        if c in fast_disp.columns:
            fast_disp[c] = fast_disp[c].map(money if c == "pnl" else two_dec)
    halluc_disp = hallucination.copy()
    for c in ["pips", "pnl"]:
        if c in halluc_disp.columns:
            halluc_disp[c] = halluc_disp[c].map(money if c == "pnl" else two_dec)

    ps_disp = outputs["place_skip"].copy()
    for c in ["avg_token_count", "avg_hedge_density", "avg_conviction_density", "self_correction_rate", "avg_numeric_claims", "avg_level_terms"]:
        if c in ps_disp.columns:
            ps_disp[c] = ps_disp[c].map(two_dec)

    size_disp = display_perf(outputs["sizing"]).head(12)
    drift_disp = outputs["drift"].tail(12).copy()
    for c in ["net_pips", "avg_hedge_density", "avg_conviction_density", "self_correction_rate"]:
        if c in drift_disp.columns:
            drift_disp[c] = drift_disp[c].map(two_dec)
    if "net_usd" in drift_disp.columns:
        drift_disp["net_usd"] = drift_disp["net_usd"].map(money)

    halluc_counts = hallucination["hallucination_verdict"].value_counts(dropna=False).reset_index()
    halluc_counts.columns = ["hallucination_verdict", "sample_n"]

    overconf_rate = (
        fast_audit["rationale_evidence_claim"].eq("overconfident_vs_weak_snapshot").mean()
        if not fast_audit.empty and "rationale_evidence_claim" in fast_audit.columns
        else 0
    )
    phase5_place_rate = phase5["decision_bucket"].eq("placed").mean() if len(phase5) else 0

    lines: list[str] = []
    lines.append("# PHASE 4 - LLM REASONING FORENSICS")
    lines.append("")
    lines.append("_Auto Fillmore forensic investigation, Apr 16-May 1. Diagnosis only. No patch recommendations are made in this phase._")
    lines.append("")
    lines.append("## Phase 4 Bottom Line")
    lines.append("")
    lines.append(
        f"The reasoning layer is indicted. The closed-trade corpus contains {len(closed)} closed rows "
        f"with {len(wins)} winners and {len(losses)} losers, and the loss asymmetry remains visible in reasoning space: "
        f"avg winner {wins['pips'].mean():.2f}p / {money(wins['pnl'].mean())}, avg loser {losses['pips'].mean():.2f}p / {money(losses['pnl'].mean())}. "
        "Source: `phase4_reasoning_corpus.csv` columns `pips`, `pnl`, `decision_bucket`."
    )
    lines.append("")
    lines.append(
        "The strongest diagnostic pattern is not that Fillmore lacks reasons. It has too many reasons, and it converts caveats into permission. "
        "`critical_level_mixed_caveat_trade`, `momentum_with_caveat_trade`, and structure-only level language dominate destructive closed cells. "
        "Source: `phase4_rationale_clusters.csv`, `phase4_destructive_cells.csv`, and `phase4_cognitive_failure_tests.csv`."
    )
    lines.append("")
    lines.append(
        f"Phase 5 is a separate caution flag: the local sidecar shows {len(phase5)} Phase 5 calls and a {phase5_place_rate:.1%} placed rate. "
        "That is evidence about reasoning shape/selectivity only, not P&L, because there are no Phase 5 closed outcomes in this forensic cut. "
        "Source: `phase4_reasoning_corpus.csv` columns `prompt_version`, `decision_bucket`, `pips`."
    )
    lines.append("")
    lines.append("## 4.1 Reasoning Corpus Construction")
    lines.append("")
    lines.append(
        f"Built `phase4_reasoning_corpus.csv` from `phase3_snapshot_flattened_sidecar.csv`, "
        f"`phase3_closed_trades_with_snapshot_fields.csv`, and `reasoning_trade_dossier.csv`. "
        f"The corpus has {len(corpus)} calls/rows, {len(placed)} placed rows, {len(skips)} explicit skips, and {len(closed)} closed rows with P&L."
    )
    lines.append("")
    lines.append("Linguistic features added per row: token count, sentence count, hedge-word density, conviction-word density, numeric-claim count, indicator-name count, self-correction phrases, level/macro/micro term counts, and a deterministic rationale archetype. Source: `phase4_reasoning_corpus.csv` feature columns.")
    lines.append("")
    lines.append("## 4.2 Rationale Cluster Discovery")
    lines.append("")
    lines.append("Embeddings are not available in the local runtime, so clustering is deterministic lexical archetyping with manual labels. That is less subtle than embeddings, but repeatable and auditable.")
    lines.append("")
    lines.append(md_table(cluster_disp, ["rationale_cluster", "calls", "placed", "closed", "win_rate", "net_pips", "net_usd", "expectancy_pips", "expectancy_usd", "mae_p75", "mfe_p75", "avg_lots", "sample_size_warning"], 14))
    lines.append("")
    lines.append("Diagnostic read:")
    lines.append("")
    lines.append("- Predictive or salvageable archetypes must show positive expectancy with `closed >= 15`; anything below that is not reliable enough to preserve on outcome evidence alone.")
    lines.append("- Toxic/confabulatory archetypes are high-frequency patterns with negative net pips or USD, especially if they use level or caveat language while still placing.")
    lines.append("- Source: `phase4_rationale_clusters.csv` columns `closed`, `win_rate`, `net_pips`, `net_usd`, `expectancy_pips`, `avg_lots`.")
    lines.append("")
    lines.append("## 4.3 Sell-CLR Reasoning Forensic")
    lines.append("")
    lines.append(
        f"Sell-CLR remains the central reasoning failure: {len(sell_clr)} closed sell-CLR rows versus {len(buy_clr)} buy-CLR rows. "
        "Phase 3 exonerated simple snapshot missingness; Phase 4 therefore asks what the model did with comparable raw inputs."
    )
    lines.append("")
    lines.append(md_table(sell_buy_disp, ["side", "calls", "placed", "closed", "win_rate", "net_pips", "net_usd", "avg_winner_pips", "avg_loser_pips", "expectancy_pips", "expectancy_usd", "token_count_mean", "hedge_density_mean", "conviction_density_mean", "self_correction_rate", "numeric_claim_mean"], 10))
    lines.append("")
    lines.append("Sell-CLR versus buy-CLR cluster mix:")
    lines.append("")
    lines.append(md_table(display_perf(outputs["cluster_side"]).head(12), ["side", "rationale_cluster", "closed", "win_rate", "net_pips", "net_usd", "expectancy_pips", "avg_lots"], 12))
    lines.append("")
    lines.append("Counterfactual framing sample: worst sell-CLR losers matched to buy-CLR winners by prompt regime / lot bucket / H1 regime where possible.")
    lines.append("")
    lines.append(md_table(matched_disp, ["sell_trade_id", "sell_prompt_regime", "sell_lots", "sell_pips", "sell_pnl", "sell_cluster", "matched_buy_trade_id", "buy_pips", "buy_pnl", "buy_cluster", "divergence_read"], 8))
    lines.append("")
    lines.append("Source: `phase4_sell_clr_matched_examples.csv`; excerpts are retained there for row-level review.")
    lines.append("")
    lines.append("### 17 CLR Fast-Failure Rationale Audit")
    lines.append("")
    lines.append(
        f"Of the 17 fast-failure CLR rows from Phase 3, {overconf_rate:.1%} contain strong/clean/fresh/material style reasoning against weak, thin, mixed, or ambiguous snapshot evidence. "
        "Source: `phase4_clr_fast_failure_rationale_audit.csv` columns `snapshot_level_evidence_read`, `rationale_evidence_claim`, `conviction_word_count`."
    )
    lines.append("")
    lines.append(md_table(fast_disp, ["trade_id", "side", "lots", "pips", "pnl", "mae_abs", "mfe", "snapshot_level_evidence_read", "rationale_cluster", "rationale_evidence_claim", "rationale_excerpt"], 17))
    lines.append("")
    lines.append("### Sell-CLR Hypothesis Verdict")
    lines.append("")
    lines.append("| Hypothesis | Evidence weight | Verdict | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| H5: Reasoning template collapse on shorts | High | Primary driver | Sell-CLR repeatedly falls into level/caveat archetypes that treat rejection/reclaim language as sufficient. See `phase4_sell_buy_clr_cluster_mix.csv`. |")
    lines.append("| H3: Side-agnostic snapshot forced support/resistance remapping | Medium-high | Contributor | Phase 3 showed raw support/resistance objects instead of side-normalized entry-wall/profit-path packets; Phase 4 shows the model misuses level language on failures. |")
    lines.append("| H4: Macro-bias text contradicted shorts | Medium | Contributor, not sole cause | Macro/policy wording appears in damaging clusters, but passive short drift was favorable and snapshot coverage was symmetric; macro text amplifies bad short reasoning rather than fully explaining it. |")
    lines.append("| H1: Prompt-side directional asymmetry | Low-medium | Regime-specific contributor | Phase 3+ prompts explicitly mention sell-side burden; earlier damage predates that. The prompt likely failed to enforce the burden more than it created the original side asymmetry. |")
    lines.append("| H2: Model-side USDJPY prior | Low | Unproven | The data cannot see model priors directly. The short failure exists despite favorable passive short drift, but that does not prove an internal USDJPY prior. |")
    lines.append("")
    lines.append("## 4.4 Cognitive Failure Mode Tests")
    lines.append("")
    lines.append(md_table(cognitive_disp, ["test", "bucket", "calls", "placed", "closed", "place_rate", "win_rate", "net_pips", "net_usd", "expectancy_pips", "expectancy_usd", "avg_lots"], 40))
    lines.append("")
    lines.append("Failure-mode reads:")
    lines.append("")
    lines.append("- Confirmation bias test: ambiguous/contradictory rows are identified from reasoning text and persisted Phase 4/5 flags. If their placed expectancy is negative, the model is not resolving ambiguity; it is narrating through it.")
    lines.append("- Hedging-as-tell and overconfidence-as-tell are measured by density bins, not subjective vibes. Source: `phase4_cognitive_failure_tests.csv` columns `test`, `bucket`, `place_rate`, `net_pips`, `net_usd`.")
    lines.append("- Self-correction follow-through is the cleanest language test: rows with `however/but/although/despite` should often skip; when they still place and lose, that is direct caveat laundering.")
    lines.append("")
    lines.append("### Manual/Heuristic Hallucination Audit - 40 Placed Rows")
    lines.append("")
    lines.append("The audit checks rationale claims against the persisted snapshot. It is conservative: M15/candle/wick claims are marked unverifiable when the stored snapshot does not retain the underlying candle evidence.")
    lines.append("")
    lines.append(md_table(halluc_counts, ["hallucination_verdict", "sample_n"], 10))
    lines.append("")
    lines.append(md_table(halluc_disp, ["trade_id", "prompt_regime", "trigger_family", "side", "pips", "pnl", "rationale_cluster", "numeric_prices_claimed", "numeric_prices_unsupported", "hallucination_verdict", "rationale_excerpt"], 12))
    lines.append("")
    lines.append("Source: `phase4_hallucination_audit_40.csv` columns `numeric_prices_claimed`, `numeric_prices_supported`, `numeric_prices_unsupported`, `hallucination_verdict`.")
    lines.append("")
    lines.append("## 4.5 Place vs Skip Reasoning Comparison")
    lines.append("")
    lines.append(md_table(ps_disp, ["decision_bucket", "trigger_family", "calls", "avg_token_count", "avg_hedge_density", "avg_conviction_density", "self_correction_rate", "avg_numeric_claims", "avg_level_terms"], 20))
    lines.append("")
    lines.append("Skip-side caveat: skip rows still lack forward outcome telemetry. This table only describes reasoning shape at decision time, not skip correctness. Source: `phase4_place_skip_reasoning_comparison.csv`.")
    lines.append("")
    lines.append("## 4.6 System Prompt Audit by Regime")
    lines.append("")
    lines.append(md_table(prompt_disp, ["prompt_version", "calls", "placed", "closed", "win_rate", "net_pips", "net_usd", "source", "clause_flagged", "diagnostic_read"], 10))
    lines.append("")
    lines.append("Prompt-regime diagnosis:")
    lines.append("")
    lines.append("- Phase 2 runner/custom-exit v3 contains runner-preservation language in source (`api/autonomous_fillmore.py:4201-4202`, `4473`, `4693-4694`), and Phase 2 already isolated this as a CLR damage regime.")
    lines.append("- Phase 3 house-edge names historical red patterns, but the observed reasoning still admits mixed/caveat trades. That is prompt-model interaction failure: the instruction identified the weakness without reliably making it terminal.")
    lines.append("- Phase 5 reasoning-quality is not a P&L verdict. Its local evidence is 69 calls and 0 placed trades, so Phase 4 can only say it changed behavior sharply toward abstention.")
    lines.append("")
    lines.append("Observable prompt-rule compliance:")
    lines.append("")
    lines.append(md_table(compliance_disp, ["rule", "prompt_version", "universe_n", "compliant_n", "violation_n", "compliance_rate", "closed_n", "violation_closed_net_pips", "violation_closed_net_usd", "source_cols"], 10))
    lines.append("")
    lines.append("Source: `phase4_prompt_rule_compliance.csv`. This is limited to rules whose fields were persisted; unpersisted prompt clauses cannot receive an honest compliance percentage.")
    lines.append("")
    lines.append("## 4.7 Sizing Reasoning Audit")
    lines.append("")
    lines.append(md_table(size_disp, ["lot_bucket", "rationale_cluster", "closed", "win_rate", "net_pips", "net_usd", "expectancy_pips", "expectancy_usd", "avg_lots", "avg_conviction_density", "explicit_size_language_rate"], 12))
    lines.append("")
    lines.append("Diagnosis: sizing language is not reliably tethered to realized edge. The corpus can audit lots, planned RR, and account/equity/margin fields, but Phase 3 already showed open exposure and rolling P&L were not persisted in the decision snapshot for most damaging rows. Source: `phase4_sizing_reasoning_audit.csv`, `phase3_snapshot_flattened_sidecar.csv` account fields.")
    lines.append("")
    lines.append("## 4.8 Reasoning x Outcome Causal Mapping")
    lines.append("")
    lines.append("Top destructive cells with `closed >= 10`:")
    lines.append("")
    lines.append(md_table(destructive_disp, ["rationale_cluster", "trigger_family", "side", "prompt_regime", "lot_bucket", "closed", "win_rate", "net_pips", "net_usd", "expectancy_pips", "expectancy_usd", "avg_lots"], 8))
    lines.append("")
    if len(outputs["destructive"]) < 5:
        lines.append(f"Only {len(outputs['destructive'])} negative-expectancy cells met `closed >= 10`; the report does not pad this list with positive cells.")
        lines.append("")
    lines.append("Mechanism narratives:")
    lines.append("")
    for _, row in outputs["destructive"].head(5).iterrows():
        lines.append(
            f"- `{row.get('rationale_cluster')}` x `{row.get('trigger_family')}` x `{row.get('side')}` x `{row.get('prompt_regime')}`: "
            f"{int(row.get('closed', 0))} closed, {one_dec(row.get('net_pips'))}p, {money(row.get('net_usd'))}. "
            "Mechanism: the model had enough language to justify the setup, but the language pattern did not correspond to positive expectancy. "
            "This is a reasoning-selection failure, and when the lot bucket is above 1 lot it also becomes sizing amplification."
        )
    lines.append("")
    lines.append("Source: `phase4_destructive_cells.csv` columns `rationale_cluster`, `trigger_family`, `side`, `prompt_regime`, `lot_bucket`, `closed`, `net_pips`, `net_usd`.")
    lines.append("")
    lines.append("## 4.9 Reasoning Drift Across the Window")
    lines.append("")
    lines.append(md_table(drift_disp, ["created_date", "prompt_regime", "calls", "placed", "skipped", "closed", "net_pips", "net_usd", "avg_hedge_density", "avg_conviction_density", "self_correction_rate"], 12))
    lines.append("")
    lines.append("Reasoning shape changes sharply at prompt boundaries, especially the Phase 5 abstention shift. Within-regime drift is less trustworthy because some regimes have short calendar windows and small N. Source: `phase4_reasoning_drift_by_date.csv`.")
    lines.append("")
    lines.append("## Reasoning Verdict")
    lines.append("")
    lines.append("Predictive rationale archetypes to preserve, pending Phase 9:")
    lines.append("")
    good = outputs["clusters"][(outputs["clusters"]["closed"] >= 15) & (outputs["clusters"]["net_pips"] > 0)].sort_values("net_pips", ascending=False)
    if good.empty:
        lines.append("- No rationale archetype clears `closed >= 15` and positive net-pips strongly enough to call it proven. This is important: the reasoning layer has not earned a broad preserve list.")
    else:
        for _, row in good.head(5).iterrows():
            lines.append(f"- `{row['rationale_cluster']}`: {int(row['closed'])} closed, {one_dec(row['net_pips'])}p, {money(row['net_usd'])}.")
    lines.append("")
    lines.append("Toxic rationale archetypes to eliminate at prompt level, pending Phase 9:")
    bad = outputs["clusters"][(outputs["clusters"]["closed"] >= 15) & ((outputs["clusters"]["net_pips"] < 0) | (outputs["clusters"]["net_usd"] < 0))].sort_values("net_usd")
    for _, row in bad.head(6).iterrows():
        lines.append(f"- `{row['rationale_cluster']}`: {int(row['closed'])} closed, {one_dec(row['net_pips'])}p, {money(row['net_usd'])}.")
    lines.append("")
    lines.append("Cognitive failure modes confirmed:")
    lines.append("")
    lines.append("- Caveat laundering: self-correction and contradiction language appears in placed losers instead of stopping the trade. Evidence: `phase4_cognitive_failure_tests.csv` and `phase4_reasoning_term_impact.csv`.")
    lines.append("- Level-language overreach: CLR fast failures often use confident level language against weak/thin/mixed level evidence. Evidence: `phase4_clr_fast_failure_rationale_audit.csv`.")
    lines.append("- Sizing confidence mismatch: larger lots are not isolated to proven positive-expectancy rationale clusters. Evidence: `phase4_sizing_reasoning_audit.csv`.")
    lines.append("- Skip-side telemetry remains incomplete: Phase 5 abstention is visible, but without forward skip outcomes it cannot be called good or bad selectivity yet.")
    lines.append("")
    lines.append("Prompt clauses indicted for Phase 9 review:")
    lines.append("")
    lines.append("- Runner-preservation clauses such as 'Preserve wider runners' / 'leave a runner' are suspect in a system with negative USD expectancy and missing open-exposure context.")
    lines.append("- House-edge clauses that name red patterns without making caveat resolution terminal are suspect; the model can recite the red pattern and still place.")
    lines.append("- Any clause that asks for a catalyst but accepts structure-only text such as support/reject/reclaim is suspect, because that language repeatedly appears in negative-expectancy clusters.")
    lines.append("")
    lines.append("Reasoning behaviors that should become explicit constraints in Phase 9, stated diagnostically:")
    lines.append("")
    lines.append("- If the rationale contains contradiction/self-correction language, the decision must prove material resolution rather than simply mention the caveat.")
    lines.append("- If the rationale is level-only, it is not a material catalyst by itself.")
    lines.append("- If the rationale cannot explain why expected loser size is smaller than expected winner size, it has not addressed the structural failure mode.")
    lines.append("")
    lines.append("## Evidence Gaps")
    lines.append("")
    lines.append("- Skip correctness is still unknowable without `price_at_expiry`, `distance_at_expiry_pips`, and post-gate MAE/MFE for skipped calls.")
    lines.append("- Hallucination audit is limited by persisted snapshot coverage. M15/candle/wick references often cannot be verified because the raw prompt snapshot did not preserve those candle packets.")
    lines.append("- Prompt history before structured prompt versions is incomplete in persisted rows; Phase A prompt-causality is weaker than Phase 2+ prompt-causality.")
    lines.append("- Model-internal attention or priors cannot be observed. H2 remains hypothesis only.")
    lines.append("- Phase 4 and Phase 5 have too few or zero closed outcomes in this local forensic cut; they can be audited for behavior shift, not profitability.")
    lines.append("")
    lines.append("## Artifacts Written")
    lines.append("")
    for name in [
        "phase4_reasoning_corpus.csv",
        "phase4_rationale_clusters.csv",
        "phase4_sell_clr_vs_buy_clr.csv",
        "phase4_sell_buy_clr_cluster_mix.csv",
        "phase4_clr_fast_failure_rationale_audit.csv",
        "phase4_cognitive_failure_tests.csv",
        "phase4_hallucination_audit_40.csv",
        "phase4_place_skip_reasoning_comparison.csv",
        "phase4_prompt_regime_audit.csv",
        "phase4_prompt_rule_compliance.csv",
        "phase4_sell_clr_matched_examples.csv",
        "phase4_sizing_reasoning_audit.csv",
        "phase4_destructive_cells.csv",
        "phase4_reasoning_drift_by_date.csv",
        "phase4_reasoning_term_impact.csv",
        "phase4_manifest.json",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    sidecar, closed, prior, fast = load_inputs()
    corpus = build_corpus(sidecar, closed, prior)
    corpus = add_linguistic_features(corpus)
    corpus = add_clusters(corpus)

    outputs = write_corpus_outputs(corpus)
    fast_out = fast_failure_audit(corpus, fast)
    cognitive = cognitive_tests(corpus)
    hallucination = hallucination_audit(corpus)
    prompt_df = prompt_audit(corpus)
    compliance_df = prompt_rule_compliance(corpus)
    matched_df = sell_clr_matched_examples(corpus)
    term_df = top_terms(corpus)
    counterfactuals = get_counterfactuals()

    report = build_report(corpus, outputs, fast_out, cognitive, hallucination, prompt_df, compliance_df, matched_df, term_df, counterfactuals)
    (OUT / "PHASE4_REASONING_FORENSICS.md").write_text(report, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "sidecar": str(SIDECAR_CSV.relative_to(ROOT)),
            "closed": str(CLOSED_CSV.relative_to(ROOT)),
            "prior_dossier": str(PRIOR_DOSSIER_CSV.relative_to(ROOT)),
            "fast_failures": str(FAST_FAILURE_CSV.relative_to(ROOT)),
        },
        "outputs": [
            "PHASE4_REASONING_FORENSICS.md",
            "phase4_reasoning_corpus.csv",
            "phase4_rationale_clusters.csv",
            "phase4_sell_clr_vs_buy_clr.csv",
            "phase4_sell_buy_clr_cluster_mix.csv",
            "phase4_clr_fast_failure_rationale_audit.csv",
            "phase4_cognitive_failure_tests.csv",
            "phase4_hallucination_audit_40.csv",
            "phase4_place_skip_reasoning_comparison.csv",
            "phase4_prompt_regime_audit.csv",
            "phase4_prompt_rule_compliance.csv",
            "phase4_sell_clr_matched_examples.csv",
            "phase4_sizing_reasoning_audit.csv",
            "phase4_destructive_cells.csv",
            "phase4_reasoning_drift_by_date.csv",
            "phase4_reasoning_term_impact.csv",
        ],
        "notes": [
            "No embedding libraries were installed; rationale archetypes are deterministic lexical labels.",
            "Phase 5 has no closed outcomes in this forensic cut; it is behavioral evidence only.",
            "Skip outcome quality remains unavailable without forward skip telemetry.",
        ],
    }
    (OUT / "phase4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {OUT / 'PHASE4_REASONING_FORENSICS.md'}")
    print(f"Corpus rows: {len(corpus)}; closed rows: {corpus['is_closed'].sum()}; explicit skips: {(corpus['decision_bucket'] == 'skipped').sum()}")


if __name__ == "__main__":
    main()
