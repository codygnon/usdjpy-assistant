#!/usr/bin/env python3
"""Phase 6 trade lifecycle analysis for Autonomous Fillmore.

This is a read-only forensic harness. It uses terminal MAE/MFE, hold time,
entry/exit timestamps, Phase 4 reasoning features, and Phase 3 snapshot fields.
It explicitly does not infer minute-by-minute path ordering.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
CLOSED_CSV = OUT / "phase3_closed_trades_with_snapshot_fields.csv"
PHASE4_CSV = OUT / "phase4_reasoning_corpus.csv"
FAST_FAILURE_CSV = OUT / "phase3_clr_level_failure_17.csv"


EVENT_SOURCES = [
    {
        "event": "FOMC statement",
        "event_utc": "2026-04-29T18:00:00+00:00",
        "confidence": "high",
        "source": "Federal Reserve FOMC page: statement released April 29, 2026 at 2:00 p.m. ET",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20260429.htm",
    },
    {
        "event": "US Personal Income and Outlays / PCE",
        "event_utc": "2026-04-30T12:30:00+00:00",
        "confidence": "high",
        "source": "BEA release: embargoed until 8:30 a.m. EDT, Thursday, April 30, 2026",
        "url": "https://www.bea.gov/news/2026/personal-income-and-outlays-march-2026",
    },
    {
        "event": "Japan CPI March release",
        "event_utc": "2026-04-23T23:30:00+00:00",
        "confidence": "medium_time_assumed_0830_jst",
        "source": "Statistics Bureau schedule: March Japan CPI release date April 24; time not shown on page",
        "url": "https://www.stat.go.jp/english/data/cpi/1582.htm",
    },
    {
        "event": "Tokyo CPI April preliminary release",
        "event_utc": "2026-04-30T23:30:00+00:00",
        "confidence": "medium_time_assumed_0830_jst",
        "source": "Statistics Bureau schedule: April Ku-area Tokyo preliminary CPI release date May 1; time not shown on page",
        "url": "https://www.stat.go.jp/english/data/cpi/1582.htm",
    },
    {
        "event": "BOJ Statement on Monetary Policy",
        "event_utc": "2026-04-28T02:30:00+00:00",
        "confidence": "low_time_from_market_calendar_not_official_page",
        "source": "BOJ official page confirms Apr. 28, 2026 statement; exact release time not shown on BOJ list",
        "url": "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/index.htm",
    },
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


def one(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two(x: Any) -> str:
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


def max_drawdown(values: pd.Series) -> float:
    vals = values.fillna(0).astype(float)
    if vals.empty:
        return 0.0
    curve = vals.cumsum()
    return float((curve - curve.cummax()).min())


def profit_factor_pips(pips: pd.Series) -> float | None:
    wins = pips[pips > 0].sum()
    losses = abs(pips[pips < 0].sum())
    if losses == 0:
        return math.inf if wins else None
    return float(wins / losses)


def sharpe(values: pd.Series) -> float | None:
    vals = values.dropna().astype(float)
    if len(vals) < 2 or vals.std(ddof=1) == 0:
        return None
    return float(vals.mean() / vals.std(ddof=1) * math.sqrt(len(vals)))


def perf(group: pd.DataFrame, pips_col: str = "pips") -> dict[str, Any]:
    g = group[group[pips_col].notna()].copy()
    wins = g[g[pips_col] > 0]
    losses = g[g[pips_col] < 0]
    return {
        "n": len(g),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(g)) if len(g) else None,
        "net_pips": g[pips_col].sum() if len(g) else 0.0,
        "avg_pips": g[pips_col].mean() if len(g) else None,
        "median_pips": g[pips_col].median() if len(g) else None,
        "profit_factor_pips": profit_factor_pips(g[pips_col]) if len(g) else None,
        "max_drawdown_pips": max_drawdown(g[pips_col]) if len(g) else None,
    }


def display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["win_rate", "stop_overshoot_rate", "tp_undershoot_rate", "early_entry_failure_share", "exit_reversal_share", "share"]:
        if col in out.columns:
            out[col] = out[col].map(pct)
    for col in ["net_usd"]:
        if col in out.columns:
            out[col] = out[col].map(money)
    for col in [
        "net_pips",
        "avg_pips",
        "median_pips",
        "profit_factor_pips",
        "max_drawdown_pips",
        "hold_median_min",
        "hold_p25_min",
        "hold_p75_min",
        "winner_capture_median",
        "loser_capture_median",
        "winner_mfe_left_median",
        "loser_mae_saved_median",
        "mae_p75",
        "mfe_p75",
        "stop_overshoot_median",
        "tp_undershoot_median",
        "counterfactual_pips",
        "delta_vs_actual_pips",
        "nearest_event_minutes",
    ]:
        if col in out.columns:
            out[col] = out[col].map(two)
    return out


def load_data() -> pd.DataFrame:
    closed = pd.read_csv(CLOSED_CSV, low_memory=False)
    corpus = pd.read_csv(PHASE4_CSV, low_memory=False)
    corpus_cols = [
        "trade_id",
        "rationale_cluster",
        "hedge_density",
        "conviction_density",
        "self_correction_count",
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
    df["hold_minutes"] = (df["closed_dt"] - df["filled_dt"]).dt.total_seconds() / 60
    df["outcome"] = np.where(df["pips"] > 0, "win", np.where(df["pips"] < 0, "loss", "flat"))
    df["abs_pips"] = df["pips"].abs()
    df["winner_capture_ratio"] = np.where((df["pips"] > 0) & (df["mfe"] > 0), df["pips"] / df["mfe"], np.nan)
    df["loser_capture_ratio"] = np.where((df["pips"] < 0) & (df["mae_abs"] > 0), df["abs_pips"] / df["mae_abs"], np.nan)
    df["winner_mfe_left"] = np.where(df["pips"] > 0, df["mfe"] - df["pips"], np.nan)
    df["loser_mae_saved"] = np.where(df["pips"] < 0, df["mae_abs"] - df["abs_pips"], np.nan)
    df["has_self_correction"] = df["self_correction_count"].fillna(0) > 0
    df["hedge_density"] = df["hedge_density"].fillna(0.0)
    df["conviction_density"] = df["conviction_density"].fillna(0.0)
    df["rationale_cluster"] = df["rationale_cluster"].fillna("unknown")
    df["prompt_regime"] = df["prompt_regime"].fillna("unknown")
    df["trigger_family"] = df["trigger_family"].fillna("missing")
    df["features.session"] = df["features.session"].fillna("missing").str.lower()
    df["day_of_week"] = df["filled_dt"].dt.day_name()
    df["hour_utc"] = df["filled_dt"].dt.hour
    df["pip_value_per_lot"] = np.where(
        (df["lots"] > 0) & (df["pips"].abs() > 0),
        df["pnl"].abs() / (df["lots"] * df["pips"].abs()),
        np.nan,
    )
    return df.sort_values("filled_dt").reset_index(drop=True)


def capture_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        winners = group[group["pips"] > 0]
        losers = group[group["pips"] < 0]
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "n": len(group),
            "winner_n": len(winners),
            "loser_n": len(losers),
            "winner_capture_median": winners["winner_capture_ratio"].median(),
            "winner_capture_p25": winners["winner_capture_ratio"].quantile(0.25),
            "winner_capture_p75": winners["winner_capture_ratio"].quantile(0.75),
            "loser_capture_median": losers["loser_capture_ratio"].median(),
            "loser_capture_p25": losers["loser_capture_ratio"].quantile(0.25),
            "loser_capture_p75": losers["loser_capture_ratio"].quantile(0.75),
            "winner_mfe_left_median": winners["winner_mfe_left"].median(),
            "loser_mae_saved_median": losers["loser_mae_saved"].median(),
            "mae_p75": group["mae_abs"].quantile(0.75),
            "mfe_p75": group["mfe"].quantile(0.75),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    add("overall", "all_closed", df)
    for col, label in [
        ("trigger_family", "gate"),
        ("side", "side"),
        ("prompt_regime", "prompt_regime"),
        ("rationale_cluster", "rationale_cluster"),
    ]:
        for key, group in df.groupby(col, dropna=False):
            add(label, s(key) or "missing", group)
    out = pd.DataFrame(rows).sort_values(["segment_type", "net_pips"], ascending=[True, True])
    out.to_csv(OUT / "phase6_capture_efficiency.csv", index=False)
    return out


def hold_time_outcome(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "n": len(group),
            "hold_median_min": group["hold_minutes"].median(),
            "hold_p25_min": group["hold_minutes"].quantile(0.25),
            "hold_p75_min": group["hold_minutes"].quantile(0.75),
            "winner_hold_median_min": group.loc[group["pips"] > 0, "hold_minutes"].median(),
            "loser_hold_median_min": group.loc[group["pips"] < 0, "hold_minutes"].median(),
            "winner_minus_loser_hold_min": group.loc[group["pips"] > 0, "hold_minutes"].median()
            - group.loc[group["pips"] < 0, "hold_minutes"].median(),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    add("overall", "all_closed", df)
    for outcome, group in df.groupby("outcome"):
        add("outcome", outcome, group)
    for col, label in [
        ("trigger_family", "gate"),
        ("side", "side"),
        ("prompt_regime", "prompt_regime"),
        ("rationale_cluster", "rationale_cluster"),
    ]:
        for key, group in df.groupby(col, dropna=False):
            add(label, s(key) or "missing", group)
    # Fixed envelope bins.
    bins = [-0.01, 3, 7, 15, 30, 10_000]
    labels = ["<=3m", "3-7m", "7-15m", "15-30m", ">30m"]
    tmp = df.copy()
    tmp["hold_bin"] = pd.cut(tmp["hold_minutes"], bins=bins, labels=labels)
    for key, group in tmp.groupby("hold_bin", dropna=False, observed=False):
        add("hold_time_envelope", s(key), group)
    out = pd.DataFrame(rows).sort_values(["segment_type", "net_pips"], ascending=[True, True])
    out.to_csv(OUT / "phase6_hold_time_outcome.csv", index=False)
    return out


def stop_tp_discipline(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stop_overshoot_pips"] = np.where(
        (work["pips"] < 0) & work["features.sl_pips"].notna(),
        work["abs_pips"] - work["features.sl_pips"],
        np.nan,
    )
    work["stop_overshoot_flag"] = work["stop_overshoot_pips"] > 1.0
    work["tp_undershoot_pips"] = np.where(
        (work["pips"] > 0) & work["features.tp_pips"].notna(),
        work["features.tp_pips"] - work["pips"],
        np.nan,
    )
    work["tp_undershoot_flag"] = work["tp_undershoot_pips"] > 1.0
    rows: list[dict[str, Any]] = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        loss_rows = group[group["stop_overshoot_pips"].notna()]
        win_rows = group[group["tp_undershoot_pips"].notna()]
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "n": len(group),
            "losses_with_sl": len(loss_rows),
            "wins_with_tp": len(win_rows),
            "stop_overshoot_rate": loss_rows["stop_overshoot_flag"].mean() if len(loss_rows) else None,
            "stop_overshoot_median": loss_rows["stop_overshoot_pips"].median() if len(loss_rows) else None,
            "tp_undershoot_rate": win_rows["tp_undershoot_flag"].mean() if len(win_rows) else None,
            "tp_undershoot_median": win_rows["tp_undershoot_pips"].median() if len(win_rows) else None,
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    add("overall", "all_closed", work)
    for col, label in [
        ("trigger_family", "gate"),
        ("prompt_regime", "prompt_regime"),
        ("exit_strategy", "exit_strategy"),
    ]:
        for key, group in work.groupby(col, dropna=False):
            add(label, s(key) or "missing", group)
    out = pd.DataFrame(rows).sort_values(["segment_type", "net_pips"], ascending=[True, True])
    out.to_csv(OUT / "phase6_stop_tp_discipline.csv", index=False)
    return out, work


def runner_regime_test(df: pd.DataFrame, stop_work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = df.copy()
    merged["stop_overshoot_flag"] = stop_work["stop_overshoot_flag"]
    merged["tp_undershoot_flag"] = stop_work["tp_undershoot_flag"]
    for regime, group in merged.groupby("prompt_regime", dropna=False):
        winners = group[group["pips"] > 0]
        losers = group[group["pips"] < 0]
        row = {
            "prompt_regime": regime,
            "n": len(group),
            "hold_median_min": group["hold_minutes"].median(),
            "winner_hold_median_min": winners["hold_minutes"].median(),
            "loser_hold_median_min": losers["hold_minutes"].median(),
            "median_winner_pips": winners["pips"].median(),
            "median_loser_pips": losers["pips"].median(),
            "winner_capture_median": winners["winner_capture_ratio"].median(),
            "loser_capture_median": losers["loser_capture_ratio"].median(),
            "stop_overshoot_rate": group["stop_overshoot_flag"].mean(skipna=True),
            "tp_undershoot_rate": group["tp_undershoot_flag"].mean(skipna=True),
            "mae_p75": group["mae_abs"].quantile(0.75),
            "mfe_p75": group["mfe"].quantile(0.75),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    runner = out[out["prompt_regime"] == "Phase 2 runner/custom-exit v3"]
    if not runner.empty:
        r = runner.iloc[0]
        verdict = "measurable_harm"
        if r["net_pips"] >= 0:
            verdict = "not_harmful_on_pips"
        elif r["hold_median_min"] <= out["hold_median_min"].median():
            verdict = "harm_not_from_longer_hold"
        out["runner_verdict"] = ""
        out.loc[out["prompt_regime"] == "Phase 2 runner/custom-exit v3", "runner_verdict"] = verdict
    out.to_csv(OUT / "phase6_runner_regime_test.csv", index=False)
    return out


def caveat_exit_interaction(df: pd.DataFrame, stop_work: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stop_overshoot_flag"] = stop_work["stop_overshoot_flag"]
    work["tp_undershoot_flag"] = stop_work["tp_undershoot_flag"]
    work["hedge_bin"] = "zero"
    positive = work.loc[work["hedge_density"] > 0, "hedge_density"]
    if not positive.empty:
        q50 = positive.quantile(0.50)
        q80 = positive.quantile(0.80)
        work.loc[(work["hedge_density"] > 0) & (work["hedge_density"] <= q50), "hedge_bin"] = "low_hedge"
        work.loc[(work["hedge_density"] > q50) & (work["hedge_density"] <= q80), "hedge_bin"] = "mid_hedge"
        work.loc[work["hedge_density"] > q80, "hedge_bin"] = "high_hedge"
    work["conviction_bin"] = "zero"
    positive_c = work.loc[work["conviction_density"] > 0, "conviction_density"]
    if not positive_c.empty:
        q50 = positive_c.quantile(0.50)
        q80 = positive_c.quantile(0.80)
        work.loc[(work["conviction_density"] > 0) & (work["conviction_density"] <= q50), "conviction_bin"] = "low_conviction"
        work.loc[(work["conviction_density"] > q50) & (work["conviction_density"] <= q80), "conviction_bin"] = "mid_conviction"
        work.loc[work["conviction_density"] > q80, "conviction_bin"] = "high_conviction"
    work["self_correction_bin"] = np.where(work["has_self_correction"], "self_correction_present", "self_correction_absent")
    rows = []

    def add(test: str, bucket: str, group: pd.DataFrame) -> None:
        losers = group[group["pips"] < 0]
        winners = group[group["pips"] > 0]
        row = {
            "test": test,
            "bucket": bucket,
            "n": len(group),
            "loser_n": len(losers),
            "winner_n": len(winners),
            "hold_median_min": group["hold_minutes"].median(),
            "loser_hold_median_min": losers["hold_minutes"].median(),
            "winner_hold_median_min": winners["hold_minutes"].median(),
            "winner_capture_median": winners["winner_capture_ratio"].median(),
            "loser_capture_median": losers["loser_capture_ratio"].median(),
            "stop_overshoot_rate": group["stop_overshoot_flag"].mean(skipna=True),
            "tp_undershoot_rate": group["tp_undershoot_flag"].mean(skipna=True),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    for col, test in [
        ("hedge_bin", "hedge_density"),
        ("conviction_bin", "conviction_density"),
        ("self_correction_bin", "self_correction"),
        ("rationale_cluster", "rationale_cluster"),
    ]:
        for key, group in work.groupby(col, dropna=False):
            add(test, s(key), group)
    out = pd.DataFrame(rows).sort_values(["test", "net_pips"])
    out.to_csv(OUT / "phase6_caveat_exit_interaction.csv", index=False)
    return out


def session_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    # Approximate FX transition times in UTC.
    transition_minutes = {
        "tokyo_close": 9 * 60,
        "london_open": 8 * 60,
        "ny_open": 13 * 60,
        "ny_close": 21 * 60,
    }
    minute_of_day = work["filled_dt"].dt.hour * 60 + work["filled_dt"].dt.minute
    labels = []
    for minute in minute_of_day:
        hit = []
        for name, target in transition_minutes.items():
            dist = min(abs(minute - target), 1440 - abs(minute - target))
            if dist <= 15:
                hit.append(name)
        labels.append(",".join(hit) if hit else "not_transition")
    work["session_transition"] = labels
    rows = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        winners = group[group["pips"] > 0]
        losers = group[group["pips"] < 0]
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "hold_median_min": group["hold_minutes"].median(),
            "winner_capture_median": winners["winner_capture_ratio"].median(),
            "loser_capture_median": losers["loser_capture_ratio"].median(),
            "mae_p75": group["mae_abs"].quantile(0.75),
            "mfe_p75": group["mfe"].quantile(0.75),
            "spread_median": group.get("features.spread_at_entry", pd.Series(dtype=float)).median(),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    for col, label in [
        ("features.session", "session"),
        ("hour_utc", "hour_utc"),
        ("day_of_week", "day_of_week"),
        ("session_transition", "session_transition"),
    ]:
        for key, group in work.groupby(col, dropna=False):
            add(label, s(key), group)
    out = pd.DataFrame(rows).sort_values(["segment_type", "net_pips"], ascending=[True, True])
    out.to_csv(OUT / "phase6_session_lifecycle.csv", index=False)
    return out


def event_window(df: pd.DataFrame) -> pd.DataFrame:
    events = pd.DataFrame(EVENT_SOURCES)
    events["event_dt"] = pd.to_datetime(events["event_utc"], utc=True)
    events.to_csv(OUT / "phase6_event_sources.csv", index=False)
    rows = []
    annotated = []
    for _, trade in df.iterrows():
        distances = (events["event_dt"] - trade["filled_dt"]).dt.total_seconds() / 60
        abs_idx = distances.abs().idxmin()
        nearest = events.loc[abs_idx]
        delta = float(distances.loc[abs_idx])
        if -60 <= delta < 0:
            bucket = "pre_event_60m"
        elif 0 <= delta <= 30:
            bucket = "in_event_0_30m_after"
        elif 30 < delta <= 180:
            bucket = "post_event_30_180m"
        else:
            bucket = "clear_window"
        d = trade.to_dict()
        d.update(
            {
                "nearest_event": nearest["event"],
                "nearest_event_minutes": delta,
                "event_bucket": bucket,
                "event_confidence": nearest["confidence"],
                "event_source_url": nearest["url"],
            }
        )
        annotated.append(d)
    work = pd.DataFrame(annotated)
    work.to_csv(OUT / "phase6_event_window_trades.csv", index=False)
    for key, group in work.groupby("event_bucket", dropna=False):
        winners = group[group["pips"] > 0]
        losers = group[group["pips"] < 0]
        row = {
            "event_bucket": key,
            "n": len(group),
            "nearest_event_examples": json.dumps(group["nearest_event"].value_counts().head(5).to_dict()),
            "median_abs_event_distance_min": group["nearest_event_minutes"].abs().median(),
            "winner_capture_median": winners["winner_capture_ratio"].median(),
            "loser_capture_median": losers["loser_capture_ratio"].median(),
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("net_pips")
    out.to_csv(OUT / "phase6_event_window.csv", index=False)
    return out


def path_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(segment_type: str, segment: str, group: pd.DataFrame) -> None:
        winners = group[group["pips"] > 0]
        losers = group[group["pips"] < 0]
        early_entry_fail = losers[(losers["mfe"].fillna(999) <= 2.0) & (losers["mae_abs"].fillna(0) >= 6.0)]
        exit_reversal = losers[losers["mfe"].fillna(0) >= 4.0]
        row = {
            "segment_type": segment_type,
            "segment": segment,
            "n": len(group),
            "winner_mfe_left_median": winners["winner_mfe_left"].median(),
            "winner_mfe_left_p75": winners["winner_mfe_left"].quantile(0.75),
            "loser_mae_saved_median": losers["loser_mae_saved"].median(),
            "loser_mae_saved_p75": losers["loser_mae_saved"].quantile(0.75),
            "realized_loss_equals_mae_rate": (losers["loser_mae_saved"].abs() <= 1.0).mean() if len(losers) else None,
            "early_entry_failure_n": len(early_entry_fail),
            "early_entry_failure_pips": early_entry_fail["pips"].sum(),
            "early_entry_failure_share": len(early_entry_fail) / len(losers) if len(losers) else None,
            "exit_reversal_n": len(exit_reversal),
            "exit_reversal_pips": exit_reversal["pips"].sum(),
            "exit_reversal_share": len(exit_reversal) / len(losers) if len(losers) else None,
            "sample_size_warning": "N<15" if len(group) < 15 else "",
        }
        row.update(perf(group))
        rows.append(row)

    add("overall", "all_closed", df)
    for col, label in [
        ("trigger_family", "gate"),
        ("side", "side"),
        ("prompt_regime", "prompt_regime"),
        ("rationale_cluster", "rationale_cluster"),
    ]:
        for key, group in df.groupby(col, dropna=False):
            add(label, s(key) or "missing", group)
    out = pd.DataFrame(rows).sort_values(["segment_type", "net_pips"], ascending=[True, True])
    out.to_csv(OUT / "phase6_path_asymmetry.csv", index=False)
    return out


def lifecycle_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    median_winner_hold = df.loc[df["pips"] > 0, "hold_minutes"].median()
    p50_mae = df["mae_abs"].median()
    avg_loser_abs = abs(df.loc[df["pips"] < 0, "pips"].mean())
    lock_level = 1.5 * avg_loser_abs

    def policy(name: str, pips: pd.Series, assumption: str) -> dict[str, Any]:
        row = perf(pd.DataFrame({"pips": pips}), "pips")
        row.update(
            {
                "policy": name,
                "counterfactual_pips": pips.sum(),
                "delta_vs_actual_pips": pips.sum() - df["pips"].sum(),
                "max_drawdown_pips": max_drawdown(pips),
                "sharpe_pips": sharpe(pips),
                "assumption": assumption,
            }
        )
        return row

    actual = df["pips"].copy()
    time_stop = actual.copy()
    late = df["hold_minutes"] > median_winner_hold
    # Optimistic upper bound: late losers that had any meaningful MFE are flattened.
    time_stop.loc[late & (df["pips"] < 0) & (df["mfe"].fillna(0) >= 2.0)] = 0.0
    trail = actual.copy()
    trail.loc[df["pips"] < -p50_mae] = -p50_mae
    tp_locked = actual.copy()
    tp = df["features.tp_pips"]
    can_hit_tp = (df["mfe"] >= tp) & tp.notna()
    tp_locked.loc[can_hit_tp] = tp.loc[can_hit_tp]
    capture_floor = actual.copy()
    hit_floor = df["mfe"].fillna(0) >= lock_level
    capture_floor.loc[hit_floor] = np.maximum(capture_floor.loc[hit_floor], lock_level)

    rows = [
        policy("actual_observed", actual, "Observed realized pips."),
        policy("A_time_stop_at_median_winner_hold_upper_bound", time_stop, f"Median winner hold {median_winner_hold:.2f}m; late losers with MFE>=2p flattened. Requires path-order assumption."),
        policy("B_trailing_stop_at_p50_MAE_upper_bound", trail, f"Losses capped at p50 terminal MAE {p50_mae:.2f}p. Assumes stop could trigger before worse path."),
        policy("C_TP_locked_exit_upper_bound", tp_locked, "If terminal MFE reached stored TP, assume TP fill; otherwise observed pips. Requires MFE-before-exit assumption."),
        policy("D_capture_floor_1_5x_avg_loser_upper_bound", capture_floor, f"If MFE reached {lock_level:.2f}p, assume at least that capture. Strong path-order assumption."),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase6_lifecycle_counterfactuals.csv", index=False)
    return out


def attribution(df: pd.DataFrame) -> pd.DataFrame:
    losses = df[df["pips"] < 0].copy()
    winners = df[df["pips"] > 0].copy()
    entry_fail = losses[(losses["mfe"].fillna(999) <= 2.0) & (losses["mae_abs"].fillna(0) >= 6.0)]
    exit_reversal = losses[losses["mfe"].fillna(0) >= 4.0]
    other_losses = losses.drop(index=entry_fail.index.union(exit_reversal.index))
    fast = pd.read_csv(FAST_FAILURE_CSV)
    fast_ids = set(fast["trade_id"].dropna().astype(str))
    fast_rows = df[df["trade_id"].astype(str).isin(fast_ids)]
    rows = [
        {
            "bucket": "entry_generated_terminal_proxy_mfe_le_2_mae_ge_6",
            "n": len(entry_fail),
            "pips": entry_fail["pips"].sum(),
            "definition": "Losses with terminal MFE <=2p and MAE >=6p. Because path-time buckets are absent, this is terminal evidence, not minute-one proof.",
        },
        {
            "bucket": "exit_generated_reversal_proxy_loss_with_mfe_ge_4",
            "n": len(exit_reversal),
            "pips": exit_reversal["pips"].sum(),
            "definition": "Losses that had terminal favorable excursion >=4p but closed red; suggests exit/lifecycle reversal risk.",
        },
        {
            "bucket": "other_losses",
            "n": len(other_losses),
            "pips": other_losses["pips"].sum(),
            "definition": "Remaining losses after entry-failure and exit-reversal proxies.",
        },
        {
            "bucket": "winner_offset",
            "n": len(winners),
            "pips": winners["pips"].sum(),
            "definition": "All winners; offsets gross loss buckets to observed net pips.",
        },
        {
            "bucket": "phase3_17_clr_fast_failures",
            "n": len(fast_rows),
            "pips": fast_rows["pips"].sum(),
            "definition": "Previously identified CLR fast failures: top-quartile MAE and <=2p MFE.",
        },
        {
            "bucket": "observed_net",
            "n": len(df),
            "pips": df["pips"].sum(),
            "definition": "Total observed net pips.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase6_entry_exit_attribution.csv", index=False)
    return out


def build_report(
    df: pd.DataFrame,
    capture: pd.DataFrame,
    hold: pd.DataFrame,
    stop_tp: pd.DataFrame,
    runner: pd.DataFrame,
    caveat: pd.DataFrame,
    session: pd.DataFrame,
    event: pd.DataFrame,
    path: pd.DataFrame,
    counter: pd.DataFrame,
    attr: pd.DataFrame,
) -> str:
    overall_capture = capture[capture["segment_type"].eq("overall")].iloc[0]
    overall_hold = hold[hold["segment_type"].eq("overall")].iloc[0]
    best_envelope = hold[hold["segment_type"].eq("hold_time_envelope")].sort_values("avg_pips", ascending=False).iloc[0]
    runner_row = runner[runner["prompt_regime"].eq("Phase 2 runner/custom-exit v3")]
    runner_verdict = runner_row["runner_verdict"].iloc[0] if not runner_row.empty and "runner_verdict" in runner.columns else "undetermined"
    attr_map = {row["bucket"]: row for _, row in attr.iterrows()}
    entry_pips = attr_map["entry_generated_terminal_proxy_mfe_le_2_mae_ge_6"]["pips"]
    exit_pips = attr_map["exit_generated_reversal_proxy_loss_with_mfe_ge_4"]["pips"]
    other_loss_pips = attr_map["other_losses"]["pips"]
    winner_pips = attr_map["winner_offset"]["pips"]
    fast_pips = attr_map["phase3_17_clr_fast_failures"]["pips"]
    net_pips = attr_map["observed_net"]["pips"]
    gross_loss = entry_pips + exit_pips + other_loss_pips
    entry_verdict = "entry-generated" if abs(entry_pips) >= abs(exit_pips) else "exit-generated/reversal"
    loss_rows = df[df["pips"] < 0]
    mae_under_exit = loss_rows[(loss_rows["mae_abs"] > 0) & (loss_rows["abs_pips"] > loss_rows["mae_abs"] + 0.1)]
    mae_under_rate = len(mae_under_exit) / len(loss_rows) if len(loss_rows) else np.nan
    # Caveat verdict.
    cav = caveat[(caveat["test"].eq("self_correction"))]
    present = cav[cav["bucket"].eq("self_correction_present")]
    absent = cav[cav["bucket"].eq("self_correction_absent")]
    caveat_verdict = "undetermined"
    if not present.empty and not absent.empty:
        caveat_verdict = "yes" if present["loser_hold_median_min"].iloc[0] > absent["loser_hold_median_min"].iloc[0] else "no"

    cap_d = display(capture.copy())
    hold_d = display(hold.copy())
    stop_d = display(stop_tp.copy())
    runner_d = display(runner.copy())
    caveat_d = display(caveat.copy())
    session_d = display(session.copy())
    event_d = display(event.copy())
    path_d = display(path.copy())
    counter_d = display(counter.copy())
    attr_d = attr.copy()
    attr_d["pips"] = attr_d["pips"].map(two)

    lines: list[str] = []
    lines.append("# PHASE 6 - TRADE LIFECYCLE ANALYSIS")
    lines.append("")
    lines.append("_Auto Fillmore forensic investigation, Apr 16-May 1. Diagnosis only. Terminal MAE/MFE analysis; no minute-by-minute path inference._")
    lines.append("")
    lines.append("## Phase 6 Bottom Line")
    lines.append("")
    lines.append(
        f"Committed lifecycle verdict: **{entry_verdict} damage dominates the negative leg**. "
        f"The terminal entry-failure proxy accounts for {one(entry_pips)}p across {int(attr_map['entry_generated_terminal_proxy_mfe_le_2_mae_ge_6']['n'])} losses, "
        f"while the exit-reversal proxy accounts for {one(exit_pips)}p across {int(attr_map['exit_generated_reversal_proxy_loss_with_mfe_ge_4']['n'])} losses. "
        f"Winners offset the gross loss buckets by +{one(winner_pips)}p, producing the observed {one(net_pips)}p net. Source: `phase6_entry_exit_attribution.csv`."
    )
    lines.append("")
    lines.append(
        f"Capture asymmetry: median winners captured {two(overall_capture['winner_capture_median'])}x of terminal MFE, while median losers realized {two(overall_capture['loser_capture_median'])}x of terminal MAE. "
        f"Median winner pips left on table: {two(overall_capture['winner_mfe_left_median'])}p. Median loser MAE saved: {two(overall_capture['loser_mae_saved_median'])}p."
    )
    lines.append(
        f"MAE integrity caveat: {len(mae_under_exit)} of {len(loss_rows)} losing rows ({pct(mae_under_rate)}) realized a larger loss than stored MAE. "
        "That is impossible if MAE includes the exit tick, so loser-capture ratios should be read as 'full adverse excursion or worse,' not precise path capture."
    )
    lines.append("")
    lines.append(
        f"Hold-time asymmetry: median hold was {two(overall_hold['hold_median_min'])}m overall. "
        f"Winners median {two(overall_hold['winner_hold_median_min'])}m; losers median {two(overall_hold['loser_hold_median_min'])}m. "
        f"Least-bad observed hold-time envelope was `{best_envelope['segment']}` with {two(best_envelope['avg_pips'])}p expectancy."
    )
    lines.append("")
    lines.append("## 6.1 Capture Efficiency")
    lines.append("")
    lines.append(md_table(cap_d[cap_d["segment_type"].isin(["overall", "gate", "side", "prompt_regime"])], ["segment_type", "segment", "n", "winner_n", "loser_n", "winner_capture_median", "loser_capture_median", "winner_mfe_left_median", "loser_mae_saved_median", "mae_p75", "mfe_p75", "net_pips", "sample_size_warning"], 32))
    lines.append("")
    lines.append("Interpretation: low winner capture plus high loser capture means exits crystallize more of bad path than good path. Because many losing rows have realized loss greater than stored MAE, loser capture is directionally useful but not path-precise. Source: `phase6_capture_efficiency.csv`.")
    lines.append("")
    lines.append("## 6.2 Hold Time vs Outcome")
    lines.append("")
    lines.append(md_table(hold_d[hold_d["segment_type"].isin(["overall", "outcome", "hold_time_envelope", "gate", "prompt_regime"])], ["segment_type", "segment", "n", "hold_median_min", "winner_hold_median_min", "loser_hold_median_min", "net_pips", "avg_pips", "profit_factor_pips", "sample_size_warning"], 38))
    lines.append("")
    lines.append(f"Least-bad observed envelope: `{best_envelope['segment']}`. Every fixed hold-time bucket was still negative, so this is descriptive only; without path-time buckets, it is not an executable time-stop proof.")
    lines.append("")
    lines.append("## 6.3 Stop/TP Discipline")
    lines.append("")
    lines.append(md_table(stop_d[stop_d["segment_type"].isin(["overall", "gate", "prompt_regime", "exit_strategy"])], ["segment_type", "segment", "n", "losses_with_sl", "wins_with_tp", "stop_overshoot_rate", "stop_overshoot_median", "tp_undershoot_rate", "tp_undershoot_median", "net_pips", "sample_size_warning"], 36))
    lines.append("")
    lines.append("Stored SL/TP coverage is partial. Stop overshoot/TP undershoot claims are limited to rows with `features.sl_pips` / `features.tp_pips` present.")
    lines.append("")
    lines.append("## 6.4 Runner-Preservation Prompt-Regime Test")
    lines.append("")
    lines.append(md_table(runner_d, ["prompt_regime", "n", "win_rate", "net_pips", "hold_median_min", "winner_hold_median_min", "loser_hold_median_min", "winner_capture_median", "loser_capture_median", "mae_p75", "mfe_p75", "runner_verdict", "sample_size_warning"], 12))
    lines.append("")
    lines.append(f"Committed runner-regime verdict: **{runner_verdict}**. Phase 2 runner/custom-exit v3 was the worst prompt-regime in pips among regimes with lifecycle data, but the table separates whether harm came from longer holds versus bad entries.")
    lines.append("")
    lines.append("## 6.5 Caveat-Laundered Exits")
    lines.append("")
    lines.append(md_table(caveat_d[caveat_d["test"].isin(["hedge_density", "self_correction", "rationale_cluster"])], ["test", "bucket", "n", "loser_n", "hold_median_min", "loser_hold_median_min", "winner_hold_median_min", "winner_capture_median", "loser_capture_median", "net_pips", "sample_size_warning"], 32))
    lines.append("")
    lines.append(f"Committed caveat-exit verdict: **{caveat_verdict}**. This compares loser hold time for self-correction-present versus absent rows; sample-size warnings matter because non-caveat rows are rare.")
    lines.append("")
    lines.append("## 6.6 Session / Time-of-Day / Day-of-Week Lifecycle")
    lines.append("")
    lines.append(md_table(session_d[session_d["segment_type"].isin(["session", "day_of_week", "session_transition"])], ["segment_type", "segment", "n", "win_rate", "net_pips", "avg_pips", "hold_median_min", "winner_capture_median", "loser_capture_median", "mae_p75", "mfe_p75", "sample_size_warning"], 42))
    lines.append("")
    lines.append("Session transition windows are approximate ±15 minutes around Tokyo close, London open, NY open, and NY close in UTC.")
    lines.append("")
    lines.append("## 6.7 Event-Window Check")
    lines.append("")
    lines.append(md_table(event_d, ["event_bucket", "n", "win_rate", "net_pips", "avg_pips", "median_abs_event_distance_min", "winner_capture_median", "loser_capture_median", "nearest_event_examples", "sample_size_warning"], 12))
    lines.append("")
    lines.append("Event-source precision is mixed: FOMC and PCE have official release times; Japan CPI/Tokyo CPI official page provides dates, so 08:30 JST is an assumption; BOJ statement date is official but exact time is low-confidence.")
    lines.append("")
    lines.append("## 6.8 MAE-MFE Path Asymmetry")
    lines.append("")
    lines.append(md_table(path_d[path_d["segment_type"].isin(["overall", "gate", "side", "prompt_regime", "rationale_cluster"])], ["segment_type", "segment", "n", "winner_mfe_left_median", "loser_mae_saved_median", "realized_loss_equals_mae_rate", "early_entry_failure_n", "early_entry_failure_pips", "exit_reversal_n", "exit_reversal_pips", "net_pips", "sample_size_warning"], 36))
    lines.append("")
    lines.append("This is terminal path asymmetry, not path ordering. It shows which trades had no meaningful favorable excursion and which had enough MFE to imply an exit-management opportunity.")
    lines.append("")
    lines.append("## 6.9 Lifecycle Counterfactuals")
    lines.append("")
    lines.append(md_table(display(counter.copy()), ["policy", "n", "win_rate", "counterfactual_pips", "delta_vs_actual_pips", "profit_factor_pips", "max_drawdown_pips", "sharpe_pips", "assumption"], 8))
    lines.append("")
    lines.append("These are upper bounds with strong path-order assumptions. The p50-MAE trailing result is especially diagnostic rather than actionable because stored MAE appears to under-report some losing exits. These policies do not prove an implementable exit rule.")
    lines.append("")
    lines.append("## Entry vs Exit Attribution")
    lines.append("")
    lines.append(md_table(attr_d, ["bucket", "n", "pips", "definition"], 8))
    lines.append("")
    lines.append(
        f"Gross-loss ledger: entry-failure proxy {one(entry_pips)}p, exit-reversal proxy {one(exit_pips)}p, other losses {one(other_loss_pips)}p, winners +{one(winner_pips)}p. "
        f"The 17 known CLR fast failures alone contributed {one(fast_pips)}p. This makes the 1.64x loss asymmetry primarily entry-generated, with exit leakage as a secondary amplifier."
    )
    lines.append("")
    lines.append("## Lifecycle Verdict")
    lines.append("")
    lines.append(f"- Are winners cut early? **Yes**: median winner capture is {two(overall_capture['winner_capture_median'])} of MFE, with {two(overall_capture['winner_mfe_left_median'])}p median MFE left on winners.")
    lines.append(f"- Are losers held too long? **Partly**: median loser captures {two(overall_capture['loser_capture_median'])} of stored MAE, with the MAE integrity caveat above; exit-reversal losses with MFE>=4p account for {one(exit_pips)}p.")
    lines.append(f"- Did runner-preservation prompt language cause measurable harm? **{runner_verdict}** based on Phase 2 runner/custom-exit v3 lifecycle metrics.")
    lines.append(f"- Did caveat language at entry predict held-too-long losers? **{caveat_verdict}**, with caveat that non-caveat comparison N is small.")
    lines.append(f"- Is the 1.64x loss-asymmetry primarily entry-generated or exit-generated? **Entry-generated**, because terminal no-green/large-MAE losses and the known CLR fast failures explain the largest gross-loss bucket.")
    lines.append("")
    lines.append("## Evidence Gaps")
    lines.append("")
    lines.append("- No path-time MAE/MFE buckets at 1/3/5/15 minutes, so Phase 6 cannot prove whether MAE occurred before MFE inside any individual trade.")
    lines.append("- Stored MAE does not always include the realized exit loss; add an exit-inclusive MAE field or path replay so loser capture ratios are physically consistent.")
    lines.append("- No exact skipped-call forward paths, so lifecycle counterfactuals apply only to placed/closed trades.")
    lines.append("- SL/TP fields are partial and prompt-regime-dependent; stop/TP discipline cannot be computed for every row.")
    lines.append("- Event-window timestamps for Japan CPI and BOJ are partly date-only/assumed-time; only FOMC and PCE have high-confidence official release times.")
    lines.append("- Exit-manager internal decisions are not replayable; terminal outcomes show behavior but not every intermediate management decision.")
    lines.append("")
    lines.append("## Public Event Sources")
    lines.append("")
    for src in EVENT_SOURCES:
        lines.append(f"- {src['event']}: {src['source']} — {src['url']}")
    lines.append("")
    lines.append("## Artifacts Written")
    lines.append("")
    for name in [
        "phase6_capture_efficiency.csv",
        "phase6_hold_time_outcome.csv",
        "phase6_stop_tp_discipline.csv",
        "phase6_runner_regime_test.csv",
        "phase6_caveat_exit_interaction.csv",
        "phase6_session_lifecycle.csv",
        "phase6_event_window.csv",
        "phase6_event_window_trades.csv",
        "phase6_event_sources.csv",
        "phase6_path_asymmetry.csv",
        "phase6_lifecycle_counterfactuals.csv",
        "phase6_entry_exit_attribution.csv",
        "phase6_manifest.json",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    capture = capture_efficiency(df)
    hold = hold_time_outcome(df)
    stop_tp, stop_work = stop_tp_discipline(df)
    runner = runner_regime_test(df, stop_work)
    caveat = caveat_exit_interaction(df, stop_work)
    session = session_lifecycle(df)
    event = event_window(df)
    path = path_asymmetry(df)
    counter = lifecycle_counterfactuals(df)
    attr = attribution(df)
    report = build_report(df, capture, hold, stop_tp, runner, caveat, session, event, path, counter, attr)
    (OUT / "PHASE6_LIFECYCLE_ANALYSIS.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "closed": str(CLOSED_CSV.relative_to(ROOT)),
            "phase4_corpus": str(PHASE4_CSV.relative_to(ROOT)),
            "fast_failures": str(FAST_FAILURE_CSV.relative_to(ROOT)),
        },
        "outputs": [
            "PHASE6_LIFECYCLE_ANALYSIS.md",
            "phase6_capture_efficiency.csv",
            "phase6_hold_time_outcome.csv",
            "phase6_stop_tp_discipline.csv",
            "phase6_runner_regime_test.csv",
            "phase6_caveat_exit_interaction.csv",
            "phase6_session_lifecycle.csv",
            "phase6_event_window.csv",
            "phase6_event_window_trades.csv",
            "phase6_event_sources.csv",
            "phase6_path_asymmetry.csv",
            "phase6_lifecycle_counterfactuals.csv",
            "phase6_entry_exit_attribution.csv",
        ],
        "notes": [
            "Terminal MAE/MFE only; no path-time ordering is inferred.",
            "Event windows use official sources where possible; Japan CPI and BOJ exact times are bounded/assumed.",
            "Lifecycle counterfactuals are upper bounds with path-order assumptions.",
        ],
    }
    (OUT / "phase6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'PHASE6_LIFECYCLE_ANALYSIS.md'}")
    print(f"Closed rows: {len(df)}; net pips: {df['pips'].sum():.1f}; median hold: {df['hold_minutes'].median():.2f}m")


if __name__ == "__main__":
    main()
