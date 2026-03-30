from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.regime_classifier import RegimeThresholds
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_v44_conservative_router as v44_router
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import validate_regime_classifier as regime_validation


@dataclass(frozen=True)
class VariantKBaselineContext:
    m1: pd.DataFrame
    m1_idx: pd.DatetimeIndex
    m5_basic: pd.DataFrame
    classified_basic: pd.DataFrame
    classified_dynamic: pd.DataFrame
    dyn_time_idx: pd.DatetimeIndex


@dataclass(frozen=True)
class VariantKBaselineOutcome:
    accepted: list[Any] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    adjustments: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _normalize_ohlc(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _build_m5_from_m1(m1_df: pd.DataFrame) -> pd.DataFrame:
    if m1_df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    temp = m1_df.set_index("time").sort_index()
    out = (
        temp.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def build_variant_k_baseline_context(data_by_tf: dict[str, Any]) -> VariantKBaselineContext:
    m1 = _normalize_ohlc(data_by_tf.get("M1"))
    if m1.empty:
        return VariantKBaselineContext(
            m1=m1,
            m1_idx=pd.DatetimeIndex([]),
            m5_basic=pd.DataFrame(columns=["time", "open", "high", "low", "close"]),
            classified_basic=pd.DataFrame(columns=["time"]),
            classified_dynamic=pd.DataFrame(columns=["time", "sf_er", "sf_delta_er"]),
            dyn_time_idx=pd.DatetimeIndex([]),
        )
    featured = regime_validation.compute_features(m1)
    classified_basic = regime_validation.classify_all_bars(featured, RegimeThresholds())
    m5_basic = _normalize_ohlc(data_by_tf.get("M5"))
    if m5_basic.empty:
        m5_basic = _build_m5_from_m1(m1)
    m5_dynamic = variant_i._compute_dynamic_features_on_m5(m5_basic)
    dynamic_cols = ["time", "sf_er", "sf_delta_er"]
    classified_dynamic = pd.merge_asof(
        classified_basic.sort_values("time"),
        m5_dynamic[dynamic_cols].sort_values("time"),
        on="time",
        direction="backward",
    )
    return VariantKBaselineContext(
        m1=m1,
        m1_idx=pd.DatetimeIndex(m1["time"]),
        m5_basic=m5_basic,
        classified_basic=classified_basic,
        classified_dynamic=classified_dynamic,
        dyn_time_idx=pd.DatetimeIndex(classified_dynamic["time"]),
    )


def _trade_row_from_candidate(candidate: Any) -> merged_engine.TradeRow:
    entry_time = pd.Timestamp(getattr(candidate, "entry_time_utc", None)).tz_convert("UTC")
    raw = dict(getattr(candidate, "raw_result", {}) or {})
    raw.setdefault("entry_profile", getattr(candidate, "strategy_tag", ""))
    return merged_engine.TradeRow(
        strategy=str(getattr(candidate, "strategy_family", "")),
        entry_time=entry_time,
        exit_time=entry_time,
        entry_session=str(getattr(candidate, "strategy_family", "")),
        side=str(getattr(candidate, "side", "")),
        pips=0.0,
        usd=0.0,
        exit_reason="runtime_open",
        standalone_entry_equity=100000.0,
        raw=raw,
        size_scale=1.0,
    )


def _reject(candidate: Any, *, reason: str, stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "identity": getattr(candidate, "identity", None),
        "strategy_tag": getattr(candidate, "strategy_tag", None),
        "intent_source": "baseline",
        "slice_id": getattr(candidate, "slice_id", None),
        "reason": reason,
        "variant_k_stage": stage,
    }
    if extra:
        payload.update(extra)
    return payload


def apply_variant_k_baseline_admission(
    baseline_candidates: list[Any],
    *,
    data_by_tf: dict[str, Any],
) -> VariantKBaselineOutcome:
    context = build_variant_k_baseline_context(data_by_tf)
    if not baseline_candidates:
        return VariantKBaselineOutcome()

    accepted: list[Any] = []
    rejected: list[dict[str, Any]] = []
    adjustments: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []

    for candidate in baseline_candidates:
        strategy_family = str(getattr(candidate, "strategy_family", "") or "")
        entry_time = pd.Timestamp(getattr(candidate, "entry_time_utc", None))
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")

        # Variant F: V44-only filter.
        if strategy_family == "v44_ny":
            trade = _trade_row_from_candidate(candidate)
            filtered = v44_router._filter_v44_trade(
                trade,
                context.classified_basic,
                context.m5_basic,
                block_breakout=True,
                block_post_breakout=True,
                block_ambiguous_non_momentum=True,
                momentum_only=False,
                exhaustion_gate=False,
                soft_exhaustion=False,
                er_threshold=0.35,
                decay_threshold=0.40,
            )
            diagnostics.append(
                {
                    "identity": getattr(candidate, "identity", None),
                    "stage": "variant_f",
                    "blocked": bool(filtered.blocked),
                    "reason": str(filtered.reason),
                    "regime_label": str(filtered.regime_label),
                    "er": float(filtered.er),
                    "decay": float(filtered.decay),
                }
            )
            if filtered.blocked:
                rejected.append(
                    _reject(
                        candidate,
                        reason=f"variant_k: {filtered.reason}",
                        stage="variant_f_v44_filter",
                        extra={
                            "regime_label": str(filtered.regime_label),
                            "er": round(float(filtered.er), 4),
                            "decay": round(float(filtered.decay), 4),
                        },
                    )
                )
                continue

        # Variant K London cluster block.
        if strategy_family == "london_v2" and not context.classified_dynamic.empty and len(context.dyn_time_idx) > 0:
            lookup_idx = context.dyn_time_idx.get_indexer([entry_time], method="ffill")[0]
            if lookup_idx >= 0:
                regime_info = variant_i._lookup_regime_with_dynamic(context.classified_dynamic, context.dyn_time_idx, entry_time)
                full_row = context.classified_dynamic.iloc[lookup_idx]
                er = float(full_row.get("sf_er", 0.5))
                if np.isnan(er):
                    er = 0.5
                er_label = variant_k._er_bucket(er)
                der_label = variant_k._der_bucket(regime_info["delta_er"])
                cluster_cell = (regime_info["regime_label"], er_label, der_label)
                diagnostics.append(
                    {
                        "identity": getattr(candidate, "identity", None),
                        "stage": "variant_k_cluster",
                        "blocked": cluster_cell in variant_k.LONDON_BLOCK_CLUSTER,
                        "cluster_cell": list(cluster_cell),
                    }
                )
                if cluster_cell in variant_k.LONDON_BLOCK_CLUSTER:
                    rejected.append(
                        _reject(
                            candidate,
                            reason=f"blocked_london_cluster_{regime_info['regime_label']}_{er_label}_{der_label}",
                            stage="variant_k_london_cluster",
                            extra={
                                "regime_label": str(regime_info["regime_label"]),
                                "er_bucket": er_label,
                                "der_bucket": der_label,
                                "delta_er": round(float(regime_info["delta_er"]), 4),
                            },
                        )
                    )
                    continue
                adjustments[str(getattr(candidate, "identity", ""))] = {
                    "raw_result_updates": {
                        "variant_k_baseline": "G_holdthrough",
                        "variant_k_holdthrough_enabled": True,
                        "variant_k_holdthrough_rule": "london_v2_be1_trail8_session_close",
                    }
                }

        # Variant I global standdown.
        if not context.classified_dynamic.empty and len(context.dyn_time_idx) > 0:
            regime_info = variant_i._lookup_regime_with_dynamic(context.classified_dynamic, context.dyn_time_idx, entry_time)
            diagnostics.append(
                {
                    "identity": getattr(candidate, "identity", None),
                    "stage": "variant_i_global",
                    "blocked": bool(variant_i._is_variant_i_blocked(regime_info["regime_label"], regime_info["delta_er"])),
                    "regime_label": str(regime_info["regime_label"]),
                    "delta_er": round(float(regime_info["delta_er"]), 4),
                }
            )
            if variant_i._is_variant_i_blocked(regime_info["regime_label"], regime_info["delta_er"]):
                rejected.append(
                    _reject(
                        candidate,
                        reason="blocked_global_pbt_der_neg",
                        stage="variant_i_global_standdown",
                        extra={
                            "regime_label": str(regime_info["regime_label"]),
                            "delta_er": round(float(regime_info["delta_er"]), 4),
                        },
                    )
                )
                continue

        accepted.append(candidate)

    return VariantKBaselineOutcome(
        accepted=accepted,
        rejected=rejected,
        adjustments=adjustments,
        diagnostics=diagnostics,
    )
