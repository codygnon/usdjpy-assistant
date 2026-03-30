from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core.ownership_table import cell_key, der_bucket, er_bucket
from core.regime_classifier import RegimeClassifier, RegimeResult
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.signal_engine import drop_incomplete_last_bar

REGIME_BB_PERIOD = 25
REGIME_BB_STD = 2.2
REGIME_BB_WIDTH_LOOKBACK = 100
REGIME_M5_EMA_FAST = 9
REGIME_SLOPE_BARS = 4
REGIME_H1_EMA_FAST = 20
REGIME_H1_EMA_SLOW = 50
REGIME_V44_BLOCK_LABELS = frozenset({"breakout", "post_breakout_trend"})
LONDON_V2_OWNERSHIP_BLOCK_CELLS = frozenset(
    {
        ("breakout", "er_low", "der_neg"),
        ("momentum", "er_mid", "der_neg"),
        ("ambiguous", "er_high", "der_neg"),
    }
)


def _drop_incomplete_tf(df: Optional[pd.DataFrame], tf: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d.dropna(subset=["time"]).sort_values("time")
    if d.empty:
        return d
    try:
        return drop_incomplete_last_bar(d, tf)  # type: ignore[arg-type]
    except Exception:
        return d


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    adx = dx.rolling(window=period, min_periods=period).mean()
    val = float(adx.iloc[-1]) if len(adx.dropna()) else 0.0
    return 0.0 if not np.isfinite(val) else val


def compute_regime_snapshot(
    m5_df: pd.DataFrame,
    m15_df: Optional[pd.DataFrame],
    h1_df: pd.DataFrame,
    pip_size: float,
    *,
    replay_m5_bars: int = 60,
    repeats_per_m5_bar: int = 5,
) -> tuple[str, dict[str, float], float]:
    min_m5 = REGIME_BB_PERIOD + REGIME_BB_WIDTH_LOOKBACK + REGIME_SLOPE_BARS + 5
    if m5_df is None or len(m5_df) < min_m5:
        return "unknown", {}, 0.0

    replay_df = m5_df.tail(max(replay_m5_bars, min_m5)).reset_index(drop=True)
    clf = RegimeClassifier()
    last_res: Optional[RegimeResult] = None

    m15_time_idx = pd.DatetimeIndex(m15_df["time"]) if m15_df is not None and "time" in m15_df.columns else None
    h1_time_idx = pd.DatetimeIndex(h1_df["time"]) if h1_df is not None and "time" in h1_df.columns else None

    for i in range(len(replay_df)):
        cur = replay_df.iloc[: i + 1]
        close = cur["close"].astype(float)

        sma = close.rolling(REGIME_BB_PERIOD, min_periods=REGIME_BB_PERIOD).mean()
        std = close.rolling(REGIME_BB_PERIOD, min_periods=REGIME_BB_PERIOD).std(ddof=0)
        bb_upper = sma + REGIME_BB_STD * std
        bb_lower = sma - REGIME_BB_STD * std
        bb_mid = sma
        bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        bb_width_pctile_s = bb_width.rolling(
            REGIME_BB_WIDTH_LOOKBACK, min_periods=REGIME_BB_WIDTH_LOOKBACK,
        ).rank(pct=True)
        bb_width_pctile = float(bb_width_pctile_s.iloc[-1])
        if np.isnan(bb_width_pctile):
            continue

        cutoff = bb_width.rolling(
            REGIME_BB_WIDTH_LOOKBACK, min_periods=REGIME_BB_WIDTH_LOOKBACK,
        ).quantile(0.80)
        cutoff_last = float(cutoff.iloc[-1]) if pd.notna(cutoff.iloc[-1]) else float("nan")
        width_last = float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else float("nan")
        if np.isnan(cutoff_last) or np.isnan(width_last):
            continue
        bb_regime = "ranging" if width_last < cutoff_last else "trending"

        ema_fast = _compute_ema(close, REGIME_M5_EMA_FAST)
        sb = max(1, REGIME_SLOPE_BARS)
        if len(ema_fast) > sb:
            m5_slope = (float(ema_fast.iloc[-1]) - float(ema_fast.iloc[-1 - sb])) / pip_size / sb
        else:
            m5_slope = 0.0
        m5_slope_abs = abs(m5_slope)

        bar_ts = pd.Timestamp(cur["time"].iloc[-1])

        adx = 0.0
        if m15_df is not None and m15_time_idx is not None and len(m15_df) >= 14:
            j = m15_time_idx.get_indexer([bar_ts], method="ffill")[0]
            if j >= 13:
                adx = _compute_adx(m15_df.iloc[: j + 1], 14)
        elif len(cur) >= 42:
            adx = _compute_adx(cur, 42)

        h1_aligned = False
        if h1_df is not None and h1_time_idx is not None and len(h1_df) >= max(REGIME_H1_EMA_FAST, REGIME_H1_EMA_SLOW) + 2:
            j = h1_time_idx.get_indexer([bar_ts], method="ffill")[0]
            if j >= max(REGIME_H1_EMA_FAST, REGIME_H1_EMA_SLOW):
                h1_cur = h1_df.iloc[: j + 1]
                h1_close = h1_cur["close"].astype(float)
                h1_fast = _compute_ema(h1_close, REGIME_H1_EMA_FAST)
                h1_slow = _compute_ema(h1_close, REGIME_H1_EMA_SLOW)
                h1_bull = float(h1_fast.iloc[-1]) > float(h1_slow.iloc[-1])
                h1_bear = float(h1_fast.iloc[-1]) < float(h1_slow.iloc[-1])
                h1_aligned = (m5_slope > 0 and h1_bull) or (m5_slope < 0 and h1_bear)

        for _ in range(max(1, repeats_per_m5_bar)):
            last_res = clf.update(adx, m5_slope_abs, h1_aligned, bb_width_pctile, bb_regime)

    if last_res is None:
        return "unknown", {}, 0.0
    return last_res.label, last_res.scores.as_dict, float(last_res.margin)


def compute_ownership_cell(
    m5_df: pd.DataFrame,
    m15_df: Optional[pd.DataFrame],
    h1_df: pd.DataFrame,
    pip_size: float,
) -> dict[str, Any]:
    out = {
        "regime_label": "unknown",
        "regime_margin": 0.0,
        "er": None,
        "delta_er": None,
        "er_bucket": None,
        "der_bucket": None,
        "ownership_cell": None,
    }
    if m5_df is None or m5_df.empty:
        return out

    label, _scores, margin = compute_regime_snapshot(m5_df, m15_df, h1_df, pip_size)
    out["regime_label"] = label
    out["regime_margin"] = round(float(margin), 4)

    er = float(compute_efficiency_ratio(m5_df, lookback=12))
    delta_er = float(compute_delta_efficiency_ratio(m5_df, lookback=12, delta_bars=3))
    if not np.isfinite(er):
        er = 0.5
    if not np.isfinite(delta_er):
        delta_er = 0.0
    out["er"] = round(er, 4)
    out["delta_er"] = round(delta_er, 4)
    eb = er_bucket(er)
    db = der_bucket(delta_er)
    out["er_bucket"] = eb
    out["der_bucket"] = db
    if label != "unknown":
        out["ownership_cell"] = cell_key(label, eb, db)
    return out


def v44_regime_block(
    m5_df: pd.DataFrame,
    m15_df: Optional[pd.DataFrame],
    h1_df: pd.DataFrame,
    pip_size: float,
) -> tuple[bool, str, str]:
    label, scores, _margin = compute_regime_snapshot(m5_df, m15_df, h1_df, pip_size)
    if label == "unknown":
        return False, "unknown", ""
    if label in REGIME_V44_BLOCK_LABELS:
        return True, label, f"regime block ({label})"
    if label == "ambiguous" and isinstance(scores, dict) and scores:
        top_regime = max(scores, key=scores.get)
        if top_regime != "momentum":
            return True, label, f"regime block (ambiguous_top_{top_regime})"
    return False, label, ""


def london_v2_ownership_cluster_block(
    m5_df: pd.DataFrame,
    m15_df: Optional[pd.DataFrame],
    h1_df: pd.DataFrame,
    pip_size: float,
) -> tuple[bool, str]:
    cell_info = compute_ownership_cell(m5_df, m15_df, h1_df, pip_size)
    label = str(cell_info.get("regime_label") or "unknown")
    if label == "unknown" or m5_df is None or len(m5_df) < 15:
        return False, ""
    cell = (
        label,
        str(cell_info.get("er_bucket") or ""),
        str(cell_info.get("der_bucket") or ""),
    )
    if cell in LONDON_V2_OWNERSHIP_BLOCK_CELLS:
        return True, f"ownership block ({label}/{cell[1]}/{cell[2]})"
    return False, ""


def global_regime_standdown(
    m5_df: pd.DataFrame,
    m15_df: Optional[pd.DataFrame],
    h1_df: pd.DataFrame,
    pip_size: float,
) -> tuple[bool, str]:
    label, _scores, _margin = compute_regime_snapshot(m5_df, m15_df, h1_df, pip_size)
    if label != "post_breakout_trend":
        return False, ""
    delta_er = compute_delta_efficiency_ratio(m5_df, lookback=12, delta_bars=3)
    if delta_er < 0:
        return True, f"global standdown (post_breakout_trend + ΔER={delta_er:.3f})"
    return False, ""


def compute_phase3_ownership_audit_for_data(data_by_tf: dict[str, Any], pip_size: float) -> dict[str, Any]:
    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    m15_df = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
    h1_df = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
    out: dict[str, Any] = {
        "schema": "phase3_ownership_audit_v1",
        "regime_label": "unknown",
        "regime_margin": 0.0,
        "er": None,
        "delta_er": None,
        "er_bucket": None,
        "der_bucket": None,
        "ownership_cell": None,
        "defensive_v44_regime_block": False,
        "defensive_v44_regime_reason": "",
        "defensive_london_cluster_block": False,
        "defensive_london_cluster_reason": "",
        "defensive_global_standdown": False,
        "defensive_global_standdown_reason": "",
    }
    if m5_df is None or m5_df.empty:
        return out

    out.update(compute_ownership_cell(m5_df, m15_df, h1_df, pip_size))

    vb, _vl, vr = v44_regime_block(m5_df, m15_df, h1_df, pip_size)
    out["defensive_v44_regime_block"] = bool(vb)
    out["defensive_v44_regime_reason"] = str(vr) if vb else ""

    lb, lr = london_v2_ownership_cluster_block(m5_df, m15_df, h1_df, pip_size)
    out["defensive_london_cluster_block"] = bool(lb)
    out["defensive_london_cluster_reason"] = str(lr) if lb else ""

    sb, sr = global_regime_standdown(m5_df, m15_df, h1_df, pip_size)
    out["defensive_global_standdown"] = bool(sb)
    out["defensive_global_standdown_reason"] = str(sr) if sb else ""

    return out
