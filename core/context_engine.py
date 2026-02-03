from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .profile import ProfileV1
from .signal_engine import compute_regime, drop_incomplete_last_bar, find_cross_events
from .timeframes import Timeframe


@dataclass(frozen=True)
class TfContext:
    regime: str
    last_cross_dir: Optional[str]
    last_cross_time: Optional[pd.Timestamp]
    last_cross_price: Optional[float]
    trend_since_cross: str
    diff: Optional[float]


# Default EMA configs for analysis timeframes not in profile
_DEFAULT_TF_INDICATORS = {
    "M1": {"ema_fast": 13, "sma_slow": 30, "ema_stack": [8, 13, 21]},
    "M3": {"ema_fast": 13, "sma_slow": 30, "ema_stack": [8, 13, 21]},
    "M5": {"ema_fast": 13, "sma_slow": 30, "ema_stack": [8, 13, 21]},
    "M15": {"ema_fast": 13, "sma_slow": 30, "ema_stack": [13, 21, 34]},
    "M30": {"ema_fast": 21, "sma_slow": 50, "ema_stack": [21, 50, 89]},
    "H1": {"ema_fast": 21, "sma_slow": 50, "ema_stack": [21, 50, 200]},
    "H4": {"ema_fast": 21, "sma_slow": 50, "ema_stack": [21, 50, 200]},
}


def _get_tf_config(profile: ProfileV1, tf: Timeframe) -> "TimeframeIndicators":
    from .profile import TimeframeIndicators
    if tf in profile.strategy.timeframes:
        return profile.strategy.timeframes[tf]
    d = _DEFAULT_TF_INDICATORS.get(tf, {"ema_fast": 13, "sma_slow": 30, "ema_stack": None})
    return TimeframeIndicators(**d)


def compute_tf_context(profile: ProfileV1, tf: Timeframe, df: pd.DataFrame) -> TfContext:
    cfg = _get_tf_config(profile, tf)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = drop_incomplete_last_bar(df, tf)
    if df.empty:
        return TfContext(regime="unknown", last_cross_dir=None, last_cross_time=None, last_cross_price=None, trend_since_cross="unknown", diff=None)

    e, s, diff = compute_regime(df, cfg.ema_fast, cfg.sma_slow)
    d_last = diff.iloc[-1]
    if pd.isna(d_last):
        regime = "unknown"
        d_last_f = None
    elif d_last > 0:
        regime = "bull"
        d_last_f = float(d_last)
    elif d_last < 0:
        regime = "bear"
        d_last_f = float(d_last)
    else:
        regime = "flat"
        d_last_f = 0.0

    cross_up, cross_dn = find_cross_events(diff)
    cross_idx = df.index[(cross_up | cross_dn)].tolist()
    if not cross_idx:
        return TfContext(regime=regime, last_cross_dir=None, last_cross_time=None, last_cross_price=None, trend_since_cross="unknown", diff=d_last_f)

    i = cross_idx[-1]
    last_cross_dir = "up" if bool(cross_up.loc[i]) else "down"
    last_cross_time = pd.to_datetime(df.loc[i, "time"], utc=True)
    last_cross_price = float(df.loc[i, "close"])

    after = df.loc[i:].copy()
    after_e, after_s, after_diff = compute_regime(after, cfg.ema_fast, cfg.sma_slow)
    after_diff = after_diff.dropna()
    if len(after_diff) < 10:
        trend_since = "unknown"
    else:
        frac_pos = float((after_diff > 0).mean())
        frac_neg = float((after_diff < 0).mean())
        if frac_pos >= 0.65:
            trend_since = "up"
        elif frac_neg >= 0.65:
            trend_since = "down"
        else:
            trend_since = "sideways"

    return TfContext(
        regime=regime,
        last_cross_dir=last_cross_dir,
        last_cross_time=last_cross_time,
        last_cross_price=last_cross_price,
        trend_since_cross=trend_since,
        diff=d_last_f,
    )

