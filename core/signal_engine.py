from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd

from .indicators import atr, ema, sma
from .profile import ProfileV1
from .timeframes import TIMEFRAMES, Timeframe


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Signal:
    signal_id: str
    profile_name: str
    symbol: str
    timeframe: Timeframe
    side: Side
    cross_time: pd.Timestamp
    confirm_time: pd.Timestamp
    entry_price_hint: float
    reasons: list[str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def drop_incomplete_last_bar(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """Drop last bar if it hasn't closed yet."""
    if df.empty:
        return df

    secs = TIMEFRAMES[tf].seconds
    t0 = pd.to_datetime(df["time"].iloc[-1], utc=True)
    bar_close = t0 + pd.Timedelta(seconds=secs)
    if pd.Timestamp.now(tz="UTC") < bar_close:
        return df.iloc[:-1].copy()
    return df


def compute_regime(df: pd.DataFrame, ema_period: int, sma_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"]
    e = ema(close, ema_period)
    s = sma(close, sma_period)
    diff = e - s
    return e, s, diff


def find_cross_events(diff: pd.Series) -> tuple[pd.Series, pd.Series]:
    cross_up = (diff.shift(1) <= 0) & (diff > 0)
    cross_dn = (diff.shift(1) >= 0) & (diff < 0)
    return cross_up.fillna(False), cross_dn.fillna(False)


def detect_latest_confirmed_cross_signal(
    *,
    profile: ProfileV1,
    df: pd.DataFrame,
    tf: Timeframe,
    ema_period: int,
    sma_period: int,
    confirm_bars: int,
    require_close_on_correct_side: bool,
    min_distance_pips: float,
    max_wait_bars: int,
) -> Optional[Signal]:
    """Detect the most recent confirmed cross and return a Signal if it confirms on the latest closed bars.

    Intended behavior for your workflow:
    - Cross occurs at bar i (closed).
    - Wait for confirmation candles after the cross.
    - Signal triggers when the last required confirmation candle closes.
    """
    if df.empty:
        return None

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = drop_incomplete_last_bar(df, tf)
    if len(df) < (sma_period + confirm_bars + 2):
        return None

    e, s, diff = compute_regime(df, ema_period, sma_period)
    cross_up, cross_dn = find_cross_events(diff)

    cross_indices = df.index[(cross_up | cross_dn)].tolist()
    if not cross_indices:
        return None

    # Evaluate crosses from latest to oldest; return first one that confirms on the most recent bars.
    last_idx = df.index[-1]
    for i in reversed(cross_indices):
        side: Side = "buy" if bool(cross_up.loc[i]) else "sell"
        cross_time = df.loc[i, "time"]

        # Confirmation window starts after the cross bar.
        start = i + 1
        end = min(i + max_wait_bars, last_idx)
        if start > end:
            continue

        for j in range(start, end + 1):
            # Need j..j+confirm_bars-1 to exist and end exactly at last closed bar to avoid old signals.
            k = j + confirm_bars - 1
            if k != last_idx:
                continue

            # Confirmation conditions must hold for each bar in the block.
            ok = True
            reasons: list[str] = []

            for t in range(j, k + 1):
                d = float(diff.loc[t])
                if pd.isna(d):
                    ok = False
                    break

                # side conditions on EMA-SMA diff
                if side == "buy" and d <= 0:
                    ok = False
                    break
                if side == "sell" and d >= 0:
                    ok = False
                    break

                # min separation
                if min_distance_pips > 0:
                    if abs(d) < (min_distance_pips * profile.pip_size):
                        ok = False
                        break

                # close on correct side of SMA (optional)
                if require_close_on_correct_side:
                    close = float(df.loc[t, "close"])
                    sma_val = float(s.loc[t])
                    if pd.isna(sma_val):
                        ok = False
                        break
                    if side == "buy" and close <= sma_val:
                        ok = False
                        break
                    if side == "sell" and close >= sma_val:
                        ok = False
                        break

            if not ok:
                continue

            confirm_time = df.loc[k, "time"]
            entry_price_hint = float(df.loc[k, "close"])

            reasons.append(f"{tf} EMA{ema_period}/SMA{sma_period} cross confirmed")
            reasons.append(f"confirm_bars={confirm_bars} max_wait_bars={max_wait_bars}")
            if require_close_on_correct_side:
                reasons.append("confirm_requires_close_on_correct_side=true")
            if min_distance_pips > 0:
                reasons.append(f"min_distance_pips={min_distance_pips}")

            signal_id = f"{profile.profile_name}:{profile.symbol}:{tf}:{side}:{cross_time.isoformat()}:{confirm_time.isoformat()}"

            return Signal(
                signal_id=signal_id,
                profile_name=profile.profile_name,
                symbol=profile.symbol,
                timeframe=tf,
                side=side,
                cross_time=cross_time,
                confirm_time=confirm_time,
                entry_price_hint=entry_price_hint,
                reasons=reasons,
            )

    return None


def compute_alignment_score(profile: ProfileV1, diffs: dict[Timeframe, float]) -> int:
    w = profile.strategy.filters.alignment.weights
    score = 0
    for tf, d in diffs.items():
        if d > 0:
            score += int(w[tf]) if tf in w else 1
        elif d < 0:
            score -= int(w[tf]) if tf in w else 1
    return score


def passes_alignment_filter(profile: ProfileV1, diffs: dict[Timeframe, float], side: Side) -> tuple[bool, str | None]:
    f = profile.strategy.filters.alignment
    if not f.enabled:
        return True, None

    score = compute_alignment_score(profile, diffs)
    if f.method == "strict":
        # Strict: require H4 and M15 to agree with side (ignore M1).
        required_tfs: list[Timeframe] = ["H4", "M15"]
        for tf in required_tfs:
            d = diffs.get(tf)
            if d is None:
                return False, f"alignment: missing {tf}"
            if side == "buy" and d <= 0:
                return False, f"alignment: {tf} not bullish"
            if side == "sell" and d >= 0:
                return False, f"alignment: {tf} not bearish"
        return True, None

    # Score-based
    min_score = int(f.min_score_to_trade)
    if side == "buy" and score < min_score:
        return False, f"alignment: score {score} < min {min_score} for BUY"
    if side == "sell" and score > -min_score:
        return False, f"alignment: score {score} > -min {min_score} for SELL"
    return True, None


def compute_latest_diffs(profile: ProfileV1, data_by_tf: dict[Timeframe, pd.DataFrame]) -> dict[Timeframe, float]:
    diffs: dict[Timeframe, float] = {}
    for tf, df in data_by_tf.items():
        # Not all loops/policies fetch only the profile's configured timeframes.
        # If a TF (e.g. M5) is present in data_by_tf but not defined in the profile,
        # skip it instead of crashing with KeyError.
        cfg = profile.strategy.timeframes.get(tf)
        if cfg is None:
            continue
        df2 = df.copy()
        df2["time"] = pd.to_datetime(df2["time"], utc=True)
        df2 = drop_incomplete_last_bar(df2, tf)
        if df2.empty:
            continue
        _, _, diff = compute_regime(df2, cfg.ema_fast, cfg.sma_slow)
        d = diff.iloc[-1]
        if pd.notna(d):
            diffs[tf] = float(d)
    return diffs


def evaluate_filters(profile: ProfileV1, data_by_tf: dict[Timeframe, pd.DataFrame], signal: Signal) -> tuple[bool, list[str]]:
    """Return (allowed, reasons). Reasons include both failures and context."""
    reasons: list[str] = []

    diffs = compute_latest_diffs(profile, data_by_tf)
    ok, reason = passes_alignment_filter(profile, diffs, signal.side)
    if not ok:
        return False, [reason] if reason else ["alignment: rejected"]
    if reason:
        reasons.append(reason)

    # Apply EMA stack + ATR filters on the signal timeframe (typically M1)
    df_tf = data_by_tf.get(signal.timeframe)
    if df_tf is not None:
        ok, r = passes_ema_stack_filter(profile, df_tf, signal.timeframe)
        if not ok:
            return False, [r or "ema_stack_filter: rejected"]
        ok, r = passes_atr_filter(profile, df_tf, signal.timeframe)
        if not ok:
            return False, [r or "atr_filter: rejected"]

    # Add context summary
    score = compute_alignment_score(profile, diffs) if diffs else 0
    reasons.append(f"alignment_score={score}")
    for tf, d in sorted(diffs.items()):
        reasons.append(f"{tf}_diff={d:+.6f}")
    return True, reasons


def passes_ema_stack_filter(profile: ProfileV1, df: pd.DataFrame, tf: Timeframe) -> tuple[bool, str | None]:
    f = profile.strategy.filters.ema_stack_filter
    if not f.enabled:
        return True, None
    if f.timeframe != tf:
        return True, None

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = drop_incomplete_last_bar(df, tf)
    if df.empty:
        return False, "ema_stack_filter: no data"

    close = df["close"]
    series = {p: ema(close, p) for p in f.periods}
    last = df.index[-1]

    # bull stack: ema8 > ema13 > ema21 ; bear stack reverse
    vals = [float(series[p].loc[last]) for p in f.periods]
    if any(pd.isna(v) for v in vals):
        return False, "ema_stack_filter: insufficient warmup"

    # Determine ordering and min separation
    sep = f.min_separation_pips * profile.pip_size
    bull = all(vals[i] > vals[i + 1] + sep for i in range(len(vals) - 1))
    bear = all(vals[i] < vals[i + 1] - sep for i in range(len(vals) - 1))

    if bull or bear:
        return True, None
    return False, "ema_stack_filter: not stacked (chop)"


def passes_atr_filter(profile: ProfileV1, df: pd.DataFrame, tf: Timeframe) -> tuple[bool, str | None]:
    f = profile.strategy.filters.atr_filter
    if not f.enabled:
        return True, None
    if f.timeframe != tf:
        return True, None

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = drop_incomplete_last_bar(df, tf)
    if len(df) < f.atr_period + 2:
        return False, "atr_filter: insufficient warmup"

    a = atr(df, f.atr_period)
    last_val = a.iloc[-1]
    if pd.isna(last_val):
        return False, "atr_filter: insufficient warmup"

    atr_pips = float(last_val / profile.pip_size)
    if atr_pips < f.min_atr_pips:
        return False, f"atr_filter: atr_pips {atr_pips:.2f} < min {f.min_atr_pips:.2f}"
    if f.max_atr_pips is not None and atr_pips > f.max_atr_pips:
        return False, f"atr_filter: atr_pips {atr_pips:.2f} > max {f.max_atr_pips:.2f}"
    return True, None

