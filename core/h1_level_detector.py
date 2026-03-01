"""H1 Support/Resistance Level Detection Module.

Detects, tracks, and manages H1 S/R levels for the Uncle Parsh H1 Breakout Scalper.
Supports yesterday high/low, swing highs/lows, and touch-frequency clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


@dataclass
class H1Level:
    price: float              # Level price (wick-based)
    level_type: str           # "yesterday_high", "yesterday_low", "swing_high", "swing_low", "cluster_resistance", "cluster_support"
    touch_count: int = 0      # Number of bounce touches
    is_broken: bool = False   # Power Close occurred
    break_direction: str = ""  # "bull" or "bear"
    break_time: Optional[str] = None  # ISO timestamp of Power Close
    velocity_type: Optional[str] = None  # "power" or "sniper" (set after 2nd M5 bar)
    velocity_checked: bool = False  # Whether velocity has been determined
    power_close_bar_index: Optional[int] = None  # M5 bar index of Power Close (for velocity timing)
    voided: bool = False      # True if 35 EMA veto killed this setup
    entry_mode: Optional[str] = None  # "power_break" or "sniper" (set after velocity check)
    state: str = "watching"   # "watching", "catalyst", "ready", "voided"

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "level_type": self.level_type,
            "touch_count": self.touch_count,
            "is_broken": self.is_broken,
            "break_direction": self.break_direction,
            "break_time": self.break_time,
            "velocity_type": self.velocity_type,
            "velocity_checked": self.velocity_checked,
            "power_close_bar_index": self.power_close_bar_index,
            "voided": self.voided,
            "entry_mode": self.entry_mode,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "H1Level":
        return cls(
            price=float(d["price"]),
            level_type=str(d.get("level_type", "unknown")),
            touch_count=int(d.get("touch_count", 0)),
            is_broken=bool(d.get("is_broken", False)),
            break_direction=str(d.get("break_direction", "")),
            break_time=d.get("break_time"),
            velocity_type=d.get("velocity_type"),
            velocity_checked=bool(d.get("velocity_checked", False)),
            power_close_bar_index=d.get("power_close_bar_index"),
            voided=bool(d.get("voided", False)),
            entry_mode=d.get("entry_mode"),
            state=str(d.get("state", "watching")),
        )


# Priority for deduplication: higher is stronger
_LEVEL_PRIORITY = {
    "yesterday_high": 3,
    "yesterday_low": 3,
    "cluster_resistance": 2,
    "cluster_support": 2,
    "swing_high": 1,
    "swing_low": 1,
}


class H1LevelDetector:
    def __init__(self, config: dict) -> None:
        self.lookback_hours: int = config.get("h1_lookback_hours", 48)
        self.swing_strength: int = config.get("h1_swing_strength", 3)
        self.cluster_tolerance_pips: float = config.get("h1_cluster_tolerance_pips", 5.0)
        self.min_touches_for_major: int = config.get("h1_min_touches_for_major", 2)
        self.min_distance_between_levels_pips: float = config.get("h1_min_distance_between_levels_pips", 10.0)
        self.pip_size: float = config.get("pip_size", 0.01)

    def detect_levels(self, h1_df: pd.DataFrame, current_date: str) -> list[H1Level]:
        """Detect all tradeable H1 levels from H1 data.

        1. Yesterday's High/Low
        2. H1 swing highs/lows (wick > N adjacent bars' wicks on each side)
        3. Touch frequency clustering
        Dedup levels within cluster_tolerance_pips of each other.
        """
        if h1_df is None or h1_df.empty or len(h1_df) < 3:
            return []

        df = h1_df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        # Limit to lookback
        lookback_bars = self.lookback_hours  # 1 H1 bar = 1 hour
        if len(df) > lookback_bars:
            df = df.iloc[-lookback_bars:].reset_index(drop=True)

        levels: list[H1Level] = []
        tol = self.cluster_tolerance_pips * self.pip_size

        # 1. Yesterday's High/Low
        yh, yl = self._detect_yesterday_hl(df, current_date)
        if yh is not None:
            levels.append(H1Level(price=yh, level_type="yesterday_high"))
        if yl is not None:
            levels.append(H1Level(price=yl, level_type="yesterday_low"))

        # 2. Swing highs/lows
        swing_levels = self._detect_swing_levels(df)
        levels.extend(swing_levels)

        # 3. Touch frequency clustering
        cluster_levels = self._detect_cluster_levels(df)
        levels.extend(cluster_levels)

        # Dedup: merge levels within tolerance, keeping stronger ones
        levels = self._dedup_levels(levels, tol)

        return levels

    def _detect_yesterday_hl(self, df: pd.DataFrame, current_date: str) -> tuple[Optional[float], Optional[float]]:
        """Find yesterday's high and low from H1 bars."""
        try:
            current = pd.Timestamp(current_date, tz="UTC").date()
        except Exception:
            current = pd.Timestamp.now(tz="UTC").date()

        yesterday_bars = df[df["time"].dt.date < current]
        if yesterday_bars.empty:
            return None, None

        # Get the last trading day's bars
        last_day = yesterday_bars["time"].dt.date.iloc[-1]
        day_bars = yesterday_bars[yesterday_bars["time"].dt.date == last_day]
        if day_bars.empty:
            return None, None

        yh = float(day_bars["high"].max())
        yl = float(day_bars["low"].min())
        return yh, yl

    def _detect_swing_levels(self, df: pd.DataFrame) -> list[H1Level]:
        """Detect swing highs and lows based on wick extremes."""
        levels: list[H1Level] = []
        n = self.swing_strength

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        for i in range(n, len(df) - n):
            # Swing high: bar's high > N bars on each side
            is_swing_high = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                levels.append(H1Level(price=float(highs[i]), level_type="swing_high"))

            # Swing low: bar's low < N bars on each side
            is_swing_low = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                levels.append(H1Level(price=float(lows[i]), level_type="swing_low"))

        return levels

    def _detect_cluster_levels(self, df: pd.DataFrame) -> list[H1Level]:
        """Detect S/R levels via touch frequency clustering.

        1. Collect all H1 wick highs and lows
        2. Cluster within tolerance
        3. Count bounces (candle touched zone but closed on opposite side)
        4. Clusters with >= min_touches bounces -> S/R level
        """
        tol = self.cluster_tolerance_pips * self.pip_size
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        opens = df["open"].astype(float).values
        closes = df["close"].astype(float).values

        # Collect all wick extremes
        all_prices = list(highs) + list(lows)
        all_prices.sort()

        if not all_prices:
            return []

        # Cluster prices within tolerance
        clusters: list[list[float]] = []
        current_cluster: list[float] = [all_prices[0]]
        for p in all_prices[1:]:
            if abs(p - current_cluster[-1]) <= tol:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
        clusters.append(current_cluster)

        levels: list[H1Level] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            centroid = sum(cluster) / len(cluster)

            # Count bounces: candle touches the zone but reverses
            bounce_count = 0
            for i in range(len(df)):
                h, l, o, c = highs[i], lows[i], opens[i], closes[i]
                touched_from_below = (h >= centroid - tol) and (l < centroid)
                touched_from_above = (l <= centroid + tol) and (h > centroid)
                if touched_from_below and c < centroid:
                    bounce_count += 1  # Touched resistance, closed below
                elif touched_from_above and c > centroid:
                    bounce_count += 1  # Touched support, closed above

            if bounce_count >= self.min_touches_for_major:
                # Determine if this is resistance or support based on where price is relative
                last_close = closes[-1] if len(closes) > 0 else centroid
                if last_close < centroid:
                    level_type = "cluster_resistance"
                else:
                    level_type = "cluster_support"
                levels.append(H1Level(
                    price=round(centroid, 5),
                    level_type=level_type,
                    touch_count=bounce_count,
                ))

        return levels

    def _dedup_levels(self, levels: list[H1Level], tolerance: float) -> list[H1Level]:
        """Remove duplicate levels within tolerance, keeping stronger ones.

        Priority: yesterday_high/low > high-touch clusters > swing points.
        Also enforce min_distance_between_levels_pips.
        """
        if not levels:
            return []

        # Sort by priority (descending) then touch count (descending)
        levels.sort(key=lambda lv: (-_LEVEL_PRIORITY.get(lv.level_type, 0), -lv.touch_count))

        min_dist = self.min_distance_between_levels_pips * self.pip_size
        result: list[H1Level] = []
        for lv in levels:
            too_close = False
            for existing in result:
                if abs(lv.price - existing.price) < max(tolerance, min_dist):
                    too_close = True
                    break
            if not too_close:
                result.append(lv)

        return result

    def check_power_close(self, level: H1Level, m5_candle: dict, pip_size: float) -> bool:
        """Check if M5 candle has 25%+ body past the level.

        Body = abs(close - open). For bull break (close > level):
        body_past = close - level. Ratio = body_past / body_total >= threshold.
        Must close BEYOND the level (not just wick through).
        """
        o = float(m5_candle.get("open", 0))
        c = float(m5_candle.get("close", 0))
        body = abs(c - o)
        if body < pip_size * 0.1:
            return False  # Doji — skip

        is_bull_candle = c > o

        if is_bull_candle and c > level.price:
            # Bull breakout above level
            body_past = c - level.price
            if body_past > 0 and (body_past / body) >= 0.25:
                return True
        elif not is_bull_candle and c < level.price:
            # Bear breakout below level
            body_past = level.price - c
            if body_past > 0 and (body_past / body) >= 0.25:
                return True

        return False

    def check_velocity(self, level: H1Level, second_m5_close: float, pip_size: float, velocity_pips: float) -> str:
        """At close of 2nd M5 bar after Power Close:
        If distance from level > velocity_pips -> "power"
        Else -> "sniper"
        """
        distance_pips = abs(second_m5_close - level.price) / pip_size
        if distance_pips > velocity_pips:
            return "power"
        return "sniper"

    def reset_daily(self) -> list[H1Level]:
        """Return empty list for new trading day."""
        return []
