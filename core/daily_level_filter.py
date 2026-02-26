"""
Daily Level Filter: blocks trades near yesterday/today high/low until breakout is confirmed.

- Buffer: block longs within buffer_pips below watched high; block shorts within buffer_pips above watched low.
- Breakout: allow trades in breakout direction only after N consecutive closed M5 candles close beyond the level.
- Rolling: once a level is confirmed broken, the watched level rolls to today's high/low; re-arms when today's level updates.
- Daily reset: at midnight UTC, reset state and load yesterday's high/low from D1.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


class DailyLevelFilter:
    """Filter that prevents taking trades right at the highs/lows of the day until breakout is confirmed."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        buffer_pips: float = 3.0,
        breakout_candles_required: int = 2,
        pip_size: float = 0.01,
    ):
        self.enabled = enabled
        self.buffer_pips = buffer_pips
        self.breakout_candles_required = breakout_candles_required
        self.pip_size = pip_size

        # State (persisted across iterations)
        self._date_utc: Optional[str] = None
        self._watched_high: Optional[float] = None
        self._watched_low: Optional[float] = None
        self._high_breakout_confirmed: bool = False
        self._low_breakout_confirmed: bool = False

    def reset(self, yesterday_high: Optional[float], yesterday_low: Optional[float]) -> None:
        """Call at start of new trading day (midnight UTC). Resets watched levels to yesterday's H/L."""
        self._watched_high = yesterday_high
        self._watched_low = yesterday_low
        self._high_breakout_confirmed = False
        self._low_breakout_confirmed = False

    def get_state_snapshot(self) -> dict[str, Any]:
        """For logging and frontend."""
        return {
            "date_utc": self._date_utc,
            "watched_high": self._watched_high,
            "watched_low": self._watched_low,
            "high_breakout_confirmed": self._high_breakout_confirmed,
            "low_breakout_confirmed": self._low_breakout_confirmed,
        }

    def should_allow_trade(
        self,
        direction: str,
        current_price: float,
        yesterday_high: Optional[float],
        yesterday_low: Optional[float],
        today_high: Optional[float],
        today_low: Optional[float],
        m5_closed_candles: pd.DataFrame,
        *,
        current_date_utc: Optional[str] = None,
        store=None,
        profile_name: str = "",
        symbol: str = "",
        mode: str = "",
        rule_id: str = "",
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        direction: "buy" or "sell"
        m5_closed_candles: M5 dataframe with only closed bars (exclude current forming candle).
        """
        if not self.enabled:
            reason = "daily_level_filter: disabled"
            self._log_decision(True, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
            return True, reason

        if yesterday_high is None and yesterday_low is None:
            reason = "daily_level_filter_skipped: no yesterday data"
            self._log_decision(True, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
            return True, reason

        # Initialize or sync watched levels from yesterday on first use or after date change
        if current_date_utc and current_date_utc != self._date_utc:
            self._date_utc = current_date_utc
            self.reset(yesterday_high, yesterday_low)

        if self._watched_high is None:
            self._watched_high = yesterday_high
        if self._watched_low is None:
            self._watched_low = yesterday_low

        buffer = self.buffer_pips * self.pip_size
        n_required = max(1, self.breakout_candles_required)

        # Rolling: if today's high is above watched_high, re-arm at today's high
        if today_high is not None and self._watched_high is not None and today_high > self._watched_high:
            self._watched_high = today_high
            self._high_breakout_confirmed = False
        # Rolling: if today's low is below watched_low, re-arm at today's low
        if today_low is not None and self._watched_low is not None and today_low < self._watched_low:
            self._watched_low = today_low
            self._low_breakout_confirmed = False

        # Check breakout confirmation from closed M5 only (last n_required consecutive closes)
        if m5_closed_candles is not None and len(m5_closed_candles) >= n_required:
            closes = m5_closed_candles["close"].astype(float)
            last_n = closes.iloc[-n_required:]
            if self._watched_high is not None and all(c > self._watched_high for c in last_n):
                self._high_breakout_confirmed = True
                # Roll to today so next time we watch today's high
                if today_high is not None:
                    self._watched_high = today_high
            if self._watched_low is not None and all(c < self._watched_low for c in last_n):
                self._low_breakout_confirmed = True
                if today_low is not None:
                    self._watched_low = today_low

        # Buffer: block longs within buffer below watched high
        if self._watched_high is not None and direction == "buy":
            if current_price >= self._watched_high - buffer and current_price < self._watched_high:
                reason = f"daily_level_filter: long blocked within {self.buffer_pips} pips below watched_high={self._watched_high:.5f}"
                self._log_decision(False, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
                return False, reason
            if not self._high_breakout_confirmed:
                # Price above watched_high but not yet confirmed
                if current_price >= self._watched_high:
                    reason = f"daily_level_filter: long blocked until {n_required} closed M5 above watched_high={self._watched_high:.5f}"
                    self._log_decision(False, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
                    return False, reason

        # Buffer: block shorts within buffer above watched low
        if self._watched_low is not None and direction == "sell":
            if current_price <= self._watched_low + buffer and current_price > self._watched_low:
                reason = f"daily_level_filter: short blocked within {self.buffer_pips} pips above watched_low={self._watched_low:.5f}"
                self._log_decision(False, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
                return False, reason
            if not self._low_breakout_confirmed:
                if current_price <= self._watched_low:
                    reason = f"daily_level_filter: short blocked until {n_required} closed M5 below watched_low={self._watched_low:.5f}"
                    self._log_decision(False, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
                    return False, reason

        reason = "daily_level_filter: allowed"
        self._log_decision(True, direction, current_price, reason, store, profile_name, symbol, mode, rule_id)
        return True, reason

    def _log_decision(
        self,
        allowed: bool,
        direction: str,
        price: float,
        reason: str,
        store,
        profile_name: str,
        symbol: str,
        mode: str,
        rule_id: str,
    ) -> None:
        print(f"[daily_level_filter] allowed={allowed} direction={direction} price={price:.5f} | {reason}")
        if store is not None and not allowed and profile_name and symbol:
            try:
                store.insert_execution(
                    {
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": profile_name,
                        "symbol": symbol,
                        "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                        "mode": mode,
                        "attempted": 1,
                        "placed": 0,
                        "reason": reason,
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
            except Exception as e:
                print(f"[daily_level_filter] insert_execution error: {e}")


def drop_incomplete_m5_for_filter(m5_df: pd.DataFrame) -> pd.DataFrame:
    """Return M5 dataframe with only closed bars (exclude last bar if it might be forming)."""
    if m5_df is None or m5_df.empty or len(m5_df) < 2:
        return m5_df if m5_df is not None else pd.DataFrame()
    return m5_df.iloc[:-1].copy()
