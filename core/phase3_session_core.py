from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

import pandas as pd

DAY_NAME_TO_WEEKDAY = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

DEFAULT_TOKYO_START_UTC = 16.0
DEFAULT_TOKYO_END_UTC = 22.0
DEFAULT_TOKYO_ALLOWED_DAYS = {1, 2, 4}
DEFAULT_LONDON_ALLOWED_DAYS = {1, 2}
DEFAULT_NY_ALLOWED_DAYS = {0, 1, 2, 3, 4}


def parse_hhmm_to_hour(value: Any, default_hour: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return float(default_hour)
    hh = int(m.group(1))
    mm = int(m.group(2))
    return float(hh) + float(mm) / 60.0


def weekday_set_from_names(values: Any, default_set: set[int]) -> set[int]:
    if not isinstance(values, (list, tuple, set)):
        return set(default_set)
    out: set[int] = set()
    for v in values:
        wd = DAY_NAME_TO_WEEKDAY.get(str(v).strip().lower())
        if wd is not None:
            out.add(wd)
    return out if out else set(default_set)


def in_hour_window(hour_frac: float, start_h: float, end_h: float) -> bool:
    if start_h <= end_h:
        return start_h <= hour_frac < end_h
    return hour_frac >= start_h or hour_frac < end_h


def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:
        d -= pd.Timedelta(days=1)
    return d


def nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=(n - 1) * 7)
    return d


def uk_london_open_utc(ts_day: pd.Timestamp) -> int:
    ts_day = pd.Timestamp(ts_day)
    if ts_day.tzinfo is None:
        ts_day = ts_day.tz_localize("UTC")
    else:
        ts_day = ts_day.tz_convert("UTC")
    y = ts_day.year
    summer_start = last_sunday(y, 3).normalize()
    summer_end = last_sunday(y, 10).normalize()
    d = ts_day.normalize()
    return 7 if summer_start <= d < summer_end else 8


def us_ny_open_utc(ts_day: pd.Timestamp) -> int:
    ts_day = pd.Timestamp(ts_day)
    if ts_day.tzinfo is None:
        ts_day = ts_day.tz_localize("UTC")
    else:
        ts_day = ts_day.tz_convert("UTC")
    y = ts_day.year
    summer_start = nth_sunday(y, 3, 2).normalize()
    summer_end = nth_sunday(y, 11, 1).normalize()
    d = ts_day.normalize()
    return 12 if summer_start <= d < summer_end else 13


def resolve_ny_window_hours(now_utc: datetime, v44_cfg: Optional[dict[str, Any]] = None) -> tuple[float, float]:
    cfg = v44_cfg if isinstance(v44_cfg, dict) else {}
    mode = str(cfg.get("ny_window_mode", "dst_auto")).strip().lower()
    if mode == "fixed_utc":
        raw_start = cfg.get("ny_start_hour", 12.0)
        raw_end = cfg.get("ny_end_hour", 15.0)
        ny_start = float(12.0 if raw_start is None else raw_start)
        ny_end = float(15.0 if raw_end is None else raw_end)
        if ny_end <= ny_start:
            ny_end = ny_start + max(1.0, float(cfg.get("ny_duration_hours", 3.0)))
        return ny_start, ny_end
    ny_start = float(us_ny_open_utc(pd.Timestamp(now_utc)))
    ny_duration = max(1.0, float(cfg.get("ny_duration_hours", 3.0)))
    return ny_start, ny_start + ny_duration


def resolve_phase3_bar_time(data_by_tf: dict[str, Any], fallback_now_utc: datetime) -> datetime:
    m1_df = data_by_tf.get("M1")
    if m1_df is None or getattr(m1_df, "empty", True) or "time" not in m1_df.columns:
        return fallback_now_utc
    try:
        ts = pd.Timestamp(m1_df["time"].iloc[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception:
        return fallback_now_utc


def classify_phase3_session(
    now_utc: datetime,
    effective_config: Optional[dict[str, Any]] = None,
) -> str | None:
    hour_frac = now_utc.hour + now_utc.minute / 60.0
    weekday = now_utc.weekday()

    if not isinstance(effective_config, dict):
        london_open = float(uk_london_open_utc(pd.Timestamp(now_utc)))
        london_end = london_open + 4.0
        ny_start, ny_end = resolve_ny_window_hours(now_utc, {})
        if in_hour_window(hour_frac, DEFAULT_TOKYO_START_UTC, DEFAULT_TOKYO_END_UTC) and weekday in DEFAULT_TOKYO_ALLOWED_DAYS:
            return "tokyo"
        if in_hour_window(hour_frac, london_open, london_end) and weekday in DEFAULT_LONDON_ALLOWED_DAYS:
            return "london"
        if in_hour_window(hour_frac, ny_start, ny_end) and weekday in DEFAULT_NY_ALLOWED_DAYS:
            return "ny"
        return None

    v14_cfg = effective_config.get("v14", {}) if isinstance(effective_config.get("v14"), dict) else {}
    ldn_cfg = effective_config.get("london_v2", {}) if isinstance(effective_config.get("london_v2"), dict) else {}
    v44_cfg = effective_config.get("v44_ny", {}) if isinstance(effective_config.get("v44_ny"), dict) else {}

    tokyo_start = parse_hhmm_to_hour(v14_cfg.get("session_start_utc", "16:00"), DEFAULT_TOKYO_START_UTC)
    tokyo_end = parse_hhmm_to_hour(v14_cfg.get("session_end_utc", "22:00"), DEFAULT_TOKYO_END_UTC)
    tokyo_days = weekday_set_from_names(v14_cfg.get("allowed_trading_days"), DEFAULT_TOKYO_ALLOWED_DAYS)

    london_open = float(uk_london_open_utc(pd.Timestamp(now_utc)))
    a_end = int(ldn_cfg.get("a_entry_end_min_after_london", 90))
    d_end = int(ldn_cfg.get("d_entry_end_min_after_london", 120))
    london_end = london_open + (max(240, a_end, d_end) / 60.0)
    london_days = weekday_set_from_names(ldn_cfg.get("active_days_utc"), DEFAULT_LONDON_ALLOWED_DAYS)

    ny_start, ny_end = resolve_ny_window_hours(now_utc, v44_cfg)

    if in_hour_window(hour_frac, tokyo_start, tokyo_end) and weekday in tokyo_days:
        return "tokyo"
    if in_hour_window(hour_frac, london_open, london_end) and weekday in london_days:
        return "london"
    if in_hour_window(hour_frac, ny_start, ny_end) and weekday in DEFAULT_NY_ALLOWED_DAYS:
        return "ny"
    return None
