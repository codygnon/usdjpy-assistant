from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

PIP = 0.01


@dataclass(frozen=True)
class V4Event:
    armed_time: str
    expiry_time: str
    side: str
    trigger_level: float
    stop_price: float
    tp_price: float | None
    confirmation_level: float
    confirmation_time: str
    spike_time: str
    spike_direction: str
    spike_high: float
    spike_low: float
    spike_range_pips: float
    prior_12_high: float
    prior_12_low: float
    stretch_atr_ratio_m5: float
    dist_from_m15_ema50_pips: float
    session_name: str
    cluster_block_until: str


def _ts_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_ohlc(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return out


def build_mtf_from_m1(m1_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = _normalize_ohlc(m1_df)
    if m1.empty:
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        return empty, empty
    temp = m1.set_index("time").sort_index()
    m5 = (
        temp.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    m15 = (
        temp.resample("15min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return m5, m15


def _session_name(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 0 <= hour < 8:
        return "tokyo"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    return "off"


def _build_featured_m5(m1_df: pd.DataFrame) -> pd.DataFrame:
    m5, m15 = build_mtf_from_m1(m1_df)
    if m5.empty or m15.empty:
        return pd.DataFrame()

    m5 = m5.copy()
    prev_close = m5["close"].shift(1).fillna(m5["close"])
    tr = pd.concat(
        [
            m5["high"] - m5["low"],
            (m5["high"] - prev_close).abs(),
            (m5["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    m5["atr14_pips"] = tr.rolling(14).mean() / PIP
    m5["ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["range_pips"] = (m5["high"] - m5["low"]) / PIP
    m5["body_pips"] = (m5["close"] - m5["open"]).abs() / PIP
    m5["range_atr_ratio"] = m5["range_pips"] / m5["atr14_pips"]
    m5["body_atr_ratio"] = m5["body_pips"] / m5["atr14_pips"]
    m5["prior_12_high"] = m5["high"].shift(1).rolling(12).max()
    m5["prior_12_low"] = m5["low"].shift(1).rolling(12).min()
    m5["broad_candidate"] = (
        (m5["range_pips"] >= 15.0)
        & (m5["range_atr_ratio"] >= 1.5)
        & (m5["body_atr_ratio"] >= 1.0)
        & (m5["close"] != m5["open"])
    )
    m5["spike_direction"] = m5["close"].gt(m5["open"]).map({True: "bullish", False: "bearish"})

    m15 = m15.copy()
    m15["m15_ema50"] = m15["close"].ewm(span=50, adjust=False).mean()
    merged = pd.merge_asof(
        m5.sort_values("time"),
        m15[["time", "m15_ema50"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    return merged


def _event_from_spike_fade(
    spike: pd.Series,
    fade: pd.Series,
    cfg: dict[str, Any],
) -> V4Event | None:
    if not bool(spike.get("broad_candidate")):
        return None
    needed = [
        spike.get("atr14_pips"),
        spike.get("ema20"),
        spike.get("m15_ema50"),
        spike.get("prior_12_high"),
        spike.get("prior_12_low"),
    ]
    if any(pd.isna(v) for v in needed):
        return None

    stretch_atr_ratio_m5 = abs((float(spike["close"]) - float(spike["ema20"])) / PIP) / float(spike["atr14_pips"])
    dist_from_m15_ema50_pips = (float(spike["close"]) - float(spike["m15_ema50"])) / PIP
    spike_mid = float(spike["low"]) + 0.5 * (float(spike["high"]) - float(spike["low"]))

    if str(spike["spike_direction"]) == "bullish":
        side = "sell"
        reclaim_mid = float(fade["close"]) <= spike_mid
        reclaim_prior = float(fade["close"]) <= float(spike["prior_12_high"])
        trigger_level = float(spike["prior_12_high"])
        stop_anchor = float(spike["high"]) + float(cfg["stop_buffer_pips"]) * PIP
    else:
        side = "buy"
        reclaim_mid = float(fade["close"]) >= spike_mid
        reclaim_prior = float(fade["close"]) >= float(spike["prior_12_low"])
        trigger_level = float(spike["prior_12_low"])
        stop_anchor = float(spike["low"]) - float(cfg["stop_buffer_pips"]) * PIP

    family_c_ok = (
        stretch_atr_ratio_m5 >= float(cfg["family_c_min_stretch_atr_ratio"])
        and abs(dist_from_m15_ema50_pips) >= float(cfg["family_c_min_dist_from_m15_ema50_pips"])
        and reclaim_mid
        and reclaim_prior
    )
    if not family_c_ok:
        return None

    trigger_spread_px = float(cfg["entry_spread_pips"]) * PIP
    fill_price = trigger_level + trigger_spread_px if side == "buy" else trigger_level - trigger_spread_px
    if side == "buy":
        raw_stop_pips = (fill_price - stop_anchor) / PIP
    else:
        raw_stop_pips = (stop_anchor - fill_price) / PIP
    raw_stop_pips = float(raw_stop_pips)
    if raw_stop_pips > float(cfg["stop_clamp_max_pips"]):
        return None
    stop_distance_pips = max(float(cfg["stop_clamp_min_pips"]), raw_stop_pips)
    if side == "buy":
        stop_price = fill_price - stop_distance_pips * PIP
    else:
        stop_price = fill_price + stop_distance_pips * PIP

    tp_fraction = cfg.get("tp_fraction")
    tp_price = None
    if tp_fraction is not None:
        tp_distance_pips = float(spike["range_pips"]) * float(tp_fraction)
        if side == "buy":
            tp_price = fill_price + tp_distance_pips * PIP
        else:
            tp_price = fill_price - tp_distance_pips * PIP

    armed_time = _ts_utc(fade["time"])
    expiry_time = armed_time + pd.Timedelta(minutes=int(cfg["confirmation_window_minutes"]))
    cluster_minutes = int(cfg.get("cluster_block_minutes", 120))
    cluster_block_until = armed_time + pd.Timedelta(minutes=cluster_minutes)
    return V4Event(
        armed_time=armed_time.isoformat(),
        expiry_time=expiry_time.isoformat(),
        side=side,
        trigger_level=trigger_level,
        stop_price=stop_price,
        tp_price=tp_price,
        confirmation_level=trigger_level,
        confirmation_time=armed_time.isoformat(),
        spike_time=_ts_utc(spike["time"]).isoformat(),
        spike_direction=str(spike["spike_direction"]),
        spike_high=float(spike["high"]),
        spike_low=float(spike["low"]),
        spike_range_pips=float(spike["range_pips"]),
        prior_12_high=float(spike["prior_12_high"]),
        prior_12_low=float(spike["prior_12_low"]),
        stretch_atr_ratio_m5=float(stretch_atr_ratio_m5),
        dist_from_m15_ema50_pips=float(dist_from_m15_ema50_pips),
        session_name=_session_name(armed_time),
        cluster_block_until=cluster_block_until.isoformat(),
    )


def default_v4_runtime_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "lots": 20.0,
        "confirmation_window_minutes": 10,
        "entry_spread_pips": 1.6,
        "stop_buffer_pips": 2.0,
        "stop_clamp_min_pips": 15.0,
        "stop_clamp_max_pips": 35.0,
        "tp_fraction": 0.5,
        "trailing_enabled": True,
        "trail_trigger_pips": 10.0,
        "trail_distance_pips": 5.0,
        "prove_it_fast_minutes": 15,
        "prove_it_fast_threshold_pips": -5.0,
        "shared_margin_cap_pct": 75.0,
        "max_active_or_pending": 1,
        "cluster_block_minutes": 120,
        "family_c_min_stretch_atr_ratio": 1.25,
        "family_c_min_dist_from_m15_ema50_pips": 20.0,
        "comment_tag": "spike_fade_v4",
    }


def detect_latest_v4_event(
    *,
    data_by_tf: dict[str, Any],
    phase3_state: dict[str, Any],
    runtime_cfg: dict[str, Any],
    now_utc: datetime | None = None,
) -> tuple[V4Event | None, dict[str, Any]]:
    now_utc = now_utc or datetime.now(timezone.utc)
    state = dict(phase3_state.get("v4_runtime") or {})
    m1 = _normalize_ohlc(data_by_tf.get("M1"))
    if m1.empty:
        state["lifecycle_state"] = "IDLE_WARMUP"
        state["warmup_ready"] = False
        return None, state

    m5 = _build_featured_m5(m1)
    if m5.empty or len(m5) < 2:
        state["lifecycle_state"] = "IDLE_WARMUP"
        state["warmup_ready"] = False
        return None, state

    last_m5_time = _ts_utc(m5.iloc[-1]["time"])
    state["last_seen_m5_end"] = last_m5_time.isoformat()

    last_bar = m5.iloc[-1]
    prev_bar = m5.iloc[-2]
    warmup_ready = bool(
        len(m5) >= 150
        and not pd.isna(last_bar.get("m15_ema50"))
        and not pd.isna(prev_bar.get("m15_ema50"))
        and not pd.isna(prev_bar.get("prior_12_high"))
        and not pd.isna(prev_bar.get("prior_12_low"))
        and not pd.isna(prev_bar.get("atr14_pips"))
    )
    state["warmup_ready"] = warmup_ready
    if not warmup_ready:
        state["lifecycle_state"] = "IDLE_WARMUP"
        return None, state

    last_processed = state.get("last_processed_m5_end")
    if last_processed and _ts_utc(last_processed) >= last_m5_time:
        return None, state

    event = _event_from_spike_fade(prev_bar, last_bar, runtime_cfg)
    state["last_processed_m5_end"] = last_m5_time.isoformat()
    if event is None:
        if state.get("lifecycle_state") == "IDLE_WARMUP":
            state["lifecycle_state"] = "READY"
        return None, state

    cooldown_until = state.get("cluster_block_until")
    if cooldown_until and _ts_utc(cooldown_until) > _ts_utc(now_utc):
        state["lifecycle_state"] = "COOLDOWN_CLUSTER_BLOCK"
        state["last_cluster_blocked_at"] = last_m5_time.isoformat()
        state["last_reject_reason"] = "cluster_block_active"
        return None, state

    state["lifecycle_state"] = "FADE_CONFIRMED"
    state["last_detected_event"] = event.__dict__
    state["last_reject_reason"] = None
    return event, state
