from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

import core.phase3_integrated_engine as phase3_engine
from core.phase3_ny_session import execute_v44_ny_session
from core.phase3_v44_evaluator import evaluate_v44_entry


def _m5_frame() -> pd.DataFrame:
    times = pd.date_range("2025-04-04T13:00:00Z", periods=30, freq="5min")
    opens = [159.00 + i * 0.01 for i in range(30)]
    closes = [o + 0.02 for o in opens]
    highs = [c + 0.01 for c in closes]
    lows = [o - 0.01 for o in opens]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _m1_frame() -> pd.DataFrame:
    times = pd.date_range("2025-04-04T13:55:00Z", periods=12, freq="1min")
    opens = [159.45 + i * 0.002 for i in range(12)]
    closes = [o + 0.003 for o in opens]
    highs = [c + 0.002 for c in closes]
    lows = [o - 0.002 for o in opens]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _h1_frame() -> pd.DataFrame:
    times = pd.date_range("2025-04-04T00:00:00Z", periods=80, freq="1h")
    opens = [158.0 + i * 0.02 for i in range(80)]
    closes = [o + 0.03 for o in opens]
    highs = [c + 0.01 for c in closes]
    lows = [o - 0.01 for o in opens]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_evaluate_v44_entry_treats_zero_max_entries_as_unlimited(monkeypatch) -> None:
    monkeypatch.setattr("core.phase3_v44_evaluator.compute_v44_h1_trend", lambda *args, **kwargs: "up")
    monkeypatch.setattr("core.phase3_v44_evaluator.compute_v44_atr_pct_filter", lambda *args, **kwargs: True)
    monkeypatch.setattr("core.phase3_v44_evaluator.compute_v44_m5_slope", lambda *args, **kwargs: 0.9)

    side, strength, reason = evaluate_v44_entry(
        _h1_frame(),
        _m5_frame(),
        tick=SimpleNamespace(bid=159.55, ask=159.56),
        pip_size=0.01,
        session="ny",
        session_state={"trade_count": 99, "consecutive_losses": 0, "cooldown_until": None},
        now_utc=datetime(2025, 4, 4, 14, 0, tzinfo=timezone.utc),
        max_entries_per_day=0,
    )

    assert side == "buy"
    assert strength == "strong"
    assert "bullish momentum" in reason


def test_execute_v44_ny_session_treats_zero_max_open_as_unlimited() -> None:
    class _Adapter:
        def place_order(self, **kwargs):
            return SimpleNamespace(order_retcode=0, order_id="order-1", deal_id="deal-1", fill_price=159.561)

    support = SimpleNamespace(
        ExecutionDecision=SimpleNamespace,
        _drop_incomplete_tf=lambda df, _tf: df,
        resolve_ny_window_hours=lambda now_utc, cfg: (12, 15),
        _as_risk_fraction=lambda value, default: float(value) / 100.0 if float(value) > 1 else float(value),
        _compute_ema=lambda series, period: series.astype(float).ewm(span=period, adjust=False).mean(),
        _compute_adx=lambda df: 30.0,
        _determine_v44_session_mode=lambda *args, **kwargs: "trend",
        _compute_v44_atr_rank=lambda *args, **kwargs: 0.4,
        _load_news_events_cached=lambda *args, **kwargs: tuple(),
        is_in_news_window=lambda *args, **kwargs: False,
        _v44_most_recent_news_event=lambda *args, **kwargs: None,
        _v44_news_trend_active_event=lambda *args, **kwargs: None,
        _account_sizing_value=lambda adapter, fallback=100000.0: 100000.0,
        compute_v44_h1_trend=lambda *args, **kwargs: "up",
        compute_v44_sl=lambda side, m5_df, entry_price, pip: entry_price - 0.08,
        evaluate_v44_entry=lambda *args, **kwargs: ("buy", "strong", "v44: H1 up + M5 strong bullish momentum"),
        v44_defensive_veto_block_from_state=lambda **kwargs: (False, ""),
        _phase3_order_confirmed=lambda adapter, profile, dec: (True, "deal-1"),
        V44_MAX_ENTRY_SPREAD=2.5,
        V44_MAX_OPEN=3,
        V44_MAX_ENTRIES_DAY=7,
        V44_SESSION_STOP_LOSSES=3,
        V44_STRONG_TP1_PIPS=2.0,
        V44_H1_EMA_FAST=20,
        V44_H1_EMA_SLOW=50,
        V44_M5_EMA_FAST=9,
        V44_M5_EMA_SLOW=21,
        V44_SLOPE_BARS=4,
        V44_STRONG_SLOPE=0.5,
        V44_WEAK_SLOPE=0.2,
        V44_MIN_BODY_PIPS=1.5,
        V44_ATR_PCT_CAP=0.67,
        V44_ATR_PCT_LOOKBACK=200,
        PIP_SIZE=0.01,
    )

    result = execute_v44_ny_session(
        adapter=_Adapter(),
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        data_by_tf={"M1": _m1_frame(), "M5": _m5_frame(), "H1": _h1_frame(), "H4": _h1_frame().tail(20)},
        tick=SimpleNamespace(bid=159.55, ask=159.56),
        phase3_state={"open_trade_count": 5},
        sizing_config={"v44_ny": {"max_open_positions": 0, "max_entries_per_day": 0, "rp_max_lot": 20.0, "max_lot": 20.0}},
        now_utc=datetime(2025, 4, 4, 13, 30, tzinfo=timezone.utc),
        store=None,
        ownership_audit={"ownership_cell": "ambiguous/er_high/der_pos", "regime_label": "ambiguous"},
        overlay_state={"defensive_veto_cells": ["ambiguous/er_low/der_neg"]},
        support=support,
    )

    assert result["decision"].placed is True
    assert result["v44_parity_context"]["max_open_unlimited"] == 1
    assert result["v44_parity_context"]["effective_max_open"] == 0
    assert result["v44_parity_context"]["open_trade_count_before"] == 5
    assert result["v44_parity_context"]["max_entries_unlimited"] == 1
    assert result["v44_exit_plan"]["mode"] == "managed_partial_runner"


def test_execute_v44_ny_session_defended_strict_policy_overrides_runtime_day_cap(monkeypatch) -> None:
    class _Adapter:
        def place_order(self, **kwargs):
            return SimpleNamespace(order_retcode=0, order_id="order-2", deal_id="deal-2", fill_price=159.561)

    support = SimpleNamespace(
        ExecutionDecision=SimpleNamespace,
        _drop_incomplete_tf=lambda df, _tf: df,
        resolve_ny_window_hours=lambda now_utc, cfg: (12, 15),
        _as_risk_fraction=lambda value, default: float(value) / 100.0 if float(value) > 1 else float(value),
        _compute_ema=lambda series, period: series.astype(float).ewm(span=period, adjust=False).mean(),
        _compute_adx=lambda df: 30.0,
        _determine_v44_session_mode=lambda *args, **kwargs: "trend",
        _compute_v44_atr_rank=lambda *args, **kwargs: 0.4,
        _load_news_events_cached=lambda *args, **kwargs: tuple(),
        is_in_news_window=lambda *args, **kwargs: False,
        _v44_most_recent_news_event=lambda *args, **kwargs: None,
        _v44_news_trend_active_event=lambda *args, **kwargs: None,
        _account_sizing_value=lambda adapter, fallback=100000.0: 100000.0,
        compute_v44_h1_trend=lambda *args, **kwargs: "up",
        compute_v44_sl=lambda side, m5_df, entry_price, pip: entry_price - 0.08,
        evaluate_v44_entry=lambda *args, **kwargs: ("buy", "strong", "v44: H1 up + M5 strong bullish momentum"),
        v44_defensive_veto_block_from_state=lambda **kwargs: (False, ""),
        _phase3_order_confirmed=lambda adapter, profile, dec: (True, "deal-2"),
        V44_MAX_ENTRY_SPREAD=2.5,
        V44_MAX_OPEN=3,
        V44_MAX_ENTRIES_DAY=7,
        V44_SESSION_STOP_LOSSES=3,
        V44_STRONG_TP1_PIPS=2.0,
        V44_H1_EMA_FAST=20,
        V44_H1_EMA_SLOW=50,
        V44_M5_EMA_FAST=9,
        V44_M5_EMA_SLOW=21,
        V44_SLOPE_BARS=4,
        V44_STRONG_SLOPE=0.5,
        V44_WEAK_SLOPE=0.2,
        V44_MIN_BODY_PIPS=1.5,
        V44_ATR_PCT_CAP=0.67,
        V44_ATR_PCT_LOOKBACK=200,
        PIP_SIZE=0.01,
    )

    import core.phase3_ny_session as ny_session
    monkeypatch.setattr(
        ny_session,
        "load_phase3_package_spec",
        lambda preset_id=None: SimpleNamespace(strict_policy={"max_entries_per_day": None}),
    )

    day = "2025-04-04"
    result = execute_v44_ny_session(
        adapter=_Adapter(),
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01, active_preset_name="phase3_integrated_v7_defended"),
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        data_by_tf={"M1": _m1_frame(), "M5": _m5_frame(), "H1": _h1_frame(), "H4": _h1_frame().tail(20)},
        tick=SimpleNamespace(bid=159.55, ask=159.56),
        phase3_state={f"session_ny_{day}": {"trade_count": 99, "consecutive_losses": 0, "stopped": False}},
        sizing_config={"v44_ny": {"max_open_positions": 3, "max_entries_per_day": 7, "rp_max_lot": 20.0, "max_lot": 20.0}},
        now_utc=datetime(2025, 4, 4, 13, 30, tzinfo=timezone.utc),
        store=None,
        ownership_audit={"ownership_cell": "ambiguous/er_high/der_pos", "regime_label": "ambiguous"},
        overlay_state={"defensive_veto_cells": ["ambiguous/er_low/der_neg"]},
        support=support,
    )
    assert result["decision"].placed is True
    assert result["v44_parity_context"]["max_entries_authority"] == "defended_strict_policy"
    assert result["v44_parity_context"]["max_entries_unlimited"] == 1


def test_apply_phase3_session_outcome_marks_ny_session_stopped_after_loss_limit() -> None:
    phase3_state = {
        "session_ny_2025-04-04": {
            "consecutive_losses": 2,
            "wins_closed": 0,
            "win_streak": 0,
        }
    }

    result = phase3_engine.apply_phase3_session_outcome(
        phase3_state=phase3_state,
        phase3_sizing_cfg={"v44_ny": {"session_stop_losses": 3, "cooldown_loss_bars": 1, "cooldown_win_bars": 1}},
        entry_session="ny",
        action="hard_sl",
        side="buy",
        entry_type="phase3:v44_ny:strong",
        is_loss=True,
        key_date="2025-04-04",
        now_utc=pd.Timestamp("2025-04-04T14:05:00Z"),
    )

    assert result["consecutive_losses"] == 3
    assert result["stopped"] is True
    assert phase3_state["session_ny_2025-04-04"]["stopped"] is True


def test_manage_v44_exit_uses_managed_tp1_pips_fallback(monkeypatch) -> None:
    class _Adapter:
        def __init__(self) -> None:
            self.partial_calls = []
            self.stop_updates = []

        def close_position(self, **kwargs):
            self.partial_calls.append(kwargs)

        def update_position_stop_loss(self, position_id, symbol, stop_price):
            self.stop_updates.append((position_id, symbol, stop_price))

    class _Store:
        def __init__(self) -> None:
            self.updates = []

        def update_trade(self, trade_id, updates):
            self.updates.append((trade_id, dict(updates)))

    monkeypatch.setattr(phase3_engine, "_phase3_position_meta", lambda position, side: (123, 1.0, None))

    adapter = _Adapter()
    store = _Store()
    result = phase3_engine._manage_v44_exit(
        adapter=adapter,
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        store=store,
        tick=SimpleNamespace(bid=150.051, ask=150.053),
        trade_row={
            "trade_id": "t1",
            "side": "buy",
            "entry_price": 150.000,
            "entry_type": "phase3:v44_ny:normal@ambiguous/er_high/der_pos",
            "target_price": None,
            "managed_tp1_pips": 5.0,
            "stop_price": 149.920,
            "tp1_partial_done": 0,
        },
        position=object(),
        v44_config={
            "be_offset_pips": 0.5,
            "normal_tp1_close_pct": 0.5,
            "normal_tp2_pips": 3.0,
            "normal_trail_buffer": 3.0,
            "trail_start_after_tp1_mult": 0.5,
        },
    )

    assert result["action"] == "tp1_partial"
    assert adapter.partial_calls, "expected a partial close at the managed TP1 fallback"
    assert store.updates and store.updates[0][1]["tp1_partial_done"] == 1


def test_manage_v44_exit_does_not_immediately_close_runner_after_tp1(monkeypatch) -> None:
    class _Adapter:
        def __init__(self) -> None:
            self.full_close_calls = 0
            self.stop_updates = []

        def update_position_stop_loss(self, position_id, symbol, stop_price):
            self.stop_updates.append((position_id, symbol, stop_price))

    class _Store:
        def __init__(self) -> None:
            self.updates = []

        def update_trade(self, trade_id, updates):
            self.updates.append((trade_id, dict(updates)))

    monkeypatch.setattr(phase3_engine, "_phase3_position_meta", lambda position, side: (123, 1.0, None))
    monkeypatch.setattr(
        phase3_engine,
        "_close_full",
        lambda adapter, profile, position_id, current_lots, side: (_ for _ in ()).throw(AssertionError("runner should not close immediately")),
    )

    adapter = _Adapter()
    store = _Store()
    # Entry=150.000, TP1 fallback=5.0p -> 150.050. Runner TP2 configured as +3.0p
    # should target total 8.0p (150.080), so at 150.051 we should not full-close.
    result = phase3_engine._manage_v44_exit(
        adapter=adapter,
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        store=store,
        tick=SimpleNamespace(bid=150.051, ask=150.053),
        trade_row={
            "trade_id": "t2",
            "side": "buy",
            "entry_price": 150.000,
            "entry_type": "phase3:v44_ny:normal@ambiguous/er_high/der_pos",
            "target_price": None,
            "managed_tp1_pips": 5.0,
            "stop_price": 149.920,
            "tp1_partial_done": 1,
            "breakeven_sl_price": 150.005,
        },
        position=object(),
        v44_config={
            "normal_tp2_pips": 3.0,
            "normal_trail_buffer": 3.0,
            "trail_start_after_tp1_mult": 0.5,
        },
    )
    assert result["action"] != "tp2_full"


def test_defended_sizing_lock_ignores_runtime_override_for_ny_window_and_day_cap(tmp_path) -> None:
    override = tmp_path / "phase3_integrated_sizing_config.json"
    override.write_text(
        '{"v44_ny":{"ny_window_mode":"auto_dst","ny_start_hour":12.0,"start_delay_minutes":0,"max_entries_per_day":7}}',
        encoding="utf-8",
    )
    cfg = phase3_engine.load_phase3_sizing_config(
        config_path=override,
        preset_id="phase3_integrated_v7_defended",
    )
    v44 = cfg.get("v44_ny", {})
    # Contract/source-of-truth wins for defended preset.
    assert v44.get("ny_window_mode") == "fixed_utc"
    assert float(v44.get("ny_start_hour")) == 13.0
    assert int(v44.get("start_delay_minutes")) == 5
    # Contract strict-policy has null => unlimited.
    assert v44.get("max_entries_per_day") is None
    assert "v44_ny.max_entries_per_day" in list((cfg.get("_meta") or {}).get("locked_keys") or [])
