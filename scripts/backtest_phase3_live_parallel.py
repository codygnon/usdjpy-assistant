#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import MarketContext
import core.phase3_integrated_engine as p3
from core.phase3_integrated_engine import (
    _compute_session_windows,
    _parse_hhmm_to_hour,
    _resolve_ny_window_hours,
    classify_session,
    load_phase3_sizing_config,
)
from core.phase3_shared_engine import evaluate_phase3_bar
from core.presets import PresetId, apply_preset, get_preset_patch
from core.profile import default_profile_for_name, load_profile_v1
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import phase3_parity_check as parity


def _install_replay_fastpaths() -> None:
    # The replay feeds clean, sorted, fully-completed bars for each timeframe.
    # Avoid re-copying / re-sorting / re-dropping them inside the engine.
    def _fast_drop_incomplete_tf(df: pd.DataFrame | None, tf: str) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        return df

    _asian_cache: dict[tuple[str, int, int], tuple[float, float, float, bool]] = {}
    _orig_compute_asian_range = p3.compute_asian_range

    def _fast_compute_asian_range(
        m1_df: pd.DataFrame,
        london_open_utc_hour: int,
        range_min_pips: float,
        range_max_pips: float,
    ):
        if m1_df is None or m1_df.empty or "time" not in m1_df.columns:
            return _orig_compute_asian_range(
                m1_df,
                london_open_utc_hour,
                range_min_pips=range_min_pips,
                range_max_pips=range_max_pips,
            )
        try:
            now = pd.Timestamp(m1_df["time"].iloc[-1])
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            else:
                now = now.tz_convert("UTC")
            key = (
                now.date().isoformat(),
                int(london_open_utc_hour),
                int(len(m1_df)),
                round(float(range_min_pips), 4),
                round(float(range_max_pips), 4),
            )
            cached = _asian_cache.get(key)
            if cached is not None:
                return cached
            result = _orig_compute_asian_range(
                m1_df,
                london_open_utc_hour,
                range_min_pips=range_min_pips,
                range_max_pips=range_max_pips,
            )
            _asian_cache[key] = result
            return result
        except Exception:
            return _orig_compute_asian_range(
                m1_df,
                london_open_utc_hour,
                range_min_pips=range_min_pips,
                range_max_pips=range_max_pips,
            )

    p3._drop_incomplete_tf = _fast_drop_incomplete_tf
    p3.compute_asian_range = _fast_compute_asian_range


_install_replay_fastpaths()


class RecordingMockAdapter(parity.MockOrderAdapter):
    def __init__(self, equity: float = 100_000.0, pip_size: float = 0.01) -> None:
        super().__init__(equity=equity)
        self.start_equity = float(equity)
        self.pip_size = float(pip_size)
        self.closed_trades: list[dict[str, Any]] = []
        self._closed_trade_by_id: dict[str, dict[str, Any]] = {}
        self.profile_name = "phase3_live_parallel_backtest"

    def realize_trade(
        self,
        order_id: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        trade = self._open_trades.get(order_id)
        if trade is None:
            return
        side = str(trade.get("side", "")).lower()
        entry_price = float(trade.get("entry_price") or trade.get("fill_price") or 0.0)
        lots = float(trade.get("size_lots") or 0.0)
        if entry_price <= 0 or lots <= 0 or side not in {"buy", "sell"}:
            usd = 0.0
            pips = 0.0
        else:
            diff = float(exit_price) - entry_price
            pips = (diff / self.pip_size) if side == "buy" else (-diff / self.pip_size)
            pip_value_per_lot = (self.pip_size / max(entry_price, 1e-9)) * 100000.0
            usd = float(pips) * pip_value_per_lot * lots
        self.balance += float(usd)
        self.equity = self.balance
        self.closed_trades.append(
            {
                "order_id": order_id,
                "entry_time": trade.get("entry_time"),
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "side": side,
                "strategy_tag": trade.get("strategy_tag"),
                "strategy": parity._strategy_from_tag(trade.get("strategy_tag")),
                "session": parity._session_from_tag(trade.get("strategy_tag")),
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "sl_price": trade.get("stop_price"),
                "tp1_price": trade.get("target_price"),
                "size_lots": lots,
                "risk_usd_planned": trade.get("risk_usd_planned"),
                "pips": float(pips),
                "usd": float(usd),
                "exit_reason": exit_reason,
                "entry_type": trade.get("entry_type"),
                "comment": trade.get("comment"),
                "session_key": trade.get("session_key"),
                "size_scale": 1.0,
                "standalone_entry_equity": float(trade.get("equity_before", self.start_equity) or self.start_equity),
                "raw": dict(trade),
            }
        )
        del self._open_trades[order_id]

    def list_open_trades(self, profile: str) -> list[parity._MockRow]:
        return list(self._open_trades.values())

    def _pip_value_per_lot(self, entry_price: float) -> float:
        return (self.pip_size / max(entry_price, 1e-9)) * 100000.0

    def _trade_price_pips_usd(
        self,
        trade: dict[str, Any],
        exit_price: float,
        close_lots: float,
    ) -> tuple[float, float]:
        side = str(trade.get("side", "")).lower()
        entry_price = float(trade.get("entry_price") or trade.get("fill_price") or 0.0)
        if entry_price <= 0 or close_lots <= 0 or side not in {"buy", "sell"}:
            return 0.0, 0.0
        diff = float(exit_price) - entry_price
        pips = (diff / self.pip_size) if side == "buy" else (-diff / self.pip_size)
        usd = float(pips) * self._pip_value_per_lot(entry_price) * close_lots
        return float(pips), float(usd)

    def close_position(self, ticket, symbol: str, volume: float, position_type: int):
        trade = self._open_trades.get(int(ticket))
        if trade is None:
            return
        close_lots = min(float(volume or 0.0), float(trade.get("size_lots") or 0.0))
        if close_lots <= 0:
            return
        side = str(trade.get("side", "")).lower()
        exit_price = float(self._tick.bid if side == "buy" else self._tick.ask)
        pips, usd = self._trade_price_pips_usd(trade, exit_price, close_lots)

        trade["realized_usd"] = float(trade.get("realized_usd") or 0.0) + usd
        trade["realized_pip_lots"] = float(trade.get("realized_pip_lots") or 0.0) + (pips * close_lots)
        trade["size_lots"] = float(trade.get("size_lots") or 0.0) - close_lots
        self.balance += usd
        self.equity = self.balance

        if float(trade.get("size_lots") or 0.0) > 1e-9:
            self._open_trades[int(ticket)] = trade
            return

        original_lots = float(trade.get("original_lots") or close_lots)
        total_usd = float(trade.get("realized_usd") or 0.0)
        total_pip_lots = float(trade.get("realized_pip_lots") or 0.0)
        total_pips = (total_pip_lots / original_lots) if original_lots > 0 else 0.0
        closed = {
            "order_id": int(ticket),
            "trade_id": str(trade.get("trade_id") or ticket),
            "entry_time": trade.get("entry_time"),
            "exit_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "side": side,
            "strategy_tag": trade.get("strategy_tag"),
            "strategy": parity._strategy_from_tag(trade.get("strategy_tag")),
            "session": parity._session_from_tag(trade.get("strategy_tag")),
            "entry_price": float(trade.get("entry_price") or 0.0),
            "exit_price": exit_price,
            "sl_price": trade.get("stop_price"),
            "tp1_price": trade.get("target_price"),
            "size_lots": original_lots,
            "risk_usd_planned": trade.get("risk_usd_planned"),
            "pips": float(total_pips),
            "usd": float(total_usd),
            "exit_reason": trade.get("exit_reason") or "managed_close",
            "entry_type": trade.get("entry_type"),
            "comment": trade.get("comment"),
            "session_key": trade.get("session_key"),
            "size_scale": 1.0,
            "standalone_entry_equity": float(trade.get("equity_before", self.start_equity) or self.start_equity),
            "raw": dict(trade),
        }
        self.closed_trades.append(closed)
        self._closed_trade_by_id[str(closed["trade_id"])] = closed
        del self._open_trades[int(ticket)]

    def update_position_stop_loss(self, ticket, symbol: str, stop_price: float):
        trade = self._open_trades.get(int(ticket))
        if trade is None:
            return
        trade["stop_price"] = float(stop_price)

    def update_trade(self, trade_id: str, updates: dict[str, Any]):
        trade_id = str(trade_id)
        for trade in self._open_trades.values():
            if str(trade.get("trade_id")) == trade_id:
                trade.update(updates or {})
                return
        closed = self._closed_trade_by_id.get(trade_id)
        if closed is not None:
            closed.update(updates or {})

    def close_trade(self, trade_id: str, updates: dict[str, Any]):
        trade_id = str(trade_id)
        closed = self._closed_trade_by_id.get(trade_id)
        if closed is not None:
            closed.update(updates or {})

    def get_trades_for_date(self, profile_name: str, day_iso: str):
        return self.get_closed_trades_for_exit_date(profile_name, day_iso)

    def get_closed_trades_for_exit_date(self, profile_name: str, day_iso: str):
        out = []
        for row in self.closed_trades:
            try:
                ts = pd.Timestamp(row.get("exit_time"))
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                if ts.date().isoformat() == str(day_iso):
                    out.append(
                        {
                            "entry_type": row.get("entry_type"),
                            "pips": row.get("pips"),
                            "profit": row.get("usd"),
                        }
                    )
            except Exception:
                continue
        return out


def _close_updates(phase3_state: dict[str, Any], session_key: str | None, ts: pd.Timestamp, won: bool) -> None:
    if not session_key:
        return
    sdat = dict(phase3_state.get(session_key, {}))
    sdat["cooldown_until"] = (ts + pd.Timedelta(minutes=5)).isoformat()
    if won:
        sdat["wins_closed"] = int(sdat.get("wins_closed", 0)) + 1
        sdat["win_streak"] = int(sdat.get("win_streak", 0)) + 1
        sdat["consecutive_losses"] = 0
    else:
        sdat["consecutive_losses"] = int(sdat.get("consecutive_losses", 0)) + 1
        sdat["win_streak"] = 0
    phase3_state[session_key] = sdat


@contextlib.contextmanager
def _patched_phase3_now(now_ts: pd.Timestamp):
    real_datetime = p3.datetime

    class ReplayDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            dt = now_ts.to_pydatetime()
            if tz is not None:
                return dt.astimezone(tz)
            return dt

    p3.datetime = ReplayDateTime
    try:
        yield
    finally:
        p3.datetime = real_datetime


def _session_has_ended(entry_type: str, t: pd.Timestamp, sizing_config: dict[str, Any] | None = None) -> bool:
    cfg = sizing_config if isinstance(sizing_config, dict) else {}
    et = str(entry_type or "").lower()
    if et.startswith("phase3:v14"):
        v14_cfg = cfg.get("v14", {}) if isinstance(cfg.get("v14"), dict) else {}
        start_h = _parse_hhmm_to_hour(v14_cfg.get("session_start_utc", "16:00"), 16.0)
        end_h = _parse_hhmm_to_hour(v14_cfg.get("session_end_utc", "22:00"), 22.0)
        hour_frac = float(t.hour) + float(t.minute) / 60.0
        if start_h <= end_h:
            return not (start_h <= hour_frac < end_h)
        return not (hour_frac >= start_h or hour_frac < end_h)
    if et.startswith("phase3:london_v2"):
        ldn_cfg = cfg.get("london_v2", {}) if isinstance(cfg.get("london_v2"), dict) else {}
        windows = _compute_session_windows(t.to_pydatetime())
        session_end = windows["ny_open"] if bool(ldn_cfg.get("hard_close_at_ny_open", True)) else windows["london_end"]
        return t.to_pydatetime() >= session_end
    if et.startswith("phase3:v44"):
        v44_cfg = cfg.get("v44_ny", {}) if isinstance(cfg.get("v44_ny"), dict) else {}
        day0 = t.normalize().to_pydatetime()
        _, ny_end_hour = _resolve_ny_window_hours(t.to_pydatetime(), v44_cfg)
        ny_end = day0 + pd.Timedelta(hours=ny_end_hour)
        return t.to_pydatetime() >= ny_end
    return False


def _manage_exits_recording(
    *,
    adapter: RecordingMockAdapter,
    profile: Any,
    data_by_tf: dict[str, pd.DataFrame],
    phase3_state: dict[str, Any],
    t: pd.Timestamp,
    sizing_config: dict[str, Any] | None = None,
) -> None:
    if not adapter._open_trades:
        return
    with _patched_phase3_now(t):
        for oid, trade in list(adapter._open_trades.items()):
            side = str(trade.get("side", "")).lower()
            position = {
                "id": int(oid),
                "currentUnits": int(round(float(trade.get("size_lots") or 0.0) * 100000.0)),
            }
            trade_row = {
                "trade_id": str(trade.get("trade_id") or oid),
                "side": side,
                "entry_price": float(trade.get("entry_price") or 0.0),
                "stop_price": trade.get("stop_price"),
                "target_price": trade.get("target_price"),
                "tp1_partial_done": int(trade.get("tp1_partial_done") or 0),
                "breakeven_sl_price": trade.get("breakeven_sl_price"),
                "entry_type": trade.get("entry_type"),
                "entry_session": parity._session_from_tag(trade.get("strategy_tag")),
                "opened_at": trade.get("entry_time"),
                "timestamp_utc": trade.get("entry_time"),
                "size_lots": float(trade.get("original_lots") or trade.get("size_lots") or 0.0),
                "pips": trade.get("pips"),
            }
            result = p3.manage_phase3_exit(
                adapter=adapter,
                profile=profile,
                store=adapter,
                tick=adapter._tick,
                trade_row=trade_row,
                position=position,
                data_by_tf=data_by_tf,
                phase3_state=phase3_state,
                sizing_config=sizing_config,
            )
            action = str(result.get("action") or "")
            if action and action not in {"none", "trail_update", "tp1_partial"}:
                if str(trade.get("trade_id") or oid) in adapter._closed_trade_by_id:
                    adapter._closed_trade_by_id[str(trade.get("trade_id") or oid)]["exit_time"] = t.isoformat()
                    adapter._closed_trade_by_id[str(trade.get("trade_id") or oid)]["exit_reason"] = action
                key_date = p3.phase3_trade_key_date(trade.get("entry_time"), t.to_pydatetime().replace(tzinfo=None))
                est = result.get("closed_pips_est")
                try:
                    pips_eval = float(est) if est is not None else 0.0
                except Exception:
                    pips_eval = 0.0
                is_loss = True if action == "hard_sl" else (pips_eval < 0.0)
                entry_session = p3.infer_phase3_entry_session(trade.get("entry_type"), parity._session_from_tag(trade.get("strategy_tag")))
                p3.apply_phase3_session_outcome(
                    phase3_state=phase3_state,
                    phase3_sizing_cfg=sizing_config or {},
                    entry_session=entry_session,
                    entry_type=trade.get("entry_type"),
                    is_loss=is_loss,
                    action=action,
                    side=side,
                    key_date=key_date,
                    now_utc=t,
                )


def _finalize_open_trades(adapter: RecordingMockAdapter, phase3_state: dict[str, Any], last_bar: pd.Series) -> None:
    t = pd.Timestamp(last_bar["time"]).tz_convert("UTC")
    px = float(last_bar["close"])
    for oid, trade in list(adapter._open_trades.items()):
        adapter.realize_trade(oid, t, px, "end_of_replay")
        _close_updates(phase3_state, trade.get("session_key"), t, adapter.closed_trades[-1]["usd"] > 0)


def _trade_rows(closed_trades: list[dict[str, Any]]) -> list[merged_engine.TradeRow]:
    rows: list[merged_engine.TradeRow] = []
    for t in closed_trades:
        rows.append(
            merged_engine.TradeRow(
                strategy=str(t["strategy"]),
                entry_time=pd.Timestamp(t["entry_time"]),
                exit_time=pd.Timestamp(t["exit_time"]),
                entry_session=str(t["session"]),
                side=str(t["side"]),
                pips=float(t["pips"]),
                usd=float(t["usd"]),
                exit_reason=str(t["exit_reason"]),
                standalone_entry_equity=float(t.get("standalone_entry_equity", 100000.0)),
                raw=dict(t.get("raw", {})),
                size_scale=float(t.get("size_scale", 1.0)),
            )
        )
    return rows


def _group_day_hour(closed_trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not closed_trades:
        return [], []
    df = pd.DataFrame(closed_trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["day_name"] = df["exit_time"].dt.day_name()
    df["hour_utc"] = df["exit_time"].dt.hour

    def _summ(group_cols: list[str]) -> list[dict[str, Any]]:
        rows = []
        for key, g in df.groupby(group_cols):
            wins = (g["usd"] > 0).sum()
            losses = (g["usd"] <= 0).sum()
            gross_win = g.loc[g["usd"] > 0, "usd"].sum()
            gross_loss = -g.loc[g["usd"] <= 0, "usd"].sum()
            pf = (gross_win / gross_loss) if gross_loss > 0 else None
            key_dict = {}
            if isinstance(key, tuple):
                for col, val in zip(group_cols, key):
                    key_dict[col] = val
            else:
                key_dict[group_cols[0]] = key
            key_dict.update(
                {
                    "trades": int(len(g)),
                    "wins": int(wins),
                    "losses": int(losses),
                    "win_rate_pct": float((wins / len(g)) * 100.0) if len(g) else 0.0,
                    "net_usd": float(g["usd"].sum()),
                    "net_pips": float(g["pips"].sum()),
                    "profit_factor": float(pf) if pf is not None else None,
                }
            )
            rows.append(key_dict)
        return rows

    return _summ(["day_name"]), _summ(["hour_utc"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel backtest of the live Phase 3 engine using local M1 CSV replay")
    p.add_argument("--input-csv", required=True, help="Local M1 CSV used for replay")
    p.add_argument(
        "--profile",
        default="",
        help="Optional profile JSON path. If omitted, builds a fresh profile from the Phase 3 integrated preset.",
    )
    p.add_argument("--output", default="", help="Output JSON path")
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--spread-pips", type=float, default=1.0)
    return p.parse_args()


def _build_phase3_profile() -> Any:
    base = default_profile_for_name("phase3_live_parallel_backtest")
    base = base.model_copy()
    base.symbol = "USDJPY"
    base.pip_size = 0.01
    base.display_currency = "USD"
    base.deposit_amount = 100000.0
    profile = apply_preset(base, PresetId.PHASE3_INTEGRATED_USD_JPY)

    # Phase 3 preset application intentionally preserves editor risk fields.
    # For research replay we want the preset's own risk block to match the live Phase 3 template.
    risk_patch = get_preset_patch(PresetId.PHASE3_INTEGRATED_USD_JPY).get("risk") or {}
    if risk_patch:
        profile = profile.model_copy(update={"risk": profile.risk.model_copy(update=risk_patch)})
    return profile


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if args.profile:
        profile = load_profile_v1(Path(args.profile))
    else:
        profile = _build_phase3_profile()
    policy = next(
        (
            pol
            for pol in profile.execution.policies
            if getattr(pol, "enabled", True) and getattr(pol, "type", "") == "phase3_integrated"
        ),
        None,
    )
    if policy is None:
        raise RuntimeError("No enabled phase3_integrated policy found in profile.")

    m1_all = parity._load_m1_csv(input_csv)
    data_by_tf_all = {
        "M1": m1_all,
        "M5": parity._resample_ohlc(m1_all, "5min"),
        "M15": parity._resample_ohlc(m1_all, "15min"),
        "H1": parity._resample_ohlc(m1_all, "1h"),
        "H4": parity._resample_ohlc(m1_all, "4h"),
        "D": parity._resample_ohlc(m1_all, "1D"),
    }
    start_ts = pd.Timestamp(m1_all["time"].min()).tz_convert("UTC")
    end_ts = pd.Timestamp(m1_all["time"].max()).tz_convert("UTC")
    phase3_preset_id = getattr(profile, "active_preset_name", None)
    sizing = load_phase3_sizing_config(preset_id=phase3_preset_id)
    adapter = RecordingMockAdapter(equity=float(args.starting_equity), pip_size=float(profile.pip_size))
    phase3_state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    tf_frames = {tf: data_by_tf_all[tf].reset_index(drop=True) for tf in data_by_tf_all}
    tf_time = {
        tf: (
            pd.to_datetime(tf_frames[tf]["time"], utc=True).to_numpy(dtype="datetime64[ns]")
            if "time" in tf_frames[tf].columns
            else np.array([], dtype="datetime64[ns]")
        )
        for tf in tf_frames
    }
    tf_history_cap = {
        "M1": 3200,
        "M5": 450,
        "M15": 240,
        "H1": 160,
        "H4": 120,
        "D": 90,
    }
    tf_last_i = {tf: -1 for tf in tf_frames}
    tf_last_slice = {tf: tf_frames[tf].iloc[:0] for tf in tf_frames}
    session_mask = np.array(
        [
            classify_session(ts.to_pydatetime(), sizing) is not None
            for ts in pd.to_datetime(m1_all["time"], utc=True)
        ],
        dtype=bool,
    )

    for row, should_eval in zip(m1_all.itertuples(index=False), session_mask):
        t = pd.Timestamp(row.time).tz_convert("UTC")
        if not should_eval and not adapter._open_trades:
            continue
        bid = float(row.close)
        ask = float(row.close) + float(args.spread_pips) * float(profile.pip_size)
        tick = SimpleNamespace(bid=bid, ask=ask)
        adapter.set_tick(tick)

        sliced = {}
        t_np = np.datetime64(t.to_datetime64())
        for tf, d in tf_frames.items():
            if d.empty or tf_time[tf].size == 0:
                sliced[tf] = d
                continue
            i = int(np.searchsorted(tf_time[tf], t_np, side="right"))
            if i == tf_last_i[tf]:
                sliced[tf] = tf_last_slice[tf]
                continue
            start_i = max(0, i - int(tf_history_cap.get(tf, i)))
            current_slice = d.iloc[start_i:i]
            tf_last_i[tf] = i
            tf_last_slice[tf] = current_slice
            sliced[tf] = current_slice

        _manage_exits_recording(
            adapter=adapter,
            profile=profile,
            data_by_tf=sliced,
            phase3_state=phase3_state,
            t=t,
            sizing_config=sizing,
        )
        phase3_state["open_trade_count"] = len(adapter._open_trades)

        if not should_eval:
            continue

        equity_before = adapter.balance
        result = evaluate_phase3_bar(
            adapter=adapter,
            profile=profile,
            log_dir=Path("runtime") / profile.profile_name,
            policy=policy,
            context=MarketContext(spread_pips=(ask - bid) / float(profile.pip_size), alignment_score=0),
            data_by_tf=sliced,
            tick=tick,
            mode="ARMED_AUTO_DEMO",
            phase3_state=phase3_state,
            store=adapter,
            sizing_config=sizing if sizing else None,
            is_new_m1=True,
            preset_id=phase3_preset_id,
        )
        updates = result.get("phase3_state_updates") or {}
        if updates:
            phase3_state.update(updates)
        risk_usd_planned = result.get("risk_usd_planned")
        if risk_usd_planned is not None:
            adapter.set_last_order_risk(float(risk_usd_planned))
        dec = result.get("decision")
        if getattr(dec, "placed", False):
            for sk in list(updates.keys()):
                if isinstance(sk, str) and (
                    sk.startswith("session_ny_") or sk.startswith("session_london_") or sk.startswith("session_tokyo_")
                ):
                    adapter.set_last_order_session(sk)
                    break
            if adapter._open_trades:
                last_oid = max(adapter._open_trades)
                adapter._open_trades[last_oid]["entry_time"] = t.isoformat()
                adapter._open_trades[last_oid]["strategy_tag"] = result.get("strategy_tag")
                adapter._open_trades[last_oid]["equity_before"] = float(equity_before)
                adapter._open_trades[last_oid]["trade_id"] = str(last_oid)
                adapter._open_trades[last_oid]["original_lots"] = float(adapter._open_trades[last_oid].get("size_lots") or 0.0)
        records.append(
            {
                "bar_time": t.isoformat(),
                "attempted": bool(getattr(dec, "attempted", False)),
                "placed": bool(getattr(dec, "placed", False)),
                "side": getattr(dec, "side", None),
                "reason": getattr(dec, "reason", ""),
                "strategy_tag": result.get("strategy_tag"),
                "strategy": parity._strategy_from_tag(result.get("strategy_tag")),
                "session": parity._session_from_tag(result.get("strategy_tag")),
                "entry_price": result.get("entry_price"),
                "sl_price": result.get("sl_price"),
                "tp1_price": result.get("tp1_price"),
                "units": result.get("units"),
            }
        )

    _finalize_open_trades(adapter, phase3_state, m1_all.iloc[-1])

    trade_rows = _trade_rows(adapter.closed_trades)
    eq_curve = merged_engine._build_equity_curve(trade_rows, float(args.starting_equity))
    summary = merged_engine._stats(trade_rows, float(args.starting_equity), eq_curve)
    by_day, by_hour = _group_day_hour(adapter.closed_trades)

    out = {
        "engine": "phase3_live_parallel_backtest",
        "profile": str(Path(args.profile).resolve()) if args.profile else "preset:phase3_integrated_usd_jpy",
        "dataset": str(input_csv.resolve()),
        "date_range": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "starting_equity": float(args.starting_equity),
        "ending_equity": float(args.starting_equity + summary["net_usd"]),
        "spread_pips": float(args.spread_pips),
        "combined": summary,
        "by_strategy": merged_engine._subset_breakdown(trade_rows, lambda t: t.strategy),
        "by_session": merged_engine._subset_breakdown(trade_rows, lambda t: t.entry_session),
        "by_month": merged_engine._group_monthly(trade_rows),
        "by_day_of_week": by_day,
        "by_hour_utc": by_hour,
        "entry_decisions": {
            "rows": int(len(records)),
            "entries": int(sum(1 for r in records if r["placed"])),
            "by_strategy": parity._value_counts(pd.DataFrame(records).query("placed == True"), "strategy"),
            "by_session": parity._value_counts(pd.DataFrame(records).query("placed == True"), "session"),
        },
        "closed_trades": adapter.closed_trades,
        "equity_curve": eq_curve,
        "notes": {
            "implementation": "Replay of current live Phase 3 engine over local M1 bars with resampled higher timeframes.",
            "session_behavior": "Uses exact live Phase 3 engine session/day logic; only trades when the live engine would trade.",
            "exit_model": "Simplified replay: closes on SL, TP1, session_end, or end_of_replay.",
            "effective_config_sources": (sizing.get("_meta", {}) or {}).get("source_paths", {}),
        },
    }

    out_dir = Path("research_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_csv.stem.lower()
        label = "1000k" if "800k_split" in stem else "500k" if "500k" in stem else stem
        out_path = out_dir / f"phase3_live_parallel_{label}_report.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "combined": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
