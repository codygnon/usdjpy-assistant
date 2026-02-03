from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from adapters import mt5_adapter
from core.execution_engine import execute_signal_demo_only
from core.execution_state import RuntimeState, load_state, save_state
from core.models import MarketContext, TradeCandidate
from core.profile import (
    ExecutionPolicyConfirmedCross,
    ExecutionPolicyIndicator,
    ExecutionPolicyPriceLevelTrend,
    ProfileV1,
    load_profile_v1,
    save_profile_v1,
)
from core.risk_engine import evaluate_trade
from core.signal_engine import Signal
from core.ta_analysis import compute_ta_multi
from storage.sqlite_store import SqliteStore


BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
LOGS_DIR = BASE_DIR / "logs"


def list_profiles() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    # include nested v1 folder if present
    return sorted([p for p in PROFILES_DIR.rglob("*.json") if p.is_file()])


def store_for(profile_name: str) -> SqliteStore:
    log_dir = LOGS_DIR / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()
    return store


def runtime_state_path(profile_name: str) -> Path:
    return LOGS_DIR / profile_name / "runtime_state.json"


def load_latest_context(store: SqliteStore, profile_name: str) -> dict | None:
    row = store.latest_snapshot(profile_name)
    return dict(row) if row is not None else None


def main() -> None:
    st.set_page_config(page_title="USDJPY Assistant v1", layout="wide")
    st.title("USDJPY Assistant v1 (MT5 Demo)")

    profiles = list_profiles()
    if not profiles:
        st.error(f"No profiles found under `{PROFILES_DIR}`.")
        st.stop()

    sel = st.sidebar.selectbox("Profile", profiles, format_func=lambda p: p.relative_to(PROFILES_DIR).as_posix())
    profile_path = Path(sel)

    # Load (migrates legacy profiles in-memory)
    profile: ProfileV1 = load_profile_v1(profile_path)
    pname = profile.profile_name

    store = store_for(pname)
    state_path = runtime_state_path(pname)
    state = load_state(state_path)

    # --- Run controls / status ---
    st.sidebar.subheader("Run controls")
    st.sidebar.caption("Choose profile, mode, and runtime options. Save state after changes.")
    st.sidebar.write({"profile": pname, "symbol": profile.symbol})

    mode = st.sidebar.selectbox("Mode", ["DISARMED", "ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"], index=["DISARMED", "ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"].index(state.mode))
    st.sidebar.caption("DISARMED: no execution. ARMED_MANUAL_CONFIRM: log pending signals for you to execute in UI. ARMED_AUTO_DEMO: auto-execute on demo only.")
    kill_switch = st.sidebar.checkbox("Kill switch (disable execution)", value=state.kill_switch)
    st.sidebar.caption("When enabled, the loop never places orders even if armed. Use as an emergency stop.")
    if st.sidebar.button("Save runtime state"):
        save_state(state_path, RuntimeState(mode=mode, kill_switch=kill_switch, last_processed_bar_time_utc=state.last_processed_bar_time_utc))
        st.sidebar.success("Saved runtime state.")

    loop_running = False
    if "loop_process" in st.session_state and st.session_state["loop_process"] is not None:
        p = st.session_state["loop_process"]
        if p.poll() is None:
            loop_running = True
        else:
            st.session_state["loop_process"] = None
            if "loop_log_file" in st.session_state and st.session_state["loop_log_file"] is not None:
                try:
                    st.session_state["loop_log_file"].close()
                except Exception:
                    pass
                st.session_state["loop_log_file"] = None

    with st.sidebar.expander("Current loop settings"):
        st.caption("The loop uses these when started with this profile. Save profile JSON and runtime state before starting the loop.")
        st.write("**Profile path:**", str(profile_path.resolve()))
        st.write("**Runtime:**", {"mode": mode, "kill_switch": kill_switch})
        ex = getattr(profile, "execution", None)
        policies_summary = []
        if ex is not None:
            for pol in ex.policies:
                policies_summary.append({"type": getattr(pol, "type", "?"), "id": getattr(pol, "id", "?"), "enabled": getattr(pol, "enabled", True)})
        settings = {
            "symbol": profile.symbol,
            "pip_size": float(profile.pip_size),
            "risk": {"max_lots": profile.risk.max_lots, "max_spread_pips": profile.risk.max_spread_pips, "require_stop": profile.risk.require_stop},
            "alignment_enabled": profile.strategy.filters.alignment.enabled,
            "loop_poll_seconds": float(ex.loop_poll_seconds) if ex is not None else None,
            "loop_poll_seconds_fast": float(ex.loop_poll_seconds_fast) if ex is not None else None,
            "policies": policies_summary,
        }
        st.json(settings)
        st.write("**Loop:**", "running" if loop_running else "not running")

    st.sidebar.markdown("Run loop from UI:")
    st.sidebar.caption("Start or stop the loop here. Uses the same venv and profile as Doctor. Output: logs/<profile>/loop.log.")
    run_clicked = st.sidebar.button("Run loop", disabled=loop_running)
    stop_clicked = st.sidebar.button("Stop loop", disabled=not loop_running)
    if run_clicked and not loop_running:
        log_dir = LOGS_DIR / pname
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "loop.log"
        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            st.sidebar.error(f"Could not create loop.log: {e}")
        else:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(BASE_DIR / "run_loop.py"), "--profile", str(profile_path.resolve())],
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            st.session_state["loop_process"] = proc
            st.session_state["loop_log_file"] = log_file
            st.session_state["loop_log_path"] = str(log_path)
            st.sidebar.success("Loop started. View loop log for output.")
            st.rerun()
    if stop_clicked and loop_running:
        p = st.session_state["loop_process"]
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        f = st.session_state.get("loop_log_file")
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        st.session_state["loop_process"] = None
        st.session_state["loop_log_file"] = None
        if "loop_log_path" in st.session_state:
            del st.session_state["loop_log_path"]
        st.sidebar.success("Loop stopped.")
        st.rerun()

    st.sidebar.markdown("Run loop in a separate terminal:")
    st.sidebar.caption("You can also start the loop via the **Run loop** button above (same venv and profile). Use a terminal if you prefer.")
    venv_scripts = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    venv_bin = BASE_DIR / ".venv" / "bin" / "python"
    venv_py = venv_scripts if venv_scripts.exists() else (venv_bin if venv_bin.exists() else None)
    py_cmd = str(venv_py) if venv_py is not None else sys.executable
    st.sidebar.code(f'"{py_cmd}" "{(BASE_DIR / "run_loop.py")}" --profile "{profile_path.resolve()}"', language="bash")

    with st.sidebar.expander("View loop log"):
        log_path = LOGS_DIR / pname / "loop.log"
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as e:
                st.caption(f"Could not read log: {e}")
            else:
                last_n = 80
                tail = lines[-last_n:] if len(lines) > last_n else lines
                log_text = "".join(tail) if tail else "(empty)"
                st.code(log_text, language="text")
        else:
            st.caption("No log yet. Start the loop to create one.")

    st.sidebar.markdown("One-off checks:")
    st.sidebar.caption("Doctor: MT5 connection, symbol, ticks, bars. Snapshot: compute and log context once.")
    if st.sidebar.button("Run Doctor"):
        out = subprocess.run([sys.executable, str(BASE_DIR / "doctor_mt5.py"), "--profile", str(profile_path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        st.sidebar.code((out.stdout + ("\n" + out.stderr if out.stderr else "")).strip(), language="text")
    if st.sidebar.button("Run Snapshot (once)"):
        out = subprocess.run([sys.executable, str(BASE_DIR / "snapshot_log.py"), "--profile", str(profile_path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        st.sidebar.code((out.stdout + ("\n" + out.stderr if out.stderr else "")).strip(), language="text")

    # --- Profile editor (grouped so context/TA/stats show first) ---
    with st.expander("Profile settings (Basic, Advanced, Trade management, Execution)", expanded=False):
        st.caption("Edit settings below, then click **Save profile JSON** to persist.")

        with st.expander("Basic (risk + symbol)", expanded=True):
            st.caption("Symbol, pip size, and risk limits. Trades are rejected if they violate these rules.")
            c1, c2, c3 = st.columns(3)
            with c1:
                profile.symbol = st.text_input("symbol", value=profile.symbol)
                st.caption("MT5 symbol (e.g. USDJPY.PRO). Must match the symbol in your terminal.")
                profile.pip_size = st.number_input("pip_size", value=float(profile.pip_size), step=0.001, format="%.3f")
                st.caption("Pip size for the pair (e.g. 0.01 for JPY). Used for spread, stops, and targets in pips.")
            with c2:
                profile.risk.max_lots = st.number_input("max_lots", value=float(profile.risk.max_lots), step=0.1)
                st.caption("Maximum position size in lots. Trades are rejected if size exceeds this.")
                profile.risk.max_spread_pips = st.number_input("max_spread_pips", value=float(profile.risk.max_spread_pips), step=0.1)
                st.caption("Max allowed spread in pips. No new trades when spread is above this.")
                profile.risk.max_trades_per_day = st.number_input("max_trades_per_day", value=int(profile.risk.max_trades_per_day), step=1)
                st.caption("Daily cap on new trades. Resets at midnight UTC.")
            with c3:
                profile.risk.require_stop = st.checkbox("require_stop", value=bool(profile.risk.require_stop))
                st.caption("If on, every trade must have a stop loss. Trades without a stop are rejected.")
                profile.risk.min_stop_pips = st.number_input("min_stop_pips", value=float(profile.risk.min_stop_pips), step=1.0)
                st.caption("Minimum stop distance in pips. Stops closer than this are rejected.")
                profile.risk.max_open_trades = st.number_input("max_open_trades", value=int(profile.risk.max_open_trades), step=1)
                st.caption("Max open positions at once. No new trades when this limit is reached.")

        with st.expander("Advanced (strategy + filters)", expanded=False):
            st.caption("EMA/SMA periods per timeframe, cross confirmation, and filters. Used by the confirmed-cross execution policy.")
            st.markdown("**Timeframe indicators**")
            st.caption("EMA fast and SMA slow define the regime (bull/bear). M1 ema_stack is used for scalping filters when enabled.")
            for tf in ["M1", "M15", "H4"]:
                cfg = profile.strategy.timeframes[tf]  # type: ignore[index]
                st.write(f"**{tf}**")
                a, b, c = st.columns(3)
                with a:
                    cfg.ema_fast = st.number_input(f"{tf}.ema_fast", value=int(cfg.ema_fast), step=1)
                with b:
                    cfg.sma_slow = st.number_input(f"{tf}.sma_slow", value=int(cfg.sma_slow), step=1)
                with c:
                    if tf == "M1":
                        stack = cfg.ema_stack or [8, 13, 21]
                        stack_s = st.text_input(f"{tf}.ema_stack (comma)", value=",".join(str(x) for x in stack))
                        st.caption("Comma-separated EMA periods for stack filter (e.g. 8, 13, 21).")
                        try:
                            cfg.ema_stack = [int(x.strip()) for x in stack_s.split(",") if x.strip()]
                        except Exception:
                            st.warning("Invalid ema_stack; keeping previous.")

            st.markdown("**Cross setup (M1 confirmation)**")
            st.caption("Confirmed cross policy: wait for EMA/SMA cross + confirmation candles before signalling.")
            setup = profile.strategy.setups["m1_cross_entry"]
            setup.enabled = st.checkbox("m1_cross_entry.enabled", value=bool(setup.enabled))
            st.caption("Enable or disable the M1 cross setup used by the confirmed_cross execution policy.")
            setup.confirmation.confirm_bars = st.number_input("confirm_bars", value=int(setup.confirmation.confirm_bars), step=1)
            st.caption("Number of closed bars after the EMA/SMA cross that must confirm the move before a signal is generated.")
            setup.confirmation.max_wait_bars = st.number_input("max_wait_bars", value=int(setup.confirmation.max_wait_bars), step=1)
            st.caption("Max bars to wait for confirmation after the cross. Older crosses are ignored.")
            setup.confirmation.require_close_on_correct_side = st.checkbox("require_close_on_correct_side", value=bool(setup.confirmation.require_close_on_correct_side))
            st.caption("If on, confirmation bars must close on the correct side of the SMA.")
            setup.confirmation.min_distance_pips = st.number_input("min_distance_pips", value=float(setup.confirmation.min_distance_pips), step=0.1)
            st.caption("Minimum EMA–SMA distance in pips for a valid confirmation.")

            st.markdown("**Filters**")
            st.caption("Alignment, EMA stack, and ATR filters. Signals that fail any enabled filter are rejected.")
            align = profile.strategy.filters.alignment
            align.enabled = st.checkbox("alignment.enabled", value=bool(align.enabled))
            st.caption("Require multi-timeframe alignment (H4/M15/M1) before trading.")
            align.method = st.selectbox("alignment.method", ["score", "strict"], index=["score", "strict"].index(align.method))
            st.caption("Score: trade when alignment score meets threshold. Strict: H4 and M15 must agree with side.")
            align.min_score_to_trade = st.number_input("alignment.min_score_to_trade", value=int(align.min_score_to_trade), step=1)
            st.caption("Min alignment score (e.g. 1) to allow a trade. Higher = stricter.")

            stack_f = profile.strategy.filters.ema_stack_filter
            stack_f.enabled = st.checkbox("ema_stack_filter.enabled", value=bool(stack_f.enabled))
            st.caption("Require EMA stack (e.g. 8 > 13 > 21 for bull) on the chosen timeframe.")
            stack_f.min_separation_pips = st.number_input("ema_stack_filter.min_separation_pips", value=float(stack_f.min_separation_pips), step=0.1)
            st.caption("Min pips between EMAs for a valid stack.")

            atr_f = profile.strategy.filters.atr_filter
            atr_f.enabled = st.checkbox("atr_filter.enabled", value=bool(atr_f.enabled))
            st.caption("Require ATR within a range (avoid very low or very high volatility).")
            atr_f.atr_period = st.number_input("atr_filter.atr_period", value=int(atr_f.atr_period), step=1)
            atr_f.min_atr_pips = st.number_input("atr_filter.min_atr_pips", value=float(atr_f.min_atr_pips), step=0.5)
            atr_f.max_atr_pips = st.number_input("atr_filter.max_atr_pips (0 = none)", value=float(atr_f.max_atr_pips or 0.0), step=0.5)
            st.caption("ATR period and min/max ATR in pips. 0 = no max.")
            atr_f.max_atr_pips = None if float(atr_f.max_atr_pips) == 0.0 else float(atr_f.max_atr_pips)

    with st.expander("Trade management", expanded=False):
        st.caption("Default take-profit and risk:reward. Used when no explicit TP is set.")
        tm = profile.trade_management.target
        tm.mode = st.selectbox("target.mode", ["fixed_pips", "rr"], index=["fixed_pips", "rr"].index(tm.mode))
        st.caption("Fixed pips: TP = entry ± pips_default. RR: TP from stop distance × rr_default.")
        if tm.mode == "fixed_pips":
            tm.pips_default = st.number_input("target.pips_default", value=float(tm.pips_default), step=1.0)
            st.caption("Default take-profit distance in pips.")
        else:
            tm.rr_default = st.number_input("target.rr_default", value=float(tm.rr_default), step=0.1)
            st.caption("Risk:reward multiple (e.g. 1.5 = 1.5× risk).")

    with st.expander("Execution (policies + loop)", expanded=False):
        st.caption("Execution policies define when and how the loop places orders. Loop poll interval controls how often we check for new bars or price levels.")
        ex = profile.execution
        ex.loop_poll_seconds = st.number_input("loop_poll_seconds", value=float(ex.loop_poll_seconds), step=0.5, min_value=1.0)
        st.caption("How often (seconds) the loop polls for new M1 bars. Lower = faster reaction but more MT5 requests.")
        ex.loop_poll_seconds_fast = st.number_input("loop_poll_seconds_fast", value=float(ex.loop_poll_seconds_fast), step=0.5, min_value=1.0)
        st.caption("Faster poll used when a price-level policy has use_pending_order off (e.g. 1–2 s).")

        st.markdown("**Execution policies**")
        st.caption("Toggle policies on/off. Add new policies or edit type-specific params. Only enabled policies run.")
        for i, pol in enumerate(list(ex.policies)):
            with st.container():
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    pol.enabled = st.checkbox(f"Policy {i+1} enabled", value=bool(pol.enabled), key=f"pol_en_{i}_{pol.id}")
                with c2:
                    st.write(f"**{pol.type}** — `{pol.id}`")
                with c3:
                    if st.button("Remove", key=f"remove_pol_{i}_{pol.id}"):
                        del ex.policies[i]
                        save_profile_v1(profile, profile_path)
                        st.rerun()
                if pol.type == "confirmed_cross":
                    pol.setup_id = st.text_input("setup_id", value=pol.setup_id, key=f"pol_setup_{i}_{pol.id}")
                    st.caption("Setup ID (e.g. m1_cross_entry) linked to strategy.setups. Confirmed cross signals use this setup.")
                elif pol.type == "price_level_trend":
                    pol.price_level = st.number_input("price_level (X)", value=float(pol.price_level), format="%.5f", key=f"pol_pl_{i}_{pol.id}")
                    st.caption("Price level (X). Buy limit below market or sell limit above market at this price.")
                    pol.side = st.selectbox("side", ["buy", "sell"], index=["buy", "sell"].index(pol.side), key=f"pol_side_{i}_{pol.id}")
                    pol.tp_pips = st.number_input("tp_pips (Y)", value=float(pol.tp_pips), step=0.5, key=f"pol_tp_{i}_{pol.id}")
                    st.caption("Take-profit distance in pips from entry.")
                    sl_val = pol.sl_pips if pol.sl_pips is not None else 0.0
                    sl_in = st.number_input("sl_pips (0 = none)", value=float(sl_val), step=0.5, key=f"pol_sl_{i}_{pol.id}")
                    pol.sl_pips = None if sl_in == 0.0 else float(sl_in)
                    st.caption("Optional stop-loss in pips. 0 = no SL.")
                    tw = pol.trend_timeframes
                    tw_s = st.text_input("trend_timeframes (comma)", value=",".join(tw), key=f"pol_tw_{i}_{pol.id}")
                    valid_tf = {"M1", "M15", "H4"}
                    try:
                        pol.trend_timeframes = [x.strip() for x in tw_s.split(",") if x.strip() and x.strip() in valid_tf] or list(tw)  # type: ignore[assignment]
                    except Exception:
                        pass
                    st.caption("One or more timeframes (e.g. M1 or M15,M1) that must match trend_direction. One = more aggressive.")
                    pol.trend_direction = st.selectbox("trend_direction", ["bearish", "bullish"], index=["bearish", "bullish"].index(pol.trend_direction), key=f"pol_td_{i}_{pol.id}")
                    mw = pol.max_wait_minutes
                    mw_in = st.number_input("max_wait_minutes (0 = default 60)", value=int(mw) if mw is not None else 0, step=5, key=f"pol_mw_{i}_{pol.id}")
                    pol.max_wait_minutes = None if mw_in == 0 else mw_in
                    st.caption("Skip placing again for this rule within this many minutes. 0 = use 60.")
                    pol.use_pending_order = st.checkbox("use_pending_order", value=bool(pol.use_pending_order), key=f"pol_upo_{i}_{pol.id}")
                    st.caption("If on, place pending limit at X. If off, poll tick and send market when price reaches X (uses fast poll).")
                elif pol.type == "indicator_based":
                    pol.timeframe = st.selectbox("timeframe", ["M1", "M15", "H4"], index=["M1", "M15", "H4"].index(pol.timeframe), key=f"pol_ind_tf_{i}_{pol.id}")
                    st.caption("Timeframe for RSI/MACD and regime.")
                    pol.regime = st.selectbox("regime", ["bull", "bear"], index=["bull", "bear"].index(pol.regime), key=f"pol_ind_reg_{i}_{pol.id}")
                    pol.side = st.selectbox("side", ["buy", "sell"], index=["buy", "sell"].index(pol.side), key=f"pol_ind_side_{i}_{pol.id}")
                    st.caption("Regime must match; RSI zone (e.g. oversold) triggers this side.")
                    pol.rsi_period = int(st.number_input("rsi_period", value=int(pol.rsi_period), step=1, key=f"pol_ind_rsi_per_{i}_{pol.id}"))
                    pol.rsi_oversold = st.number_input("rsi_oversold", value=float(pol.rsi_oversold), step=1.0, key=f"pol_ind_rsi_os_{i}_{pol.id}")
                    pol.rsi_overbought = st.number_input("rsi_overbought", value=float(pol.rsi_overbought), step=1.0, key=f"pol_ind_rsi_ob_{i}_{pol.id}")
                    pol.rsi_zone = st.selectbox("rsi_zone", ["oversold", "overbought", "neutral"], index=["oversold", "overbought", "neutral"].index(pol.rsi_zone), key=f"pol_ind_zone_{i}_{pol.id}")
                    st.caption("RSI thresholds and zone that must match (e.g. oversold for dip buy).")
                    pol.use_macd_cross = st.checkbox("use_macd_cross", value=bool(pol.use_macd_cross), key=f"pol_ind_macd_{i}_{pol.id}")
                    if pol.use_macd_cross:
                        pol.macd_fast = int(st.number_input("macd_fast", value=int(pol.macd_fast), step=1, key=f"pol_ind_mf_{i}_{pol.id}"))
                        pol.macd_slow = int(st.number_input("macd_slow", value=int(pol.macd_slow), step=1, key=f"pol_ind_ms_{i}_{pol.id}"))
                        pol.macd_signal = int(st.number_input("macd_signal", value=int(pol.macd_signal), step=1, key=f"pol_ind_macds_{i}_{pol.id}"))
                    pol.tp_pips = st.number_input("tp_pips", value=float(pol.tp_pips), step=0.5, key=f"pol_ind_tp_{i}_{pol.id}")
                    sl_val = pol.sl_pips if pol.sl_pips is not None else 0.0
                    sl_in = st.number_input("sl_pips (0 = none)", value=float(sl_val), step=0.5, key=f"pol_ind_sl_{i}_{pol.id}")
                    pol.sl_pips = None if sl_in == 0.0 else float(sl_in)
                    st.caption("Take-profit and optional stop-loss in pips.")
                st.divider()

        with st.expander("Add new policy"):
            st.caption("Create a new execution policy. It is saved to the profile immediately.")
            t = st.selectbox("New policy type", ["confirmed_cross", "price_level_trend", "indicator_based"], key="add_pol_type")
            if t == "confirmed_cross":
                new_id = st.text_input("id", value="confirmed_cross_1", key="add_pol_id")
                setup_id = st.text_input("setup_id", value="m1_cross_entry", key="add_pol_setup")
                if st.button("Create confirmed_cross policy", key="add_pol_submit_cc"):
                    ex.policies.append(ExecutionPolicyConfirmedCross(id=new_id, enabled=True, setup_id=setup_id))
                    save_profile_v1(profile, profile_path)
                    st.rerun()
            elif t == "price_level_trend":
                new_id = st.text_input("id", value="price_level_trend_1", key="add_pl_id")
                pl_val = st.number_input("price_level", value=150.0, format="%.5f", key="add_pl_pl")
                side_val = st.selectbox("side", ["buy", "sell"], key="add_pl_side")
                tp_val = st.number_input("tp_pips", value=10.0, step=0.5, key="add_pl_tp")
                sl_in = st.number_input("sl_pips (0 = none)", value=0.0, step=0.5, key="add_pl_sl")
                sl_val = None if sl_in == 0.0 else float(sl_in)
                tw_s = st.text_input("trend_timeframes (comma)", value="M15,M1", key="add_pl_tw")
                valid_tf = {"M1", "M15", "H4"}
                tw_val = [x.strip() for x in tw_s.split(",") if x.strip() and x.strip() in valid_tf] or ["M15", "M1"]
                td_val = st.selectbox("trend_direction", ["bearish", "bullish"], key="add_pl_td")
                use_po_val = st.checkbox("use_pending_order", value=True, key="add_pl_upo")
                if st.button("Create price_level_trend policy", key="add_pol_submit_pl"):
                    ex.policies.append(
                        ExecutionPolicyPriceLevelTrend(
                            id=new_id,
                            enabled=True,
                            price_level=float(pl_val),
                            side=side_val,  # type: ignore[arg-type]
                            tp_pips=float(tp_val),
                            sl_pips=sl_val,
                            trend_timeframes=tw_val,  # type: ignore[arg-type]
                            trend_direction=td_val,  # type: ignore[arg-type]
                            use_pending_order=use_po_val,
                        )
                    )
                    save_profile_v1(profile, profile_path)
                    st.rerun()
            else:
                new_id = st.text_input("id", value="indicator_based_1", key="add_ind_id")
                tf_val = st.selectbox("timeframe", ["M1", "M15", "H4"], index=1, key="add_ind_tf")
                reg_val = st.selectbox("regime", ["bull", "bear"], key="add_ind_reg")
                side_val = st.selectbox("side", ["buy", "sell"], key="add_ind_side")
                rsi_per = int(st.number_input("rsi_period", value=14, step=1, key="add_ind_rsi_per"))
                rsi_os = st.number_input("rsi_oversold", value=30.0, step=1.0, key="add_ind_rsi_os")
                rsi_ob = st.number_input("rsi_overbought", value=70.0, step=1.0, key="add_ind_rsi_ob")
                zone_val = st.selectbox("rsi_zone", ["oversold", "overbought", "neutral"], key="add_ind_zone")
                use_macd = st.checkbox("use_macd_cross", value=False, key="add_ind_macd")
                macd_f = int(st.number_input("macd_fast", value=12, step=1, key="add_ind_mf")) if use_macd else 12
                macd_s = int(st.number_input("macd_slow", value=26, step=1, key="add_ind_ms")) if use_macd else 26
                macd_sig = int(st.number_input("macd_signal", value=9, step=1, key="add_ind_macds")) if use_macd else 9
                tp_val = st.number_input("tp_pips", value=10.0, step=0.5, key="add_ind_tp")
                sl_in = st.number_input("sl_pips (0 = none)", value=0.0, step=0.5, key="add_ind_sl")
                sl_val = None if sl_in == 0.0 else float(sl_in)
                if st.button("Create indicator_based policy", key="add_pol_submit_ind"):
                    ex.policies.append(
                        ExecutionPolicyIndicator(
                            id=new_id,
                            enabled=True,
                            timeframe=tf_val,  # type: ignore[arg-type]
                            regime=reg_val,  # type: ignore[arg-type]
                            side=side_val,  # type: ignore[arg-type]
                            rsi_period=rsi_per,
                            rsi_oversold=rsi_os,
                            rsi_overbought=rsi_ob,
                            rsi_zone=zone_val,  # type: ignore[arg-type]
                            use_macd_cross=use_macd,
                            macd_fast=macd_f,
                            macd_slow=macd_s,
                            macd_signal=macd_sig,
                            tp_pips=float(tp_val),
                            sl_pips=sl_val,
                        )
                    )
                    save_profile_v1(profile, profile_path)
                    st.rerun()

    if st.button("Save profile JSON"):
        save_profile_v1(profile, profile_path)
        st.success(f"Saved profile to {profile_path}")

    # --- Latest context + Technical analysis (side by side) ---
    latest = load_latest_context(store, pname)
    col_ctx, col_ta = st.columns(2)
    with col_ctx:
        st.subheader("Latest context")
        st.caption("Latest snapshot: spread, alignment, H4/M15/M1 regime. Run loop or Snapshot to refresh.")
        if latest is None:
            st.info("No snapshots yet. Run the loop or run a snapshot manually.")
        else:
            st.json({k: latest[k] for k in ["timestamp_utc", "spread_pips", "alignment_score", "h4_regime", "m15_regime", "m1_regime"] if k in latest})
    with col_ta:
        st.subheader("Technical analysis")
        st.caption("RSI, MACD, ATR per timeframe. Helps interpret trends beyond bull/bear.")
        try:
            mt5_adapter.initialize()
            try:
                mt5_adapter.ensure_symbol(profile.symbol)
                data_by_tf = {
                    "H4": mt5_adapter.get_bars(profile.symbol, "H4", 800),
                    "M15": mt5_adapter.get_bars(profile.symbol, "M15", 2000),
                    "M1": mt5_adapter.get_bars(profile.symbol, "M1", 3000),
                }
                ta = compute_ta_multi(profile, data_by_tf)
                for tf in ["H4", "M15", "M1"]:
                    s = ta.get(tf)
                    if s is None:
                        st.write(f"**{tf}**: no data.")
                        continue
                    st.markdown(f"**{tf}**: {s.summary}")
                    st.caption(f"regime={s.regime} rsi={s.rsi_value} ({s.rsi_zone}) atr={s.atr_state}")
            finally:
                mt5_adapter.shutdown()
        except Exception as e:
            st.caption(f"Could not compute technical analysis: {e}")

    # --- Pending signals (manual confirm) ---
    st.subheader("Pending signals (manual confirm)")
    st.caption("Signals waiting for your approval when mode is ARMED_MANUAL_CONFIRM. Select one and Execute to place the order (demo only).")
    pend = store.list_pending_signals(pname)
    if not pend:
        st.caption("No pending signals.")
    else:
        pend_df = pd.DataFrame([dict(r) for r in pend])
        st.dataframe(pend_df[["timestamp_utc", "signal_id", "timeframe", "side", "confirm_time", "entry_price_hint"]], use_container_width=True)

        sel_sig = st.selectbox("Select pending signal_id to execute", [r["signal_id"] for r in pend])
        if st.button("Execute selected signal now (demo only)"):
            if kill_switch:
                st.error("Kill switch is enabled. Disable it first.")
            else:
                row = next(r for r in pend if r["signal_id"] == sel_sig)
                reasons = []
                if row["reasons_json"]:
                    try:
                        reasons = json.loads(row["reasons_json"])
                    except Exception:
                        reasons = []

                sig = Signal(
                    signal_id=str(row["signal_id"]),
                    profile_name=pname,
                    symbol=profile.symbol,
                    timeframe=str(row["timeframe"]),
                    side=str(row["side"]),
                    cross_time=pd.to_datetime(row["cross_time"], utc=True) if row["cross_time"] else pd.Timestamp.now(tz="UTC"),
                    confirm_time=pd.to_datetime(row["confirm_time"], utc=True) if row["confirm_time"] else pd.Timestamp.now(tz="UTC"),
                    entry_price_hint=float(row["entry_price_hint"] or 0.0),
                    reasons=[str(x) for x in reasons],
                )

                # Use latest snapshot for context
                ctx = latest or {}
                mkt = MarketContext(
                    spread_pips=float(ctx["spread_pips"]) if ctx.get("spread_pips") is not None else None,
                    alignment_score=int(ctx["alignment_score"]) if ctx.get("alignment_score") is not None else None,
                )
                trades_df = store.read_trades_df(pname)
                log_dir = LOGS_DIR / pname

                mt5_adapter.initialize()
                try:
                    mt5_adapter.ensure_symbol(profile.symbol)
                    dec = execute_signal_demo_only(
                        profile=profile,
                        log_dir=log_dir,
                        signal=sig,
                        context=mkt,
                        trades_df=trades_df,
                        mode="ARMED_AUTO_DEMO",
                    )
                finally:
                    mt5_adapter.shutdown()

                st.write({"attempted": dec.attempted, "placed": dec.placed, "reason": dec.reason, "order_id": dec.order_id, "deal_id": dec.deal_id})
                if dec.placed:
                    store.delete_pending_signal(sel_sig)
                    st.success("Executed and removed from pending.")

    # --- Log a trade (manual entry with risk checks) ---
    st.subheader("Manual trade log (uses risk engine + latest snapshot)")
    st.caption("Log a manual trade. Risk rules (max lots, spread, stop, etc.) are applied. Latest snapshot is attached.")
    ctx = latest or {}
    trades_df = store.read_trades_df(pname)
    mkt = MarketContext(
        spread_pips=float(ctx["spread_pips"]) if ctx.get("spread_pips") is not None else None,
        alignment_score=int(ctx["alignment_score"]) if ctx.get("alignment_score") is not None else None,
    )

    with st.form("log_trade_form"):
        side = st.selectbox("side", ["buy", "sell"])
        entry = st.number_input("entry_price", value=0.0, format="%.3f")
        stop = st.number_input("stop_price (0 = blank)", value=0.0, format="%.3f")
        target = st.number_input("target_price (0 = blank)", value=0.0, format="%.3f")
        size = st.number_input("size_lots (0 = blank)", value=0.0, step=0.1)
        notes = st.text_input("notes", value="")
        submit = st.form_submit_button("Pre-trade check + log")

    if submit:
        if not ctx:
            st.error("No latest snapshot available. Run the loop or snapshot first.")
        else:
            cand = TradeCandidate(
                symbol=profile.symbol,
                side=side,  # type: ignore[arg-type]
                entry_price=float(entry),
                stop_price=None if stop == 0 else float(stop),
                target_price=None if target == 0 else float(target),
                size_lots=None if size == 0 else float(size),
            )
            decision = evaluate_trade(profile=profile, candidate=cand, context=mkt, trades_df=trades_df)
            st.write({"allow": decision.allow, "hard_reasons": decision.hard_reasons, "warnings": decision.warnings})
            if decision.allow:
                snap = store.latest_snapshot(pname)
                trade_id = f"{pd.Timestamp.now(tz='UTC').isoformat()}_{profile.symbol}_{side}"
                store.insert_trade(
                    {
                        "trade_id": trade_id,
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": pname,
                        "symbol": profile.symbol,
                        "side": side,
                        "config_json": json.dumps(profile.model_dump()),
                        "entry_price": float(entry),
                        "stop_price": None if stop == 0 else float(stop),
                        "target_price": None if target == 0 else float(target),
                        "size_lots": None if size == 0 else float(size),
                        "notes": notes,
                        "snapshot_id": int(snap["id"]) if snap is not None else None,
                        "preset_name": profile.active_preset_name or "Manual",
                    }
                )
                st.success(f"Logged trade {trade_id} to DB.")

    # --- Close a trade ---
    st.subheader("Close a trade (DB)")
    open_trades = store.list_open_trades(pname)
    if not open_trades:
        st.info("No open trades to close.")
    else:
        options = [r["trade_id"] for r in open_trades]
        trade_id_sel = st.selectbox("Open trade_id", options)
        exit_price = st.number_input("exit_price", value=0.0, format="%.3f")
        exit_reason = st.text_input("exit_reason", value="")
        if st.button("Close selected trade"):
            if exit_price == 0.0:
                st.error("Enter a non-zero exit_price.")
            else:
                trades_df = store.read_trades_df(pname)
                row = trades_df[trades_df["trade_id"] == trade_id_sel].tail(1)
                if row.empty:
                    st.error("Trade not found in DB.")
                else:
                    r0 = row.iloc[0]
                    side0 = str(r0["side"])
                    entry0 = float(r0["entry_price"])
                    stop0 = r0["stop_price"] if pd.notna(r0["stop_price"]) else None
                    stop0 = float(stop0) if stop0 is not None else None

                    pip_size = float(profile.pip_size)
                    pips = (exit_price - entry0) / pip_size if side0 == "buy" else (entry0 - exit_price) / pip_size
                    risk_pips = abs(entry0 - stop0) / pip_size if stop0 is not None else None
                    r_mult = (pips / risk_pips) if (risk_pips is not None and risk_pips != 0) else None

                    store.close_trade(
                        trade_id=trade_id_sel,
                        updates={
                            "exit_price": float(exit_price),
                            "exit_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "exit_reason": exit_reason or None,
                            "pips": float(pips),
                            "risk_pips": float(risk_pips) if risk_pips is not None else None,
                            "r_multiple": float(r_mult) if r_mult is not None else None,
                        },
                    )
                    r_disp = round(r_mult, 3) if r_mult is not None else None
                    st.success(f"Closed {trade_id_sel} (pips={round(pips, 3)}, R={r_disp})")

    # --- Execution rejection breakdown ---
    st.subheader("Execution rejection breakdown")
    st.caption("Why execution attempts were rejected (last 200). Use to tune alignment, risk, or spread.")
    execs_df = store.read_executions_df(pname).tail(200)
    if execs_df.empty or "reason" not in execs_df.columns:
        st.caption("No executions yet.")
    else:
        def _reason_key(r: str) -> str:
            if pd.isna(r) or not r:
                return "unknown"
            s = str(r).strip()
            if ":" in s:
                return s.split(":")[0].strip()
            if " " in s:
                return s.split()[0].strip()
            return s[:40]

        execs_df = execs_df.copy()
        execs_df["reason_group"] = execs_df["reason"].map(_reason_key)
        breakdown = execs_df.groupby("reason_group", dropna=False).size().sort_values(ascending=False)
        st.write(breakdown.to_dict())

    # --- Tables ---
    st.subheader("Logs")
    st.caption("Snapshots, trades, and executions from the loop. Use for review and debugging.")

    def _num_config(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        return {c: st.column_config.NumberColumn(c, format="%.3f") for c in df.columns if hasattr(df[c].dtype, "kind") and df[c].dtype.kind == "f"}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Snapshots (last 20)**")
        snaps = store.read_snapshots_df(pname).tail(20)
        st.dataframe(snaps, use_container_width=True, column_config=_num_config(snaps))
    with c2:
        st.markdown("**Trades (last 20)**")
        trades = store.read_trades_df(pname).tail(20)
        st.dataframe(trades, use_container_width=True, column_config=_num_config(trades))
    with c3:
        st.markdown("**Executions (last 20)**")
        execs = store.read_executions_df(pname).tail(20)
        st.dataframe(execs, use_container_width=True, column_config=_num_config(execs))

    st.subheader("Quick stats")
    st.caption("Win rate and average pips over closed trades.")
    trades_all = store.read_trades_df(pname)
    if trades_all.empty:
        st.info("No trades yet.")
    else:
        closed = trades_all[pd.to_numeric(trades_all.get("exit_price"), errors="coerce").notna()].copy() if "exit_price" in trades_all.columns else pd.DataFrame()
        if closed.empty:
            st.info("No closed trades yet.")
        else:
            pips = pd.to_numeric(closed.get("pips"), errors="coerce")
            wins = int((pips > 0).sum())
            total = int(pips.notna().sum())
            avg_pips = float(pips.mean()) if total > 0 else 0.0
            wr = round(wins / total, 3) if total > 0 else None
            st.write(
                {
                    "closed_trades": total,
                    "win_rate_pips": wr,
                    "avg_pips": round(avg_pips, 3),
                }
            )


if __name__ == "__main__":
    main()

