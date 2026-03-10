#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.broker import get_adapter
from core.models import MarketContext
from core.phase3_integrated_engine import (
    execute_phase3_integrated_policy_demo_only,
    load_phase3_sizing_config,
)
from core.profile import load_profile_v1

ENTRY_BAR_MATCH_MIN = 98.5
SIDE_MATCH_MIN = 99.5
SL_P95_MAX = 0.5
TP1_P95_MAX = 0.5
SESSION_MATCH_REQUIRED = 100.0


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def _fetch_prefetch_bars(adapter, symbol: str, tf: str, count: int) -> pd.DataFrame:
    df = adapter.get_bars(symbol, tf, count)
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ("time", "open", "high", "low", "close", "volume") if c in df.columns]
    return _to_utc(df[keep])


def _load_m1_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"CSV missing required columns {sorted(need)}: {path}")
    out = df[[c for c in ("time", "open", "high", "low", "close", "volume") if c in df.columns]].copy()
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return _to_utc(out)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    d = df.copy().set_index("time").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in d.columns:
        agg["volume"] = "sum"
    out = d.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return _to_utc(out)


class _MockRow(dict):
    """Dict subclass with attribute access so callers can use row["key"] or row.key."""
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def keys(self):  # noqa: D401
        return super().keys()


class MockOrderAdapter:
    """Minimal broker adapter for parity replay.

    Tracks open positions in-memory so London V2's open-risk ledger
    (execute_london_v2_entry → store.list_open_trades) uses actual placed
    risk rather than the count-based fallback.
    """

    def __init__(self, equity: float = 100_000.0) -> None:
        self.equity = float(equity)
        self.balance = float(equity)
        self._order_id = 1
        self._deal_id = 1
        self._tick = SimpleNamespace(bid=0.0, ask=0.0)
        self.placed_orders: list[dict[str, Any]] = []
        # In-memory open trades for open-risk cap (keyed by order_id).
        self._open_trades: dict[int, _MockRow] = {}

    def set_tick(self, tick) -> None:
        self._tick = tick

    def get_account_info(self):
        return SimpleNamespace(equity=self.equity, balance=self.balance, margin_used=0.0)

    def place_order(self, **kwargs):
        side = str(kwargs.get("side", "buy")).lower()
        fill = self._tick.ask if side == "buy" else self._tick.bid
        oid = self._order_id
        ret = SimpleNamespace(
            order_retcode=0,
            order_id=oid,
            deal_id=self._deal_id,
            fill_price=float(fill),
        )
        comment = kwargs.get("comment") or ""
        # Infer entry_type from order comment (mirrors run_loop.py strategy_tag logic).
        entry_type: str | None = None
        for tag in ("v14_mean_reversion", "london_v2_arb", "london_v2_d", "v44_ny"):
            if tag in str(comment):
                entry_type = f"phase3:{tag}"
                break
        order_rec = {
            "order_id": oid,
            "deal_id": self._deal_id,
            "side": side,
            "fill_price": float(fill),
            "entry_price": float(fill),
            "stop_price": kwargs.get("stop_price"),
            "target_price": kwargs.get("target_price"),
            "size_lots": kwargs.get("lots"),
            "comment": comment,
            "entry_type": entry_type,
            "risk_usd_planned": None,  # populated after order if caller provides
        }
        self.placed_orders.append(order_rec)
        self._open_trades[oid] = _MockRow(order_rec)
        self._order_id += 1
        self._deal_id += 1
        return ret

    def set_last_order_risk(self, risk_usd_planned: float) -> None:
        """Called by parity loop after getting risk_usd_planned from exec_result."""
        if self._open_trades:
            last_oid = max(self._open_trades)
            self._open_trades[last_oid]["risk_usd_planned"] = float(risk_usd_planned)

    def set_last_order_session(self, session_key: str) -> None:
        """Tag the most recently placed order with its phase3_state session_key for exit simulation."""
        if self._open_trades:
            last_oid = max(self._open_trades)
            self._open_trades[last_oid]["session_key"] = session_key

    def list_open_trades(self, profile: str) -> list[_MockRow]:
        """Return open trade rows for London open-risk ledger (store-like interface)."""
        return list(self._open_trades.values())

    def is_demo(self):
        return True


def _simulate_exits(
    bar: Any,
    open_trades: dict,
    phase3_state: dict,
    t: pd.Timestamp,
) -> None:
    """Simulate SL/TP1 hits for all open mock trades against the current M1 bar.

    Removes closed trades from open_trades in-place and updates phase3_state
    session dicts (cooldown_until, consecutive_losses, wins_closed) so the
    entry gates re-arm correctly on the next bar.

    Uses a fixed 5-minute post-close cooldown (= 1 M5 bar minimum, matching
    V44 cooldown_win/loss config default of 1 bar).
    """
    try:
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
    except (KeyError, ValueError, TypeError):
        return

    # Phase 3 integrated session windows (UTC) used by the live engine.
    # If a trade is outside its owning strategy session, close it before
    # SL/TP checks so open-count gates don't carry stale positions.
    hour_frac = float(t.hour) + float(t.minute) / 60.0

    def _outside_session(entry_type: str) -> bool:
        et = str(entry_type or "").lower()
        if et.startswith("phase3:v14"):
            return not (16.0 <= hour_frac < 22.0)
        if et.startswith("phase3:london_v2"):
            return not (8.0 <= hour_frac < 13.0)
        if et.startswith("phase3:v44"):
            return not (13.0 <= hour_frac < 16.0)
        return False

    closed_oids: list[int] = []
    for oid, trade in list(open_trades.items()):
        entry_type = str(trade.get("entry_type") or "")
        if _outside_session(entry_type):
            closed_oids.append(oid)
            continue

        side = str(trade.get("side", "")).lower()
        sl = trade.get("stop_price")
        tp = trade.get("target_price")
        if sl is None and tp is None:
            continue  # no SL/TP (kill-switch style) — leave open

        hit_sl, hit_tp = False, False
        if side == "buy":
            if sl is not None and bar_low <= float(sl):
                hit_sl = True
            elif tp is not None and bar_high >= float(tp):
                hit_tp = True
        elif side == "sell":
            if sl is not None and bar_high >= float(sl):
                hit_sl = True
            elif tp is not None and bar_low <= float(tp):
                hit_tp = True

        if not hit_sl and not hit_tp:
            continue

        closed_oids.append(oid)
        session_key = trade.get("session_key")
        if session_key:
            sdat = dict(phase3_state.get(session_key, {}))
            # Always apply a minimum post-close cooldown (5 min = 1 M5 bar).
            sdat["cooldown_until"] = (t + pd.Timedelta(minutes=5)).isoformat()
            if hit_sl:
                sdat["consecutive_losses"] = int(sdat.get("consecutive_losses", 0)) + 1
                sdat["win_streak"] = 0
            else:
                sdat["wins_closed"] = int(sdat.get("wins_closed", 0)) + 1
                sdat["win_streak"] = int(sdat.get("win_streak", 0)) + 1
                sdat["consecutive_losses"] = 0
            phase3_state[session_key] = sdat

    for oid in closed_oids:
        del open_trades[oid]


def _pick_time_col(df: pd.DataFrame) -> str | None:
    for c in ("bar_time", "time", "timestamp", "timestamp_utc", "entry_time", "entry_datetime"):
        if c in df.columns:
            return c
    return None


def _strategy_from_tag(tag: Any) -> str:
    s = str(tag or "").lower()
    if "v14" in s:
        return "v14"
    if "london_v2" in s:
        return "london_v2"
    if "v44" in s:
        return "v44_ny"
    return "unknown"


def _session_from_tag(tag: Any) -> str:
    s = str(tag or "").lower()
    if "v14" in s:
        return "tokyo"
    if "london_v2" in s:
        return "london"
    if "v44" in s:
        return "ny"
    return "unknown"


def _bool_series(df: pd.DataFrame, candidates: list[str], default: bool = False) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(False).astype(bool)
    return pd.Series(default, index=df.index, dtype=bool)


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _value_counts(rows: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    if col not in rows.columns:
        return []
    vc = rows[col].fillna("unknown").astype(str).value_counts().sort_index()
    return [{"key": str(k), "count": int(v)} for k, v in vc.items()]


def _compare_reports(parity_df: pd.DataFrame, compare_df: pd.DataFrame, pip_size: float) -> tuple[dict[str, Any], pd.DataFrame]:
    left = parity_df.copy()
    right = compare_df.copy()

    lc = _pick_time_col(left)
    rc = _pick_time_col(right)
    if lc is None or rc is None:
        return ({"error": "missing time column in one of the files"}, pd.DataFrame())

    left["bar_time"] = pd.to_datetime(left[lc], utc=True, errors="coerce")
    right["bar_time"] = pd.to_datetime(right[rc], utc=True, errors="coerce")
    left = left.dropna(subset=["bar_time"]).copy()
    right = right.dropna(subset=["bar_time"]).copy()

    if "strategy_tag" not in right.columns and "strategy" in right.columns:
        right["strategy_tag"] = right["strategy"].astype(str)

    left["strategy"] = left.get("strategy_tag", "").map(_strategy_from_tag)
    left["session"] = left.get("strategy_tag", "").map(_session_from_tag)
    right["strategy"] = right.get("strategy_tag", "").map(_strategy_from_tag)
    right["session"] = right.get("strategy_tag", "").map(_session_from_tag)

    placed_live = _bool_series(left, ["placed"], default=False)
    placed_cmp = _bool_series(right, ["placed", "attempted", "is_entry"], default=True)
    left["placed"] = placed_live
    right["placed"] = placed_cmp

    merged = left.merge(right, on="bar_time", how="outer", suffixes=("_live", "_cmp"))
    live_ent = merged.get("placed_live", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
    cmp_ent = merged.get("placed_cmp", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
    both = live_ent & cmp_ent

    side_live = merged.get("side_live", pd.Series("", index=merged.index)).astype(str).str.lower()
    side_cmp = merged.get("side_cmp", pd.Series("", index=merged.index)).astype(str).str.lower()
    session_live = merged.get("session_live", pd.Series("", index=merged.index)).astype(str).str.lower()
    session_cmp = merged.get("session_cmp", pd.Series("", index=merged.index)).astype(str).str.lower()

    sl_live = _safe_numeric(merged, "sl_price_live")
    sl_cmp = _safe_numeric(merged, "sl_price_cmp")
    tp_live = _safe_numeric(merged, "tp1_price_live")
    tp_cmp = _safe_numeric(merged, "tp1_price_cmp")

    sl_diff = (sl_live - sl_cmp).abs() / float(pip_size)
    tp_diff = (tp_live - tp_cmp).abs() / float(pip_size)

    matched = int(both.sum())
    entries_live = int(live_ent.sum())
    entries_cmp = int(cmp_ent.sum())

    entry_bar_match_pct = (matched / entries_cmp * 100.0) if entries_cmp > 0 else None

    side_mask = both & side_live.ne("") & side_cmp.ne("")
    side_match_pct = ((side_live[side_mask] == side_cmp[side_mask]).mean() * 100.0) if side_mask.any() else None

    # Filter wrong-orientation baseline SL: V14 backtest uses limit-order format where
    # sl_price is placed on the WRONG side of entry (sl > entry for buy, sl < entry for sell).
    # These pairs must be excluded from P95 — the semantics are incompatible with the
    # integrated engine's pivot-based market-order SL.
    entry_cmp = _safe_numeric(merged, "entry_price_cmp")
    _has_orient_data = entry_cmp.notna() & sl_cmp.notna()
    _wrong_orient = (
        (_has_orient_data & (side_cmp == "buy") & (sl_cmp >= entry_cmp))
        | (_has_orient_data & (side_cmp == "sell") & (sl_cmp <= entry_cmp))
    )
    wrong_orient_skipped = int((both & _wrong_orient).sum())

    sl_mask = both & sl_diff.notna() & (~_wrong_orient)
    sl_p95 = float(sl_diff[sl_mask].quantile(0.95)) if sl_mask.any() else None

    tp_mask = both & tp_diff.notna() & (~_wrong_orient)
    tp_p95 = float(tp_diff[tp_mask].quantile(0.95)) if tp_mask.any() else None

    sess_mask = both & session_live.ne("") & session_cmp.ne("")
    session_match_pct = ((session_live[sess_mask] == session_cmp[sess_mask]).mean() * 100.0) if sess_mask.any() else None

    mismatch_rows: list[dict[str, Any]] = []
    for _, r in merged.loc[(~live_ent) & cmp_ent].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "missing_entry",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
        })
    for _, r in merged.loc[live_ent & (~cmp_ent)].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "extra_entry",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
        })
    for _, r in merged.loc[both & (side_live != side_cmp)].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "side_mismatch",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
            "side_live": r.get("side_live", ""),
            "side_cmp": r.get("side_cmp", ""),
        })
    for _, r in merged.loc[both & sl_diff.gt(0.5).fillna(False)].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "sl_diff_gt_0p5",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
            "sl_live": r.get("sl_price_live"),
            "sl_cmp": r.get("sl_price_cmp"),
        })
    for _, r in merged.loc[both & tp_diff.gt(0.5).fillna(False)].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "tp1_diff_gt_0p5",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
            "tp1_live": r.get("tp1_price_live"),
            "tp1_cmp": r.get("tp1_price_cmp"),
        })
    for _, r in merged.loc[both & (session_live != session_cmp)].iterrows():
        mismatch_rows.append({
            "bar_time": pd.Timestamp(r["bar_time"]).isoformat(),
            "mismatch_type": "session_mismatch",
            "strategy_live": r.get("strategy_live", ""),
            "strategy_cmp": r.get("strategy_cmp", ""),
            "session_live": r.get("session_live", ""),
            "session_cmp": r.get("session_cmp", ""),
        })

    mismatches_df = pd.DataFrame(mismatch_rows)
    summary = {
        "entries_live": entries_live,
        "entries_compare": entries_cmp,
        "matched_entries": matched,
        "entry_bar_match_pct": entry_bar_match_pct,
        "side_match_pct": side_match_pct,
        "sl_diff_p95_pips": sl_p95,
        "tp1_diff_p95_pips": tp_p95,
        "session_tag_match_pct": session_match_pct,
        "wrong_orientation_baseline_skipped": wrong_orient_skipped,
        "missing_entries": int(((~live_ent) & cmp_ent).sum()),
        "extra_entries": int((live_ent & (~cmp_ent)).sum()),
        "by_strategy_live": _value_counts(left[left["placed"]], "strategy"),
        "by_strategy_compare": _value_counts(right[right["placed"]], "strategy"),
        "by_session_live": _value_counts(left[left["placed"]], "session"),
        "by_session_compare": _value_counts(right[right["placed"]], "session"),
        "mismatch_breakdown": _value_counts(mismatches_df, "mismatch_type"),
    }

    # Per-strategy + per-session mismatch breakdown for targeted diagnosis.
    if not mismatches_df.empty:
        _strat_col = "strategy_cmp" if "strategy_cmp" in mismatches_df.columns else "strategy_live"
        _sess_col = "session_cmp" if "session_cmp" in mismatches_df.columns else "session_live"
        by_strat_sess: dict[str, Any] = {}
        for _strat in ["v14", "london_v2", "v44_ny", "unknown"]:
            _sm = mismatches_df[mismatches_df.get(_strat_col, pd.Series("", index=mismatches_df.index)).fillna("").astype(str) == _strat] if _strat_col in mismatches_df.columns else pd.DataFrame()
            if _sm.empty:
                continue
            by_strat_sess[_strat] = {
                "total": int(len(_sm)),
                "by_type": _value_counts(_sm, "mismatch_type"),
                "by_session": _value_counts(_sm, _sess_col) if _sess_col in _sm.columns else [],
            }
        summary["mismatch_by_strategy"] = by_strat_sess

        # Per-session aggregate (session+type) for direct diagnosis.
        by_sess: dict[str, Any] = {}
        for _sess in ["tokyo", "london", "ny", "unknown"]:
            _ssm = mismatches_df[mismatches_df.get(_sess_col, pd.Series("", index=mismatches_df.index)).fillna("").astype(str) == _sess] if _sess_col in mismatches_df.columns else pd.DataFrame()
            if _ssm.empty:
                continue
            by_sess[_sess] = {
                "total": int(len(_ssm)),
                "by_type": _value_counts(_ssm, "mismatch_type"),
                "by_strategy": _value_counts(_ssm, _strat_col) if _strat_col in _ssm.columns else [],
            }
        summary["mismatch_by_session"] = by_sess

    # Over-fire diagnosis: explain large extra_entry counts per strategy.
    extra_live = int((live_ent & (~cmp_ent)).sum())
    if extra_live > 0:
        notes: list[str] = []
        for _strat, _live_n, _cmp_n in [
            ("v14", entries_live, entries_cmp),
        ]:
            pass  # per-strategy counts are already in by_strategy_live/compare
        v14_live_n = sum(
            int(x["count"]) for x in summary.get("by_strategy_live", []) if x.get("key") == "v14"
        )
        v14_cmp_n = sum(
            int(x["count"]) for x in summary.get("by_strategy_compare", []) if x.get("key") == "v14"
        )
        v44_live_n = sum(
            int(x["count"]) for x in summary.get("by_strategy_live", []) if x.get("key") == "v44_ny"
        )
        v44_cmp_n = sum(
            int(x["count"]) for x in summary.get("by_strategy_compare", []) if x.get("key") == "v44_ny"
        )
        if v14_live_n > v14_cmp_n:
            notes.append(
                f"V14 over-fires {v14_live_n} vs {v14_cmp_n} baseline: parity replay resamples "
                f"M1→M15/H4/D producing different indicator values (BB/ATR/SAR/pivots) than "
                f"the OANDA-fetched bars used in the standalone V14 backtest. Use --save-baseline "
                f"for a semantically aligned reference."
            )
        if v44_live_n > v44_cmp_n:
            notes.append(
                f"V44 over-fires {v44_live_n} vs {v44_cmp_n} baseline: integrated Phase 3 V44 "
                f"momentum detection differs from the standalone session_momentum backtest engine "
                f"(different signal timing). max_open_positions=3 also allows {min(3, v44_live_n)} "
                f"consecutive-bar entries per signal cluster (by design). Use --save-baseline for "
                f"regression testing instead of cross-engine comparisons."
            )
        if notes:
            summary["over_fire_diagnosis"] = notes

    return summary, mismatches_df


def _acceptance_gate(compare_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    fails: list[str] = []
    eb = compare_metrics.get("entry_bar_match_pct")
    if eb is not None and eb < ENTRY_BAR_MATCH_MIN:
        fails.append(f"entry_bar_match_pct {eb:.2f} < {ENTRY_BAR_MATCH_MIN}")

    sm = compare_metrics.get("side_match_pct")
    if sm is not None and sm < SIDE_MATCH_MIN:
        fails.append(f"side_match_pct {sm:.2f} < {SIDE_MATCH_MIN}")

    sl95 = compare_metrics.get("sl_diff_p95_pips")
    if sl95 is not None and sl95 > SL_P95_MAX:
        fails.append(f"sl_diff_p95_pips {sl95:.3f} > {SL_P95_MAX}")

    tp95 = compare_metrics.get("tp1_diff_p95_pips")
    if tp95 is not None and tp95 > TP1_P95_MAX:
        fails.append(f"tp1_diff_p95_pips {tp95:.3f} > {TP1_P95_MAX}")

    sess = compare_metrics.get("session_tag_match_pct")
    if sess is not None and sess < SESSION_MATCH_REQUIRED:
        fails.append(f"session_tag_match_pct {sess:.2f} < {SESSION_MATCH_REQUIRED}")

    return len(fails) == 0, fails


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 parity replay checker")
    p.add_argument("--start", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--profile", default="config/profiles/usdjpy_assistant.json", help="Profile JSON path")
    p.add_argument("--compare", default="", help="Optional CSV to compare against")
    p.add_argument("--bars-count", type=int, default=120000, help="Prefetch bars per timeframe")
    p.add_argument("--spread-pips", type=float, default=1.0, help="Synthetic spread for replay ticks")
    p.add_argument("--equity", type=float, default=100000.0)
    p.add_argument("--input-csv", default="", help="Optional local M1 CSV (offline mode, bypass broker adapter)")
    p.add_argument(
        "--save-baseline", default="",
        help="Save this replay's placed entries as a self-derived baseline CSV (recommended over "
             "per-strategy backtest exports, which have different indicator semantics). "
             "Example: research_out/phase3_baseline_self_2025-01-01_2025-01-31.csv",
    )
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    profile_path = Path(args.profile)
    profile = load_profile_v1(profile_path)
    policy = next((pol for pol in profile.execution.policies if getattr(pol, "enabled", True) and getattr(pol, "type", "") == "phase3_integrated"), None)
    if policy is None:
        raise RuntimeError("No enabled phase3_integrated policy found in profile.")

    if args.input_csv:
        m1_all = _load_m1_csv(Path(args.input_csv))
        data_by_tf_all = {
            "M1": m1_all,
            "M5": _resample_ohlc(m1_all, "5min"),
            "M15": _resample_ohlc(m1_all, "15min"),
            "H1": _resample_ohlc(m1_all, "1h"),
            "H4": _resample_ohlc(m1_all, "4h"),
            "D": _resample_ohlc(m1_all, "1D"),
        }
    else:
        real_adapter = get_adapter(profile)
        try:
            if hasattr(real_adapter, "initialize"):
                real_adapter.initialize()
            if hasattr(real_adapter, "ensure_symbol"):
                real_adapter.ensure_symbol(profile.symbol)

            data_by_tf_all = {
                "M1": _fetch_prefetch_bars(real_adapter, profile.symbol, "M1", args.bars_count),
                "M5": _fetch_prefetch_bars(real_adapter, profile.symbol, "M5", max(5000, args.bars_count // 4)),
                "M15": _fetch_prefetch_bars(real_adapter, profile.symbol, "M15", max(5000, args.bars_count // 10)),
                "H1": _fetch_prefetch_bars(real_adapter, profile.symbol, "H1", 2000),
                "H4": _fetch_prefetch_bars(real_adapter, profile.symbol, "H4", 1000),
                "D": _fetch_prefetch_bars(real_adapter, profile.symbol, "D", 400),
            }
        finally:
            if hasattr(real_adapter, "shutdown"):
                try:
                    real_adapter.shutdown()
                except Exception:
                    pass

    # Trim each timeframe to a bounded pre-window for faster replay while preserving indicator history.
    tf_lookback = {
        "M1": pd.Timedelta(days=7),
        "M5": pd.Timedelta(days=21),
        "M15": pd.Timedelta(days=30),
        "H1": pd.Timedelta(days=90),
        "H4": pd.Timedelta(days=180),
        "D": pd.Timedelta(days=730),
    }
    for tf, df in list(data_by_tf_all.items()):
        if df is None or df.empty:
            continue
        lb = tf_lookback.get(tf, pd.Timedelta(days=30))
        lo = start_ts - lb
        data_by_tf_all[tf] = df[(df["time"] >= lo) & (df["time"] <= end_ts)].copy()

    m1 = data_by_tf_all["M1"]
    m1 = m1[(m1["time"] >= start_ts) & (m1["time"] <= end_ts)].copy()
    if m1.empty:
        raise RuntimeError("No M1 bars in requested range.")

    sizing = load_phase3_sizing_config()
    mock_adapter = MockOrderAdapter(equity=args.equity)
    phase3_state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    tf_frames = {tf: data_by_tf_all[tf].reset_index(drop=True) for tf in data_by_tf_all}
    tf_idx = {tf: 0 for tf in tf_frames}
    tf_time = {tf: (tf_frames[tf]["time"].to_numpy() if "time" in tf_frames[tf].columns else np.array([])) for tf in tf_frames}

    for _, row in m1.iterrows():
        t = pd.Timestamp(row["time"]).tz_convert("UTC")
        bid = float(row["close"])
        ask = float(row["close"]) + args.spread_pips * float(profile.pip_size)
        tick = SimpleNamespace(bid=bid, ask=ask)
        mock_adapter.set_tick(tick)

        # Simulate SL/TP exits against this bar before evaluating new entries.
        _simulate_exits(row, mock_adapter._open_trades, phase3_state, t)
        phase3_state["open_trade_count"] = len(mock_adapter._open_trades)

        sliced = {}
        for tf, d in tf_frames.items():
            if d.empty or tf_time[tf].size == 0:
                sliced[tf] = d
                continue
            i = tf_idx[tf]
            n = len(d)
            while i < n and pd.Timestamp(tf_time[tf][i]) <= t:
                i += 1
            tf_idx[tf] = i
            sliced[tf] = d.iloc[:i]

        result = execute_phase3_integrated_policy_demo_only(
            adapter=mock_adapter,
            profile=profile,
            log_dir=Path("runtime") / profile.profile_name,
            policy=policy,
            context=MarketContext(spread_pips=(ask - bid) / float(profile.pip_size), alignment_score=0),
            data_by_tf=sliced,
            tick=tick,
            mode="ARMED_AUTO_DEMO",
            phase3_state=phase3_state,
            store=mock_adapter,  # provides list_open_trades for London open-risk cap
            sizing_config=sizing if sizing else None,
            is_new_m1=True,
        )
        updates = result.get("phase3_state_updates") or {}
        if updates:
            phase3_state.update(updates)
        # Propagate risk_usd_planned so London's open-risk ledger is accurate.
        _rup = result.get("risk_usd_planned")
        if _rup is not None:
            mock_adapter.set_last_order_risk(float(_rup))

        dec = result.get("decision")
        if getattr(dec, "placed", False):
            # Tag the newly opened mock trade with its session_key so _simulate_exits can
            # decrement the right session bucket when it hits SL or TP1.
            for _sk in list(updates.keys()):
                if isinstance(_sk, str) and (
                    _sk.startswith("session_ny_")
                    or _sk.startswith("session_london_")
                    or _sk.startswith("session_tokyo_")
                ):
                    mock_adapter.set_last_order_session(_sk)
                    break
        strategy_tag = result.get("strategy_tag")
        records.append(
            {
                "bar_time": t.isoformat(),
                "attempted": bool(getattr(dec, "attempted", False)),
                "placed": bool(getattr(dec, "placed", False)),
                "side": getattr(dec, "side", None),
                "reason": getattr(dec, "reason", ""),
                "strategy_tag": strategy_tag,
                "strategy": _strategy_from_tag(strategy_tag),
                "session": _session_from_tag(strategy_tag),
                "entry_price": result.get("entry_price"),
                "sl_price": result.get("sl_price"),
                "tp1_price": result.get("tp1_price"),
                "units": result.get("units"),
            }
        )

    out_dir = Path("research_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(records)
    out_csv = out_dir / f"parity_phase3_{args.start}_{args.end}.csv"
    out_df.to_csv(out_csv, index=False)

    # Self-baseline: save placed-entries as a future comparison baseline.
    # This approach is semantically correct because it uses the same indicator
    # computation path (resampled M1) as the parity replay itself, eliminating
    # indicator drift that occurs when comparing against per-strategy backtest exports.
    if args.save_baseline:
        bl_path = Path(args.save_baseline)
        bl_path.parent.mkdir(parents=True, exist_ok=True)
        bl_cols = ["bar_time", "placed", "side", "strategy_tag", "strategy", "session",
                   "entry_price", "sl_price", "tp1_price"]
        placed_only = out_df[out_df["placed"].fillna(False).astype(bool)].copy()
        placed_only[[c for c in bl_cols if c in placed_only.columns]].to_csv(bl_path, index=False)
        print(f"Self-baseline saved: {len(placed_only)} entries → {bl_path}", file=sys.stderr)

    _placed_mask = out_df["placed"].fillna(False).astype(bool) if "placed" in out_df.columns else pd.Series(False, index=out_df.index)
    summary: dict[str, Any] = {
        "output_csv": str(out_csv),
        "rows": int(len(out_df)),
        "entries": int(_placed_mask.sum()),
        "by_strategy_live": _value_counts(out_df[_placed_mask], "strategy"),
        "by_session_live": _value_counts(out_df[_placed_mask], "session"),
        "effective_config_hash": (sizing.get("_meta", {}) or {}).get("effective_hash"),
        "effective_config_sources": (sizing.get("_meta", {}) or {}).get("source_paths", {}),
    }
    if args.save_baseline:
        summary["self_baseline_saved"] = args.save_baseline

    mismatches_path = out_dir / f"parity_phase3_{args.start}_{args.end}_mismatches.csv"
    compare_ok = True
    compare_fails: list[str] = []

    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            cmp_df = pd.read_csv(compare_path)
            cmp_metrics, mismatches_df = _compare_reports(out_df, cmp_df, float(profile.pip_size))
            summary["compare"] = cmp_metrics
            if not mismatches_df.empty:
                mismatches_df.to_csv(mismatches_path, index=False)
            else:
                pd.DataFrame(columns=["bar_time", "mismatch_type", "strategy_live", "strategy_cmp", "session_live", "session_cmp"]).to_csv(mismatches_path, index=False)
            compare_ok, compare_fails = _acceptance_gate(cmp_metrics)
            summary["acceptance"] = {
                "pass": bool(compare_ok),
                "thresholds": {
                    "entry_bar_match_pct_min": ENTRY_BAR_MATCH_MIN,
                    "side_match_pct_min": SIDE_MATCH_MIN,
                    "sl_diff_p95_max": SL_P95_MAX,
                    "tp1_diff_p95_max": TP1_P95_MAX,
                    "session_tag_match_pct_min": SESSION_MATCH_REQUIRED,
                },
                "fail_reasons": compare_fails,
                "mismatches_csv": str(mismatches_path),
            }
        else:
            summary["compare"] = {"error": f"compare file not found: {compare_path}"}
            pd.DataFrame(columns=["bar_time", "mismatch_type", "strategy_live", "strategy_cmp", "session_live", "session_cmp"]).to_csv(mismatches_path, index=False)
            summary["acceptance"] = {
                "pass": False,
                "fail_reasons": ["compare file not found"],
                "mismatches_csv": str(mismatches_path),
            }
            compare_ok = False
            compare_fails = ["compare file not found"]
    else:
        pd.DataFrame(columns=["bar_time", "mismatch_type", "strategy_live", "strategy_cmp", "session_live", "session_cmp"]).to_csv(mismatches_path, index=False)
        summary["acceptance"] = {
            "pass": None,
            "fail_reasons": [],
            "mismatches_csv": str(mismatches_path),
            "note": "No --compare supplied; acceptance gate not evaluated",
        }

    summary_path = out_dir / f"parity_phase3_{args.start}_{args.end}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.compare and (not compare_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
