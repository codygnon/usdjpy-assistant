#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.admission import AdmissionFilter
from core.regime_backtest_engine.cross_asset_confluence import CrossAssetConfluence
from core.regime_backtest_engine.cross_asset_data import CrossAssetDataLoader
from core.regime_backtest_engine.data import load_market_data
from core.regime_backtest_engine.engine import BacktestEngine
from core.regime_backtest_engine.manifest import evaluate_summary_against_manifest, freeze_manifest
from core.regime_backtest_engine.margin import MarginModel
from core.regime_backtest_engine.models import (
    AdmissionConfig,
    BacktestResult,
    ClosedTrade,
    FixedSpreadConfig,
    InstrumentSpec,
    PortfolioState,
    RunConfig,
    Signal,
    SlippageConfig,
    SpreadConfig,
    closed_trade_to_row,
    portfolio_snapshot_from_state,
)
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView


class ProgressBacktestEngine(BacktestEngine):
    """Same as BacktestEngine.run but prints every 100k bars."""

    def run(self, config: RunConfig) -> BacktestResult:
        missing = [name for name in config.active_families if name not in self.strategies]
        if missing:
            raise ValueError(f"missing strategies for active families: {missing}")

        loaded = load_market_data(config)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = None
        if config.manifest is not None:
            manifest_path = output_dir / "run_manifest.json"
            freeze_manifest(config.manifest, manifest_path)

        margin_model = MarginModel(config.instrument)
        admission = AdmissionFilter(config.admission, margin_model)
        state = PortfolioState(
            balance=float(config.initial_balance),
            equity=float(config.initial_balance),
            unrealized_pnl=0.0,
            margin_used=0.0,
            available_margin=float(config.initial_balance),
        )

        bar_rows: list[dict[str, Any]] = []
        arbitration_rows: list[dict[str, Any]] = []
        n = len(loaded.store)
        t0 = time.perf_counter()

        for idx in range(n):
            if idx > 0 and idx % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"Progress: bar_index={idx} / {n} ({elapsed:.1f}s elapsed)", flush=True)

            current_bar = BarView(loaded.store, idx)
            fill_rejections = self._fill_pending_orders(state, current_bar, margin_model, config)
            for rejection in fill_rejections:
                arbitration_rows.append(
                    {
                        "bar_index": idx,
                        "timestamp": current_bar.timestamp,
                        "candidate_families": [rejection.signal.family],
                        "candidate_directions": [rejection.signal.direction],
                        "outcome": "fill_rejected",
                        "accepted_families": [],
                        "rejected_families": [rejection.signal.family],
                        "rejection_reasons": [rejection.reason],
                    }
                )

            self._mark_to_market(state, current_bar, margin_model)
            self._check_exits(state, current_bar, margin_model, config)
            self._mark_to_market(state, current_bar, margin_model)

            snapshot = portfolio_snapshot_from_state(state)
            history = HistoricalDataView(loaded.store, idx)
            signals: list[Signal] = []
            for family in config.active_families:
                strat = self.strategies[family]
                signals.extend(strat.evaluate_signals(current_bar, history, snapshot))

            decision = admission.decide(
                bar_index=idx,
                timestamp=current_bar.timestamp,
                signals=signals,
                portfolio=snapshot,
                reference_price=float(current_bar.mid_close),
            )
            state.pending_signals = list(decision.accepted)
            if decision.record is not None:
                arbitration_rows.append(
                    {
                        "bar_index": decision.record.bar_index,
                        "timestamp": decision.record.timestamp,
                        "candidate_families": list(decision.record.candidate_families),
                        "candidate_directions": list(decision.record.candidate_directions),
                        "outcome": decision.record.outcome,
                        "accepted_families": list(decision.record.accepted_families),
                        "rejected_families": list(decision.record.rejected_families),
                        "rejection_reasons": list(decision.record.rejection_reasons),
                    }
                )

            bar_rows.append(
                {
                    "timestamp": current_bar.timestamp,
                    "bar_index": idx,
                    "balance": state.balance,
                    "unrealized_pnl": state.unrealized_pnl,
                    "equity": state.equity,
                    "margin_used": state.margin_used,
                    "available_margin": state.available_margin,
                    "open_position_count": len(state.open_positions),
                    "pending_order_count": len(state.pending_signals),
                    "accepted_signal_count": len(decision.accepted),
                    "rejected_signal_count": len(decision.rejected) + len(fill_rejections),
                    **self._per_family_counts(state, config.active_families),
                }
            )

        state.pending_signals = []
        self._mark_to_market(state, BarView(loaded.store, len(loaded.store) - 1), margin_model)

        trade_columns = [field.name for field in fields(ClosedTrade)]
        trade_df = pd.DataFrame([closed_trade_to_row(t) for t in state.closed_trades], columns=trade_columns)
        bar_df = pd.DataFrame(bar_rows)
        arbitration_df = pd.DataFrame(arbitration_rows)
        summary = self._build_summary(config, loaded.contract.synthetic_bid_ask, loaded.frame, trade_df, bar_df, arbitration_df, state)
        if config.manifest is not None:
            summary["manifest_evaluation"] = evaluate_summary_against_manifest(summary, config.manifest)

        config_snapshot_path = output_dir / "run_config.json"
        config_snapshot_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        trade_log_path = output_dir / "trade_log.csv"
        trade_df.to_csv(trade_log_path, index=False)
        shutil.copy2(trade_log_path, output_dir / "trades.csv")

        bar_log_path = output_dir / f"bar_state_log.{ 'parquet' if config.bar_log_format == 'parquet' else 'csv' }"
        if config.bar_log_format == "parquet":
            try:
                bar_df.to_parquet(bar_log_path, index=False)
            except Exception as exc:
                raise RuntimeError("bar_log_format='parquet' requires pyarrow or fastparquet") from exc
        else:
            bar_df.to_csv(bar_log_path, index=False)

        if not bar_df.empty and "equity" in bar_df.columns:
            bar_df[["timestamp", "bar_index", "equity"]].to_csv(output_dir / "equity_curve.csv", index=False)

        arbitration_log_path = None
        if not arbitration_df.empty:
            arbitration_log_path = output_dir / "arbitration_log.csv"
            arbitration_df.to_csv(arbitration_log_path, index=False)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        return BacktestResult(
            summary=summary,
            trade_log_path=trade_log_path,
            bar_log_path=bar_log_path,
            config_snapshot_path=config_snapshot_path,
            manifest_path=manifest_path,
            arbitration_log_path=arbitration_log_path,
            final_portfolio=portfolio_snapshot_from_state(state),
        )


def _entry_hour_utc(ts: Any) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.hour)


def _infer_initial_units(trade_df: pd.DataFrame, trade_id: int) -> int:
    sub = trade_df.loc[trade_df["trade_id"] == trade_id].sort_values("exit_bar")
    if sub.empty:
        return 0
    r0 = sub.iloc[0]
    return int(r0["closed_units"] + r0["remaining_units"])


def _size_tier(units: int) -> str:
    if units >= 275_000:
        return "size_mult_1.5"
    if units >= 150_000:
        return "size_mult_1.0"
    return "size_mult_0.5"


def _extend_summary(summary: dict[str, Any], trade_df: pd.DataFrame, bar_df: pd.DataFrame) -> dict[str, Any]:
    out = dict(summary)
    if trade_df.empty:
        out.update(
            {
                "avg_winner": None,
                "avg_loser": None,
                "largest_winner": None,
                "largest_loser": None,
                "avg_hold_bars": None,
                "longest_win_streak": 0,
                "longest_lose_streak": 0,
                "london_trades": 0,
                "london_win_rate": None,
                "ny_trades": 0,
                "ny_win_rate": None,
                "average_position_size": None,
                "bias_distribution": {},
                "net_pnl": float(out.get("net_pnl_usd") or 0.0),
            }
        )
        return out

    tl = trade_df.groupby(["trade_id", "family"], as_index=False).agg({"pnl_usd": "sum", "bars_held": "max"})
    wins = tl.loc[tl["pnl_usd"] > 0, "pnl_usd"]
    losses = tl.loc[tl["pnl_usd"] < 0, "pnl_usd"]
    out["avg_winner"] = float(wins.mean()) if len(wins) else None
    out["avg_loser"] = float(losses.mean()) if len(losses) else None
    out["largest_winner"] = float(wins.max()) if len(wins) else None
    out["largest_loser"] = float(losses.min()) if len(losses) else None
    out["avg_hold_bars"] = float(tl["bars_held"].mean()) if len(tl) else None

    first_exit = trade_df.groupby("trade_id")["exit_bar"].min()
    tl = tl.assign(_fe=tl["trade_id"].map(first_exit)).sort_values("_fe")
    streak_w = streak_l = cur_w = cur_l = 0
    for pnl in tl["pnl_usd"]:
        if pnl > 0:
            cur_w += 1
            cur_l = 0
            streak_w = max(streak_w, cur_w)
        elif pnl < 0:
            cur_l += 1
            cur_w = 0
            streak_l = max(streak_l, cur_l)
        else:
            cur_w = cur_l = 0
    out["longest_win_streak"] = int(streak_w)
    out["longest_lose_streak"] = int(streak_l)

    first_rows = trade_df.sort_values("exit_bar").groupby("trade_id", as_index=False).first()
    london_mask = first_rows["entry_time"].map(_entry_hour_utc).between(7, 10, inclusive="both")
    ny_mask = first_rows["entry_time"].map(_entry_hour_utc).between(12, 16, inclusive="both")
    london_ids = set(first_rows.loc[london_mask, "trade_id"])
    ny_ids = set(first_rows.loc[ny_mask, "trade_id"])

    def _wr(ids: set[int]) -> float | None:
        if not ids:
            return None
        sub = tl[tl["trade_id"].isin(ids)]
        if sub.empty:
            return None
        w = int((sub["pnl_usd"] > 0).sum())
        return float(w / len(sub) * 100.0)

    out["london_trades"] = int(len(london_ids))
    out["london_win_rate"] = _wr(london_ids)
    out["ny_trades"] = int(len(ny_ids))
    out["ny_win_rate"] = _wr(ny_ids)

    sizes: list[int] = []
    bias_dist: dict[str, int] = {}
    for tid in tl["trade_id"].unique():
        u = _infer_initial_units(trade_df, int(tid))
        sizes.append(u)
        tier = _size_tier(u)
        bias_dist[tier] = bias_dist.get(tier, 0) + 1
    out["average_position_size"] = float(sum(sizes) / len(sizes)) if sizes else None
    out["bias_distribution"] = bias_dist
    out["net_pnl"] = float(out.get("net_pnl_usd") or 0.0)
    return out


def _benchmark_evaluation(ext: dict[str, Any]) -> dict[str, Any]:
    tc = int(ext.get("trade_count") or 0)
    wr = float(ext.get("win_rate") or 0.0)
    pf = float(ext.get("profit_factor") or 0.0)
    dd = float(ext.get("max_drawdown_pct") or 0.0)
    trade_count_pass = tc >= 100
    win_rate_pass = (wr / 100.0) >= 0.55
    profit_factor_pass = pf >= 1.50
    drawdown_pass = dd <= 20.0
    if tc < 100:
        status = "INSUFFICIENT DATA"
        all_pass = False
    else:
        status = "PASS" if all((trade_count_pass, win_rate_pass, profit_factor_pass, drawdown_pass)) else "FAIL"
        all_pass = status == "PASS"
    return {
        "minimum_trade_count": 100,
        "minimum_win_rate": 0.55,
        "minimum_profit_factor": 1.50,
        "maximum_drawdown_pct": 20.0,
        "trade_count_pass": trade_count_pass,
        "win_rate_pass": win_rate_pass,
        "profit_factor_pass": profit_factor_pass,
        "drawdown_pass": drawdown_pass,
        "all_pass": all_pass,
        "status": status,
    }


def main() -> None:
    uj = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    cross = ROOT / "research_out/cross_assets"
    paths = {
        "usdjpy": uj,
        "brent": cross / "BCO_USD_H1_OANDA.csv",
        "eurusd": cross / "EUR_USD_H1_OANDA.csv",
        "gold": cross / "XAU_USD_D_OANDA.csv",
        "silver": cross / "XAG_USD_D_OANDA.csv",
    }
    for label, p in paths.items():
        if not p.is_file():
            print(f"ERROR: missing {label} data file: {p}", file=sys.stderr)
            sys.exit(1)

    try:
        n_bars = sum(1 for _ in open(uj, encoding="utf-8")) - 1
    except OSError as e:
        print(f"ERROR: cannot read USDJPY CSV: {e}", file=sys.stderr)
        sys.exit(1)

    data_loader = CrossAssetDataLoader(
        usdjpy_path=str(paths["usdjpy"]),
        brent_path=str(paths["brent"]),
        eurusd_path=str(paths["eurusd"]),
        gold_path=str(paths["gold"]),
        silver_path=str(paths["silver"]),
    )
    uj_rows = len(data_loader.get_usdjpy_bars())
    if uj_rows != n_bars:
        print(f"Note: loader reports {uj_rows} USDJPY rows; line count was {n_bars}.", flush=True)

    OUTPUT_DIR = ROOT / "research_out/cross_asset_confluence_real"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run_log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_path.write_text("", encoding="utf-8")
    log(f"Starting cross_asset_confluence_real at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"USDJPY bars (loader): {uj_rows}")

    strategy = CrossAssetConfluence(data_loader=data_loader)
    family = strategy.family_name
    cfg = RunConfig(
        hypothesis="cross_asset_confluence_real",
        data_path=paths["usdjpy"],
        output_dir=OUTPUT_DIR,
        mode="standalone",
        active_families=(family,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        # Mid-only USDJPY CSV: engine requires fixed/model spread (not from_data without bid/ask).
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=1,
            max_open_positions_per_family={family: 1},
            max_total_units=500_000,
            max_units_per_family={family: 500_000},
            family_priority=(family,),
        ),
        initial_balance=100_000.0,
        bar_log_format="csv",
    )

    engine = ProgressBacktestEngine({family: strategy})
    t0 = time.perf_counter()
    result = engine.run(cfg)
    log(f"Finished in {time.perf_counter() - t0:.1f}s")

    trade_df = pd.read_csv(result.trade_log_path)
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["benchmark_evaluation"] = _benchmark_evaluation(ext)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")
    log(json.dumps(ext, indent=2, default=str))
    print("summary:", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
