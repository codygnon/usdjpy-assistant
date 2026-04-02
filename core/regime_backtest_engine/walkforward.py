from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .engine import BacktestEngine
from .manifest import freeze_manifest, evaluate_summary_against_manifest
from .models import RunConfig, WalkForwardConfig, WalkForwardResult, WalkForwardSegmentResult, WalkForwardWindow
from .strategy import HistoricalDataView, StrategyFamily
from .data import load_market_data


StrategyFactory = Callable[[], StrategyFamily]


def build_walk_forward_windows(
    *,
    total_bars: int,
    in_sample_bars: int,
    out_sample_bars: int,
    step_bars: int | None = None,
    anchored: bool = False,
) -> tuple[WalkForwardWindow, ...]:
    if total_bars <= 0:
        raise ValueError("total_bars must be positive")
    if in_sample_bars <= 0 or out_sample_bars <= 0:
        raise ValueError("in_sample_bars and out_sample_bars must be positive")
    step = out_sample_bars if step_bars is None else step_bars
    if step <= 0:
        raise ValueError("step_bars must be positive")

    windows: list[WalkForwardWindow] = []
    in_start = 0
    in_end = in_sample_bars - 1
    out_start = in_end + 1
    out_end = out_start + out_sample_bars - 1
    segment_idx = 1

    while out_end < total_bars:
        windows.append(
            WalkForwardWindow(
                label=f"wf_{segment_idx:03d}",
                in_sample_start_index=in_start,
                in_sample_end_index=in_end,
                out_sample_start_index=out_start,
                out_sample_end_index=out_end,
            )
        )
        segment_idx += 1
        if anchored:
            in_end += step
        else:
            in_start += step
            in_end += step
        out_start += step
        out_end += step

    if not windows:
        raise ValueError("walk-forward sizing produced zero windows")
    return tuple(windows)


class WalkForwardRunner:
    def __init__(self, strategy_factories: dict[str, StrategyFactory]) -> None:
        self.strategy_factories = strategy_factories

    def run(self, base_config: RunConfig, walk_forward: WalkForwardConfig) -> WalkForwardResult:
        missing = [name for name in base_config.active_families if name not in self.strategy_factories]
        if missing:
            raise ValueError(f"missing strategy factories for active families: {missing}")

        output_dir = Path(walk_forward.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = None
        if base_config.manifest is not None:
            manifest_path = output_dir / "walk_forward_manifest.json"
            freeze_manifest(base_config.manifest, manifest_path)

        segment_results: list[WalkForwardSegmentResult] = []
        oos_trade_frames: list[pd.DataFrame] = []
        oos_bar_frames: list[pd.DataFrame] = []
        rolling_balance = float(base_config.initial_balance)

        for window in walk_forward.windows:
            in_strategies = {name: factory() for name, factory in self.strategy_factories.items() if name in base_config.active_families}
            out_strategies = {name: factory() for name, factory in self.strategy_factories.items() if name in base_config.active_families}
            self._fit_strategies(base_config, window, in_strategies)
            self._fit_strategies(base_config, window, out_strategies)

            segment_dir = output_dir / window.label
            in_cfg = base_config.model_copy(
                update={
                    "output_dir": segment_dir / "in_sample",
                    "start_index": window.in_sample_start_index,
                    "end_index": window.in_sample_end_index,
                    "manifest": None,
                    "initial_balance": float(base_config.initial_balance),
                }
            )
            out_cfg = base_config.model_copy(
                update={
                    "output_dir": segment_dir / "out_of_sample",
                    "start_index": window.out_sample_start_index,
                    "end_index": window.out_sample_end_index,
                    "manifest": None,
                    "initial_balance": float(rolling_balance),
                }
            )

            in_result = BacktestEngine(in_strategies).run(in_cfg)
            out_result = BacktestEngine(out_strategies).run(out_cfg)
            rolling_balance = float(out_result.summary["ending_balance"])

            segment_results.append(
                WalkForwardSegmentResult(
                    label=window.label,
                    in_sample_result=in_result,
                    out_of_sample_result=out_result,
                )
            )
            oos_trade_frames.append(pd.read_csv(out_result.trade_log_path))
            oos_bar_frames.append(self._read_frame(out_result.bar_log_path))

        aggregate_summary = self._aggregate_out_of_sample(base_config, walk_forward, segment_results, oos_trade_frames, oos_bar_frames)
        if base_config.manifest is not None:
            aggregate_summary["manifest_evaluation"] = evaluate_summary_against_manifest(aggregate_summary, base_config.manifest)

        summary_path = output_dir / "walk_forward_summary.json"
        summary_path.write_text(json.dumps(aggregate_summary, indent=2, default=str), encoding="utf-8")

        segments_payload = [
            {
                "label": segment.label,
                "in_sample_summary": segment.in_sample_result.summary,
                "out_of_sample_summary": segment.out_of_sample_result.summary,
                "in_sample_trade_log_path": str(segment.in_sample_result.trade_log_path),
                "out_of_sample_trade_log_path": str(segment.out_of_sample_result.trade_log_path),
            }
            for segment in segment_results
        ]
        segments_path = output_dir / "walk_forward_segments.json"
        segments_path.write_text(json.dumps(segments_payload, indent=2, default=str), encoding="utf-8")

        return WalkForwardResult(
            summary=aggregate_summary,
            summary_path=summary_path,
            manifest_path=manifest_path,
            segments_path=segments_path,
            segment_results=tuple(segment_results),
        )

    def _fit_strategies(
        self,
        base_config: RunConfig,
        window: WalkForwardWindow,
        strategies: dict[str, StrategyFamily],
    ) -> None:
        fit_cfg = base_config.model_copy(
            update={
                "start_index": window.in_sample_start_index,
                "end_index": window.in_sample_end_index,
            }
        )
        loaded = load_market_data(fit_cfg)
        history = HistoricalDataView(loaded.store, len(loaded.store) - 1)
        for strategy in strategies.values():
            fit_fn = getattr(strategy, "fit", None)
            if callable(fit_fn):
                fit_fn(history)

    def _aggregate_out_of_sample(
        self,
        base_config: RunConfig,
        walk_forward: WalkForwardConfig,
        segment_results: list[WalkForwardSegmentResult],
        trade_frames: list[pd.DataFrame],
        bar_frames: list[pd.DataFrame],
    ) -> dict[str, Any]:
        all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        all_bars = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame()

        net_pnl = float(all_trades["pnl_usd"].sum()) if not all_trades.empty else 0.0
        gross_win = float(all_trades.loc[all_trades["pnl_usd"] > 0, "pnl_usd"].sum()) if not all_trades.empty else 0.0
        gross_loss = abs(float(all_trades.loc[all_trades["pnl_usd"] < 0, "pnl_usd"].sum())) if not all_trades.empty else 0.0
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        wins = int((all_trades["pnl_usd"] > 0).sum()) if not all_trades.empty else 0
        total = int(len(all_trades))

        peak = float(base_config.initial_balance)
        max_dd = 0.0
        for eq in all_bars["equity"].tolist() if not all_bars.empty else [base_config.initial_balance]:
            peak = max(peak, float(eq))
            max_dd = max(max_dd, peak - float(eq))
        max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

        by_family = []
        if not all_trades.empty:
            for family, group in all_trades.groupby("family"):
                by_family.append(
                    {
                        "family": str(family),
                        "trade_count": int(len(group)),
                        "net_pnl_usd": float(group["pnl_usd"].sum()),
                    }
                )

        if all_bars.empty:
            raise ValueError("walk-forward produced no out-of-sample bars")

        return {
            "hypothesis": base_config.manifest.hypothesis if base_config.manifest is not None else base_config.hypothesis,
            "mode": "walk_forward",
            "aggregate_scope": "out_of_sample_only",
            "active_families": list(base_config.active_families),
            "window_count": len(walk_forward.windows),
            "processed_bar_count": int(len(all_bars)),
            "processed_start_time": pd.Timestamp(all_bars["timestamp"].iloc[0]).isoformat(),
            "processed_end_time": pd.Timestamp(all_bars["timestamp"].iloc[-1]).isoformat(),
            "initial_balance": float(base_config.initial_balance),
            "ending_balance": float(segment_results[-1].out_of_sample_result.summary["ending_balance"]),
            "ending_equity": float(segment_results[-1].out_of_sample_result.summary["ending_equity"]),
            "net_pnl_usd": net_pnl,
            "profit_factor": profit_factor,
            "trade_count": total,
            "win_rate": float(wins / total * 100.0) if total else None,
            "max_drawdown_usd": float(max_dd),
            "max_drawdown_pct": float(max_dd_pct),
            "max_concurrent_positions": int(all_bars["open_position_count"].max()),
            "by_family": by_family,
            "walk_forward_segments": [window.model_dump() for window in walk_forward.windows],
        }

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
