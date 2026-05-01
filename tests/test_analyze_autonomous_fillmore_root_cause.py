"""Tests for scripts/analyze_autonomous_fillmore_root_cause.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_autonomous_fillmore_root_cause.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("afrc", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["afrc"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


afrc = _load_module()


def _phase3_row(**overrides):
    base = {
        "suggestion_id": overrides.get("suggestion_id", "s1"),
        "created_utc": "2026-04-30T10:00:00+00:00",
        "profile": "newera8",
        "prompt_version": afrc.PHASE3_VERSION,
        "action": "placed",
        "decision": "trade",
        "side": "buy",
        "lots": 1.0,
        "closed_at": "2026-04-30T10:20:00+00:00",
        "pips": 2.0,
        "pnl": 10.0,
        "trade_id": "ai_autonomous:test",
        "trigger_family": "critical_level_reaction",
        "trigger_reason": "support_reclaim:HALF_YEN:160.50",
        "timeframe_alignment": "aligned",
        "trigger_fit": "level_reaction",
        "session": "['London', 'NY']",
        "h1_regime": "bull",
        "zone_memory_read": "fresh_setup",
        "repeat_trade_case": "none",
        "planned_rr_estimate": 1.1,
        "named_catalyst": "reclaimed_support at HALF_YEN 160.50 with micro confirmation",
        "trade_thesis": "Clean aligned support reclaim.",
        "rationale": "AUTONOMOUS clean setup",
        "max_adverse_pips": -0.5,
        "max_favorable_pips": 3.0,
        "exit_strategy": "tp1_be_m5_trail",
        "exit_reason": "hit_breakeven",
    }
    base.update(overrides)
    return base


def _write_live_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"total": len(rows), "limit": len(rows), "offset": 0, "items": rows}))


def _headline_fixture() -> list[dict]:
    rows: list[dict] = []
    idx = 0

    # 43 winners at +2p / +$10 = +86p / +$430.
    for i in range(43):
        rows.append(_phase3_row(
            suggestion_id=f"w{i}",
            trade_id=f"ai_autonomous:w{i}",
            side="buy" if i < 30 else "sell",
            timeframe_alignment="mixed" if i < 30 else "aligned",
            pips=2.0,
            pnl=10.0,
            named_catalyst="reclaimed_support at HALF_YEN 160.50 with micro confirmation",
        ))
        idx += 1

    # 28 losers totaling -138p / -$1680.19. Overall: -52.0p / -$1250.19.
    for i in range(28):
        rows.append(_phase3_row(
            suggestion_id=f"l{i}",
            trade_id=f"ai_autonomous:l{i}",
            side="buy" if i < 6 else "sell",
            lots=8.0 if i < 5 else 1.0,
            timeframe_alignment="mixed" if i < 23 else "aligned",
            pips=-3.0 if i == 27 else -5.0,
            pnl=-60.19 if i == 27 else -60.0,
            trigger_family="momentum_continuation" if i in (3, 4, 5) else "critical_level_reaction",
            named_catalyst="reclaimed_support at HALF_YEN 160.50",
            trade_thesis="Structure-only catalyst without material context.",
            max_adverse_pips=-8.0,
            max_favorable_pips=0.6,
            exit_reason="hit_stop_loss",
        ))
        idx += 1

    # Two placed/open rows.
    for i in range(2):
        row = _phase3_row(suggestion_id=f"open{i}", trade_id=f"ai_autonomous:open{i}")
        row["closed_at"] = None
        row["pips"] = None
        row["pnl"] = None
        rows.append(row)

    # 22 skipped calls.
    for i in range(22):
        row = _phase3_row(suggestion_id=f"skip{i}", trade_id="")
        row["action"] = "skip"
        row["decision"] = "skip"
        row["lots"] = 0.0
        row["closed_at"] = None
        row["pips"] = None
        row["pnl"] = None
        rows.append(row)

    assert len(rows) == 95
    return rows


def test_catalyst_scoring_deterministic_examples():
    assert afrc.score_catalyst("")[0] == 0
    assert afrc.score_catalyst("level reject")[0] == 0
    assert afrc.score_catalyst("reclaimed_support at HALF_YEN 160.50")[0] == 1
    assert afrc.score_catalyst("reclaimed_support at HALF_YEN 160.50 with micro confirmation")[0] == 2
    assert afrc.score_catalyst("BoJ intervention warning plus liquidity sweep after failed prior low")[0] == 3


def test_phase3_filtering_matches_known_headline_totals(tmp_path: Path):
    live = tmp_path / "live.json"
    out = tmp_path / "out"
    _write_live_json(live, _headline_fixture())

    result = afrc.run_analysis(evidence=tmp_path / "missing", timeline=tmp_path / "timeline", prior=tmp_path / "prior", out=out, live_jsons=[live])
    summary = result["phase3_summary"]

    assert summary["calls"] == 95
    assert summary["placed"] == 73
    assert summary["closed"] == 71
    assert summary["win_rate"] == pytest.approx(43 / 71)
    assert summary["net_pips"] == pytest.approx(-52.0)
    assert summary["net_pnl"] == pytest.approx(-1250.19)
    assert summary["placement_rate"] == pytest.approx(73 / 95)


def test_counterfactual_counts_saved_and_missed_pips():
    rows = [
        _phase3_row(suggestion_id="good", pips=4.0, pnl=40.0, timeframe_alignment="mixed",
                    named_catalyst="reclaimed_support at HALF_YEN 160.50 with micro confirmation"),
        _phase3_row(suggestion_id="bad", pips=-7.0, pnl=-70.0, timeframe_alignment="mixed",
                    named_catalyst="reclaimed_support at HALF_YEN 160.50",
                    trade_thesis="", rationale=""),
    ]
    row = afrc.counterfactual_row("mixed_rule", rows, afrc.rule_mixed_score2_green)
    assert row["blocked_trades"] == 1
    assert row["blocked_losers"] == 1
    assert row["blocked_winners"] == 0
    assert row["saved_loss_pips"] == pytest.approx(7.0)
    assert row["missed_winner_pips"] == pytest.approx(0.0)
    assert row["net_delta_pips"] == pytest.approx(7.0)


def test_missing_optional_fields_do_not_crash(tmp_path: Path):
    live = tmp_path / "live.json"
    out = tmp_path / "out"
    rows = [
        {
            "suggestion_id": "minimal",
            "created_utc": "2026-04-30T10:00:00+00:00",
            "prompt_version": afrc.PHASE3_VERSION,
            "action": "placed",
            "decision": "trade",
            "lots": 1,
            "closed_at": "2026-04-30T10:05:00+00:00",
            "pips": -1,
            "pnl": -10,
            "trade_id": "ai_autonomous:minimal",
        }
    ]
    _write_live_json(live, rows)

    result = afrc.run_analysis(evidence=tmp_path / "missing", timeline=tmp_path / "timeline", prior=tmp_path / "prior", out=out, live_jsons=[live])
    assert result["phase3_summary"]["closed"] == 1
    assert (out / "REPORT.md").exists()


def test_outputs_have_stable_required_columns(tmp_path: Path):
    live = tmp_path / "live.json"
    out = tmp_path / "out"
    _write_live_json(live, _headline_fixture())
    afrc.run_analysis(evidence=tmp_path / "missing", timeline=tmp_path / "timeline", prior=tmp_path / "prior", out=out, live_jsons=[live])

    expected_files = [
        "REPORT.md",
        "phase3_loser_dossier.csv",
        "phase3_winner_loser_contrast.csv",
        "catalyst_quality_audit.csv",
        "mixed_alignment_audit.csv",
        "family_side_matrix.csv",
        "loss_asymmetry.csv",
        "rule_counterfactuals.csv",
        "SOLUTION_IDEAS.md",
    ]
    for name in expected_files:
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 0, name

    dossier_header = (out / "phase3_loser_dossier.csv").read_text().splitlines()[0]
    for col in ("catalyst_score", "weakness_signals", "proposed_preventable_rule"):
        assert col in dossier_header

    counter_header = (out / "rule_counterfactuals.csv").read_text().splitlines()[0]
    for col in ("saved_loss_pips", "missed_winner_pips", "net_delta_pips", "new_placement_rate"):
        assert col in counter_header


def test_report_includes_required_headline_facts(tmp_path: Path):
    live = tmp_path / "live.json"
    out = tmp_path / "out"
    _write_live_json(live, _headline_fixture())
    afrc.run_analysis(evidence=tmp_path / "missing", timeline=tmp_path / "timeline", prior=tmp_path / "prior", out=out, live_jsons=[live])

    report = (out / "REPORT.md").read_text()
    assert "WR 60.6%" in report
    assert "-52.0p" in report
    assert "$-1,250.19" in report
    assert "placement rate 76.8%" in report
    assert "Sell side" in report
    assert "Mixed alignment" in report
    assert "Large lots" in report
