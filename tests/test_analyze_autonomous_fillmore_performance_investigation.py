"""Tests for scripts/analyze_autonomous_fillmore_performance_investigation.py.

Uses small fixture CSVs in tmp_path so the analysis logic can be exercised
deterministically without depending on a live evidence pack.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_autonomous_fillmore_performance_investigation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("afpi", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["afpi"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


afpi = _load_module()


def _row(**overrides):
    base = {
        "profile": "newera8",
        "suggestion_id": "s1",
        "trade_id": "t1",
        "created_utc": "2026-04-20T10:00:00+00:00",
        "closed_at": "2026-04-20T10:30:00+00:00",
        "minutes_open": "30",
        "action": "placed",
        "decision": "trade",
        "side": "buy",
        "lots": "2.0",
        "win_loss": "win",
        "pips": "5.0",
        "pnl_usd": "100.0",
        "max_adverse_pips": "-2.0",
        "max_favorable_pips": "+7.0",
        "trigger_family": "critical_level_reaction",
        "trigger_reason": "support_reclaim:HALF_YEN:159.50",
        "conviction_rung": "B",
        "zone_memory_read": "fresh_setup",
        "repeat_trade_case": "none",
        "timeframe_alignment": "aligned",
        "trigger_fit": "level_reaction",
        "planned_rr_estimate": "1.4",
        "exit_strategy": "tp1_be_m5_trail",
        "prompt_version": "vTest",
        "session": "['London', 'NY']",
        "vol_label": "Normal",
        "h1_regime": "bull",
        "m5_regime": "bull",
        "m1_regime": "bull",
        "ledger_exit_reason": "hit_take_profit",
        "why_trade_despite_weakness": "",
    }
    base.update(overrides)
    return base


# --- cohort labeling determinism --------------------------------------------


def test_high_quality_cohort_labels_deterministic():
    r = _row(pips="5.0", pnl_usd="100.0", max_adverse_pips="-2.0", max_favorable_pips="+7.0")
    labels = afpi.label_cohorts(r)
    assert "high_quality" in labels
    assert "poor_quality" not in labels
    # Idempotent
    assert afpi.label_cohorts(r) == labels


def test_high_quality_requires_clean_path():
    # pips ok and pnl ok, but MAE -5 and MFE +5 -> not clean enough
    r = _row(pips="4.5", pnl_usd="50.0", max_adverse_pips="-5.0", max_favorable_pips="+5.0")
    assert "high_quality" not in afpi.label_cohorts(r)


def test_poor_quality_by_pips_or_mae_or_sl():
    assert "poor_quality" in afpi.label_cohorts(
        _row(pips="-7.0", pnl_usd="-150.0", max_adverse_pips="-7.0", max_favorable_pips="+0.5",
             win_loss="loss", ledger_exit_reason="hit_breakeven")
    )
    assert "poor_quality" in afpi.label_cohorts(
        _row(pips="-3.0", pnl_usd="-50.0", max_adverse_pips="-9.0", max_favorable_pips="+1.0",
             win_loss="loss", ledger_exit_reason="hit_breakeven")
    )
    assert "poor_quality" in afpi.label_cohorts(
        _row(pips="-1.0", pnl_usd="-20.0", max_adverse_pips="-3.0", max_favorable_pips="+1.0",
             win_loss="loss", ledger_exit_reason="hit_stop_loss")
    )


def test_large_usd_loss_independent_of_pips():
    r = _row(pips="-3.0", pnl_usd="-200.0", win_loss="loss",
             max_adverse_pips="-3.0", max_favorable_pips="+1.0",
             ledger_exit_reason="user_closed_early")
    labels = afpi.label_cohorts(r)
    assert "large_usd_loss" in labels
    # pips/MAE/SL didn't trigger poor_quality
    assert "poor_quality" not in labels


def test_high_conviction_loser():
    r = _row(pips="-3.0", pnl_usd="-50.0", win_loss="loss", conviction_rung="A",
             max_adverse_pips="-3.0", max_favorable_pips="+1.0",
             ledger_exit_reason="user_closed_early")
    assert "high_conviction_loser" in afpi.label_cohorts(r)


# --- weakness traded anyway -------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("zone_memory_read", "failing_zone"),
        ("zone_memory_read", "unresolved_chop"),
        ("repeat_trade_case", "blind_retry"),
        ("planned_rr_estimate", "0.8"),
        ("timeframe_alignment", "mixed"),
        ("timeframe_alignment", "countertrend"),
        ("why_trade_despite_weakness", "this is a probe even though the zone failed"),
    ],
)
def test_weakness_traded_anyway_detected(field, value):
    r = _row(**{field: value})
    assert "weakness_traded_anyway" in afpi.label_cohorts(r)


def test_weakness_not_flagged_when_skipped():
    r = _row(action="", decision="skip", lots="0", zone_memory_read="failing_zone")
    labels = afpi.label_cohorts(r)
    assert "weakness_traded_anyway" not in labels
    assert "skip" in labels


def test_weakness_clean_setup_not_flagged():
    assert "weakness_traded_anyway" not in afpi.label_cohorts(_row())


# --- pips-first, USD-second summary -----------------------------------------


def test_cohort_summary_uses_pips_and_mae_first():
    rows = [
        _row(pips="5.0", pnl_usd="50.0", max_adverse_pips="-2.0", max_favorable_pips="+7.0"),
        _row(pips="6.0", pnl_usd="60.0", max_adverse_pips="-1.0", max_favorable_pips="+8.0"),
    ]
    s = afpi.cohort_summary(rows)
    assert s["count"] == 2
    assert s["closed_count"] == 2
    assert s["win_rate"] == 1.0
    assert s["avg_pips"] == pytest.approx(5.5)
    assert s["net_pips"] == pytest.approx(11.0)
    assert s["avg_mae_pips"] == pytest.approx(-1.5)
    assert s["avg_mfe_pips"] == pytest.approx(7.5)


# --- gate/prompt mismatch ---------------------------------------------------


def test_gate_prompt_mismatch_only_flags_placed_with_weakness():
    rows = [
        _row(),  # clean placed -> not a mismatch
        _row(zone_memory_read="failing_zone", repeat_trade_case="blind_retry"),
        _row(action="", decision="skip", lots="0", zone_memory_read="failing_zone"),  # skip, ignored
        _row(trigger_family="", zone_memory_read="failing_zone"),  # no gate family, ignored
    ]
    out = afpi.gate_prompt_mismatch_rows(rows)
    assert len(out) == 1
    assert "zone=failing_zone" in out[0]["weakness_signals"]
    assert "repeat=blind_retry" in out[0]["weakness_signals"]


# --- end-to-end with missing optional fields --------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_end_to_end_with_missing_optional_fields(tmp_path):
    evidence = tmp_path / "ev"
    evidence.mkdir()
    out = tmp_path / "out"

    rows = [
        _row(suggestion_id="hq1", pips="6.0", pnl_usd="120", max_adverse_pips="-1", max_favorable_pips="+8"),
        _row(
            suggestion_id="pq1",
            pips="-8",
            pnl_usd="-160",
            win_loss="loss",
            max_adverse_pips="-9",
            max_favorable_pips="+0.5",
            ledger_exit_reason="hit_stop_loss",
            zone_memory_read="failing_zone",
            repeat_trade_case="blind_retry",
            timeframe_alignment="mixed",
            planned_rr_estimate="0.6",
            conviction_rung="C",
        ),
        # Skip row, mostly empty
        {
            "profile": "newera8",
            "suggestion_id": "skip1",
            "action": "",
            "decision": "skip",
            "lots": "0",
        },
    ]

    _write_csv(evidence / "combined_autonomous_joined.csv", rows)
    _write_csv(evidence / "combined_autonomous_suggestions_flat.csv", rows)

    rc = afpi.main(["--evidence", str(evidence), "--out", str(out)])
    assert rc == 0

    expected = [
        "REPORT.md",
        "trade_cohorts.csv",
        "common_denominators.csv",
        "gate_prompt_mismatch.csv",
        "patch_recommendations.md",
    ]
    for name in expected:
        p = out / name
        assert p.exists(), f"missing output: {name}"
        assert p.stat().st_size > 0

    # Sanity-check report sections
    report = (out / "REPORT.md").read_text()
    assert "high-quality" in report.lower()
    assert "poor-quality" in report.lower()
    assert "out of scope" in report.lower()
    assert "daily-loss" in report.lower()

    # Patch recommendations are concrete (mention specific files)
    patches = (out / "patch_recommendations.md").read_text()
    assert "api/autonomous_fillmore.py" in patches
    assert "api/autonomous_performance.py" in patches
    assert "AUTONOMOUS_PROMPT_VERSION" in patches
