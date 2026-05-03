"""Step 1 acceptance test: paper-trade a fake gate event end-to-end.

Verifies that every priority-list field is captured, stored, and replayable
from logs alone. If we cannot reconstruct what the LLM saw and what the system
did from logs only, the telemetry layer is not done.

Also verifies the Step 1 audit requirements:
  - v1 and v2 rows distinguishable by engine_version alone, no joins
  - Halt logic kicks in after HALT_STRIKE_THRESHOLD consecutive missing-blocking polls
  - Schema hash changes when field set or PROMPT_VERSION changes (silent-drift catch)
  - Sizing function referential transparency placeholder (real test ships in Step 4)
  - Zero cross-imports between fillmore_v2 and runner_score
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when pytest is invoked from the project dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import suggestion_tracker  # noqa: E402
from api.fillmore_v2 import (  # noqa: E402
    ENGINE_VERSION,
    PROMPT_VERSION,
    SNAPSHOT_SCHEMA_HASH_CURRENT,
    SNAPSHOT_SCHEMA_HASH_V2_0_0,
    SNAPSHOT_VERSION,
    persistence,
    snapshot as snap_mod,
    state as v2_state,
    telemetry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(p)
    persistence.init_v2_schema(p)
    return p


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Telemetry: capture functions
# ---------------------------------------------------------------------------

def test_open_lots_by_side_splits_correctly():
    positions = [
        telemetry.OpenPositionRow(side="buy", units=200_000, unrealized_pnl_usd=120.0),
        telemetry.OpenPositionRow(side="buy", units=50_000, unrealized_pnl_usd=-15.0),
        telemetry.OpenPositionRow(side="sell", units=100_000, unrealized_pnl_usd=42.0),
    ]
    assert telemetry.open_lots_by_side(positions) == (2.5, 1.0)
    assert telemetry.unrealized_pnl_by_side(positions) == (105.0, 42.0)


def test_pip_value_per_lot_usdjpy():
    # At 150.00, pip value = (0.01 / 150) * 100_000 ≈ $6.67
    assert telemetry.pip_value_per_lot(150.0) == pytest.approx(6.6667, abs=0.001)


def test_risk_after_fill_usd_basic():
    pv = telemetry.pip_value_per_lot(150.0)
    risk = telemetry.risk_after_fill_usd(proposed_lots=2.0, sl_pips=8.0, pip_value_per_lot_usd=pv)
    # 2 lots * 8 pips * $6.6667 ≈ $106.67
    assert risk == pytest.approx(106.67, abs=0.05)


def test_rolling_pnl_uses_last_n_only():
    trades = [
        telemetry.ClosedTradeRow(closed_at_utc=f"2026-04-01T{h:02d}:00Z", pnl_usd=10.0, pips=5.0, lots=1.0)
        for h in range(25)
    ]
    # Add a recent loser
    trades.append(
        telemetry.ClosedTradeRow(closed_at_utc="2026-05-01T00:00Z", pnl_usd=-100.0, pips=-20.0, lots=2.0)
    )
    pnl, lw = telemetry.rolling_pnl(trades, window=20)
    # last 20 = 19 winners (10/5/1) + 1 loser (-100/-20/2)
    assert pnl == pytest.approx(19 * 10.0 - 100.0, abs=0.01)
    assert lw == pytest.approx(19 * 5.0 - 40.0, abs=0.01)


def test_volatility_regime_thresholds():
    assert telemetry.classify_volatility_regime(5.1) == "normal"
    assert telemetry.classify_volatility_regime(10.0) == "elevated"
    assert telemetry.classify_volatility_regime(None) == "unknown"


def test_build_level_packet_buy_normalizes_distance_sign():
    pkt = telemetry.build_level_packet(
        proposed_side="buy",
        nearest_support={"price": 149.50, "quality_score": 78, "structural_origin": "h1_swing_low"},
        nearest_resistance={"price": 150.30, "quality_score": 60, "structural_origin": "pdh"},
        tick_mid=150.00,
        profit_path_blocker_pips=30.0,
    )
    assert pkt is not None
    assert pkt.side == "buy_support"
    assert pkt.level_quality_score == 78
    # Buy: support at 149.50, mid 150.00 → distance is -50p (level is below mid)
    assert pkt.distance_pips == pytest.approx(-50.0, abs=0.1)


def test_build_level_packet_sell_uses_resistance():
    pkt = telemetry.build_level_packet(
        proposed_side="sell",
        nearest_support={"price": 149.50, "quality_score": 78},
        nearest_resistance={"price": 150.30, "quality_score": 86},
        tick_mid=150.00,
    )
    assert pkt is not None
    assert pkt.side == "sell_resistance"
    assert pkt.level_quality_score == 86


def test_build_level_packet_returns_none_when_side_level_missing():
    pkt = telemetry.build_level_packet(
        proposed_side="buy",
        nearest_support=None,
        nearest_resistance={"price": 150.30, "quality_score": 86},
        tick_mid=150.00,
    )
    assert pkt is None


# ---------------------------------------------------------------------------
# Snapshot identity, versioning, schema-hash drift detection
# ---------------------------------------------------------------------------

def test_snapshot_default_versions_and_hash():
    s = snap_mod.Snapshot(
        snapshot_id=snap_mod.new_snapshot_id(),
        created_utc=snap_mod.now_utc_iso(),
    )
    assert s.engine_version == ENGINE_VERSION == "v2"
    assert s.snapshot_version == SNAPSHOT_VERSION
    assert s.prompt_version == PROMPT_VERSION
    # Hash deterministic across two captures with no changes
    s2 = snap_mod.Snapshot(
        snapshot_id="other",
        created_utc=snap_mod.now_utc_iso(),
    )
    assert s.snapshot_schema_hash == s2.snapshot_schema_hash
    assert len(s.snapshot_schema_hash) == 16


def test_schema_hash_matches_current_baseline():
    """Baseline is pinned durably, not only computed at runtime.

    On schema bumps: add a new SNAPSHOT_SCHEMA_HASH_V2_SNAP_N constant in
    api.fillmore_v2.__init__, point SNAPSHOT_SCHEMA_HASH_CURRENT at it, and
    leave the older constant as a historical anchor. This test then asserts
    the new current hash; old constants stay grep-able for evolution audits.
    """
    assert snap_mod.compute_schema_hash() == SNAPSHOT_SCHEMA_HASH_CURRENT


def test_historical_v2_0_0_hash_constant_preserved():
    """The original Step 1 hash must remain in source as an immutable anchor."""
    assert SNAPSHOT_SCHEMA_HASH_V2_0_0 == "03b4e69ff188c61a"


def test_schema_hash_changes_when_prompt_version_changes(monkeypatch):
    """Silent-drift catch: bumping prompt without bumping schema must change hash."""
    h1 = snap_mod.compute_schema_hash()
    monkeypatch.setattr(snap_mod, "PROMPT_VERSION", "v2.prompt.99-test")
    h2 = snap_mod.compute_schema_hash()
    assert h1 != h2


# ---------------------------------------------------------------------------
# Blocking-field halt logic
# ---------------------------------------------------------------------------

def test_check_blocking_fields_flags_missing():
    s = snap_mod.Snapshot(
        snapshot_id="x",
        created_utc=snap_mod.now_utc_iso(),
        # Intentionally leave out open_lots_buy, risk_after_fill_usd, pip_value, rendered_prompt
    )
    missing = snap_mod.check_blocking_fields(s)
    assert "open_lots_buy" in missing
    assert "risk_after_fill_usd" in missing
    assert "pip_value_per_lot" in missing
    assert "rendered_prompt" in missing
    # snapshot_version is set by default → not missing
    assert "snapshot_version" not in missing


def test_check_blocking_fields_clear_when_all_present():
    s = _full_snapshot()
    assert snap_mod.check_blocking_fields(s) == []


def test_clr_required_fields_demanded_when_flagged():
    s = _full_snapshot()
    s.level_packet = None
    missing = snap_mod.check_blocking_fields(s, require_clr=True)
    assert "level_packet" in missing


def test_halt_strikes_after_three_consecutive(profile_dir: Path):
    snap_mod.reset_blocking_strikes(profile_dir)
    halt1, _ = snap_mod.register_blocking_result(profile_dir, ["risk_after_fill_usd"])
    halt2, _ = snap_mod.register_blocking_result(profile_dir, ["risk_after_fill_usd"])
    halt3, reason = snap_mod.register_blocking_result(profile_dir, ["risk_after_fill_usd"])
    assert (halt1, halt2, halt3) == (False, False, True)
    assert "risk_after_fill_usd" in (reason or "")
    is_halt, _ = snap_mod.is_halted(profile_dir)
    assert is_halt is True


def test_halt_resets_when_blocking_clears(profile_dir: Path):
    snap_mod.reset_blocking_strikes(profile_dir)
    snap_mod.register_blocking_result(profile_dir, ["pip_value_per_lot"])
    snap_mod.register_blocking_result(profile_dir, ["pip_value_per_lot"])
    halt, _ = snap_mod.register_blocking_result(profile_dir, [])
    assert halt is False
    is_halt, _ = snap_mod.is_halted(profile_dir)
    assert is_halt is False


# ---------------------------------------------------------------------------
# End-to-end: paper-trade a fake gate event, replay from logs alone
# ---------------------------------------------------------------------------

def _full_snapshot() -> snap_mod.Snapshot:
    s = snap_mod.Snapshot(
        snapshot_id=snap_mod.new_snapshot_id(),
        created_utc=snap_mod.now_utc_iso(),
    )
    s.tick_mid = 150.00
    s.tick_bid = 149.99
    s.tick_ask = 150.01
    s.spread_pips = 1.0
    s.account_equity = 100_000.0
    s.open_lots_buy = 1.5
    s.open_lots_sell = 0.0
    s.unrealized_pnl_buy = 42.0
    s.unrealized_pnl_sell = 0.0
    s.pip_value_per_lot = telemetry.pip_value_per_lot(150.0)
    s.risk_after_fill_usd = telemetry.risk_after_fill_usd(
        proposed_lots=2.0, sl_pips=8.0, pip_value_per_lot_usd=s.pip_value_per_lot
    )
    s.rolling_20_trade_pnl = -25.0
    s.rolling_20_lot_weighted_pnl = -10.0
    s.level_packet = telemetry.build_level_packet(
        proposed_side="buy",
        nearest_support={"price": 149.50, "quality_score": 78, "structural_origin": "h1_swing_low"},
        nearest_resistance=None,
        tick_mid=150.00,
        profit_path_blocker_pips=30.0,
    )
    s.level_age_metadata = telemetry.build_level_age_metadata(
        level_age_minutes=240.0, touch_count=3, broken_then_reclaimed=False
    )
    s.all_gate_candidates = [
        snap_mod.GateCandidate(gate_id="critical_level_reaction", score=0.82, eligible=True),
        snap_mod.GateCandidate(gate_id="momentum_continuation", score=0.41, eligible=False, veto_reason="below_threshold"),
    ]
    s.selected_gate_id = "critical_level_reaction"
    s.rendered_prompt = "SYSTEM: ...\nUSER: pre-cleared CLR buy packet at 149.50, score 78."
    s.rendered_context_json = json.dumps({"setup": "buy_clr", "level_score": 78})
    s.volatility_regime = "normal"
    s.proposed_side = "buy"
    s.sl_pips = 8.0
    s.tp_pips = 16.0
    s.timeframe_alignment = "mixed"
    s.macro_bias = "bullish"
    s.catalyst_category = "material"
    s.active_sessions = ["london", "ny"]
    s.session_overlap = "london_ny"
    s.path_time_mae_mfe = snap_mod.PathTimeMaeMfe()
    return s


def test_end_to_end_fake_gate_event_replayable_from_logs(db_path: Path):
    """Paper-trade a fake gate event end-to-end. Replay from logs alone."""
    s = _full_snapshot()
    # Step: telemetry layer says we're clear to place
    assert snap_mod.check_blocking_fields(s, require_clr=True) == []

    suggestion_id = "test-v2-suggestion-001"
    persistence.insert_v2_row(
        db_path,
        suggestion_id=suggestion_id,
        profile="test_profile",
        model="gpt-5.4-mini",
        snapshot=s,
        side="buy",
        lots=2.0,
        limit_price=150.00,
        decision="place",
        rationale="Pre-cleared buy-CLR packet, level score 78, 30p path room.",
        sl=149.92,
        tp=150.16,
        trigger_family="critical_level_reaction",
        deterministic_lots=2.0,
    )

    # Read it back. Every blocking telemetry field must round-trip.
    row = persistence.fetch_v2_row(db_path, suggestion_id)
    assert row is not None
    assert row["engine_version"] == "v2"
    assert row["snapshot_version"] == SNAPSHOT_VERSION
    assert row["snapshot_schema_hash"] == s.snapshot_schema_hash
    assert row["prompt_version"] == PROMPT_VERSION
    assert row["rendered_prompt"] == s.rendered_prompt
    assert row["rendered_context_json"] == s.rendered_context_json
    assert row["open_lots_buy"] == 1.5
    assert row["open_lots_sell"] == 0.0
    assert row["unrealized_pnl_buy"] == 42.0
    assert row["pip_value_per_lot"] == s.pip_value_per_lot
    assert row["risk_after_fill_usd"] == s.risk_after_fill_usd
    assert row["rolling_20_trade_pnl"] == -25.0
    assert row["rolling_20_lot_weighted_pnl"] == -10.0
    assert row["volatility_regime"] == "normal"
    assert row["timeframe_alignment"] == "mixed"
    assert row["macro_bias"] == "bullish"
    assert row["catalyst_category"] == "material"
    assert json.loads(row["active_sessions_json"]) == ["london", "ny"]
    assert row["session_overlap"] == "london_ny"
    assert row["deterministic_lots"] == 2.0

    # Level packet round-trips with side normalization preserved
    pkt = json.loads(row["level_packet_json"])
    assert pkt["side"] == "buy_support"
    assert pkt["level_quality_score"] == 78
    assert pkt["distance_pips"] == -50.0

    # Level age metadata
    age = json.loads(row["level_age_metadata_json"])
    assert age["touch_count"] == 3
    assert age["broken_then_reclaimed"] is False

    # All gate candidates preserved (not just selected)
    cands = json.loads(row["gate_candidates_json"])
    assert len(cands) == 2
    cand_ids = {c["gate_id"] for c in cands}
    assert {"critical_level_reaction", "momentum_continuation"} == cand_ids


def test_v1_and_v2_rows_distinguishable_by_engine_version_alone(db_path: Path):
    """No joins required to separate v1/v2 history."""
    # Simulate a v1 row: insert directly without engine_version (v1 doesn't write it)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """INSERT INTO ai_suggestions (suggestion_id, created_utc, profile, model, side,
               limit_price, lots, entry_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("v1-row-1", "2026-04-15T00:00Z", "p", "gpt-5.4-mini", "buy", 150.0, 1.0, "ai_autonomous"),
        )
        conn.commit()
    # And a v2 row via the v2 path
    s = _full_snapshot()
    persistence.insert_v2_row(
        db_path,
        suggestion_id="v2-row-1",
        profile="p",
        model="gpt-5.4-mini",
        snapshot=s,
        side="buy",
        lots=2.0,
        limit_price=150.00,
        decision="place",
    )
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        v2_only = conn.execute(
            "SELECT suggestion_id FROM ai_suggestions WHERE engine_version = 'v2'"
        ).fetchall()
        v1_only = conn.execute(
            "SELECT suggestion_id FROM ai_suggestions WHERE engine_version IS NULL"
        ).fetchall()
    assert {r["suggestion_id"] for r in v2_only} == {"v2-row-1"}
    assert {r["suggestion_id"] for r in v1_only} == {"v1-row-1"}


def test_state_file_is_separate_from_v1(profile_dir: Path):
    """v2 state lives in runtime_state_fillmore_v2.json, not runtime_state.json."""
    v2_state.update_state(profile_dir, foo=1, bar="x")
    assert (profile_dir / "runtime_state_fillmore_v2.json").exists()
    assert not (profile_dir / "runtime_state.json").exists()
    assert v2_state.load_state(profile_dir) == {"foo": 1, "bar": "x"}


# ---------------------------------------------------------------------------
# Import-graph audit (Step 1 audit requirement #5)
# ---------------------------------------------------------------------------

def test_no_runner_score_imports_in_fillmore_v2():
    """fillmore_v2 must not import core.runner_score (either direction)."""
    import api.fillmore_v2 as v2_pkg
    pkg_dir = Path(v2_pkg.__file__).parent
    offenders: list[str] = []
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text()
        if "runner_score" in text:
            offenders.append(str(py))
    assert offenders == [], (
        f"fillmore_v2 must not reference runner_score; found in: {offenders}"
    )


def test_runner_score_does_not_import_fillmore_v2():
    text = (REPO_ROOT / "core" / "runner_score.py").read_text()
    assert "fillmore_v2" not in text


def test_persistence_module_init_is_idempotent(db_path: Path):
    """Re-running init_v2_schema on an already-migrated DB is a no-op."""
    persistence.init_v2_schema(db_path)
    persistence.init_v2_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_suggestions)").fetchall()}
    for col, _ in persistence.V2_COLUMNS:
        assert col in cols, f"v2 column missing after migration: {col}"


def test_init_v2_schema_fails_loudly_without_base_table(tmp_path: Path):
    """Defensive: if suggestion_tracker.init_db wasn't called, raise — don't silently create."""
    p = tmp_path / "empty.sqlite"
    with pytest.raises(RuntimeError, match="ai_suggestions table missing"):
        persistence.init_v2_schema(p)
