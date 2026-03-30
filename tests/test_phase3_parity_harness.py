from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase3_parity_harness import (
    _extract_trace_additive_rows,
    compare_trace_to_diagnostics,
    compare_trace_to_offline_additive_artifact,
    parse_phase3_diagnostics_log,
)


def test_parse_phase3_diagnostics_log_reads_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "phase3.log"
    log_path.write_text(
        "2026-03-27T12:00:01+00:00\tbar=2025-06-10T14:05:00+00:00\tplaced=1\tsession=ny\tstrategy=v44_ny\treason='entry_ok'\townership_cell=momentum/er_mid/der_neg\n",
        encoding="utf-8",
    )

    rows = parse_phase3_diagnostics_log(log_path)

    assert len(rows) == 1
    assert rows[0]["bar"] == "2025-06-10T14:05:00+00:00"
    assert rows[0]["strategy"] == "v44_ny"


def test_compare_trace_to_diagnostics_detects_reason_drift(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    diag_path = tmp_path / "phase3.log"
    out_path = tmp_path / "diff.json"

    trace_path.write_text(json.dumps({
        "trace": [
            {
                "bar_time_utc": "2025-06-10T14:05:00+00:00",
                "session": "ny",
                "strategy_tag": "phase3:v44_ny:strong@momentum/er_mid/der_neg",
                "placed": True,
                "reason": "v44: entry ok",
                "ownership_cell": "momentum/er_mid/der_neg",
            }
        ]
    }), encoding="utf-8")
    diag_path.write_text(
        "2026-03-27T12:00:01+00:00\tbar=2025-06-10T14:05:00+00:00\tplaced=1\tsession=ny\tstrategy=v44_ny\treason='different'\townership_cell=momentum/er_mid/der_neg\n",
        encoding="utf-8",
    )

    payload = compare_trace_to_diagnostics(trace_path, diag_path, out_path)

    assert payload["diff_count"] == 1
    assert any("reason:" in msg for msg in payload["diffs"][0]["mismatches"])


def test_compare_trace_to_offline_additive_artifact_detects_slice_scale_drift(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    artifact_path = tmp_path / "artifact.json"
    out_path = tmp_path / "offline_diff.json"

    trace_path.write_text(json.dumps({
        "trace": [
            {
                "package_id": "v7_pfdd__followup",
                "attribution": {
                    "package_spec": {
                        "strict_policy": {
                            "allow_internal_overlap": True,
                            "allow_opposite_side_overlap": True,
                        }
                    },
                    "additive_envelope": {
                        "baseline_intents": [],
                        "offensive_intents": [{"slice_id": "T3_ambig_mid_pos_sell", "size_scale": 0.5}],
                        "accepted": [{"intent_source": "offensive", "slice_id": "T3_ambig_mid_pos_sell"}],
                    },
                },
            }
        ]
    }), encoding="utf-8")
    artifact_path.write_text(json.dumps({
        "package_id": "v7_pfdd__followup",
        "base_slice_scales": {"T3_ambig_mid_pos_sell": 0.25},
        "strict_policy": {
            "allow_internal_overlap": True,
            "allow_opposite_side_overlap": True,
        },
    }), encoding="utf-8")

    payload = compare_trace_to_offline_additive_artifact(trace_path, artifact_path, out_path)

    assert payload["matches"] is False
    assert any("slice_scale[T3_ambig_mid_pos_sell]" in row for row in payload["mismatches"])


def test_extract_trace_additive_rows_uses_accepted_entries_only() -> None:
    rows = _extract_trace_additive_rows([
        {
            "attribution": {
                "additive_envelope": {
                    "accepted": [
                        {
                            "entry_time_utc": "2026-03-30T14:00:00+00:00",
                            "intent_source": "offensive",
                            "slice_id": "T3_ambig_mid_pos_sell",
                            "strategy_family": "v14",
                            "side": "sell",
                            "ownership_cell": "ambiguous/er_mid/der_pos",
                            "strategy_tag": "phase3:v14_mean_reversion@ambiguous/er_mid/der_pos",
                        }
                    ]
                }
            }
        }
    ])

    assert rows == [
        {
            "entry_time_utc": "2026-03-30T14:00:00+00:00",
            "source": "offensive",
            "slice_id": "T3_ambig_mid_pos_sell",
            "strategy_family": "v14",
            "side": "sell",
            "ownership_cell": "ambiguous/er_mid/der_pos",
            "strategy_tag": "phase3:v14_mean_reversion@ambiguous/er_mid/der_pos",
        }
    ]
