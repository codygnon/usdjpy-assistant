from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase3_parity_harness import compare_trace_to_diagnostics, parse_phase3_diagnostics_log


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
