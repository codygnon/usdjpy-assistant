"""Auto Fillmore v2 — deterministic shell with a constrained LLM inside it.

Built per `research_out/autonomous_fillmore_forensic_20260501/PHASE9_OVERHAUL_BLUEPRINT.md`.
Lives in parallel to api/autonomous_fillmore.py (v1). Process-level engine flag
selects which loop runs; no mid-session swaps. v2 reads/writes its own runtime
state file and adds columns to the shared ai_suggestions table tagged with
`engine_version='v2'` so v1 and v2 rows are distinguishable without joins.
"""
from __future__ import annotations

ENGINE_VERSION = "v2"

# Bump SNAPSHOT_VERSION on any change to the captured field set. Every v2
# suggestion row records the version active at capture time so historical
# replays can use the snapshot schema that was in force.
SNAPSHOT_VERSION = "v2.snap.2"

# Pinned schema hashes per snapshot version. The CURRENT pin is asserted by
# tests so non-intentional changes to the field set or the hash algorithm
# cannot silently redefine the baseline. Historical pins are kept as anchors
# so a schema-evolution audit can trace the chain of intentional bumps.
SNAPSHOT_SCHEMA_HASH_V2_0_0 = "03b4e69ff188c61a"  # Step 1 baseline (v2.snap.1, prompt stub)
SNAPSHOT_SCHEMA_HASH_V2_SNAP_2 = "66e8d0344cae9740"  # Step 3: + pre-decision veto inputs (prompt stub)
SNAPSHOT_SCHEMA_HASH_V2_PROMPT_1 = "c1f6863ebcc2c8ca"  # Step 6: PROMPT_VERSION bump to v2.prompt.1
SNAPSHOT_SCHEMA_HASH_CURRENT = SNAPSHOT_SCHEMA_HASH_V2_PROMPT_1

# Bump PROMPT_VERSION on any change to the rendered system or user prompt.
# Pinned at v2.prompt.1 by Step 6 — see api/fillmore_v2/system_prompt.py.
PROMPT_VERSION = "v2.prompt.1"

__all__ = [
    "ENGINE_VERSION",
    "SNAPSHOT_VERSION",
    "SNAPSHOT_SCHEMA_HASH_V2_0_0",
    "SNAPSHOT_SCHEMA_HASH_V2_SNAP_2",
    "SNAPSHOT_SCHEMA_HASH_V2_PROMPT_1",
    "SNAPSHOT_SCHEMA_HASH_CURRENT",
    "PROMPT_VERSION",
]
