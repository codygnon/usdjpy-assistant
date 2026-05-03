# Fillmore v2 Schema Hash Baseline

## v2.0.0 / Step 1

- `SNAPSHOT_VERSION`: `v2.snap.1`
- `PROMPT_VERSION`: `v2.prompt.0-stub`
- `SNAPSHOT_SCHEMA_HASH_V2_0_0`: `03b4e69ff188c61a`

This is the reference hash for the Step 1 snapshot field set. If the hash
function changes later, compare against this value before accepting a new
baseline.

## v2.snap.2 / Step 3

- `SNAPSHOT_VERSION`: `v2.snap.2`
- `PROMPT_VERSION`: `v2.prompt.0-stub`
- `SNAPSHOT_SCHEMA_HASH_V2_SNAP_2`: `66e8d0344cae9740`
- `SNAPSHOT_SCHEMA_HASH_CURRENT`: `66e8d0344cae9740`

This is the active reference hash after adding the pre-decision veto inputs.
The Step 1 anchor remains above for schema-evolution audits.

## v2.prompt.1 / Step 6

- `SNAPSHOT_VERSION`: `v2.snap.2`
- `PROMPT_VERSION`: `v2.prompt.1`
- `SNAPSHOT_SCHEMA_HASH_V2_PROMPT_1`: `c1f6863ebcc2c8ca`
- `SNAPSHOT_SCHEMA_HASH_CURRENT`: `c1f6863ebcc2c8ca`

This is the active reference hash after replacing the prompt stub with the
Phase 9 system prompt. The snapshot field set did not change from `v2.snap.2`;
the hash changed because `PROMPT_VERSION` is part of the schema-hash payload.
