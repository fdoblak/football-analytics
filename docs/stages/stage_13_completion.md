# Stage 13 completion — Target events pipeline

## Gate

`PASS_WITH_FINDINGS — TARGET EVENTS PIPELINE ACTIVE; STAGE 13 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Delivered

| Sub-stage | Deliverable |
|-----------|-------------|
| **13A** | `replay_candidates` contract + baseline; live/replay eligibility; camera_position for supported views only; duplicate suppression helpers |
| **13B** | Period/half-scoped attack-direction resolver (`events/attack_direction.py`; Stage 11 wrapper updated) |
| **13C** | `target_event_ledger` + `event_revisions` append-only ledger merging Stage 10–12 sources |
| **13D** | Coverage-aware target event metrics aggregation for requested product metrics |
| **13E** | Fusion pipeline (`configs/events/events_pipeline.yaml`) |

## Core rules enforced

- Never invent live if replay uncertain
- camera_position only for supported view classes; else unknown
- Conflict / unknown attack direction → directional metrics `not_evaluable`
- Never invent real team names (anonymous team ids only)
- Append-only ledger; no destructive merge; source preservation; lineage
- Revision/revocation append-only; evaluation leakage guard
- Automatic ceiling = `provisional`
- Without reviewed GT → `NOT_EVALUATED_NO_REVIEWED_TARGET_EVENTS_GROUND_TRUTH`

## Registry

Contract count after Stage 13: **45** (`replay_candidates`, `target_event_ledger`, `event_revisions`)

## Validators

| Script | Gate |
|--------|------|
| `scripts/check_events_contracts.py` | `PASS — TARGET EVENTS CONTRACTS ACTIVE` |
| `scripts/check_replay_candidate_baseline.py` | `PASS — REPLAY CANDIDATE BASELINE ACTIVE` |
| `scripts/check_attack_direction_resolver.py` | `PASS — ATTACK DIRECTION RESOLVER ACTIVE` |
| `scripts/check_target_event_ledger.py` | `PASS — TARGET EVENT LEDGER ACTIVE` |
| `scripts/check_target_event_metrics.py` | `PASS — TARGET EVENT METRICS ACTIVE` |
| `scripts/check_events_pipeline.py` | Stage 13 close gate (above) |

## CLI

```text
football-analytics events contracts validate
football-analytics events compute --fixture-smoke --output-dir ...
football-analytics events integrate --fixture-smoke --output-dir ...
football-analytics events pipeline-validate
```

## Runtime roots

- `/home/fdoblak/workspace/events_contract_checks`
- `/home/fdoblak/workspace/replay_candidate_checks`
- `/home/fdoblak/workspace/attack_direction_checks`
- `/home/fdoblak/workspace/target_event_ledger_checks`
- `/home/fdoblak/workspace/target_event_metrics_checks`
- `/home/fdoblak/workspace/events_pipeline_checks`

## Evidence

JSON-only under `artifacts/evidence/stage_13/`.

## Next

Do not start Stage 14 without an explicit user prompt. Do not invent Opta or real-match accuracy claims.
