# Stage 12 completion — Duels / competitive events pipeline

## Gate

`PASS_WITH_FINDINGS — DUELS EVENTS PIPELINE ACTIVE; STAGE 12 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Delivered

| Sub-stage | Deliverable |
|-----------|-------------|
| **12A** | Contracts: `take_on_attempts`, `ground_duel_candidates`, `aerial_duel_candidates`, `tackle_events`, `recovery_events`, `turnover_events`, `clearance_events` + request/receipt/quality/evaluation/review JSON |
| **12B** | Take-on baseline (`configs/duels/take_on_baseline.yaml`) |
| **12C** | Ground/tackle/recovery/turnover baseline (`configs/duels/ground_duel_baseline.yaml`) |
| **12D** | Aerial/clearance baseline (`configs/duels/aerial_clearance_baseline.yaml`) |
| **12E** | Fusion pipeline (`configs/duels/duels_pipeline.yaml`) |

## Core rules enforced

- Nearby opponent alone ≠ take-on
- Nearest / track switch alone ≠ duel outcome
- Monocular aerial is conservative → candidate/unknown/not_evaluable; no exact 3D height claim
- Long ball alone ≠ clearance
- Automatic ceiling = `provisional`
- `metric_origin: project_generated`, `definition_style: opta_style_metric_definition`
- Without reviewed GT → `NOT_EVALUATED_NO_REVIEWED_DUELS_EVENTS_GROUND_TRUTH`

## Validators

| Script | Gate |
|--------|------|
| `scripts/check_duels_contracts.py` | `PASS — DUELS CONTRACTS ACTIVE` |
| `scripts/check_take_on_baseline.py` | `PASS — TAKE-ON BASELINE ACTIVE` |
| `scripts/check_ground_duel_baseline.py` | `PASS — GROUND DUEL BASELINE ACTIVE` |
| `scripts/check_aerial_clearance_baseline.py` | `PASS — AERIAL CLEARANCE BASELINE ACTIVE` |
| `scripts/check_duels_pipeline.py` | Stage 12 close gate (above) |

## CLI

```text
football-analytics duels contracts validate
football-analytics duels compute --fixture-smoke --output-dir ...
football-analytics duels integrate --fixture-smoke --output-dir ...
football-analytics duels pipeline-validate
```

## Runtime roots

- `/home/fdoblak/workspace/duels_contract_checks` (12A)
- `/home/fdoblak/workspace/take_on_checks` (12B)
- `/home/fdoblak/workspace/ground_duel_checks` (12C)
- `/home/fdoblak/workspace/aerial_clearance_checks` (12D)
- `/home/fdoblak/workspace/duels_pipeline_checks` (12E)

## Evidence

JSON-only under `artifacts/evidence/stage_12/`.

## Next

Do not start Stage 13 without an explicit user prompt. Do not invent Opta or real-match accuracy claims.
