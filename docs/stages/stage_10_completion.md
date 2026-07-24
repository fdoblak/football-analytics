# Stage 10 completion — Human-ball interaction pipeline

## Gate

`PASS_WITH_FINDINGS — HUMAN BALL INTERACTION PIPELINE ACTIVE; STAGE 10 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Delivered

| Sub-stage | Deliverable |
|-----------|-------------|
| **10A** | Contracts: `human_ball_proximity`, `ball_contact_candidates`, `possession_hypotheses` + request/receipt/quality/evaluation/review JSON |
| **10B** | Proximity + contact-candidate baseline (`configs/interaction/human_ball_proximity_contact_baseline.yaml`) |
| **10C** | Possession/control finite-state baseline (`configs/interaction/possession_control_baseline.yaml`) |
| **10D** | Fusion pipeline (`configs/interaction/human_ball_interaction_pipeline.yaml`) |

## Core rules enforced

- Nearest player ≠ possession owner
- Single-frame proximity ≠ contact
- Contact ≠ controlled possession / completed event
- Missing ball ≠ no possession / ≠ loose ball
- Hard gap / cut / replay terminate (no carry)
- Automatic ceiling = `provisional` (never auto `confirmed`)
- Multi-player ambiguity → `contested`
- Operational metrics ≠ event accuracy claims
- Without reviewed GT → `NOT_EVALUATED_…`

## Validators

| Script | Gate |
|--------|------|
| `scripts/check_human_ball_interaction_contracts.py` | Stage 10A contracts |
| `scripts/check_human_ball_proximity_contact_baseline.py` | `PASS — HUMAN BALL PROXIMITY CONTACT BASELINE ACTIVE` |
| `scripts/check_possession_control_baseline.py` | `PASS — POSSESSION CONTROL BASELINE ACTIVE` |
| `scripts/check_human_ball_interaction_pipeline.py` | Stage 10 close gate (above) |

## CLI

```text
football-analytics interaction contracts validate
football-analytics interaction proximity compute|validate
football-analytics interaction possession compute|validate
football-analytics interaction integrate|pipeline-validate
```

## Runtime roots

- `/home/fdoblak/workspace/human_ball_contract_checks` (10A)
- `/home/fdoblak/workspace/human_ball_proximity_contact_checks` (10B)
- `/home/fdoblak/workspace/possession_control_checks` (10C)
- `/home/fdoblak/workspace/human_ball_interaction_pipeline_checks` (10D)

## Evidence

JSON-only under `artifacts/evidence/stage_10/` (and mapped `stage_10a`–`stage_10d` workspace dirs).

## Next

**Aşama 11A — Pas / Reception / Top İlerletme Sözleşmeleri**

Do not start Stage 11 without an explicit user prompt. Do not invent Opta or real-match accuracy claims.
