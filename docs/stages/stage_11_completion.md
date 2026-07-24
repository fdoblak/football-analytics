# Stage 11 completion — Passing / reception / progression pipeline

## Gate

`PASS_WITH_FINDINGS — PASSING PIPELINE ACTIVE; STAGE 11 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Delivered

| Sub-stage | Deliverable |
|-----------|-------------|
| **11A** | Contracts: `pass_candidates`, `reception_candidates`, `pass_outcomes`, `ball_progression_segments`, `target_ball_touches` + request/receipt/quality/evaluation/attack_direction/review JSON |
| **11B** | Pass/reception baseline (`configs/passing/pass_reception_baseline.yaml`) |
| **11C** | Target metrics + attack-direction stub (`configs/passing/passing_metrics_baseline.yaml`) |
| **11D** | Fusion pipeline (`configs/passing/passing_pipeline.yaml`) |

## Core rules enforced

- Owner change alone ≠ completed pass
- Cut / replay / hard gap → no pass
- Attack direction unknown → directional metrics `not_evaluable`
- Penalty presence ≠ box touch (requires possession/contact + pitch mapping + playable)
- Automatic ceiling = `provisional`
- `metric_origin: project_generated`, `definition_style: opta_style_metric_definition`
- Without reviewed GT → `NOT_EVALUATED_NO_REVIEWED_PASSING_GROUND_TRUTH`

## Validators

| Script | Gate |
|--------|------|
| `scripts/check_passing_contracts.py` | `PASS — PASSING CONTRACTS ACTIVE` |
| `scripts/check_pass_reception_baseline.py` | `PASS — PASS RECEPTION BASELINE ACTIVE` |
| `scripts/check_passing_metrics_baseline.py` | `PASS — PASSING METRICS BASELINE ACTIVE` |
| `scripts/check_passing_pipeline.py` | Stage 11 close gate (above) |

## CLI

```text
football-analytics passing contracts validate
football-analytics passing compute --fixture-smoke --output-dir ...
football-analytics passing integrate --fixture-smoke --output-dir ...
football-analytics passing pipeline-validate
```

## Runtime roots

- `/home/fdoblak/workspace/passing_contract_checks` (11A)
- `/home/fdoblak/workspace/passing_reception_checks` (11B)
- `/home/fdoblak/workspace/passing_metrics_checks` (11C)
- `/home/fdoblak/workspace/passing_pipeline_checks` (11D)

## Evidence

JSON-only under `artifacts/evidence/stage_11/`.

## Next

Do not start Stage 12 without an explicit user prompt. Do not invent Opta or real-match accuracy claims.
