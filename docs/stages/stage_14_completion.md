# Stage 14 completion — Single-player pipeline and reporting

## Gate

`PASS_WITH_FINDINGS — SINGLE PLAYER PIPELINE ACTIVE; STAGE 14 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Delivered

| Sub-stage | Deliverable |
|-----------|-------------|
| **14A** | `football_analytics.orchestration` plan/run/resume with stage chain + receipts |
| **14B** | Unified review hub prepare/apply (CAS append-only, audit, revoke, stale) |
| **14C** | Canonical `single_player_report` builder (JSON schema; no team summary) |
| **14D** | `visualization.report_renderer` consolidated PNG (synthetic test + cleanup) |
| **14E** | CLI `pipeline|review|report` + `scripts/check_single_player_pipeline.py` |

## Core rules enforced

- Fingerprint checks + deterministic cache keys
- Resume/restart + no-overwrite + failure isolation + partial status
- Stale artifact / stale decision rejection
- Cancellation receipt
- Bounded-memory policy (no unbounded frame lists in orchestrator)
- Cleanup stage-owned temp only; never mutate user video
- No confirmed without review when required
- No team summary in report
- Stage 16 reserved finals documented only — synthetic renders must not be committed as customer finals

## Registry

Arrow contract count unchanged: **45** (Stage 14 prefers JSON schemas for orchestration/review/report).

## Validators

| Script | Gate |
|--------|------|
| `scripts/check_single_player_pipeline.py` | Stage 14 close gate (above) |

## CLI

```text
football-analytics pipeline plan|run|resume|validate
football-analytics review prepare|apply
football-analytics report data|render
```

## Reserved Stage 16 final visual paths (do not treat synthetic as final)

- `/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png`
- `artifacts/final/single_player_analysis_summary.png`

## Runtime roots

- `/home/fdoblak/workspace/single_player_pipeline_checks`

## Evidence

JSON-only under `artifacts/evidence/stage_14/`.

## Next

Stage 15 is closed. Do not start Stage 16 without an explicit user prompt. Do not invent Opta or real-match accuracy claims. See `docs/stages/stage_15_completion.md` and `docs/stages/stage_16_acceptance_runbook.md`.
