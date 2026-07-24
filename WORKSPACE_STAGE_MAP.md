# Workspace stage map

Quick orientation for which stage owns which runtime/workspace roots and close status.

| Stage | Status | Package / focus | Runtime roots (checks) | Evidence |
|-------|--------|-----------------|------------------------|----------|
| 00–02 | CLOSED | Foundation / storage / contracts | various | `artifacts/evidence/stage_0*` |
| 03 | CLOSED | Video ingest / normalize | video checks | `stage_03/` |
| 04 | CLOSED | Broadcast / shots / camera | broadcast checks | `stage_04/` |
| 05 | CLOSED | Detection | perception checks | `stage_05/` |
| 06 | CLOSED | Tracking | tracking checks | `stage_06/` |
| 07 | CLOSED | Identity / target | identity checks | `stage_07/` |
| 08 | CLOSED | Calibration / projection | calibration checks | `stage_08/` |
| 09 | CLOSED | Physical metrics | physical checks | `stage_09/` |
| 10 | CLOSED | Human–ball interaction | interaction checks | `stage_10/` |
| 11 | CLOSED | Passing / reception / progression | `passing_*_checks` | `stage_11/` |
| 12 | CLOSED | Duels / take-on / tackle / recovery / turnover / aerial / clearance | `duels_*_checks` | `stage_12/` |
| 13 | CLOSED | Target event ledger / metrics aggregation | `events_*_checks` | `stage_13/` |
| **14** | **CLOSED** | **Single-player E2E orchestration / review / report / render** | `single_player_pipeline_checks` | `stage_14/` |

## Stage 14 close gate

`PASS_WITH_FINDINGS — SINGLE PLAYER PIPELINE ACTIVE; STAGE 14 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Notes

- Registry Arrow contract count remains **45** (Stage 14 uses JSON schemas)
- Do not start Stage 15 without an explicit user prompt
- Real football / Opta accuracy is not validated
- Stage 16 reserved finals: `/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png` and `artifacts/final/single_player_analysis_summary.png`
