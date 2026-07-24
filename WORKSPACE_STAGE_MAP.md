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
| **12** | **CLOSED** | **Duels / take-on / tackle / recovery / turnover / aerial / clearance** | `duels_contract_checks`, `take_on_checks`, `ground_duel_checks`, `aerial_clearance_checks`, `duels_pipeline_checks` | `stage_12/` |

## Stage 12 close gate

`PASS_WITH_FINDINGS — DUELS EVENTS PIPELINE ACTIVE; STAGE 12 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Notes

- Registry contract count after Stage 12: **42**
- Do not start Stage 13 without an explicit user prompt
- Real football / Opta accuracy is not validated
