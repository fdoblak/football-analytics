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
| 14 | CLOSED | Single-player E2E orchestration / review / report / render | `single_player_pipeline_checks` | `stage_14/` |
| **15** | **CLOSED** | **Pre-release hardening (15A–15G)** | `prerelease_hardening_checks` | `stage_15/` |
| 16 | CLOSED (tech preview) | Self-contained technical acceptance + reference report (R4); video-event accuracy not validated | offline acceptance CLI | `stage_16/`, `stage_16_r4/`, `stage_16_real_video_pilot/` |

## Stage 16-R4 gate

`PASS_WITH_FINDINGS — VIDEO-BACKED TECHNICAL PREVIEW CONSOLIDATED; RELEASE TREE CLEAN; VIDEO-EVENT ACCURACY NOT VALIDATED`

Tag: `single-player-analytics-technical-preview-v0.16.1` (not `single-player-analytics-v1.0.0`).

## Stage 15 close gate

`PASS_WITH_FINDINGS — STAGE 15 PRE-RELEASE COMPLETE; ALL IMPLEMENTATION STAGES CLOSED; ONLY REAL-MATCH ACCEPTANCE STAGE 16 REMAINS`

## Notes

- Registry Arrow contract count remains **45** (Stage 15 did not bump)
- SoccerTrack v2 panoramic video is an **optional** external validation source (not a release dependency)
- Hugging Face token/login is not required for acceptance CLI, tests, build, or technical-preview release
- Real football / Opta / video-event accuracy is not validated
- Stage 16 final customer visual: `/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png` and `artifacts/final_delivery/ (canonical customer outputs)`
- Capability matrix: `docs/architecture/capability_matrix.md`
