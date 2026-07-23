# SoccerNet detection integration matrix (Stage 5A)

Read-only review of locked repos under `/home/fdoblak/projects/soccernet`.
No clone mutation, package install, model download, or inference in Stage 5A.

| Repo | Classification | Notes for detection layer |
|------|----------------|---------------------------|
| `sn-tracking` | **adapter_candidate** | Multi-object tracking annotations; player/ball boxes useful for detector eval adapters later (5B+). |
| `SoccerNet-v3` | **adapter_candidate** | Bounding boxes / actions; strong reference for entity boxes. Confirm license before deep adapter. |
| `sn-gamestate` | **reference_only** | Game-state / team localization context; not a primary generic detector source. |
| `sn-reid` | **reference_only** | Re-identification crops; identity stage, not detection contracts. |
| `sn-jersey` | **reference_only** | Jersey number OCR; identity/attributes after detection. |
| `sn-mvfoul` | **license_review** | Multi-view foul; may include person boxes — GPL/license fit needed before adapter. |
| `sn-trackeval` | **reference_only** | Tracking metrics tooling; evaluation harness later. |
| `sn-teamspotting` | **reference_only** | Team spotting; not core bbox detection. |
| `sn-banner` | **out_of_scope** | Field keypoints/lines (already in model registry); not player/ball detection. |
| `sn-calibration` | **out_of_scope** | Pitch calibration; Stage 6+ context. |
| `sn-spotting` / `ActiveSpotting` / `PTS-baseline` | **out_of_scope** | Action spotting timelines, not frame object detection. |
| `sn-caption` / `sn-grounding` / `sn-echoes` / `sn-depth` / `sn-nvs` / `SoccerNet` | **out_of_scope** | Captioning, grounding, depth, novel view — not Stage 5A detection contracts. |

## Summary

- **adapter_candidate:** `sn-tracking`, `SoccerNet-v3`
- **reference_only:** `sn-gamestate`, `sn-reid`, `sn-jersey`, `sn-trackeval`, `sn-teamspotting`
- **license_review:** `sn-mvfoul`
- **out_of_scope (this stage):** calibration/banner/spotting/caption/depth/nvs family

Adapters, if any, live only inside `football-analytics`. Original SoccerNet clones stay locked and clean.
