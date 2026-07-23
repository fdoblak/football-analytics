# Stage 3 — Broadcast video foundation (contracts → probe → normalize → frames)

Stage 2 (`foundation-v0.1.0`) is closed. Stage 3 builds the **safe video input
layer** that later stages need for single-`target_player` evidence.

Stage 3 does **not** compute player metrics, run detection/tracking/ReID, or
claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **3A** | Safe video ingest contracts & fixture design | JSON schemas, policy, typed models, path/hash safety, synthetic fixture design, validators/tests. | **CLOSED** |
| **3B** | Safe FFprobe & media validation | Run FFprobe against allowed local sources; map tool JSON into the Stage 3A probe contract; hard-fail vs warning policy; atomic probe outputs. | **CLOSED** |
| **3C** | Normalization | Execute normalize **plans** from 3A (non-destructive, no overwrite, aspect-safe). | **CLOSED** (this stage) |
| **3D** | Frame extraction / time mapping & Stage 3 close | Frame table time mapping (`video_time_us`), close Stage 3 gates. | not started |

## Product link

All Stage 3 work must keep later single-player heatmaps, duels, passes, sprints,
and evidence/coverage feasible — without inventing metrics in Stage 3.

## Out of scope for Stage 3

- SoccerNet dataset download / model download
- Player identity / tracking / events / metrics
- Opta scraping or fake Opta
- GUI/API / paid services
- Continuous automation / Codex supervisor
