# Stage 3 completion — Broadcast video foundation

Stage 3 delivers the safe local video input layer: contracts → probe →
normalize → frame timeline. No player metrics, detection, tracking, ReID, or
Opta claims.

## Sub-stages

| Sub-stage | Status | Gate |
|-----------|--------|------|
| **3A** Safe video ingest contracts & fixture design | **CLOSED** | `PASS — SAFE VIDEO INGEST CONTRACTS ACTIVE` |
| **3B** Safe FFprobe & media validation | **CLOSED** | `PASS — SAFE MEDIA PROBE ACTIVE` |
| **3C** Safe FFmpeg normalization | **CLOSED** | `PASS — SAFE VIDEO NORMALIZATION ACTIVE` |
| **3D** Frame timeline / time mapping & Stage 3 close | **CLOSED** | `PASS — DETERMINISTIC FRAME TIMELINE ACTIVE; STAGE 3 CLOSED` |

## Starting checkpoint (3D)

`f38ea05cf1483cb9d2805ee13678b58c73c12663`

## Evidence pointers

- Contracts: `docs/stages/stage_03a_ingest_contracts.md`
- Probe: `docs/stages/stage_03b_media_probe.md`
- Normalization: `docs/stages/stage_03c_video_normalization.md`
- Frame timeline: `docs/stages/stage_03d_frame_timeline.md`
- Operator docs: `docs/video/`

## Final counts (Phase C)

- Baseline → final: **549 → 572** pytest PASS (**+23** Stage 3D)
- validators: video contracts/probe/normalization/frame_timeline PASS;
  runtime/data/stage-cache/storage/registries/secrets PASS;
  `check_project.py --profile local --deep` → PASS_WITH_WARNINGS (dirty tree)
- `python -m build` PASS; protected packages unchanged
- Runtime report:
  `/home/fdoblak/workspace/frame_timeline_checks/frame_timeline_validation_20260723T074303Z.json`

## Tag planned

Annotated `video-ingest-v0.3.0` (Stage 2 convention: `foundation-v0.1.0`).

## Gate

`PASS — DETERMINISTIC FRAME TIMELINE ACTIVE; STAGE 3 CLOSED`

## Next stage (name only)

`Aşama 4A — Shot Sınırları ve Kamera Görüşü Sınıflandırma Sözleşmeleri`

## Out of scope (unchanged)

SoccerNet download, identity/tracking/events/metrics, Opta scraping, GUI/API,
Codex/continuous automation.

## Overall Stage 3 gate

`PASS — DETERMINISTIC FRAME TIMELINE ACTIVE; STAGE 3 CLOSED`
