# Stage 5B — Human detection baseline (player/official boxes as generic humans)

## Status

**CLOSED** with findings (2026-07-23).

Gate: `PASS_WITH_FINDINGS — HUMAN DETECTION BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Purpose

Run a bounded, network-off Ultralytics YOLO11n person detector on analysis-window–eligible
frames, emit Stage 5A detection contracts, and provide a deterministic IoU evaluator.
Generic COCO `person` maps to `entity_type=human`, `role_label=unknown` only
(never player / referee / GK). Ball is deferred to Stage 5C.

## Model selection

| Candidate | Local present? | Size | License | Decision |
|-----------|----------------|------|---------|----------|
| SoccerNet / NBJW SV_kp / SV_lines | yes (keypoints/lines) | ~253 MiB each | review_required | unfit (not person detector) |
| Local YOLO `.pt` inventory | **none** | — | — | no suitable weight |
| Ultralytics `yolo11n.pt` (v8.3.0 asset) | placed in model_archive | 5,613,764 B | **AGPL-3.0** | **selected** (evaluation_only) |
| Ultralytics `yolov8n.pt` | not used | ~6 MiB | AGPL-3.0 | not selected (YOLO11n preferred) |

Selected URL:
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`

- SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- Path: `/home/fdoblak/football_data/model_archive/yolo11n.pt`
- Provenance: `yolo11n.pt.provenance.json` sibling
- Registry id: `ultralytics_yolo11n_coco_person`
- COCO person class id: `0`
- `approval: evaluation_only`, `production_approved: false`

## Coordinate space

Ultralytics `YOLO.predict()` on the original BGR image returns `boxes.xyxy` in
**source-frame** coordinates. Stage 5B records letterbox transform fingerprints
for provenance but does **not** apply `inverse_bbox` (would double-transform).
Clip / min-area / max-aspect filters run in source space.

## Outputs

Under a run output directory (runtime root
`/home/fdoblak/workspace/human_detection_checks/` only):

- `detections.parquet`
- `detection_frame_status.parquet`
- `detection_attributes.parquet`
- `detection_run_receipt.json` (`ball_detection_count` always 0)
- `evaluation.json` → `NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH` when no reviewed GT

## CLI

```
football-analytics perception humans detect --source --timeline --analysis-windows --output-dir --config
football-analytics perception humans evaluate --predictions --ground-truth --output [--config]
```

## Findings (open)

1. **AGPL-3.0** Ultralytics weight/code distribution risk — evaluation only.
2. **No reviewed football ground truth** — mAP/F1 on real match footage not claimed.
3. **GPU** — host CUDA may be unavailable in agent context (`AGENT_CONTEXT_GPU_UNVERIFIABLE`); CPU smoke covered.

## Out of scope

- Training / fine-tuning
- Ball detection (Stage 5C)
- Role upgrade to player/referee/GK
- SoccerNet repo changes
- Package upgrades
