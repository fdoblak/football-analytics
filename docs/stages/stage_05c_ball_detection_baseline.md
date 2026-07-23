# Stage 5C — Ball detection baseline (COCO sports ball)

## Status

**CLOSED** with findings (2026-07-23).

Gate: `PASS_WITH_FINDINGS — BALL DETECTION BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Purpose

Run a bounded, network-off Ultralytics YOLO11n **sports ball** detector on
analysis-window–eligible frames, emit Stage 5A detection contracts, and provide
a deterministic IoU evaluator for ball entities. COCO `sports ball` (class id 32)
maps to `entity_type=ball`, `role_label=unknown` only (never player ownership).

Small-object strategy: deterministic tiling + class-aware merge of full-frame and
tile candidates (`inference_mode: hybrid` default).

## Model reuse (no new download)

| Field | Value |
|-------|-------|
| Path | `/home/fdoblak/football_data/model_archive/yolo11n.pt` |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Size | 5,613,764 B |
| Registry id | `ultralytics_yolo11n_coco_person` (unchanged) |
| Sports ball | id=32, name=`sports ball` (verified at load) |
| Approval | `evaluation_only`, `production_approved=false` |

Same physical artifact as Stage 5B. Person adapter still rejects ball; ball
adapter rejects person.

## Coordinate space

- Full-frame Ultralytics `boxes.xyxy` → **source-frame** space.
- Tile-crop predictions → **tile-local** → `map_tile_bbox_to_source` adds tile origin.

## Outputs

Under `/home/fdoblak/workspace/ball_detection_checks/` run directories:

- `detections.parquet`
- `detection_frame_status.parquet`
- `detection_attributes.parquet`
- `ball_detection_run_receipt.json` (`human_detection_count` always 0)
- `ball_evaluation.json` → `NOT_EVALUATED_NO_REVIEWED_BALL_GROUND_TRUTH` when no reviewed GT

## CLI

```
football-analytics perception ball detect --source --timeline --analysis-windows --output-dir --config
football-analytics perception ball evaluate --predictions --ground-truth --output [--config]
```

## Findings (open)

1. **AGPL-3.0** Ultralytics weight/code distribution risk — evaluation only (reuse).
2. **No reviewed football ball ground truth** — real match mAP/F1 not claimed.
3. **GPU** — host CUDA may be unavailable in agent context; CPU smoke covered.
4. Nano COCO sports-ball recall on real broadcast footage is unverified (tiling helps but is not a trained ball specialist).

## Out of scope

- Training / fine-tuning / new model download
- Role classification (Stage 5D)
- Tracking / events / possession ownership
- SoccerNet repo changes
- Package upgrades
