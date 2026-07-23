# Stage 5E — Detection fusion, quality gates, Stage 5 close

## Status

**CLOSED** with findings (2026-07-23).

Gate: `PASS_WITH_FINDINGS — DETECTION PIPELINE ACTIVE; STAGE 5 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Purpose

Fuse Stage 5B human, 5C ball, and 5D role artifacts into one deterministic,
validated detection bundle with operational quality gates and sampled review
queue. **No new detector/classifier.** No tracking / ReID / team / identity /
events / physical metrics.

## Fusion rules

- Align `run_id` / `video_id` / timeline fingerprint / source video SHA or fail
  with explicit error (no silent fill).
- Merge human + ball detections; remap ball `detection_id` when needed for
  uniqueness within a frame.
- **No cross-class NMS** between human and ball.
- Role attributes overwrite human dets only; ball stays `role_label=unknown`.
- Frame status merge: prefer `processed*` over `skipped` when either ran;
  preserve `not_eligible`; never invent zero-as-processed for unprocessed;
  eligibility conflict (`not_eligible` vs `processed*`) fails hard.
- Validate with `validate_detection_bundle`; counts recalculated from tables
  must match pipeline receipt.

## Outputs

Under `/home/fdoblak/workspace/detection_pipeline_checks/`:

- `detections.parquet`
- `detection_frame_status.parquet`
- `detection_attributes.parquet`
- `detection_pipeline_receipt.json`
- `detection_quality_report.json`
- `review_queue.json` (sampled; does not spam every unknown)

## CLI

```
football-analytics perception integrate ...
football-analytics perception validate [--frames N] [--config ...]
```

## Eval

`NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH` — operational quality only.

## Findings (open)

1. No reviewed football detection/role/ball GT — real match accuracy not claimed.
2. Upstream AGPL Ultralytics (5B/5C) and kit-heuristic role limits (5D) remain.
3. Stage 5 is a technical detection baseline, not the final product.
