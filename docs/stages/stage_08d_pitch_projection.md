# Stage 8D — Pitch projection pipeline and Stage 8 close

## 1. Purpose

Project human/ball **track observations** onto canonical **pitch metres** using
Stage 8C `calibration_segments`:

- Human source: `bbox_bottom_centre` (footpoint approximation; no pose model)
- Ball source: `bbox_centre` (image-plane only; airborne unknown)
- `image_to_pitch` homography only (never `H_inv` for the primary map)
- Mapping status + uncertainty + physical / target-customer eligibility
- Outputs: `projected_positions` + receipt / quality / eval / review queue

**Stage 8 closed** at tag `calibration-baseline-v0.8.0`.

## 2. Starting SHA

`44631a4d1b79037078d58aacddb7948d96879f5d`

## 3. Segment selection

Half-open `[start_us, end_us)`. Unique `valid` + `physical_metric_eligible`
segment per observation time. Overlaps → hard conflict. Gaps / degraded /
uncertain / interpolated → `not_calibrated` (not physical-eligible mapping).

## 4. Eligibility

| Entity | Physical metric | Event metric | Customer target metric |
|--------|-----------------|--------------|------------------------|
| Human (observed, mapped, in-bounds, unc OK, playable) | may be eligible | n/a | only with confirmed target interval |
| Predicted / extrapolated / provisional | not eligible | — | not eligible |
| Ball | **always false** | **always false** | — |

## 5. Package / CLI / runtime

| Artifact | Path |
|----------|------|
| Config | `configs/calibration/pitch_projection_pipeline.yaml` |
| Package | `src/football_analytics/calibration/pitch_projection*.py` |
| Validator | `scripts/check_pitch_projection_pipeline.py` |
| Runtime | `/home/fdoblak/workspace/pitch_projection_checks/` |

CLI:

- `football-analytics calibration project tracks`
- `football-analytics calibration project validate`
- `football-analytics calibration projection evaluate`

## 6. Evaluation

Without reviewed projected-position GT:

`NOT_EVALUATED_NO_REVIEWED_PROJECTED_POSITION_GROUND_TRUTH`

Synthetic known-H ≠ football accuracy.

## 7. Explicit non-claims

- No distance / speed / sprint / heatmap / events
- Attack direction remains `unknown`
- Human footpoint ≠ guaranteed ground contact
- Ball projection ≠ grounded possession / pass / shot
- NBJW/SV adapter unchanged: `evaluation_only` / GPL linking risk
- Frozen contract fingerprints unchanged (`calibrations` / `projected_positions`)

## 8. Gate / next

Gate:
`PASS_WITH_FINDINGS — PITCH PROJECTION PIPELINE ACTIVE; STAGE 8 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Next (name only):
`Aşama 9A — Hedef Futbolcu Saha Zaman Serisi ve Fiziksel Metrik Sözleşmeleri`
