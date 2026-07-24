# Stage 8 roadmap — pitch calibration and field coordinates

## Goal

Map human and ball **image** observations onto a canonical **pitch metre**
frame so future physical metrics (distance, sprint, heatmap) have an explicit,
reviewable calibration basis.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **8A** | Pitch template, coordinate systems, calibration features, homography rules, calibration segments, projected positions, request/receipt/eval stubs | **CLOSED** |
| **8B** | Pitch keypoint / line detection baseline (SV_kp / SV_lines adapters) | **CLOSED** (evaluation_only; accuracy not validated) |
| **8C** | Homography solve, calibration segments, quality gates, evaluation stub | **CLOSED** (synthetic known-H; real accuracy not validated) |
| **8D** | Projected human/ball pitch coordinates, quality gates, Stage 8 close | **CLOSED** |

## Stage 8 status

**Stage 8 is CLOSED** (`calibration-baseline-v0.8.0`). Contracts → pitch
features → homography/segments → projected positions are in-tree. Attack
direction remains `unknown`. Ball never physical/event metric-eligible.
Human footpoint is an approximation. Real football coordinate accuracy is
**not** validated. GPL SV adapter unchanged (`evaluation_only`).

## Explicit non-goals (until later stages)

- Attack direction / team-side invention
- Running distance / sprint / heatmap production
- Claiming projected positions as physical truth for airborne ball
- Stage 9A target pitch time-series / physical metric contracts (not started)

## Dependencies

- Existing `calibrations` v1 (frozen fingerprint; extended via sidecars)
- Stage 8B `calibration_features`
- Stage 8C `calibration_segments`
- `frames`, camera-view / shot segments / analysis windows
- Stage 7 identity eligibility when target metrics require confirmed identity

## Next after Stage 8

`Aşama 9A — Hedef Futbolcu Saha Zaman Serisi ve Fiziksel Metrik Sözleşmeleri`
