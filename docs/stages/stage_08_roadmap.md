# Stage 8 roadmap — pitch calibration and field coordinates

## Goal

Map human and ball **image** observations onto a canonical **pitch metre**
frame so future physical metrics (distance, sprint, heatmap) have an explicit,
reviewable calibration basis.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **8A** | Pitch template, coordinate systems, calibration features, homography rules, calibration segments, projected positions, request/receipt/eval stubs | **IN TREE** (contracts only) |
| **8B** | Pitch keypoint / line detection baseline (SV_kp / SV_lines adapters) | **IN TREE** (evaluation_only; accuracy not validated) |
| **8C** | Homography solve, calibration segments, quality gates, evaluation stub | **IN TREE** (synthetic known-H; real accuracy not validated) |
| **8D** | Projected human/ball pitch coordinates, quality gates, Stage 8 close | **not started** |

## Stage 8C status

Normalized DLT + OpenCV RANSAC homography from mapped correspondences,
frame-level quality classes, shot-cut/drift/gap segment builder with medoid
representative, `calibrations` + `calibration_segments` outputs are in-tree.
**No** projected player/ball positions, **no** physical metrics, **no** attack
direction invention, **no** reviewed GT accuracy claim. GPL-2.0 NBJW adapter
unchanged (`evaluation_only`; `production_approved=false`).

## Explicit non-goals (until later stages)

- Projected human/ball positions as product pipeline (8D)
- Attack direction / team-side invention
- Running distance / sprint / heatmap production
- Claiming projected positions as physical truth for airborne ball

## Dependencies

- Existing `calibrations` v1 (frozen fingerprint; extended via sidecars)
- Stage 8B `calibration_features`
- `frames`, camera-view / shot segments / analysis windows
- Stage 7 identity eligibility when target metrics require confirmed identity

## Next after Stage 8C

`Aşama 8D — İnsan ve Top Saha Koordinatı Projeksiyonu, Kalite Kapıları ve Aşama 8 Kapanışı`
