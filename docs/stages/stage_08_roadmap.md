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
| **8C+** | Homography solve, calibration segments on match video, temporal stability, physical metrics | **not started** |

## Stage 8B status

SV_kp / SV_lines lazy importlib HRNet adapter, stretch preprocess, peak/line
postprocess, `calibration_features` outputs, and bounded smoke are in-tree.
**No** Stage 8C homography product pipeline, **no** physical metrics, **no**
reviewed GT accuracy claim. GPL-2.0 architecture linking → evaluation_only;
`production_approved=false`.

## Explicit non-goals (until later stages)

- Homography solve as production pipeline (8C)
- Attack direction / team-side invention
- Running distance / sprint / heatmap production
- Claiming projected positions as physical truth for airborne ball

## Dependencies

- Existing `calibrations` v1 (frozen fingerprint; extended via sidecars)
- `track_observations`, `frames`, camera-view / shot segments
- Stage 7 identity eligibility when target metrics require confirmed identity

## Next after Stage 8B

`Aşama 8C — Homografi Çözümü, Kalibrasyon Segmentleri ve Değerlendirme`
