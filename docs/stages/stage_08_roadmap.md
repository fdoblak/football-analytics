# Stage 8 roadmap — pitch calibration and field coordinates

## Goal

Map human and ball **image** observations onto a canonical **pitch metre**
frame so future physical metrics (distance, sprint, heatmap) have an explicit,
reviewable calibration basis.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **8A** | Pitch template, coordinate systems, calibration features, homography rules, calibration segments, projected positions, request/receipt/eval stubs | **IN TREE** (contracts only) |
| **8B** | Pitch keypoint / line detection baseline (SV_kp / SV_lines adapters) | **not started** |
| **8C+** | Real calibration solve on match video, temporal stability, physical metrics | **not started** |

## Stage 8A status

Contracts, synthetic geometry, and validators are in-tree. **No** SV_kp /
SV_lines inference, **no** real video calibration, **no** physical metrics.

## Explicit non-goals (until later stages)

- Real keypoint / line detection
- Attack direction / team-side invention
- Running distance / sprint / heatmap production
- Claiming projected positions as physical truth for airborne ball

## Dependencies

- Existing `calibrations` v1 (frozen fingerprint; extended via sidecars)
- `track_observations`, `frames`, camera-view / shot segments
- Stage 7 identity eligibility when target metrics require confirmed identity

## Next after Stage 8A

`Aşama 8B — Saha Anahtar Noktası ve Çizgi Algılama Baseline`
