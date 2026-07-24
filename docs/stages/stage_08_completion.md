# Stage 8 completion — pitch calibration and field coordinates

## Status

**CLOSED** at annotated tag `calibration-baseline-v0.8.0`.

Gate:
`PASS_WITH_FINDINGS — PITCH PROJECTION PIPELINE ACTIVE; STAGE 8 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Real football coordinate accuracy is **not** validated (no reviewed projected-position GT).

## Sub-stages

| Stage | Deliverable | Status |
|-------|-------------|--------|
| **8A** | Pitch template, coordinate systems, calibration features, homography rules, segments, projected-position contracts | CLOSED |
| **8B** | Pitch keypoint / line detection baseline (SV_kp / SV_lines; evaluation_only) | CLOSED |
| **8C** | Homography solve + calibration segments (synthetic known-H) | CLOSED |
| **8D** | Human/ball pitch projection, quality gates, Stage 8 close | CLOSED |

## Explicit non-claims (carry forward)

- Attack direction remains `unknown`
- Human footpoint is a bbox_bottom_centre approximation (no pose/foot model)
- Ball projection is image-plane centre; airborne/grounded unknown; never physical/event metric-eligible in Stage 8
- Homography is pitch-plane only
- No running distance / sprint / heatmap / event production in Stage 8
- GPL NBJW/SV adapter remains `evaluation_only` / `production_approved=false`
- Synthetic known-H ≠ match accuracy

## Next (name only)

`Aşama 9A — Hedef Futbolcu Saha Zaman Serisi ve Fiziksel Metrik Sözleşmeleri`

Do not start Stage 9A without an explicit user prompt.
