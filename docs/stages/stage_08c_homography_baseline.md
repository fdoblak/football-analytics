# Stage 8C — Homography solving and calibration segments

## 1. Purpose

Bounded **homography solve + calibration segment** baseline from Stage 8B
`calibration_features`:

- Canonical-ID correspondences (keypoints, stable line intersections, hybrid)
- Normalized DLT + optional OpenCV RANSAC (`image_to_pitch`)
- Quality classes: `valid|degraded|uncertain|invalid|not_available`
- Shot-cut / drift / gap-aware segments with **medoid** representative (not H average)
- Outputs: `calibrations` + `calibration_segments` + receipt/quality/eval

**Not in this stage:** projected player/ball positions (8D), physical metrics,
attack direction, new models/videos, Stage 8 close.

## 2. Starting SHA

`6bf46940bf40ec5c65a45878b08b54eff7e046cf`

## 3. Correspondence

Only mapped `canonical_pitch_feature_id` features. Reject unknown / low-score /
unsuitable / NaN / OOB / duplicate canonical or image points. Line intersections
require geometric angle/length support; near-parallel rejected. Raw model scores
are **not** calibrated probabilities.

## 4. Solver / quality

- Direction: `image_to_pitch` (row-major 3×3); inverse + round-trip checked
- Reject mirror / singular / ill-conditioned / high reproj / low coverage
- Physical-mapping eligible **only** for policy `valid` (default)
- `degraded` / `uncertain` / interpolated → **not** physical-eligible

## 5. Segments

Half-open `[start_us, end_us)`. Shot cut terminates. Drift / pan-zoom opens a
new segment. Gaps are reported, never silently filled. Overlaps → hard conflict /
review. Representative = medoid by pitch test-point projection distance.

## 6. Package / CLI / runtime

| Artifact | Path |
|----------|------|
| Config | `configs/calibration/homography_baseline.yaml` |
| Package | `src/football_analytics/calibration/homography_*.py`, `correspondence.py` |
| Validator | `scripts/check_homography_baseline.py` |
| Runtime | `/home/fdoblak/workspace/homography_checks/` |

CLI:

- `football-analytics calibration homography solve`
- `football-analytics calibration segments build`
- `football-analytics calibration homography evaluate`
- (kept) `contracts|features|homography validate|project validate`

## 7. Evaluation

Without reviewed homography GT:

`NOT_EVALUATED_NO_REVIEWED_HOMOGRAPHY_GROUND_TRUTH`

Synthetic known-H ≠ football accuracy.

## 8. Explicit non-claims

- Feature detection ≠ correct homography
- Homography is pitch-plane only (not 3D)
- Attack direction remains `unknown`
- No projected positions / physical metrics in 8C
- NBJW/SV adapter unchanged: `evaluation_only` / GPL linking risk
- `calibrations` FP `41360b19…` unchanged

## 9. Gate / next

Gate:
`PASS_WITH_FINDINGS — HOMOGRAPHY AND CALIBRATION SEGMENT BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Next (name only):
`Aşama 8D — İnsan ve Top Saha Koordinatı Projeksiyonu, Kalite Kapıları ve Aşama 8 Kapanışı`
