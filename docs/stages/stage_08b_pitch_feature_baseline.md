# Stage 8B — Pitch keypoint and line detection baseline

## 1. Purpose

Bounded **SV_kp / SV_lines** pitch feature detection baseline:

- Stretch-resize preprocess (960×540 RGB, ToTensor [0,1], no mean/std)
- Lazy HRNet architecture load via `importlib` from locked NBJW paths
- Heatmap peak decode (scale=2) → source-image keypoints/lines
- `calibration_features` parquet + receipts/quality/evaluation

**Not in this stage:** Stage 8C homography solve as product pipeline, projected
player/ball positions, physical metrics, attack direction, training/download.

## 2. Starting SHA

`98983c03ac84566c635a996ee266a1f2f32a91eb`

## 3. Adapter selection matrix

| Option | Status |
|--------|--------|
| ai-dev lazy importlib HRNet (`cls_hrnet.py` / `cls_hrnet_l.py`) | **SELECTED** — evaluation_only |
| sn-calibration isolated env | available fallback (not used) |
| Copy NBJW into tree | **REJECTED** — GPL-2.0 |
| Postprocessor-only without weights | insufficient for 8B goal |

**GPL-2.0 linking risk:** Architecture is loaded from
`/home/fdoblak/projects/soccernet/sn-banner/camera_calibration/No_Bells_Just_Whistles/model/`
without vendoring. Weight redistribution license remains `review_required`.
Registry: `approval: evaluation_only`, `production_approved: false`.

## 4. Models

| Registry id | Path | SHA-256 | Size |
|-------------|------|---------|------|
| `sn_banner_sv_kp` | `/home/fdoblak/models/soccernet/sn-banner/SV_kp.pth` | `7ea78fa7…5113` | 264964645 |
| `sn_banner_sv_lines` | `/home/fdoblak/models/soccernet/sn-banner/SV_lines.pth` | `27512429…cac1` | 264857893 |

- Heatmaps: kp `(B,58,270,480)` → use `[:,:-1]` = 57 joints; lines `(B,24,…)` → 23
- Peak decode scale=2 → coords in 960×540 model space; inverse stretch to source WxH
- `confidence` always **null**; raw peak scores in `provenance_json`

## 5. Line name mapping

NBJW `lines_list` (23). **Trailing space** on `Goal left post left `.

Verified planar mappings → Stage 8A template ids (e.g. Middle line →
`halfway_line`). Goal posts/crossbars and ambiguous keypoint channels stay
`canonical_pitch_feature_id=null`.

## 6. Frame statuses

`processed` | `processed_no_features` | `not_eligible` | `skipped` | `failed`

No-feature ≠ failure. Calibration-ineligible / graphics / replay / close-up
frames are not processed by default.

## 7. Package / CLI / runtime

| Artifact | Path |
|----------|------|
| Config | `configs/calibration/pitch_feature_baseline.yaml` |
| Package | `src/football_analytics/calibration/pitch_feature_*.py` |
| Validator | `scripts/check_pitch_feature_baseline.py` |
| Runtime | `/home/fdoblak/workspace/pitch_feature_checks/` |

CLI:

- `football-analytics calibration features detect`
- `football-analytics calibration features evaluate`
- `football-analytics calibration features validate`
- (kept) `contracts|homography|project validate`

## 8. Evaluation

Without reviewed pitch-feature GT:

`NOT_EVALUATED_NO_REVIEWED_PITCH_FEATURE_GROUND_TRUTH`

Synthetic/model smoke ≠ football accuracy.

## 9. Explicit non-claims

- Feature detection ≠ homography
- Smoke ≠ match accuracy
- SV weights not production-approved
- No attack direction / projected positions / physical metrics in 8B
- `calibrations` FP `41360b19…` unchanged

## 10. Gate / next

Gate:
`PASS_WITH_FINDINGS — PITCH FEATURE DETECTION BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Next (name only):
`Aşama 8C — Homografi Çözümü, Kalibrasyon Segmentleri ve Değerlendirme`
