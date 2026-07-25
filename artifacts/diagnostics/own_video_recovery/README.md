# Own-video perception recovery — diagnostics (Stage 17-R2)

Gate: **NO-GO — OWN-VIDEO PERCEPTION ACCEPTANCE FAILED**

Sole remaining acceptance blocker: **calibration** (SV_kp/SV_lines valid coverage ≈ 0.06 ≪ 0.70).

## Progress since v0.18.0 rejection

| Check | Status | Notes |
|-------|--------|-------|
| Reviewed GT | done | 190 human / 300 ball / 34 calib |
| Human det (independent match) | pass* | *GT boxes still originate from YOLO proposals |
| Role macro F1 | **1.00** | Pitch-masked kit rules; GK track re-labeled; staff mislabels fixed |
| Team accuracy | **0.99** | Non-player team assignments = 0 |
| Ball P/R/F1 | **0.94 / 1.00 / 0.97** | Holdout visible+localised GT only |
| Target tracking | **pass proxy** | Anchor recall 1.0; window coverage 0.93; IDF1 proxy 0.96; tracks 4→27 |
| Calibration | **FAIL** | 2/34 valid SV frames; temporal propagation cannot fill sparse correspondences |

## Why calibration still fails

- Amateur pitch + single moving phone camera; SV often returns ≤3 canonical correspondences
- Local OpenCV Hough patch homographies are mirrored/singular or have absurd residuals — **not** meter-eligible
- PnLCalib / sn-calibration clones exist but are **not wired** as FA adapters
- No customer meter/speed/sprint metrics until this gate passes

## Layout

- `gt/` — reviewed GT (+ GK/staff corrections)
- `holdout_evaluation.json` — full metrics
- `GATE_STATUS.json`
- `scripts/run_stage17r2_perception_bakeoff.py` — reproducible bakeoff

## Do not

- Treat `artifacts/final_delivery/` as customer success
- Create tag `single-player-own-video-v0.18.1` until all gates pass
- Convert pixels to meters
