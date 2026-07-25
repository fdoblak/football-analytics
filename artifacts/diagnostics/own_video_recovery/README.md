# Own-video perception recovery — diagnostics (Stage 17-R2)

Gate: **NO-GO — OWN-VIDEO PERCEPTION ACCEPTANCE FAILED**

## What was completed

- Reviewed GT: **190 human** + **300 ball** frames (train/dev/locked-holdout)
- Calibration structure review: **34** frames with line/keypoint candidates
- Holdout evaluation of YOLO11n baseline + role/team + ball
- Tracker comparison: IoU vs appearance vs ByteTrack (eval-only AGPL)
- Referee/staff team-assignment regression tests
- v0.18.0 customer delivery remains rejected (not active final)

## Why NO-GO

Failed acceptance checks:

- `role_macro_f1` ≈ 0.67 < 0.90 (player/referee/GK/staff confusion)
- `calibration` not valid for meter conversion (unstable/rejected local homographies; SV multi-segment not finished)
- `target_tracking` not confirmed (jersey numbers mostly illegible; no false assignment claimed)

Passed on holdout (with caveats):

- Ball P/R/F1 meets numeric thresholds on reviewed holdout
- Team accuracy high among matched player boxes; non-player team assignment = 0
- Appearance tracker fragmentation ~1.82/s (better than IoU ~4.4/s)

Human box P/R/F1≈1.0 is **circular** (GT boxes from detector proposals) — not independent detector proof.

## Layout

- `gt/` — reviewed GT JSON + seed decisions + contact sheet samples
- `holdout_evaluation.json` — full metrics
- `metrics_charts.png`
- `tracker_comparison.json`
- `GATE_STATUS.json`

## Next blockers

1. Football-specific detector/role head (or SoccerNet GSR path) to raise role macro F1
2. Track identity GT + TrackEval HOTA/IDF1
3. SV_kp/SV_lines per-shot calibration with real pitch correspondences
4. Explicit jersey-5 visibility labels on holdout for target coverage
