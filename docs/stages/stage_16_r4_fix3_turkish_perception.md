# Stage 16-R4-FIX3 — Turkish player report + perception repair

## Verdict

`PASS_WITH_FINDINGS — TURKISH SINGLE-PLAYER REPORT COMPLETE; REAL-VIDEO PERCEPTION PILOT IMPROVED; VIDEO-EVENT ACCURACY NOT VALIDATED`

## Root cause (proof clutter)

Primary: evaluation-style overlay drew all prediction boxes plus GT Track 7 (dual box on target) with no confirmed-track gate, no team colors, and no ball pipeline.

## Perception (TeamTrack F_20200220_1_0330_0360)

- Selected detector: `candidate_v1` (stricter NMS + center merge + size/aspect filters)
- Full F1 ≈ 0.618 (baseline ≈ 0.620); target coverage 0.996; mean IoU ≈ 0.673
- Tracker: confirmed IoU CV (ByteTrack compared; AGPL rejected for product)
- ID switches / fragmentation on target: 0 / 0
- Team: track-level sticky kit clusters; flip count 0; within-track consistency 1.0
- Ball: YOLO11n class 32 tiled + gated OpenCV support; **no ball GT** → not_evaluable; rare observed/candidate only; no fake ball drawn otherwise

## Delivery

`artifacts/final_delivery/` Turkish PDF/JSON/PNG + `real_video_analysis_proof.mp4` + HTML; Windows mirror hash-equal.

Namespaces isolated: TeamTrack Track 7 ≠ SoccerTrack Player 506469.
