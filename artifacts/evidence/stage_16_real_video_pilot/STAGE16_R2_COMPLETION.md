# Stage 16-R2 — Real-video pilot completion

| Field | Value |
|-------|-------|
| Stage | 16-R2 |
| Start HEAD (prompt expected) | `8d8370f…` (superseded by R1) |
| Actual start HEAD | `4c7bdf2eaf1b329fa37324392712dd5a2421350c` |
| Gate | **`PASS_WITH_FINDINGS — REAL-VIDEO TRACKING PILOT COMPLETE; FULL EVENT ACCEPTANCE BLOCKED`** |
| Tag | `real-video-pilot-v0.16.0` (not `single-player-analytics-v1.0.0`) |

## SoccerTrack v2 isolation

- Authoritative target preserved: **left / jersey 24 / player_id 506469**
- Deprecated invalid late-job target marked: jersey 11 / `506466` → `artifacts/evidence/stage_16/deprecated_invalid_target_506466.json`
- GSR/BAS retained; not applied to TeamTrack video
- No v1.0 release

## Source selection

1. TeamTrack Kaggle metadata: license **MIT**, version 6 — verified before download
2. Kaggle CLI/auth absent → official Google Drive mirror used for individual files only
3. Selected smallest complete `soccer_side` sequence with MP4+gt+seqinfo under 15 GiB: `F_20200220_1_0330_0360` (~13.1 MiB video)
4. SoccerTrack v1 / official demo not required (Source A succeeded)

## Pilot results (attempt 2 — horizontal tiles)

- Device: CUDA RTX 3050 Laptop GPU (4 GB); tiled inference on native 6500×1000
- Detections: 14374
- Detection precision/recall/F1 (greedy IoU@0.5): ~0.682 / 0.568 / **0.620**
- Target Track 7 coverage: **1.0**, mean IoU ~0.675, mean center error ~4.0 px
- HOTA/MOTA/IDF1: `not_evaluable` (lightweight pilot evaluator)
- Pitch meters / event metrics: `not_evaluable`
- Sequence length 30 s = full sequence (bounded pilot == full selected sequence)

## Artifacts

- Report: `artifacts/evidence/stage_16_real_video_pilot/real_video_pilot_report.json`
- PNG dual paths with identical SHA-256 (see `png_hashes.json`)
- Final customer visual `artifacts/final/single_player_analysis_summary.png` **not** created

## Remaining Stage 16 blockers

- SoccerTrack v2 official panoramic match video still unavailable (HF 401 / Drive quota)
- Event/possession/BAS acceptance not covered by TeamTrack
- Broadcast production validation not covered
