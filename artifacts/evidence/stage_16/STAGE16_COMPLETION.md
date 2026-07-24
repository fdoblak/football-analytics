# Stage 16 completion status

## Gate

`NO-GO — REAL-MATCH ACCEPTANCE FAILURE`

## Stage 16-R2 resume gate

`PASS_WITH_FINDINGS — REAL-VIDEO TRACKING PILOT COMPLETE; FULL EVENT ACCEPTANCE BLOCKED`

See `artifacts/evidence/stage_16_real_video_pilot/STAGE16_R2_COMPLETION.md`. SoccerTrack v2 panoramic MP4 still blocked; TeamTrack official `soccer_side` sequence used for tracking pilot only. Tag `real-video-pilot-v0.16.0` — **not** `single-player-analytics-v1.0.0`.


## Why

Official SoccerTrack v2 source and CC BY 4.0 license were verified. Match `128057` GSR+BAS(+calibration metadata) were downloaded from the official Google Drive mirror and hashed. Target selection, leakage namespaces, and held-out BAS scaffold completed.

**Blocking failure:** panoramic match MP4 download repeatedly returned Google Drive **Quota exceeded** / Too many users after virus-scan confirm. Hugging Face canonical dataset URL returned **401** at access time. Without the official video, pilot/full-match prediction pipeline, held-out prediction evaluation, final customer report from pipeline predictions, and dual-path final PNG cannot be completed honestly.

## Retained for resume (not deleted)

- `/home/fdoblak/football_data/datasets/soccertrack_v2/source/gsr/128057/*.json` (~5.2 GiB total source tree)
- `/home/fdoblak/football_data/datasets/soccertrack_v2/source/bas/128057/*.json`
- `/home/fdoblak/football_data/datasets/soccertrack_v2/source/raw/128057/*` (partial)
- `/home/fdoblak/football_data/datasets/soccertrack_v2/source/license/*`
- Run namespaces under `/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/`

## Completed productization pieces

- Acceptance adapter package + leakage hard-fail tests (13 passed)
- Dataset registry entry `soccertrack_v2_single_match_acceptance`
- External lock pin for SoccerTrack-v2 @ `3ee38e481aab9de0f1d099c1cdde15302eb63f49`
- CC BY attribution in notices / license inventory
- Selected target: SoccerTrack v2 Match 128057 / Team left / Jersey 24 (`player_id` 506469)
- Checkpoint commit: `fbdcb6f19a06cfeaee4ffb280167725dff4d3bdf`

## Not claimed

- Real-match video processed
- Final `single-player-analytics-v1.0.0` tag
- Final dual-path PNG
- Broadcast production validation
- Official Opta accuracy
