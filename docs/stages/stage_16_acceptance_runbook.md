# Stage 16 acceptance runbook — Technical preview (R4)

**Status:** `PASS_WITH_FINDINGS — SELF-CONTAINED TECHNICAL ACCEPTANCE COMPLETE; REAL-VIDEO TRACKING VALIDATED; VIDEO-EVENT ACCURACY NOT VALIDATED`

SoccerTrack v2 panoramic MP4 is **optional** and is **not** required for release gates, CLI defaults, tests, or builds. Hugging Face token/login is **not** required.

## Three evidence tracks (do not mix)

1. **TeamTrack real-video pilot** (`teamtrack_real_video_pilot`) — real video ingest/GPU/detection/tracking.
2. **SoccerTrack v2 reference** (`soccertrack_v2_reference_analysis`) — GSR/BAS annotation-derived metrics for match `128057` / jersey `24` / `506469`. **Not** video prediction.
3. **Self-contained deterministic acceptance** (`self_contained_deterministic_acceptance`) — offline contract/metric/report arithmetic.

## Offline commands

```bash
football-analytics acceptance generate --output-dir artifacts/evidence/stage_16_r4/self_contained
football-analytics acceptance run --output-dir artifacts/evidence/stage_16_r4/self_contained
football-analytics acceptance validate --dir-a /tmp/a --dir-b /tmp/b
football-analytics acceptance reference soccertrack-v2
football-analytics report render-final
```

## Final customer outputs

- `artifacts/final/single_player_analysis_summary.json`
- `artifacts/final/single_player_analysis_summary.png`
- twin: `/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png`

Only one customer final PNG is retained in the current tree.
