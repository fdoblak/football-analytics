# Own-video perception recovery — diagnostics (Stage 17-R2)

Gate: **WAITING — REAL FRAME REVIEW REQUIRED**

Previous v0.18.0 customer delivery reclassified as
`NO-GO — OWN-VIDEO PERCEPTION NOT VALIDATED` (see `artifacts/rejected_v0.18.0/`).

## Why waiting

Reviewed ground truth is incomplete:

- human frames reviewed << 180
- ball frames reviewed << 300
- no pitch line/keypoint review set yet

Auto prelabels remain `auto_candidate` and must not be treated as GT.

## How to continue

```bash
python scripts/review_own_video_frames.py list --kind human --split holdout
python scripts/review_own_video_frames.py show --kind human --frame 709 --open
python scripts/review_own_video_frames.py export-gt
```

Overlays live under `/home/fdoblak/workspace/own_video_analysis/stage17r2/` (not Git).

## Files

- `gt/gt_human.json`, `gt/gt_ball.json`
- `seed_metrics.json` (seed-only; not acceptance)
- `PREVIOUS_RESULT_RECLASSIFIED.json`
- `recovery_plan_status.json`
