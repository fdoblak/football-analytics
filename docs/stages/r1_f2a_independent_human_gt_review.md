# R1-F2-A — Independent football human GT review

**Gate (tool ready):** `PASS — INDEPENDENT HUMAN GT REVIEW TOOL READY`

This stage prepares a **blind/independent human labeling tool**. It is **not** detector acceptance, fine-tuning, or R2.

## What is ready

| Item | Location |
|------|----------|
| Review server | `scripts/r1_independent_gt_review_server.py` (127.0.0.1 only) |
| Prepare runtime | `scripts/prepare_r1_f2a_independent_gt.py` |
| Schema / freeze validator | `src/football_analytics/annotation/independent_gt.py` |
| Frame selection | `src/football_analytics/annotation/frame_selection.py` |
| Frame manifest (git) | `artifacts/evidence/reboot_01/independent_gt/selected_frames.json` |
| Runtime drafts (NOT git) | `/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4/` |
| Windows launcher | `Desktop/Football Analytics Validation/R1 Independent GT/START_GT_REVIEW.bat` |

## Splits (time isolation)

- **train** 0–12 s — 40 frames — YOLO11n-hybrid **proposals only** (not GT)
- **dev** 12–22 s — 20 frames — fully blind
- **holdout** 22–34 s — 20 frames — fully blind

## User action

```text
Desktop → Football Analytics Validation → R1 Independent GT
→ START_GT_REVIEW.bat
```

## Freeze

Freeze is **not** performed in this stage. `validate_freeze_ready()` hard-fails without explicit later user approval. Prior agent/YOLO drafts remain non-eligible.
