# Stage 9D — Heatmap, neutral zones, and physical activity baseline

## Scope

Time-weighted heatmap, neutral Goal A/B zones, and trajectory motion-class
activity from Stage 9B/9C eligible **filtered** trajectories.

## Rules

- Heatmap weighting: **time** (`video_time_us`), not frame/point counts
- Attack direction remains **`unknown`** (no hücum/savunma thirds)
- Penalty dwell = **physical presence only** (not touch/possession/event)
- Missing coverage is **not** inactive
- Activity index: `project_generated`, not official Opta
- **No SVG/PNG committed to GitHub** (workspace temp visuals only)

## Evaluation

`NOT_EVALUATED_NO_REVIEWED_HEATMAP_ZONE_ACTIVITY_GROUND_TRUTH`

## Runtime

`/home/fdoblak/workspace/heatmap_activity_checks/`

## CLI

```bash
python -m football_analytics.cli physical spatial compute --output-dir ... --fixture-smoke
python -m football_analytics.cli physical spatial evaluate --output ... --fixture-smoke
python -m football_analytics.cli physical spatial validate
```
