# Stage 9C — Distance, speed, and sprint baseline

## Scope

Compute **measured** distance, robust speed, and project-defined sprint bouts
from Stage 9B eligible target trajectories.

## Primary layer

**`filtered`** — quality-gated observed pitch points with
`metric_eligibility=eligible`. `raw_observed` and `resampled` are diagnostic
only (resampled remains non-customer-primary / derived).

## Semantics

| Concept | Meaning |
|---------|---------|
| Measured distance | Sum of Euclidean pitch metres **inside** continuous eligible segments |
| Full-match distance | **Not** produced; uncovered time is never extrapolated |
| Robust speed | Median-window smoothed m/s from `video_time_us` deltas |
| Diagnostic raw peak | Instantaneous max before outlier exclusion — not customer peak |
| Sprint | Hysteresis state machine; `metric_origin: project_generated` |

## Sprint metadata (not official Opta)

- `metric_origin: project_generated`
- `definition_style: opta_style_metric_definition`
- Configurable entry/exit thresholds, min duration/samples/distance
- Hard gap / segment boundary terminates; no gap merge

## Evaluation

`NOT_EVALUATED_NO_REVIEWED_DISTANCE_SPEED_SPRINT_GROUND_TRUTH`

Synthetic fixtures prove math/pipeline only — **not** real football accuracy.

## Runtime

`/home/fdoblak/workspace/distance_speed_sprint_checks/`

## CLI

```bash
python -m football_analytics.cli physical motion compute --output-dir ... --fixture-smoke
python -m football_analytics.cli physical motion evaluate --output ... --fixture-smoke
python -m football_analytics.cli physical motion validate
```
