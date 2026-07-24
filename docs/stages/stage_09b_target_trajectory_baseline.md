# Stage 9B — Target trajectory cleaning, resampling, and quality baseline

## Purpose

Prepare confirmed-target pitch trajectories:

- `raw_observed` (immutable)
- `filtered` (quality gates + reason codes)
- `resampled` (time grid within continuous segments only)

**No customer distance/speed/sprint/heatmap/activity metrics** (Stage 9C+).

## Config

`configs/physical/target_trajectory_baseline.yaml`

## Package / CLI / validator

- `src/football_analytics/physical/trajectory_*.py`
- `football-analytics physical trajectory prepare --fixture-smoke`
- `football-analytics physical trajectory validate`
- `scripts/check_target_trajectory_baseline.py`
- Runtime: `/home/fdoblak/workspace/target_trajectory_checks/`

## Evaluation

`NOT_EVALUATED_NO_REVIEWED_TARGET_TRAJECTORY_GROUND_TRUTH`

## Evidence

Stage outputs recorded under `artifacts/evidence/stage_09b/` with index updates.
See `artifacts/evidence/README.md`.
