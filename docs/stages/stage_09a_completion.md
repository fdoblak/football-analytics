# Stage 9A completion — Target trajectory and physical metric contracts

## Gate

`PASS — TARGET TRAJECTORY AND PHYSICAL METRIC CONTRACTS ACTIVE`

## Delivered

- Arrow contracts: `target_trajectory_samples`, `target_trajectory_segments`,
  `trajectory_gaps`, `physical_metric_results`
- JSON: request / receipt / evaluation
- Policies: trajectory + physical metrics (contract-only, placeholders)
- Semantics: eligibility, gaps, segments, distance/speed/sprint/heatmap/activity
- Neutral Goal A/B zones; attack direction remains unknown
- Synthetic E2E validator + cleanup
- CLI: `physical contracts|request|receipt validate`

## Not delivered (by design)

- Real distance / speed / sprint / heatmap / activity values
- Production filter / resample execution
- Events / possession / pass / dribble
- Ball metrics
- Attack-relative progression

## Evaluation

`NOT_EVALUATED_NO_REVIEWED_PHYSICAL_METRIC_GROUND_TRUTH`

## Next

**Aşama 9B — Hedef Futbolcu Trajectory Temizleme, Resampling ve Kalite Baseline**
(not started in this task)
