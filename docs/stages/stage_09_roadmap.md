# Stage 9 roadmap — Target-player physical metrics

## Status

| Sub-stage | Status |
|-----------|--------|
| **9A** Trajectory + physical metric **contracts** | CLOSED |
| **9B** Trajectory cleaning / resampling / quality baseline | CLOSED |
| **9C** Distance / speed / sprint computation | **ACTIVE (this stage)** |
| 9D Heatmap / activity / coverage | NOT STARTED |
| 9E Physical metric fusion + Stage 9 close | NOT STARTED |

## 9A scope

Contracts, semantics, policies, validation, synthetic fixtures only.

**Does not** compute real distance, speed, sprint, heatmap, or activity values.

## Inputs (from prior stages)

- Stage 7: confirmed target identity + metric-eligibility timeline
- Stage 8: `projected_positions` with physical-metric eligibility
- Pitch template / coordinate frame (attack direction still `unknown`)

## Outputs (contracts)

- `target_trajectory_samples`
- `target_trajectory_segments`
- `trajectory_gaps`
- `physical_metric_results` (stub/not_evaluable rows)
- Request / receipt / evaluation JSON schemas

## Explicit non-goals (Stage 9A)

- Real metric computation or production filter/resample runs
- Event / pass / dribble / possession
- Ball physical metrics
- Attack-relative zones or progression
- New models / datasets / purchases
