# Stage 9A — Target trajectory and physical metric contracts

## 1. Purpose

Define machine-readable contracts for confirmed-target pitch trajectories and
future physical metrics (distance, speed, sprint, heatmap, activity/coverage).

**Contracts only.** No real distance/speed/sprint/heatmap/activity computation,
no production smoothing/resampling, no events, no ball metrics, no attack
direction invention.

## 2. Starting SHA

`b05c54052c2090e9df3354459ee9abf06dec7a00` (`calibration-baseline-v0.8.0`)

## 3. Frozen upstream fingerprints

| Contract | Fingerprint |
|----------|-------------|
| `projected_positions` v1 | `1860638e…` |
| `track_identity_assignments` v1 | `235e7888…` |
| `calibrations` v1 | `41360b19…` |
| `calibration_segments` v1 | `9ce13ae0…` |

## 4. Trajectory input eligibility

A point may enter target trajectory measurement only if:

- Confirmed target identity interval
- Human entity; observed track observation
- `mapping_status=mapped` and `physical_metric_eligible=true`
- Valid calibration; no extrapolation; uncertainty under threshold
- Playable/non-replay window; no revoked/conflicted assignment
- Fingerprints match

Ineligible times are retained as gap/coverage/status records — never silently dropped.

## 5. New contracts

### Arrow

- `target_trajectory_samples` v1
- `target_trajectory_segments` v1
- `trajectory_gaps` v1
- `physical_metric_results` v1

### JSON

- `schemas/physical/physical_metric_request.schema.json`
- `schemas/physical/physical_metric_run_receipt.schema.json`
- `schemas/physical/physical_metric_evaluation.schema.json`

## 6. Sample layers

| Layer | Rule |
|-------|------|
| `raw_observed` | Immutable Stage 8 validated metres |
| `filtered` | Derived; raw preserved; rejection reasons required |
| `resampled` | Derived grid; cannot fill calibration/identity gaps |

Filter/resample algorithms are **placeholders** in Stage 9A (`enabled=false`).

## 7. Gap / segment semantics

Gap types: detection, tracking, identity, calibration, non_playable, shot_boundary,
track_boundary, manual_exclusion, unknown.

- No distance bridge across gaps
- Half-open `[start_us,end_us)`
- Time via `video_time_us` (VFR-safe); never fps
- Single-sample segment ≠ metric-sufficient
- Overlapping confirmed target intervals → hard conflict

## 8. Metric definitions (contracts)

- **Distance**: 2D Euclidean metres within one continuous eligible segment
- **Speed**: m/s canonical; optional km/h display; Δt from microseconds
- **Sprint**: configurable entry/exit/hysteresis/min duration/distance; not universal standard
- **Heatmap**: absolute pitch grid/kernel; time-weighted ≠ sample count; PNG not canonical
- **Activity/coverage**: separate components; low coverage ≠ low activity; composite disabled

`0`, `null`, `not_evaluable`, and `not_observed` are distinct.

## 9. Attack direction

Default remains `unknown`. Attack-relative thirds/progression forbidden.
Neutral geometric zones only: `goal_a_third`, `middle_third`, `goal_b_third`.

## 10. Evaluation

Without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_PHYSICAL_METRIC_GROUND_TRUTH`

Synthetic fixtures are not real match accuracy.

## 11. Package / CLI / validator

| Artifact | Path |
|----------|------|
| Package | `src/football_analytics/physical/` |
| Policies | `configs/physical/trajectory_policy.yaml`, `physical_metrics_policy.yaml` |
| Validator | `scripts/check_physical_metric_contracts.py` |
| Runtime | `/home/fdoblak/workspace/physical_metric_contract_checks/` |
| CLI | `football-analytics physical contracts validate` |
| | `football-analytics physical request validate` |
| | `football-analytics physical receipt validate` |

## 12. Explicit non-claims

- Real physical metrics were **not** computed in Stage 9A
- Metrics are **not** official Opta data
- Real physical accuracy is **not** validated
- Predicted/provisional/extrapolated points are not customer-metric eligible
