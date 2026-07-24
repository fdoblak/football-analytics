# Trajectory and physical metric semantics (Stage 9A)

Contract-level definitions for future physical analytics. **No real metrics are
computed in Stage 9A.** Thresholds are configurable project policy — not
official Opta or universal sports-science standards.

## Trajectory sample

Canonical pitch metres from Stage 8 `projected_positions` that pass confirmed-target
eligibility. Layers: `raw_observed` (immutable), `filtered`, `resampled`.

## Gaps

Explicit coverage holes. Distance must not bridge gaps. Short-gap interpolation,
if enabled later, is derived and ineligible-by-default. Long gaps start a new
continuous segment. Time uses `video_time_us` only.

## Distance

2D Euclidean metres between consecutive eligible points in the **same** continuous
segment. Not 3D body motion, energy, or invisible-time estimation.

## Speed

`m/s` canonical (= distance / Δt_us). Optional `km/h` display only. No fps-based
time. Single two-point spikes are not peak speed.

## Sprint

Configurable entry/exit (hysteresis), min duration/distance, max internal gap.
Single-frame spikes are not sprints. Gaps/cuts/boundaries terminate episodes.
Threshold profile must appear in reports; not claimed as universal truth.

## Heatmap

Absolute pitch bins/kernel; time-weighted dwell distinct from sample counts.
Unseen time ≠ zero activity. No attack flip while direction is unknown.
PNG is visualization only; table/manifest is canonical.

## Activity / coverage

Components are separate. Low data coverage must not be labelled low activity.
Composite score disabled by default in Stage 9A.

## Coverage / not-evaluable policy

| Symbol | Meaning |
|--------|---------|
| `0` | Observed zero |
| `null` | Value absent |
| `not_evaluable` | Cannot evaluate under policy/coverage |
| `not_observed` | Target/time not observed |

## Attack direction

Unknown → absolute coordinates only; no `first_third` / `final_third` progression.
