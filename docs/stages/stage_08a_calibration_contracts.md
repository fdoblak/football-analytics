# Stage 8A — Pitch calibration, homography, and coordinate contracts

## 1. Purpose

Define machine-readable contracts for:

- Canonical pitch template and configurable dimensions
- Source-image vs pitch metre coordinate systems
- Calibration feature (keypoint / line) observations
- Image→pitch homography solve / quality / rejection rules
- Time-scoped calibration segments
- Projected positions + physical-metric eligibility
- Request / receipt / evaluation stubs

**Contracts only.** No SV_kp/SV_lines inference, no real keypoint/line
detection, no physical metric computation, no attack-direction invention.

## 2. Starting SHA

`59bb64457d17f1804be30d03392d2995bae69f73` (`identity-baseline-v0.7.0`)

## 3. Frozen upstream fingerprint

| Contract | Fingerprint |
|----------|-------------|
| `calibrations` v1 | `41360b19ae034f361949a75d8e773c265f0792b2603b69f612ab5863662ac871` |

`calibrations` is reused **by reference** (extended via new sidecars), not mutated.

## 4. Coordinate systems

### Source image

- Origin: top-left
- `x_px` right, `y_px` down
- Source-frame pixels (not crop/model space)
- Finite floats; frame-bounds validated

### Canonical pitch

- `x_m ∈ [0, pitch_length_m]`, `y_m ∈ [0, pitch_width_m]`
- Default template example: **105 m × 68 m** (configurable; not claimed official)
- FIFA/IFAB range validation available; not an official-size claim
- Axes are **not** attack direction; Goal A/B labels are neutral
- Attack direction default: `unknown` (no team-side invent)

Every artifact carries a `coordinate_frame_id`. Mixing frames is forbidden.

## 5. New contracts

### Arrow (registry)

- `calibration_features` v1
- `calibration_segments` v1 — time-scoped `image_to_pitch` H
- `projected_positions` v1

### JSON

- `schemas/calibration/calibration_request.schema.json`
- `schemas/calibration/calibration_run_receipt.schema.json`
- `schemas/calibration/calibration_evaluation.schema.json`

## 6. Homography rules

- Direction: `image_to_pitch` (row-major 3×3)
- ≥4 non-collinear, non-duplicate correspondences
- Reject singular / ill-conditioned / mirrored
- Invert + round-trip required
- Planar pitch only; airborne ball ≠ guaranteed pitch point
- Old H must not auto-transfer across pan/zoom/cut

## 7. Calibration segments

- Shot cut → terminate segment
- Pan/zoom → new calibration or invalid
- Replay / non-playable → `not_eligible`
- Unknown camera view → abstain/review
- Overlapping intervals → conflict
- Gaps are **not** silently filled
- Interpolated H → not physical-metric eligible

## 8. Projected positions / eligibility

Source points:

- Human: bbox bottom-centre (not guaranteed foot contact)
- Ball: bbox centre (airborne → not metric-eligible)

Mapping statuses: `mapped|outside_pitch|extrapolated|uncertain|not_calibrated|not_eligible|failed`

Physical-metric eligible only when observed source + valid calibration +
in-bounds + no extrapolation + sufficient quality (+ identity if required).

Predicted/interpolated observations are never customer physical-metric eligible.

## 9. Package / CLI / validator

| Artifact | Path |
|----------|------|
| Package | `src/football_analytics/calibration/` |
| Policy | `configs/calibration/calibration_contract_policy.yaml` |
| Coords | `configs/calibration/pitch_coordinate_system.yaml` |
| Validator | `scripts/check_calibration_contracts.py` → `/home/fdoblak/workspace/calibration_contract_checks/` |

CLI (contract-safe only — **no** `calibration run`):

- `football-analytics calibration contracts validate`
- `football-analytics calibration homography validate`
- `football-analytics calibration project validate`

## 10. Evaluator

Without reviewed calibration ground truth:

`NOT_EVALUATED_NO_REVIEWED_CALIBRATION_GROUND_TRUTH`

Null metrics + reasons. Synthetic fixtures ≠ football calibration accuracy.

## 11. Explicit non-claims

- No real calibration inference in this stage
- SV_kp / SV_lines not executed
- Attack direction unknown
- Projected position ≠ physical truth guarantee
- No running distance / sprint / heatmap yet

## 12. Gate / next

Gate: `PASS — CALIBRATION AND PITCH COORDINATE CONTRACTS ACTIVE`

Next (name only): `Aşama 8B — Saha Anahtar Noktası ve Çizgi Algılama Baseline`
