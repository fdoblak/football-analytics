# Human-ball interaction semantics (Stage 10A)

## Result levels

`candidate` → `provisional` → `confirmed` (audit trail required)

Also: `contested`, `unknown`, `not_evaluable`, `rejected`

Automatic producers stop at `provisional`.

## Evidence spaces

- Image-space distance ≠ pitch-space distance
- Pitch distance usable only with valid calibration **and** known grounded ball
- Unknown/airborne ball state blocks pitch-space contact/possession evidence

## Observation states

Distinguish observed vs predicted/interpolated for both human and ball.
Predicted/interpolated alone cannot carry ownership.

## Ball status

Primary / ambiguous / missing are distinct.
Missing observation is coverage/`not_evaluable`, not `possession=false`.

## Event readiness

Contracts may hold evidence references for later event extractors.
Stage 10A produces **zero** event counts.
