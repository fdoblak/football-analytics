# Duels / competitive events semantics (Stage 12)

Project-generated Opta-style metric definitions. **Not** official Opta. Real football accuracy is not validated.

## Separations

| Claim | Truth |
|-------|-------|
| Nearby opponent alone | ≠ take-on |
| Nearest / track switch alone | ≠ duel outcome |
| Monocular aerial | → candidate / unknown / not_evaluable; no exact 3D height |
| Long ball alone | ≠ clearance |
| Automatic baseline ceiling | `provisional` only |

## Evaluation

Without reviewed ground truth → `NOT_EVALUATED_NO_REVIEWED_DUELS_EVENTS_GROUND_TRUTH`.

## Scope

Single target player. Synthetic fixtures only in Stage 12.
