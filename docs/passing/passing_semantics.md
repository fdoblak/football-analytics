# Passing / reception / progression semantics (Stage 11)

Project-generated Opta-style metric definitions. **Not** official Opta. Real football accuracy is not validated.

## Separations

| Claim | Truth |
|-------|-------|
| Owner change alone | ≠ completed pass |
| Cut / replay / hard gap | → no pass candidate |
| Attack direction unknown | → 1→2 / 2→3 `not_evaluable` |
| Penalty-area presence | ≠ box touch |
| Box touch candidate | requires possession/contact + pitch mapping + playable |
| Automatic baseline ceiling | `provisional` only |

## Neutral zones

Without resolved attack direction, only Goal A / Middle / Goal B geometric zones are used.

## Attack direction

Resolved only from manual or config evidence. Conflict → `unknown`. Never invented.
