# Stage 13 roadmap — Target event ledger and metrics

## Status

| Sub-stage | Status |
|-----------|--------|
| **13A** Broadcast context backlog (replay / live / camera_position / dedup) | **CLOSED** |
| **13B** Attack direction resolver (period/half scoped) | **CLOSED** |
| **13C** Canonical target event ledger | **CLOSED** |
| **13D** Target event metrics aggregation | **CLOSED** |
| **13E** Fusion + Stage 13 close | **CLOSED** |

**Stage 13 status: CLOSED** (see `docs/stages/stage_13_completion.md`)

## Scope delivered

- 13A: Conservative replay candidate baseline; live eligibility; supported-class camera_position; duplicate suppression
- 13B: Explainable period/half attack-direction resolver; conflict → unknown; no team-name invention
- 13C: Append-only `target_event_ledger` + `event_revisions`; temporal dedup; lineage; leakage guard
- 13D: Coverage-aware product event metrics for one target player
- 13E: Fusion package + Stage 13 close gate

## Explicitly not delivered

- Real video / Opta accuracy validation
- Invented live when replay uncertain
- Invented real team names
- Automatic `confirmed` events without review

## Next (do not start without explicit prompt)

Stage 14+ — wait for explicit user prompt.
