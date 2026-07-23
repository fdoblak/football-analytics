# Stage 7 roadmap — ReID, identity, target player

## Goal

Attribute tracks and evidence to **one paying `target_player`** so downstream
metrics never silently report the wrong athlete.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **7A** | Identity evidence, ReID candidate links, track identity assignments, target-player request/receipt, policy, metric eligibility, review/audit, eval stubs | **in progress / contracts** |
| **7B** | Appearance embedding + tracklet ReID baseline | not started |
| **7C+** | Team/jersey evidence producers, confirmation UX, production identity run | later |

## Explicit non-goals (until later stages)

- Face recognition / biometric identity (**forbidden**)
- Auto target selection from lineup alone
- Physical track merge from ReID links
- Customer metric reports

## Dependencies

- Stage 6 tracking bundle (`track_*`)
- `team_assignments` / `jersey_observations` as **reference-only** evidence sources
- Manual review queue + append-only audit

## Next after 7A

`Aşama 7B — Görünüş Embedding ve Tracklet ReID Baseline`
