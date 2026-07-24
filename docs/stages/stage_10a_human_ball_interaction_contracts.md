# Stage 10A — Human-ball interaction, ownership, and contact-candidate contracts

## 1. Purpose

Define machine-readable contracts for future human-ball interaction metrics:

- ball control / possession hypotheses
- ball-contact candidates
- pass / reception / turnover / dribble / duel / clearance / box-touch **candidates** (evidence refs only)
- player-ball interaction coverage

**Contracts only.** No real possession, contact, or football-event inference.

## 2. Starting SHA

`1cd67c00324f7d7b46d50654bdf11e518064abf6` (`physical-metrics-baseline-v0.9.0`)

## 3. Core truth principles

These alone are **not** possession/contact evidence:

- nearest player to the ball
- bbox overlap
- single-frame proximity
- same-team visibility
- ball near a player
- primary ball track selected
- predicted ball/player position
- pixel distance alone

Every result carries an explicit level:

`candidate | provisional | confirmed | contested | unknown | not_evaluable | rejected`

Automatic baseline may produce at most **`provisional`**.
`confirmed` requires strong multi-evidence or scoped manual review.

## 4. New contracts

### Arrow

- `human_ball_proximity` v1
- `ball_contact_candidates` v1
- `possession_hypotheses` v1

### JSON

- `schemas/interaction/human_ball_interaction_request.schema.json`
- `schemas/interaction/human_ball_interaction_run_receipt.schema.json`
- `schemas/interaction/human_ball_interaction_evaluation.schema.json`
- `schemas/interaction/human_ball_interaction_quality.schema.json`
- `schemas/interaction/human_ball_interaction_manual_review_queue.schema.json`

### Policy

- `configs/interaction/human_ball_interaction_policy.yaml`

## 5. Separations

| Claim | Rule |
|-------|------|
| proximity ≠ contact | true |
| contact ≠ controlled possession | true |
| possession transition ≠ completed pass | true |
| ball leaving ≠ ball loss | true |
| approaching opponent ≠ duel | true |
| penalty presence ≠ box touch | true |
| ball near head ≠ aerial duel | true |
| direction change ≠ dribble/take-on | true |

## 6. Lifecycle

- Time base: `video_time_us` only; intervals `[start_us,end_us)`
- Terminate on shot cut, replay, non-playable, ball loss, hard gap, track termination
- No possession carry across hard gaps
- Multi-owner overlap → contested/ambiguous
- Owner change requires evidence
- Source observations immutable; decisions append-only
- Cross run/video scope → hard fail

## 7. Coverage

Separate: human observed, ball observed, joint observed, calibration, playable,
target confirmed, ambiguous-ball, contested, not-observed.

Missing ball ≠ possession=false. Low joint coverage → `not_evaluable`.

## 8. Evaluation

Without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_HUMAN_BALL_INTERACTION_GROUND_TRUTH`

## 9. Validator

`scripts/check_human_ball_interaction_contracts.py`

Runtime: `/home/fdoblak/workspace/human_ball_contract_checks/` (fixtures cleaned).

## 10. Explicit non-goals (10A)

- Real possession/contact/event inference
- Pass/dribble/duel/turnover/box-touch counts
- Nearest-player = owner
- Automatic confirmed possession
- New models/datasets/downloads
- Opta or real accuracy claims
- Stage 10B baseline
