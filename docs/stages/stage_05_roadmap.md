# Stage 5 — Player / official / ball detection (contracts → baseline)

Stage 4 (`broadcast-understanding-v0.4.0`) closed safe analysis-window routing.
Stage 5 adds the detection layer needed before tracking / identity / ball
analysis for the single-`target_player` product.

Stage 5 does **not** invent player metrics or claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **5A** | Player, goalkeeper, referee, and ball detection contracts | Arrow sidecars, taxonomy/policy, bbox transforms, receipts, validators, SoccerNet matrix, synthetic tests. **No inference.** | **CLOSED** |
| **5B** | Player/official detection baseline, model selection, evaluation | Detector choice, synthetic/real eval harness, adapters — **not started**. | **NOT STARTED** |

## Product link

Detection contracts separate visual entity boxes from football roles and record
whether a frame was processed or skipped so downstream tracking never confuses
“no players found” with “frame not run”.

## Out of scope for Stage 5A (closed)

- Torch / Ultralytics inference
- SoccerNet download or model execution
- Real match labeling campaigns
- Team / jersey / `target_player` identity
- Continuous automation / Codex supervisor
- **Stage 5B** — not started

## Next stage (name only)

`Stage 5B — Player and official detection baseline, model selection, and evaluation`
