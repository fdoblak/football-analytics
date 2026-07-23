# Stage 5 — Player / official / ball detection (contracts → baseline)

Stage 4 (`broadcast-understanding-v0.4.0`) closed safe analysis-window routing.
Stage 5 adds the detection layer needed before tracking / identity / ball
analysis for the single-`target_player` product.

Stage 5 does **not** invent player metrics or claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **5A** | Player, goalkeeper, referee, and ball detection contracts | Arrow sidecars, taxonomy/policy, bbox transforms, receipts, validators, SoccerNet matrix, synthetic tests. **No inference.** | **CLOSED** |
| **5B** | Player/official detection baseline, model selection, evaluation | Ultralytics YOLO11n person → human/unknown; IoU eval harness; adapters; bounded smoke. Ball deferred. | **CLOSED** (with findings) |
| **5C** | Ball detection baseline | YOLO11n COCO sports ball (id 32) + tiling/merge; ball/unknown only. | **CLOSED** (with findings) |
| **5D** | Human role classification baseline | Weightless HSV/kit clustering → player/GK/ref/staff/unknown via `detection_attributes`; abstention-first. | **CLOSED** (with findings) |
| **5E** | Detection fusion, quality gates, Stage 5 close | Fuse human+ball+role; pipeline receipt; operational quality; sampled review; Stage 5 tag. | **CLOSED** (with findings) |

## Product link

Detection contracts separate visual entity boxes from football roles and record
whether a frame was processed or skipped so downstream tracking never confuses
“no players found” with “frame not run”. Stage 5B emits generic humans only;
Stage 5C emits ball boxes without ownership claims; Stage 5D may refine human
`role_label` conservatively without inventing identity or teams; Stage 5E fuses
those artifacts into one validated bundle.

## Stage 5 closed

Gate: `PASS_WITH_FINDINGS — DETECTION PIPELINE ACTIVE; STAGE 5 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Tag: `detection-baseline-v0.5.0`

See `docs/stages/stage_05_completion.md`.

## Findings carried forward

- AGPL-3.0 Ultralytics distribution risk (`evaluation_only`) — reused in 5B/5C
- `NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH` (humans) and
  `NOT_EVALUATED_NO_REVIEWED_BALL_GROUND_TRUTH` (balls)
- `NOT_EVALUATED_NO_REVIEWED_HUMAN_ROLE_GROUND_TRUTH` (roles)
- `NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH` (fused pipeline)
- GPU host gate may remain unverifiable in agent contexts
- Small-object ball recall on real broadcast footage not yet validated
- Role kit heuristics are technical only; real football role accuracy not validated

## Next stage (name only)

`Aşama 6A — Çok Nesneli Takip ve Track Yaşam Döngüsü Sözleşmeleri`
