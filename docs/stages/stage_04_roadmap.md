# Stage 4 — Broadcast shot & camera analysis (contracts → detection)

Stage 3 (`video-ingest-v0.3.1`) closed the safe video input layer. Stage 4 adds
broadcast structure needed for single-`target_player` evidence routing.

Stage 4 does **not** invent player metrics or claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **4A** | Shot boundaries & camera-view classification contracts | Canonical Arrow contracts, typed models, suitability semantics, validators, synthetic fixtures, docs. | **CLOSED** |
| **4B** | Shot boundary detection baseline & evaluation | Rule-based OpenCV streaming detector, evaluation, fixtures, CLI, validator. | **CLOSED** |
| **4C** | Camera-view classification baseline | View/framing/graphics/motion/playability suitability classifiers (rule-based). | **CLOSED** |
| **4D** | Segment fusion, playability routing, Stage 4 closure | Interval fusion → `analysis_windows`, routing policy, review queue, safety eval, Stage 4 completion docs. | **CLOSED** |

Stage 4 is closed as a broadcast-understanding baseline. See
`docs/stages/stage_04_completion.md`.

## Product link

Camera/shot contracts and analysis windows mark which intervals may feed
tracking, calibration, ReID, and distance metrics — and which must be excluded
(replay, crowd, graphics, …).

## Out of scope for Stage 4 (closed)

- Torch / learned camera or replay models
- SoccerNet download or model execution
- Real match labeling campaigns
- Inventing `camera_position` / unsupported view families without evidence
- Continuous automation / Codex supervisor
- **Stage 5** (detection contracts) — not started

## Next stage (name only)

`Stage 5A — Player, goalkeeper, referee, and ball detection contracts`
