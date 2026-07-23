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
| **4D** | (next) | Not started. | not started |

## Product link

Camera/shot contracts mark which intervals may feed tracking, calibration, ReID,
and distance metrics — and which must be excluded (replay, crowd, graphics, …).

## Out of scope for Stage 4C

- Torch / learned camera models
- SoccerNet download or model execution
- Real match labeling campaigns
- Inventing `camera_position` / `replay_status` / unsupported view families
- Continuous automation / Codex supervisor
