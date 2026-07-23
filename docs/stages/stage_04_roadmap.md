# Stage 4 — Broadcast shot & camera analysis (contracts → detection)

Stage 3 (`video-ingest-v0.3.1`) closed the safe video input layer. Stage 4 adds
broadcast structure needed for single-`target_player` evidence routing.

Stage 4 does **not** invent player metrics or claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **4A** | Shot boundaries & camera-view classification contracts | Canonical Arrow contracts, typed models, suitability semantics, validators, synthetic fixtures, docs. | **CLOSED** (this stage) |
| **4B** | Shot boundary detection baseline & evaluation | Detectors / heuristics / evaluation against labeled fixtures — **not started**. | not started |

## Product link

Camera/shot contracts mark which intervals may feed tracking, calibration, ReID,
and distance metrics — and which must be excluded (replay, crowd, graphics, …).

## Out of scope for Stage 4A

- Real shot detectors / OpenCV / Torch inference
- SoccerNet download or model execution
- Real match video evaluation
- Stage 4B implementation
- Continuous automation / Codex supervisor
