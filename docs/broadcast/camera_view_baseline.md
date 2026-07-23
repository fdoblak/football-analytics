# Camera-view classification baseline

Stage 4C delivers a **rule-based** OpenCV sample classifier for broadcast
camera-view axes. It does not train models, download SoccerNet weights, or claim
Opta labels.

## Config

`configs/broadcast/camera_view_baseline.yaml`

- Strict loader + fingerprint: `load_camera_view_config` /
  `camera_config_fingerprint`
- Pitch HSV ranges are **heuristic broadcast greens**, not stadium ground truth
- Thresholds are tuned on **development** fixtures only; `frozen_evaluation` is
  frozen for gates

## Supported axes (only)

| Axis | Classes |
|------|---------|
| `view_family` | `main_broadcast`, `player_isolation`, `graphics`, `unknown` |
| `framing_scale` | `wide`, `medium`, `close_up`, `unknown` |
| `camera_motion` | `static`, `pan`, `zoom`, `compound`, `unstable`, `unknown` |
| `graphics_status` | `none`, `partial_overlay`, `dominant_overlay`, `full_screen`, `unknown` |
| `playability` | `playable`, `partially_playable`, `non_playable`, `uncertain` |

**Always left `unknown` (not invented):** `camera_position`, `replay_status`.

Contract `confidence` is always **null**; heuristic scores live in
`provenance_json.heuristic_score`.

## Limitation

**One `camera_view_segment` per shot.** Intra-shot view changes are not split in
Stage 4C.

## Pipeline

1. Deterministic equal-time samples in `[start+margin, end-margin)` using
   timeline PTS (`camera_sampling`) — no fps invention
2. Per-sample features at analysis resolution (`camera_features`): pitch green
   coverage/spread, entropy, edges, overlay-like fraction, temporal diff,
   optional Farneback flow summary (finite-only)
3. Per-sample multi-axis scores with abstain→`unknown`
4. Shot aggregation via coverage + disagreement (not naive majority alone)
5. Conservative suitability rule IDs in provenance

## CLI

```bash
football-analytics broadcast camera classify \
  --source /abs/video.mp4 \
  --timeline /abs/frames.parquet \
  --shots /abs/shot_segments.parquet \
  --output-dir /home/fdoblak/workspace/camera_view_checks/run \
  --config configs/broadcast/camera_view_baseline.yaml

football-analytics broadcast camera evaluate \
  --predictions /abs/camera_view_segments.parquet \
  --ground-truth /abs/ground_truth.json \
  --output /abs/metrics.json \
  --config configs/broadcast/camera_view_baseline.yaml
```

## Validator

`scripts/check_camera_view_baseline.py` materializes synthetic fixtures under
`/home/fdoblak/workspace/camera_view_checks/` and enforces frozen thresholds:

- supported view/framing macro F1 ≥ 0.85
- graphics macro F1 ≥ 0.90
- motion macro F1 ≥ 0.80
- playability macro F1 ≥ 0.90
- unsafe playable FP rate = 0
- OOD abstention rate ≥ 0.80
- deterministic repeat exact

Report: `/home/fdoblak/workspace/camera_view_checks/camera_view_validation_<UTC>.json`

## RISK-029

Sample-frame OpenCV decode (seek-by-index) is used for features. This does **not**
close RISK-029: `validate_broadcast_bundle` pylist paths remain.
