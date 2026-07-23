# Stage 4C completion — Camera-view classification baseline & evaluation

## 1. Purpose

Deliver a rule-based OpenCV camera-view classifier, evaluation harness,
synthetic fixtures, CLI, validator, and docs — without ML / SoccerNet / Opta.

## 2. Starting SHA

`0efc96a4d8815263f73d0f1e3d14ec9596cf8568`
(`Implement shot boundary baseline and evaluation`)

## 3. Baseline

- pytest before: **633 passed**
- OpenCV: **5.0.0** (no pip install)

## 4. Delivered

| Area | Paths |
|------|-------|
| Config | `configs/broadcast/camera_view_baseline.yaml` |
| Modules | `camera_config`, `camera_sampling`, `camera_features`, `camera_classification`, `camera_evaluation`, `camera_service`, `camera_fixtures` |
| CLI | `broadcast camera classify` / `broadcast camera evaluate` |
| Validator | `scripts/check_camera_view_baseline.py` |
| Tests | `tests/broadcast/test_camera_*.py` |
| Docs | this file + `docs/broadcast/camera_view_baseline.md` |

## 5. Algorithm (honest)

- Equal-time samples from timeline PTS within edge margins
- Features: pitch HSV heuristic coverage/spread, entropy, edges, overlay-like
  fraction, frame diff, optional Farneback flow summary
- Abstain→`unknown` on low margin / disagreement; OOD crowd-like → abstain
- Always unknown: `camera_position`, `replay_status`
- `confidence=null`; `heuristic_score` in provenance
- **One camera segment per shot** (documented limitation)

## 6. Gate targets

- view/framing macro F1 ≥ 0.85
- graphics macro F1 ≥ 0.90
- motion macro F1 ≥ 0.80
- playability macro F1 ≥ 0.90
- unsafe playable FP rate = 0
- OOD abstention ≥ 0.80
- deterministic repeat

## 7. RISK-029

Sample-frame decode path is in place. Do **not** mark RISK-029 closed:
`validate_broadcast_bundle` pylist paths remain.

## 8. Next stage (name only)

`Aşama 4D` — not started

## 9. Out of scope

Torch models, SoccerNet download/execution, real-match labeling campaigns,
continuous automation / Codex supervisor, Opta scraping, inventing
`camera_position` / `replay_status` / unsupported view families.
