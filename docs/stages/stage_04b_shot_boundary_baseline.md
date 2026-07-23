# Stage 4B completion — Shot boundary detection baseline & evaluation

## 1. Purpose

Deliver a rule-based OpenCV streaming shot-boundary detector, evaluation harness,
synthetic fixtures, CLI, validator, and docs — without ML / SoccerNet / Opta.

## 2. Starting SHA

`c141f0f8b47327e46bd793184e12dcbbc1fdd362`
(`Update Stage 4 roadmap documentation`)

## 3. Baseline

- pytest before: **616 passed**
- OpenCV: **5.0.0** (no pip install)

## 4. Delivered

| Area | Paths |
|------|-------|
| Config | `configs/broadcast/shot_boundary_baseline.yaml` |
| Modules | `shot_config`, `shot_features`, `shot_detection`, `shot_evaluation`, `shot_service`, `shot_fixtures` |
| CLI | `broadcast shots detect` / `broadcast shots evaluate` |
| Validator | `scripts/check_shot_boundary_baseline.py` |
| Tests | `tests/broadcast/test_shot_*.py` |
| Docs | this file + `docs/broadcast/shot_boundary_baseline.md` |

## 5. Algorithm (honest)

- Features: luma MAE, HSV hist correlation distance, edge magnitude change
- Hard cut: high instantaneous score, not gradual-like
- Gradual: elevated window mean + low peak sharpness; dissolve/fade heuristics
- Flash: brief spike returning to similar luma baseline within max duration
- Times from timeline parquet; segments cover `[0, duration)`

## 6. Gate targets

- hard-cut F1 ≥ 0.95
- gradual F1 ≥ 0.80
- overall F1 ≥ 0.90
- negative FP = 0
- deterministic repeat

## 7. RISK-029

Streaming decode path for features is in place. Do **not** mark RISK-029 closed:
`validate_broadcast_bundle` pylist paths remain.

## 8. Next stage (name only)

`Aşama 4C — Camera-view classification baseline` (**CLOSED** in Stage 4C doc)

## 9. Out of scope

Torch models, SoccerNet download/execution, real-match labeling campaigns,
continuous automation / Codex supervisor, Opta scraping.
