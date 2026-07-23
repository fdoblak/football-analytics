# Stage 5A completion — Player, official, and ball detection contracts

## 1. Purpose

Establish immutable detection contracts for humans (player/official roles) and
ball — plus taxonomy/policy, preprocessing transforms, frame processing status,
run receipts, validators, and synthetic fixtures — without running detectors or
real video.

## 2. Starting SHA

`d6befd45fb30f47b18c8006258a0be3452254ec9`
(`broadcast-understanding-v0.4.0`)

## 3. Baseline

- pytest: **674 passed** → **710 passed** after Stage 5A
- Tag intact: `broadcast-understanding-v0.4.0`
- Canonical `detections.json` SHA unchanged:
  `957a41ca2ded9580bc18d39bc7902e133b34ec866077ccc944ab334b9e2681fd`

## 4. Decision — detections v1 unchanged

`detections` v1 already carries run/video/frame, detection id, model class,
bbox, confidence, and model id. It does **not** safely carry entity/role,
processing status, or preprocessing. **Keep v1 unchanged**; add sidecars.

## 5. Contracts

| Contract | Spec |
|----------|------|
| `detection_frame_status` | `schemas/data/v1/detection_frame_status.json` |
| `detection_attributes` | `schemas/data/v1/detection_attributes.json` |
| `detection_run_receipt` | `schemas/perception/detection_run_receipt.schema.json` |
| `preprocessing_transform` | `schemas/perception/preprocessing_transform.schema.json` |

Registered Arrow contracts: **15** total in `configs/data/schema_registry.yaml`.
Synthetic 9-table Stage 2C bundle unchanged.

## 6. Package

`src/football_analytics/perception/`:

- `types.py` — enums + frozen dataclasses
- `taxonomy.py` / `policy.py` — strict loaders + fingerprints + routing
- `transforms.py` — letterbox/stretch forward/inverse bbox
- `validation.py` — `validate_detection_bundle(...)`
- `contracts.py` — registry / JSON-schema helpers

## 7. Semantics delivered

- Entity vs role axes; person ≠ player
- Processed empty ≠ skipped/failed
- BBox xyxy half-open; NaN/Inf/zero-area rejected; inverse transform tested
- Analysis-window routing for human/ball eligibility
- Receipt count integrity; model SHA null when unavailable
- SoccerNet integration matrix (read-only)

## 8. Validator / tests / docs

- `scripts/check_detection_contracts.py` →
  `/home/fdoblak/workspace/detection_contract_checks/`
- Tests: `tests/perception/`
- Docs: `docs/perception/detection_contracts.md`,
  `docs/perception/soccernet_detection_integration.md`, this file,
  `docs/stages/stage_05_roadmap.md`

## 9. Gate

`PASS — DETECTION CONTRACTS ACTIVE`

## 10. Next stage (name only)

`Aşama 5B — Oyuncu ve Resmî Algılama Baseline, Model Seçimi ve Değerlendirme`

## 11. Out of scope (unchanged)

Detectors, real video, SoccerNet inference, Opta scraping, GUI/API,
Codex/continuous automation, Stage 5B.
