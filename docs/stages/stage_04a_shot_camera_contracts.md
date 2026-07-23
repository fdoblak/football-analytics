# Stage 4A completion — Shot boundaries & camera-view contracts

## 1. Purpose

Establish immutable broadcast contracts for shot boundaries, shot segments, and
camera-view classification — plus typed models, suitability semantics, validators,
and synthetic fixtures — without running detectors or real video.

## 2. Starting SHA

`7e992c8aa7f68bb340b58ce8d620793106e04229`
(`Align frame time mapping quality taxonomy (Stage 3D-F1).`)

## 3. Baseline

- pytest: **588 passed** → **616 passed** after Stage 4A
- Tags intact: `video-ingest-v0.3.0`, `video-ingest-v0.3.1`
- Canonical `frames.json` SHA unchanged:
  `8fd233af820aa6242d575a2005f6552e43b37dd535140f26858782cc116d0437`

## 4. Contracts

| Contract | Spec |
|----------|------|
| `shot_boundaries` | `schemas/data/v1/shot_boundaries.json` |
| `shot_segments` | `schemas/data/v1/shot_segments.json` |
| `camera_view_segments` | `schemas/data/v1/camera_view_segments.json` |

Registered in `configs/data/schema_registry.yaml` (12 active contracts total).
Synthetic 9-table Stage 2C bundle unchanged.

## 5. Package

`src/football_analytics/broadcast/`:

- `types.py` — enums + frozen dataclasses (`ShotBoundary`, `ShotSegment`,
  `CameraViewSegment`); reuses `MappingQuality` from `video.types`
- `contracts.py` — registry load / compile / fingerprints
- `validation.py` — `validate_broadcast_bundle(...)`

## 6. Semantics delivered

- Half-open shot/camera intervals; positive duration; active no-overlap
- Boundary ascending order per video; optional null edge boundary ids
- Camera axes kept separate (view/framing/position/motion/replay/graphics/…)
- Hard rules: crowd/bench/studio/graphics → non-playable; full-screen graphics
  → unsuitable tracking/calibration; replay ≠ fully live-playable
- Confidence/coverage bounds; no NaN; no fake evidence URIs

## 7. Validator / tests / docs

- `scripts/check_broadcast_contracts.py` → runtime report under
  `/home/fdoblak/workspace/broadcast_contract_checks/`
- Tests: `tests/broadcast/`
- Docs: `docs/broadcast/shot_camera_contracts.md`, this file,
  `docs/stages/stage_04_roadmap.md`

## 8. Gate

`PASS — SHOT AND CAMERA VIEW CONTRACTS ACTIVE`

## 9. Next stage (name only)

`Aşama 4B — Shot Sınırı Algılama Baseline ve Değerlendirme`

## 10. Out of scope (unchanged)

Detectors, real video, SoccerNet inference, Opta scraping, GUI/API,
Codex/continuous automation.
