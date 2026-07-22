# Stage 2C — Canonical PyArrow data contracts and schema migrations

**Date:** 2026-07-22
**Start HEAD:** `b2ef601b74f4f78748558729fdc6d6ed6bc6fa08`
**Commit message:** `Add canonical Arrow contracts and schema migrations`

## 1. Purpose

Define versioned canonical tabular contracts, PyArrow compilation, strict validation,
atomic Parquet I/O, explicit migrations (detections 0→1), migration receipts, CLI, and
standalone validator — without pipelines, cache orchestration, or Stage 2D work.

## 2. Starting SHA

`b2ef601b74f4f78748558729fdc6d6ed6bc6fa08` on clean `main` with
`local main = origin/main = ls-remote main`.

## 3. Baseline

pytest **215** passed; ruff/black/isort/mypy PASS; runtime foundation PASS; storage PASS;
registry PASS_WITH_WARNINGS (license/access); secrets 0; pip check clean.

## 4. Prior review

Reused Stage 2B `canonical_json` / `sha256_file` / `write_json_record` / `validate_run_id`
and path-containment patterns. CLI extended with lazy data imports. No eager PyArrow on
`football_analytics` root import.

## 5. Single-source approach

Authoritative JSON contract specs under `schemas/data/v0|v1/` compiled to PyArrow schemas
in Python. No dual hand-maintained Python field lists.

## 6. Schema registry

`configs/data/schema_registry.yaml` + `schemas/data/schema_registry.schema.json`.
Rejects duplicate contracts/versions, missing specs, dangling/cyclic migration edges,
unknown status, and contract/spec name mismatches.

## 7. Contract list

**Nine v1 canonical contracts:** videos, frames, detections, track_observations,
track_summaries, calibrations, team_assignments, jersey_observations, events.

**Legacy fixture:** detections v0 (compatibility fixture — not claimed as production history).

## 8–9. Field / PK / FK summary

Documented in `docs/data/canonical_contracts.md`.

## 10. BBox / time / null policy

- BBox: image_pixels, xyxy, half-open; x2>x1, y2>y1, non-negative origins.
- Time: video-relative microseconds; wall-clock UTC only where explicitly documented.
- Null: distinct from empty string / zero; `quality_flags` prefer empty list over null.
- Confidence: `[0.0, 1.0]`; no NaN/Infinity.

## 11. PyArrow compiler

`football_analytics.data.compiler` — allowlisted Arrow types, bounded nesting, deterministic
field order, path containment, symlink rejection.

## 12. Fingerprint

SHA-256 over canonical UTF-8 JSON of normalized contract specification (semantic rules
included; machine paths excluded). Stamped into Parquet metadata.

## 13–14. Structural / semantic / bundle validation

Strict column set/order/types/nullability; PK uniqueness; semantic enums/bounds;
cross-table FK and summary consistency via `validate_contract_bundle`.

## 15. Parquet I/O

Atomic same-directory write, zstd, no default overwrite, symlink/path rejection,
metadata contract/version/fingerprint verification on read.

## 16–17. Legacy v0 and migration 0→1

xywh→xyxy; `class_id` map; `is_interpolated=false`; `quality_flags=[]`;
source file hash unchanged; destination no-overwrite; failure cleans partial destination.

## 18. Receipt

`schemas/data/migration_receipt.schema.json` — atomic JSON, no overwrite, sanitized paths.

## 19. CLI

`football-analytics contracts list|show|fingerprint|validate|migrate`.
Bare `football-analytics --version` preserved without colliding with
`contracts … --version N`.

## 20. Validator

`scripts/check_data_contracts.py` — registry compile, fingerprints, optional synthetic
roundtrip + migration smoke; exit codes 0/1/2/3.

## 21–22. Test counts

- **New tests:** 96 (80 `tests/data` + 10 CLI contracts + 6 validator script)
- **Total pytest:** **311** passed

## 23. Synthetic E2E

PASS under `/home/fdoblak/workspace/data_contract_checks/` with fixture cleanup.

## 24. Runtime report path

`/home/fdoblak/workspace/data_contract_checks/data_contract_validation_20260722T192341Z.json`
(Git-outside; fixture cleaned).

## 25. Fixture cleanup

Confirmed `fixture_cleaned=true`; fixture directory removed after smoke.

## 26. Quality tools

ruff / black / isort / mypy / pip check PASS.

## 27. Build

`python -m build` PASS. Sdist includes `src/football_analytics/data/**`,
`configs/data/schema_registry.yaml`, and `schemas/data/**` via `MANIFEST.in`.
No synthetic Parquet/runtime JSON/secrets in artifacts.

## 28. Storage / runtime / registry / security

runtime foundation PASS; storage PASS; registry PASS_WITH_WARNINGS; secrets findings=0.

## 29. Protected packages

Unchanged: pyarrow 25.0.0, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
torchaudio 2.11.0+cu128, numpy 2.2.6, pandas 2.3.3, opencv-python 5.0.0.93,
opencv-python-headless 5.0.0.93, ultralytics 8.4.91, SoccerNet 0.1.62.

## 30. Changed files (summary)

configs/data, schemas/data, src/football_analytics/data, cli.py,
scripts/check_data_contracts.py, tests/data|package|scripts, docs/data|stages|decisions|risks,
README.md, .gitignore, MANIFEST.in.

## 31. Findings

- api.github.com Cursor proxy 403 (carry-over)
- Registry license/access warnings (carry-over)
- GPU AGENT_CONTEXT_GPU_UNVERIFIABLE (carry-over)
- RISK-029: semantic validation may use pylist (memory pressure on huge tables)

## 32. Acceptance

All Stage 2C acceptance criteria met; Stage 2D not started.

## 33. Gate

`PASS_WITH_FINDINGS — CANONICAL DATA CONTRACTS ACTIVE`

## 34. Next stage (name only)

Aşama 2D — Stage Arayüzü, Cache, Project Validator, CI ve Foundation Kapanışı
