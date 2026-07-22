# Stage 2B — Runtime identity, config, logging, hash, environment records

**Date:** 2026-07-22
**Start HEAD:** `7f9ec52fc4ee0eda49c8918f1aa3400712ab4e08`
**Target commit:** `Add runtime identity and provenance foundation`

## 1. Purpose

Establish canonical run IDs, deterministic config merge/fingerprint, SHA-256 helpers,
secret-safe structured logging, environment/run-context records, atomic JSON writers,
schemas, CLI helpers, and a foundation validator — without pipelines, PyArrow table
schemas, or Stage 2D orchestration.

## 2. Starting SHA

`7f9ec52fc4ee0eda49c8918f1aa3400712ab4e08` (`Record Stage 2A GitHub synchronization`)
with `main` = `origin/main` = `git ls-remote` and clean working tree.

## 3. Baseline results

| Check | Result |
|-------|--------|
| pytest | 124 passed |
| ruff/black/isort/mypy | PASS |
| pip check | clean |
| storage | PASS |
| registry | PASS_WITH_WARNINGS (documented license/access) |
| secrets | 0 findings |

## 4. Existing contract review

- Stage 1D `run_manifest.schema.json` remains archive-oriented; not replaced.
- Archive `run_id` policy pattern extended to accept Stage 2B IDs **and** Stage 1 fixtures.
- `archive_safety.write_json_atomic` / `sha256_file` left in place for archive tools;
  Stage 2B adds package-level `core.*` APIs for runtime foundation (no unsafe duplication of
  archive policy semantics).
- CLI `--version` / `info` preserved; foundation subcommands added.
- Secret policy and storage `paths.yaml` unchanged in role.

## 5. Run ID contract

`run_YYYYMMDDTHHMMSSffffffZ_<12_hex>` — UTC, sortable, injectable clock/suffix,
path/shell-safe validation (`football_analytics.core.run_id`).

## 6. Config precedence

`defaults < user YAML < allowlisted env < explicit overrides`.

## 7. Config schema

`configs/project/defaults.yaml` + `schemas/resolved_config.schema.json` (`schema_version: 1`).

## 8. Fingerprint

SHA-256 canonical JSON; record fields `algorithm`, `canonicalization_version`, `digest`.

## 9. Hashing

Streaming file hash + deterministic directory manifest (`football_analytics.core.hashing`).

## 10. Redaction

`[REDACTED]` marker; nested keys; bearer/token; URL userinfo (`core.redaction`).

## 11. Structured logging

`core.structured_logging` — human + JSONL; no root-logger mutation; symlink-safe paths.

## 12. Environment record

`schemas/environment_record.schema.json`; allowlisted `importlib.metadata` versions;
GPU=`AGENT_CONTEXT_GPU_UNVERIFIABLE` without torch import.

## 13. Run context

`schemas/run_context.schema.json`; `initialize_run_context` transactional layout under a
caller-provided `runs_root` (tests/synthetic only).

## 14. Atomic writer

`write_json_record` — no-overwrite default, fsync, `0600`, containment.

## 15. CLI

`run-id`, `config validate|fingerprint`, `environment show` (+ existing `--version`/`info`).

## 16. Validator

`scripts/check_runtime_foundation.py` with `--synthetic-run` under
`/home/fdoblak/workspace/foundation_checks/`.

## 17. Schema files

- `schemas/resolved_config.schema.json`
- `schemas/environment_record.schema.json`
- `schemas/run_context.schema.json`

## 18. New tests

≥50 new tests under `tests/core/`, `tests/package/test_cli_runtime_foundation.py`,
`tests/scripts/test_check_runtime_foundation.py`.

## 19. Total test result

Recorded at gate close: **215 passed** (unittest/pytest).

## 20. Synthetic E2E

Validator `--synthetic-run` creates then removes a timestamped tree under
`foundation_checks/`; report JSON written outside Git.

## 21. Runtime report path

`/home/fdoblak/workspace/foundation_checks/runtime_foundation_validation_20260722T170311Z.json`

## 22–24. Quality / build / validators

- Ruff / Black / isort / mypy: PASS
- pytest / unittest: **215 passed**
- `check_runtime_foundation` (+ `--synthetic-run`): PASS
- storage: PASS; registry: PASS_WITH_WARNINGS (license/access); secrets: 0; pip check: clean
- `python -m build`: PASS (core modules present; no secrets/runtime/workspace artifacts)

## 25. Protected packages

Unchanged vs Stage 2A pins (torch/vision/audio, numpy, pandas, opencv*, ultralytics,
SoccerNet, pyarrow). No new package installs in Stage 2B.

## 26. Changed files (expected)

`src/football_analytics/core/**`, CLI, defaults/schemas, validator script, tests, docs
(ADR-0004, runtime_foundation, stage_02b, risk_register, README), archive_policy pattern,
`.gitignore`.

## 27. Findings

- Cursor `api.github.com` proxy 403 carry-over (RISK-024); Git smart HTTPS used for push.
- Registry license/access `PASS_WITH_WARNINGS` (pre-existing, documented).
- GPU remains `AGENT_CONTEXT_GPU_UNVERIFIABLE`.

## 28. Acceptance criteria

All Stage 2B brief acceptance items met at close (see final agent report).

## 29. Gate

`PASS_WITH_FINDINGS — RUNTIME FOUNDATION ACTIVE`

Findings are non-critical and documented (API proxy; registry license warnings; GPU unverifiable).

## 30. Next stage (name only)

Aşama 2C — Canonical PyArrow Veri Sözleşmeleri ve Schema Migration
