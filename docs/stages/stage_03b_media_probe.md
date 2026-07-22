# Stage 3B completion — Safe FFprobe media validation

## 1. Purpose

Bind Stage 3A contracts to a real, safe FFprobe execution/parsing layer with
policy validation, mutation detection, atomic outputs, CLI, and tests.

## 2. Starting SHA

`dfdc9c0b1360c44e7d1712d0f103a4f0c1efc933`

## 3. Baseline

506 pytest PASS; video contracts PASS; runtime/data/stage-cache/storage PASS;
registries PASS_WITH_WARNINGS (known); secrets=0.

## 4. Reused helpers

Stage 3A types/policy/validation/fixtures; Stage 2 `sha256_file`,
`write_json_record`, `hash_canonical_json`; Stage 1 path containment.

## 5. FFprobe binary/version

`/usr/bin/ffprobe` (allowlisted realpath); version captured into probe provenance.

## 6. Safe subprocess

`shell=False`, argv list, sanitized env, stdin DEVNULL, process-group kill on timeout.

## 7. Resource limits

Policy `ffprobe_policy`: 30s timeout, 1MiB stdout, 64KiB stderr, stream/json caps;
`count_frames: false`; raw JSON not persisted.

## 8. Parser

`probe_parser.py` pure mapping; `N/A`/`0/0`/`N:D`/`N/D`; domain `ProbeError` codes.

## 9. Time conversion

Integer microseconds via `Decimal` ROUND_HALF_EVEN; no float drift assumption.

## 10. Stream selection

Attached pic ignored; max area; min index; audio default-disposition then min index.

## 11. Rotation/SAR/DAR

Rotate tag + display matrix; normalize to {0,90,180,270}; SAR/DAR rational (`:` or `/`).

## 12. Policy validation

`media_validation.py` — accepted/rejected with stable error/warning codes.

## 13. Source mutation

Pre/post size+SHA+dev/ino/mtime; mismatch ⇒ `SOURCE_MUTATED_DURING_PROBE`.

## 14. Atomic outputs

`video_probe.json`, `media_validation.json`, `probe_execution_receipt.json`
via `write_json_record` (0600, no overwrite).

## 15. CLI

`football-analytics video probe --source … --output-dir … [--policy] [--contain-root]`

## 16. Validator

`scripts/check_video_probe.py`

## 17. Synthetic E2E

Tiny CFR / CFR+audio under `/home/fdoblak/workspace/video_probe_checks/`; cleanup verified.

## 18–19. Tests

New Stage 3B tests: **22** (`test_ffprobe_runner`, `test_probe_parser`,
`test_probe_service` covering runner/parser/policy/CLI/security/integration).
Total pytest after Stage 3B: **528** (was 506).

## 20–21. Quality / Build

Ruff/Black/isort/mypy PASS; `python -m build` PASS; wheel/sdist include probe
modules; sdist grafts `configs/video` + `schemas/video`; no media/runtime JSON.

## 22. Validators

video contracts + video probe + storage/runtime/data/stage-cache/registry/secrets/project.

## 23. Runtime report

`/home/fdoblak/workspace/video_probe_checks/video_probe_validation_20260722T215236Z.json`
(validator writes `video_probe_validation_<UTC>.json` by default).

## 24. Fixture cleanup

Session dirs removed in tests/validator `finally`.

## 25. Protected packages

Unchanged (torch/pyarrow/… pins).

## 26. Changed files

See commit list (ffprobe/parser/validation/service, policy, CLI, scripts, tests, docs).

## 27. Findings

Pre-existing registry license/access warnings only.

## 28. Acceptance

Stage 3B acceptance criteria met.

## 29. Gate

`PASS — SAFE FFPROBE MEDIA VALIDATION ACTIVE`

## 30. Next stage (name only)

Aşama 3C — Güvenli Video Normalizasyonu
