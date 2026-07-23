# Stage 3C completion — Safe video normalization

## 1. Purpose

Execute Stage 3A normalize **plans** with a safe FFmpeg runner: non-destructive,
no overwrite, aspect-safe, CPU `libx264`, receipted, and policy-bounded.

## 2. Starting SHA

`fadc9e3a9f3d6e570f154c644286b7fb24bb3ee3`
(`Implement safe FFprobe media validation`)

## 3. Baseline

528 pytest PASS (Stage 3B close); video contracts PASS; ruff/black/isort/mypy PASS;
runtime/data/stage-cache/storage PASS; project check PASS_WITH_WARNINGS (dirty tree).

## 4. Reused helpers

Stage 3A/3B types/policy/validation/fixtures/ffprobe/probe_service;
Stage 2 `sha256_file`, `write_json_record`, `hash_canonical_json`;
Stage 1 path containment.

## 5. FFmpeg binary/version

`/usr/bin/ffmpeg` and `/usr/bin/ffprobe` (allowlisted realpaths);
`4.4.2-0ubuntu0.22.04.1`; `libx264` capability check.

## 6. Safe subprocess

`shell=False`, argv list, sanitized env, stdin DEVNULL, `-nostdin`,
process-group kill on timeout. Never `-y`.

## 7. Resource limits

Policy `ffmpeg_policy`: timeouts, stderr/progress caps, threads=2,
`maximum_parallel_normalizations=1`, free-space floors, CRF/preset.

## 8. Planner

`normalization.py` pure planner → `NormalizePlan` + `PlannedNormalization`.

## 9. Transforms

Internal filters only: transpose / hflip+vflip / scale / setsar=1;
`-r` + `-vsync cfr` only when frame-rate conversion performed.

## 10. Audio

`copy_if_present_else_drop`: AAC copy; else transcode to AAC; absent if none.

## 11. Conformance

`normalization_validation.py` — container/codec/pix_fmt/dims/rotation/SAR/audio/drift.

## 12. Source mutation

Pre/post snapshot; mismatch ⇒ `SOURCE_MUTATED_DURING_NORMALIZATION`.

## 13. Atomic publish

Temp sibling `*.tmp.<token>.mp4` → fsync → `os.replace` → fsync parent;
exclusive `*.norm.lock` via `O_EXCL`.

## 14. Receipt

`normalization_receipt.schema.json` / `NormalizationReceipt` (provenance stage `3C`).

## 15. CLI

`football-analytics video normalize --source … --output … [--execute] …`

## 16. Validator

`scripts/check_video_normalization.py`

## 17. Synthetic E2E

Tiny MPEG-4 / CFR under `/home/fdoblak/workspace/video_normalization_checks/`;
cleanup verified.

## 18–19. Tests

New Stage 3C tests: **21** (`test_normalization_planner`,
`test_ffmpeg_runner`, `test_normalization_service`).
Total video pytest after Stage 3C: **84**.
Total pytest after Stage 3C: **549** (was 528).

## 20–21. Quality / Build

Ruff/Black/isort/mypy PASS; `python -m build` PASS; wheel/sdist include
normalization modules + `normalization_receipt.schema.json`; sdist grafts
`configs/video` + `schemas/video`; no media/runtime JSON.

## 22. Validators

video contracts + video normalization + storage/runtime/data/stage-cache/CI PASS;
`check_project.py --profile local --deep` → `PASS_WITH_WARNINGS` (dirty tree only).

## 23. Runtime report

`/home/fdoblak/workspace/video_normalization_checks/video_normalization_validation_20260723T012833Z.json`

## 24. Fixture cleanup

Session dirs removed in tests/validator `finally`.

## 25. Protected packages

Unchanged (no new pip packages; no GPU/NVENC):
torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · torchaudio 2.11.0+cu128 ·
numpy 2.2.6 · pandas 2.3.3 · opencv-python / headless 5.0.0.93 ·
ultralytics 8.4.91 · SoccerNet 0.1.62 · pyarrow 25.0.0.

## 26. Out of scope / Stage 3D

Frame extraction and time mapping remain Stage 3D (not started).

## 27. Gate

`PASS — SAFE VIDEO NORMALIZATION ACTIVE`

## 28. Next stage (name only)

Aşama 3D — Deterministik Frame Çıkarma, PTS Zaman Haritası ve Aşama 3 Kapanışı

## Status

**CLOSED** (this stage).
