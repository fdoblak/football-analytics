# Stage 3D completion — Frame timeline, time mapping & Stage 3 close

## 1. Purpose

Deterministic streaming frame timeline (`frames` v1) with PTS→`video_time_us`
mapping, optional materialization, Stage 3 gate close.

## 2. Starting SHA

`f38ea05cf1483cb9d2805ee13678b58c73c12663`
(`Implement safe FFmpeg video normalization`)

## 3. Baseline

549 pytest PASS; ruff/black/isort/mypy PASS; video contracts/probe/normalization
PASS; runtime/data/stage-cache/storage/registries/secrets PASS;
`check_project.py --profile local --quick` → PASS_WITH_WARNINGS (git metadata).

## 4. Reused helpers

Stage 3A–3C types/policy/validation/fixtures/ffprobe/ffmpeg/services;
Stage 2 `sha256_file`, `write_json_record`, contract parquet/registry/compiler;
Stage 1 path containment.

## 5. Frames contract

Unchanged `schemas/data/v1/frames.json` v1 fields:
`video_time_us`, `pts`, `frame_index`, `duration_us`, `is_key_frame`, `decode_status`.

## 6. New schemas

- `schemas/video/frame_timeline_receipt.schema.json`
- `schemas/video/frame_artifact_manifest.schema.json` (header; rows in JSONL)

## 7. Streaming Parquet

`write_contract_parquet_streaming` in `data/parquet.py` — iterator of
RecordBatch/Table, per-batch validate, `pq.ParquetWriter` zstd, atomic publish.

## 8. FFprobe frames

Line-oriented compact dump on FFmpeg 4.4.2:
`-show_frames -print_format compact=nk=1:p=0` with bounded line size,
`shell=False`, process-group cleanup. Stream index from probe/normalization.

## 9. Time mapping

`time_mapping.py` — Decimal/rational PTS mapping; quality
`exact|good|degraded|unreliable|failed`. Never invent from index/fps.

## 10. Materialization

`frame_extraction.py` — sampled/all_frames require `--execute-materialize`;
PNG (policy) + JSONL manifest.

## 11. Policy

`frame_timeline_policy` in `configs/video/ingest_policy.yaml` + loader validation.
Runtime root: `/home/fdoblak/workspace/frame_timeline_checks/`.

## 12. CLI

`football-analytics video frames --source … --output-dir … --mode …`

## 13. Validator

`scripts/check_frame_timeline.py`

## 14. Tests

New Stage 3D tests: **23** across
`test_time_mapping`, `test_frame_timeline`, `test_frame_extraction`,
`test_frame_timeline_service` (+ contracts schema-count assertion 6→8).
Total pytest after Stage 3D: **572** (was 549).

## 15–16. Quality / Build

Ruff/Black/isort/mypy PASS; `python -m build` PASS; wheel includes
frame_timeline modules; sdist grafts `configs/video` + `schemas/video`
(new receipt/manifest schemas); `git diff --check` PASS.

## 17. RISK-029

**Mitigated for the Stage 3D frame timeline streaming write path** (batched
ParquetWriter; no full-frame pylist on timeline write). Remains **open** for
general contract semantic validation pylist paths and the post-write materialize
metadata join (`pq.read_table(…).to_pylist()` bounded by `maximum_frames`).

## 18. Tag planned

Annotated `video-ingest-v0.3.0` (Stage 2 convention: `foundation-v0.1.0`).

## 19. Runtime report

`/home/fdoblak/workspace/frame_timeline_checks/frame_timeline_validation_20260723T074303Z.json`

## 20. Gate

`PASS — DETERMINISTIC FRAME TIMELINE ACTIVE; STAGE 3 CLOSED`

## 21. Next stage (name only)

`Aşama 4A — Shot Sınırları ve Kamera Görüşü Sınıflandırma Sözleşmeleri`

## Status

**CLOSED** (this stage).
