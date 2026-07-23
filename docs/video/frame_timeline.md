# Frame timeline & time mapping (Stage 3D)

Stage 3D builds a **streaming** `frames` v1 Parquet timeline from a local
normalized (or accepted) video, mapping packet PTS → `video_time_us` with
exact rationals. Optional image materialization is explicit and off by default.

## Modes

| Mode | Default | Images | Flag |
|------|---------|--------|------|
| `timeline_only` | yes | no | — |
| `sampled` | no | yes | `--execute-materialize` |
| `all_frames` | no | yes | `--execute-materialize` |

## Mapping rules

- `video_time_us = round_half_even(pts × time_base × 1e6)` via `Decimal`
- Missing PTS → `decode_status=skipped`, **no** invent from `index×fps`
- Duplicate PTS → keep + warn
- Non-monotonic PTS → `decode_status=unknown`, carry prior mapped time
- VFR: never invent CFR index timeline

## Mapping quality taxonomy (receipt `schema_version=2`)

Quality is derived from frame stats **plus** normalization-receipt evidence.
Execution failure uses receipt `status` (`failed`/`rejected`); it is **not** a
`mapping_quality` value. Failed/rejected receipts use `mapping_quality=not_available`.

| Value | Meaning |
|-------|---------|
| `exact_identity` | Proven identity (e.g. normalize skipped / already canonical; no time rewrite) |
| `timestamp_preserved` | Transcode may have occurred; PTS preserved; no frame-rate resampling |
| `derived_with_constant_offset` | Mapping uses a **proven** constant offset (µs); never invent offset |
| `derived_with_resampling` | CFR force / VFR→CFR / drop-dup / significant non-monotonic / stage-3D remap required |
| `uncertain` | Missing norm receipt, conflicting metadata, incomplete PTS, or insufficient evidence |
| `not_available` | Invented index/fps mapping, no usable frames, or execution did not produce a mapping |

Legacy v0.3.0 values (`exact`/`good`/`degraded`/`unreliable`/`failed`) are read via
`coerce_mapping_quality` / `normalize_legacy_receipt_payload`: without provenance,
legacy qualities become `uncertain` (`failed` → `not_available`). No blind map
`exact` → `exact_identity`.

## Streaming I/O

FFprobe compact line stream → Arrow batches → `write_contract_parquet_streaming`
(zstd, atomic temp→final). Does not load all frames as one Python list for the
timeline write path.

## CLI

```bash
football-analytics video frames \
  --source /abs/normalized.mp4 \
  --output-dir /home/fdoblak/workspace/frame_timeline_checks/run \
  --mode timeline_only|sampled|all_frames \
  [--execute-materialize] [--sample-every N] \
  [--normalization-receipt PATH] \
  [--expected-source-sha256 HEX]
```

## Outputs

- `frames.parquet` — canonical `frames` v1 contract
- `frame_timeline_receipt.json` — Stage 3D receipt (`schema_version=2`)
- optional `frame_artifact_manifest.json` + `frame_artifacts.jsonl` + `frames/`

## Validator

`scripts/check_frame_timeline.py` — synthetic generate → probe → normalize →
timeline_only + sampled → cleanup.

Runtime root: `/home/fdoblak/workspace/frame_timeline_checks/`
