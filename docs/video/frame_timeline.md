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
- `frame_timeline_receipt.json` — Stage 3D receipt
- optional `frame_artifact_manifest.json` + `frame_artifacts.jsonl` + `frames/`

## Validator

`scripts/check_frame_timeline.py` — synthetic generate → probe → normalize →
timeline_only + sampled → cleanup.

Runtime root: `/home/fdoblak/workspace/frame_timeline_checks/`
