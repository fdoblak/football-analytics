# Safe FFprobe media probe (Stage 3B)

## Security model

- Local regular files only; network URL/protocol ingest rejected.
- Exact binary `/usr/bin/ffprobe` (realpath allowlisted); never PATH lottery.
- `subprocess` with `shell=False`, argv list, sanitized env, stdin closed.
- Timeout + bounded stdout/stderr; kill process group on timeout.
- Source pre/post size, SHA-256, device/inode, mtime_ns; mutation ⇒ reject.
- No arbitrary FFprobe option passthrough; `-count_frames` disabled by default.
- Atomic JSON outputs (`0600`), no overwrite; raw FFprobe JSON not persisted.

## Invocation

```text
/usr/bin/ffprobe
  -hide_banner -v error
  -print_format json
  -show_format -show_streams
  -protocol_whitelist file,crypto,data
  [--] <absolute_validated_source>
```

## Resource limits (policy `ffprobe_policy`)

| Key | Default |
|-----|---------|
| probe_timeout_seconds | 30 |
| maximum_stdout_bytes | 1 MiB |
| maximum_stderr_bytes | 64 KiB |
| maximum_stream_count | 32 |
| maximum_json_depth | 12 |
| count_frames | false |
| persist_raw_ffprobe_json | false |

## Mapping

Pure parser (`probe_parser.py`) maps FFprobe JSON → Stage 3A `VideoProbe`.
Rationals accept `N/D` and `N:D`; `0/0`/`N/A` → null; invalid den → error.
Durations → integer microseconds via `Decimal` (ROUND_HALF_EVEN).

## Stream selection

Same as Stage 3A: ignore attached pictures; max area; min index.
Audio: prefer default disposition, else lowest index. Audio optional.

## Frame count

Prefer `nb_frames`; else null (never invent `duration×fps`). No default `-count_frames`.

## VFR/CFR/unknown

Compare `r_frame_rate` vs `avg_frame_rate`; unequal ⇒ `vfr`; missing ⇒ `unknown`.

## Policy validation

Separate from parsing. Stable codes include
`UNSUPPORTED_VIDEO_CODEC`, `DIMENSIONS_OUT_OF_RANGE`, `DURATION_OUT_OF_RANGE`,
`NO_USABLE_VIDEO_STREAM`, `SOURCE_MUTATED_DURING_PROBE`, `PROBE_TIMEOUT`, …

## CLI

```text
football-analytics video probe \
  --source <local-video> \
  --output-dir <runtime-output> \
  --policy configs/video/ingest_policy.yaml \
  [--contain-root <root>]
```

Exit: 0 accepted · 1 policy reject · 2 config · 3 integrity · 4 ffprobe failure.

## TOCTOU

Pre/post hash+stat around FFprobe. Descriptor-based open is not required on this
host; residual race documented. Mutation ⇒ `SOURCE_MUTATED_DURING_PROBE`.

## Stage 3C boundary

Normalization / transcode execution is **not** Stage 3B.
