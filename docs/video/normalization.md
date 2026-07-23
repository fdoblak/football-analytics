# Safe video normalization (Stage 3C)

## Security model

- Local regular files only; network URL/protocol ingest rejected.
- Exact binary `/usr/bin/ffmpeg` (realpath allowlisted); never PATH lottery.
- `subprocess` with `shell=False`, argv list, sanitized env, stdin `DEVNULL`, `-nostdin`.
- Timeout + bounded stderr/progress; kill process group on timeout.
- Never `-y`; exclusive `.norm.lock` via `O_EXCL`; temp sibling then `os.replace`.
- Source pre/post size, SHA-256, device/inode, mtime_ns; mutation ⇒ reject.
- No user filter/extra argv passthrough; filters built internally only.
- Hardware acceleration disabled; CPU `libx264` only (no NVENC/GPU).
- Disk preflight against `ffmpeg_policy` free-space floors.
- Atomic receipt JSON (`0600`), no overwrite of existing outputs.

## Planner (pure)

`plan_normalization` maps a Stage 3A/3B `VideoProbe` + policy into a fingerprintable
`NormalizePlan` and `PlannedNormalization` flags. No subprocess or filesystem writes.

Stable reason codes include `CONTAINER_NOT_CANONICAL`, `VIDEO_CODEC_NOT_CANONICAL`,
`PIXEL_FORMAT_NOT_CANONICAL`, `DIMENSIONS_REQUIRE_NORMALIZATION`,
`ROTATION_REQUIRES_BAKE`, `SAR_REQUIRES_NORMALIZATION`,
`FRAME_RATE_REQUIRES_NORMALIZATION`, `AUDIO_REQUIRES_TRANSCODE`,
`AUDIO_WILL_BE_DROPPED`, `ALREADY_CANONICAL`.

`required=false` only with `ALREADY_CANONICAL`.

## Invocation shape

```text
/usr/bin/ffmpeg
  -hide_banner -nostdin -loglevel error
  -i [--] <absolute_source>
  -map 0:v:N [-map 0:a:M]
  [-vf transpose|hflip|vflip|scale|setsar]
  -c:v libx264 -pix_fmt yuv420p -preset <policy> -crf <policy> -threads <policy>
  [-r num/den -vsync cfr]   # only when CFR conversion performed
  -c:a copy|aac | -an
  [-movflags +faststart]
  [--] <temp_output>
```

## Resource limits (policy `ffmpeg_policy`)

| Key | Default intent |
|-----|----------------|
| maximum_parallel_normalizations | 1 |
| timeout_base_seconds / per media second | bounded wall clock |
| maximum_stderr_bytes / maximum_progress_bytes | capped capture |
| minimum_free_space_bytes | hard floor before encode |
| video_crf / video_preset / ffmpeg_threads | CPU encode profile |

## Conformance

`validate_normalized_output` compares container/codec/pix_fmt/dims/rotation/SAR/audio
and duration drift. Failure ⇒ exit 5; caller does not publish.

## CLI

```text
football-analytics video normalize \
  --source <local-video> \
  --output <normalized-mp4> \
  --policy configs/video/ingest_policy.yaml \
  [--expected-source-sha256 <hex>] \
  [--execute] \
  [--contain-root <root>] \
  [--receipt-dir <dir>]
```

Default is dry-run (`planned` / `skipped` receipt, no output file).
`--execute` requires `--expected-source-sha256`.

Exit: 0 planned/skipped/succeeded · 1 policy reject · 2 usage · 3 integrity ·
4 ffmpeg fail · 5 conformance.

## Stage 3D boundary

Frame extraction / `video_time_us` mapping is **not** Stage 3C. VFR→CFR conversions
set `requires_stage3d_mapping` on the receipt for later work.
