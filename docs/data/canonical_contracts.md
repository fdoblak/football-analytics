# Canonical data contracts (Stage 2C)

Authoritative sources live under `schemas/data/v*/` and are indexed by
`configs/data/schema_registry.yaml`. Python compiles them to PyArrow schemas;
contracts are **not** duplicated as hand-written Python field lists.

## Contracts (v1)

| Contract | PK | Notable FKs |
|----------|----|-------------|
| videos | run_id, video_id | — |
| frames | run_id, video_id, frame_index | → videos |
| detections | run_id, video_id, frame_index, detection_id | → frames |
| track_observations | run_id, video_id, frame_index, track_id | → frames; nullable → detections |
| track_summaries | run_id, video_id, track_id | — |
| calibrations | run_id, video_id, frame_index, calibration_id | → frames |
| calibration_features | run_id, video_id, frame_index, feature_id | → frames |
| calibration_segments | run_id, video_id, segment_id | — (time-scoped H) |
| projected_positions | run_id, video_id, frame_index, projection_id | → frames |
| target_trajectory_samples | run_id, video_id, target_player_id, sample_id | → frames |
| target_trajectory_segments | run_id, video_id, target_player_id, trajectory_segment_id | — |
| trajectory_gaps | run_id, video_id, target_player_id, gap_id | — |
| physical_metric_results | run_id, video_id, target_player_id, metric_result_id | — |
| team_assignments | run_id, video_id, assignment_id | → track_summaries |
| jersey_observations | run_id, video_id, frame_index, observation_id | → track_summaries |
| events | run_id, video_id, event_id | actor tracks → track_summaries (bundle) |

## Policies

- **run_id**: Stage 2B canonical format
- **BBox**: image pixels, xyxy, half-open (`x2>x1`, `y2>y1`, origins ≥ 0)
- **Time**: video-relative microseconds (`video_time_us`), not wall-clock UTC for frame timing
- **Confidence**: `[0.0, 1.0]`; null means unknown where nullable
- **Null vs empty**: null ≠ 0 ≠ `""`; `quality_flags` uses empty list (no null items)
- **Identifiers**: ASCII safe IDs; no path separators / `..` / credentials
- **NaN/Infinity**: forbidden
- **Fingerprint**: SHA-256 of normalized contract JSON (fields/PK/FK/rules/metadata; excludes description/source_path)

## Parquet metadata

- `football_analytics.contract_name`
- `football_analytics.contract_version`
- `football_analytics.schema_fingerprint`
- `football_analytics.created_by_version`

Compression: `zstd`. Atomic write; default no-overwrite; symlink rejection.

## Validation levels

1. Structural (columns/types/nullability/order/PK)
2. Semantic (rules in each JSON spec)
3. Bundle FK / bbox-vs-video / actor integrity

## CLI

```bash
football-analytics contracts list
football-analytics contracts show detections --version 1
football-analytics contracts fingerprint detections --json
football-analytics contracts validate detections path.parquet
football-analytics contracts migrate detections src.parquet dst.parquet --from-version 0 --to-version 1
```

## Limitations

No video ingest, model inference, or Stage 2D cache/orchestration in this stage.
