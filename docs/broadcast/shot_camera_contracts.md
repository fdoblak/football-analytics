# Shot boundaries and camera-view contracts

Stage 4A defines **contracts only** for broadcast shot structure and camera-view
classification. No shot detector, OpenCV/Torch inference, or real video evaluation
ships in this stage.

## Product role

Shot and camera segments are intermediate context for a single `target_player`
pipeline. They tell later stages which time ranges are usable for:

- visibility / tracking coverage
- ReID continuity
- pitch calibration
- ball/event evidence
- distance and sprint measurement

They are **not** team or player performance results. Close-up, replay, crowd,
bench, studio, and full-screen graphics must not invent pitch metrics.

## Contracts

| Contract | PK | Interval |
|----------|----|----------|
| `shot_boundaries` | `(run_id, video_id, boundary_id)` | Instant `boundary_time_us` |
| `shot_segments` | `(run_id, video_id, shot_id)` | Half-open `[start_time_us, end_time_us)` |
| `camera_view_segments` | `(run_id, video_id, camera_segment_id)` | Half-open `[start_time_us, end_time_us)` |

All three FK to `videos` v1 via `(run_id, video_id)`. Canonical time unit is
integer microseconds (`video_relative`).

## Shot semantics

- Transition types are physical edit cues only: `hard_cut|dissolve|fade|wipe|flash|unknown`.
  Replay / playability are **not** transition types.
- Shot intervals are half-open; `duration_us == end_time_us - start_time_us` and
  must be `> 0`.
- Active production shots must not overlap. Gaps are allowed only when explained
  by `segment_status` (`gap_coverage` / `incomplete`).
- First/last boundary ids may be null.
- `timeline_mapping_quality` reuses Stage 3 `MappingQuality`:
  `exact_identity|timestamp_preserved|derived_with_constant_offset|derived_with_resampling|uncertain|not_available`.
- Missing PTS/frame indices stay null — never invent from index/fps.

## Camera-view axes (separate enums)

Do not collapse these into one mega-label:

- `view_family`, `framing_scale`, `camera_position`, `camera_motion`
- `replay_status`, `graphics_status`, `playability`
- `calibration_suitability`, `tracking_suitability`, `target_identity_suitability`

Suitability values: `suitable|conditionally_suitable|unsuitable|unknown`.

## Single-player suitability rules

- Tracking, calibration, and target-identity suitability are **independent**.
- Wide live broadcast may be suitable for tracking/calibration while only
  conditionally suitable for identity.
- Close-up may help identity/ReID but is usually unsuitable for pitch distance.
- Replay is not live match time — cannot be fully `playable` as live.
- `crowd|bench|studio|graphics` view families must be `non_playable`.
- `graphics_status=full_screen` ⇒ `non_playable` and tracking/calibration
  `unsuitable`.
- Unknown views should reduce coverage rather than invent metrics.
- Camera classification alone does **not** prove the target player is on pitch.

## Confidence / coverage / review

- Confidence `[0,1]` or null (never fake `0` when unknown).
- Coverage required `[0,1]` (denominator: classified duration over segment).
- NaN/Infinity forbidden.
- `review_status`: `unreviewed|reviewed|accepted|rejected|needs_review`.
- `evidence_ref` / `evidence_refs` use safe identifiers only — no invented URIs.
- Sources: `model|rule|manual|imported`.

## Validation entry points

- Table: `football_analytics.data.validation.validate_table`
- Broadcast bundle: `football_analytics.broadcast.validate_broadcast_bundle`
- Optional inclusion in `validate_contract_bundle` when broadcast tables are present
- CLI: `scripts/check_broadcast_contracts.py`

## Out of scope (Stage 4B+)

Shot boundary detectors, camera classifiers, histogram/SSIM/optical-flow
pipelines, and real-match evaluation.
