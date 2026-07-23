# Broadcast routing and analysis windows

Stage 4D fuses shot segments with camera-view segments into non-overlapping
`analysis_windows`, then applies a versioned routing policy that decides
per-task eligibility.

## Pipeline

```text
frames + shot_boundaries + shot_segments + camera_view_segments
  → interval-sweep fusion
  → playability / routing policy
  → analysis_windows.parquet
  → review_queue.json
  → pipeline_receipt.json
```

## Interval fusion

- Shots are the temporal backbone.
- Camera edges inside a shot split atomic windows.
- Gaps (no camera) become `unknown` / `uncertain` windows with `CAMERA_GAP`
  (never silently filled).
- Conflicting overlapping camera labels mark conflict + manual review.
- Camera intervals extending outside their shot raise a fusion integrity error.
- Adjacent windows merge only when all decision fields (including camera ids)
  match.

## Routing axes

Independent eligibility values (`eligible` | `conditionally_eligible` |
`ineligible` | `unknown`) for:

- tracking
- calibration
- identity
- ball_analysis
- live_event
- physical_metric

Policy file: `configs/broadcast/broadcast_routing_policy.yaml` (strict load +
fingerprint).

## Critical semantics

| Condition | Effect |
|-----------|--------|
| Full-screen / dominant graphics or non-playable | Pitch/live tasks ineligible |
| Wide main-broadcast playable | Tracking/calibration candidate |
| Close-up / player-isolation | Identity candidate; calibration/physical ineligible |
| Replay **unknown** | Live-event never auto-`eligible`; manual review |
| Replay confirmed | Live-event + physical ineligible |
| Mapping quality uncertain / not_available | Physical metric unsafe |
| Unknown / low coverage / conflict / gap | Manual review; never auto-eligible |

Eligibility of a window does **not** prove the target player is present.

## Manual review queue

Non-canonical JSON (`schemas/broadcast/manual_review_queue.schema.json`) lists
uncertain windows with reason codes, priority, and safe source refs. Items stay
`pending` until a human resolves them — routing never auto-accepts.

## CLI

```bash
football-analytics broadcast integrate \
  --timeline /abs/frames.parquet \
  --boundaries /abs/shot_boundaries.parquet \
  --shots /abs/shot_segments.parquet \
  --camera-views /abs/camera_view_segments.parquet \
  --output-dir /abs/runtime-out \
  --policy configs/broadcast/broadcast_routing_policy.yaml
```

## Honesty bounds

Synthetic fixture metrics measure routing safety, not real-match accuracy.
Replay detection is not implemented; confirmed-replay cases must be supplied as
synthetic camera metadata.
