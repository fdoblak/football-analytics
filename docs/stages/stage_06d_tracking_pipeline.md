# Stage 6D — Human+ball tracking fusion, quality gates, Stage 6 close

## Status

**CLOSED** with findings (2026-07-23).

Gate: `PASS_WITH_FINDINGS — TRACKING PIPELINE ACTIVE; STAGE 6 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Purpose

Fuse Stage 5E detection bundle refs with Stage 6B human and Stage 6C ball track
artifacts into one deterministic, validated tracking bundle with operational
quality gates and a sampled review queue.

**No new tracker/model.** No ReID, possession, events, physical metrics,
identity, jersey, or human–ball relationship tables.

## Fusion rules

- Align `run_id` / `video_id` / source SHA / timeline / detection /
  analysis-window fingerprints or hard-fail (no silent fill).
- Namespace human vs ball tracks; remap ball `track_id` after `max(human)+1`
  for compound uniqueness across the bundle.
- Merge observations / summaries / lifecycle without cross-entity FK violations.
- Preserve predicted/interpolated flags (`physical_metric_ineligible`,
  `event_ineligible`); unknown human roles; ambiguous ball not upgraded.
- No cross-cut continuation; terminated tracks do not reopen.
- Recount from tables must match pipeline receipt; atomic no-overwrite;
  failure cleans partial outputs.

## Outputs

Under `/home/fdoblak/workspace/tracking_pipeline_checks/`:

- `track_observations.parquet`
- `track_summaries.parquet`
- `track_lifecycle.parquet`
- `ball_primary_candidates.json` (remapped; not true-ball identity)
- `tracking_pipeline_receipt.json`
- `tracking_quality_report.json`
- `tracking_bundle_manifest.json`
- `review_queue.json` (sampled)

## CLI

```
football-analytics tracking integrate ...
football-analytics tracking validate [--frames N] [--config ...]
```

Humans / ball / contracts commands remain available.

## Eval

`NOT_EVALUATED_NO_REVIEWED_TRACKING_GROUND_TRUTH` — operational quality only.
Human/ball-specific not-evaluated codes retained in receipt provenance.

## Findings (open)

1. No reviewed football tracking GT — real match accuracy not claimed.
2. Track ID is not player identity; no ReID / cross-shot stitch.
3. Primary ball candidate is not a true-ball guarantee.
4. Possession / events / physical metrics / target-player report not produced.
