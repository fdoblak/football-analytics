# Shot boundary detection baseline (Stage 4B)

Rule-based OpenCV streaming detector for broadcast shot transitions.
Contracts from Stage 4A (`shot_boundaries`, `shot_segments`) are filled by this
baseline — no ML models, no SoccerNet download.

## Config

`configs/broadcast/shot_boundary_baseline.yaml`

- Analysis size (default 160×90), feature weights (luma / HSV hist / edge)
- Hard-cut and gradual thresholds, flash suppression, peak merge, min shot duration
- Decode / resource limits, evaluation matching tolerance, deterministic seed
- Loader: `load_shot_boundary_config` + `shot_config_fingerprint` (canonical JSON hash)
- Security: `overwrite_allowed`, `symlinks_allowed`, `network_sources_allowed` all false

Thresholds are tuned on **development** fixtures only. Evaluation fixtures are
regenerated with fixed generators but must not be used for threshold search.

## Pipeline

1. `shot_features` — stream decode, resize, consecutive-frame features; times from
   `frames.parquet` timeline (never invent PTS from fps alone for contract times)
2. `shot_detection` — weighted score → hard cut / gradual (dissolve|fade|unknown),
   flash suppression, peak merge, min duration → `ShotBoundary` + covering
   `ShotSegment` rows (`detection_source=rule`, `confidence=null`)
3. `shot_service` — path safety, pre/post source SHA, write parquet + receipt +
   optional `scores.jsonl`, `validate_broadcast_bundle`
4. `shot_evaluation` — greedy one-to-one match within tolerance → P/R/F1, timing,
   FP/min, over/under segmentation

## CLI

```bash
football-analytics broadcast shots detect \
  --source /abs/video.mp4 \
  --timeline /abs/frames.parquet \
  --output-dir /abs/out \
  --config configs/broadcast/shot_boundary_baseline.yaml \
  --contain-root /home/fdoblak/workspace/shot_boundary_checks

football-analytics broadcast shots evaluate \
  --predictions /abs/shot_boundaries.parquet \
  --ground-truth /abs/ground_truth.json \
  --output /abs/metrics.json
```

## Fixtures / validator

Runtime root: `/home/fdoblak/workspace/shot_boundary_checks/`

`scripts/check_shot_boundary_baseline.py` builds synthetic hard-cut / dissolve /
fade / flash / static / pan fixtures, runs detect+evaluate, checks:

| Gate | Threshold |
|------|-----------|
| Hard-cut F1 | ≥ 0.95 |
| Gradual F1 | ≥ 0.80 |
| Overall F1 | ≥ 0.90 |
| Negative-control FP | = 0 |
| Deterministic repeat | exact boundary times+types |

## RISK-029

Detection uses **streaming OpenCV decode** (no full-res frame buffer). Bundle
validation still uses `to_pylist()` on small synthetic tables; RISK-029 remains
open for general large-table semantic validation.
