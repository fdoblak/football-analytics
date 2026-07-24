# Stage 9E — Physical metric fusion, quality gates, Stage 9 close

## Scope

Fuse Stage 9B–9D outputs into one **confirmed-target** physical analysis package
with per-metric status / coverage / provenance. Does **not** recompute core
distance/speed/heatmap math.

## Outputs

- `target_physical_metric_summary.json`
- `target_physical_metric_quality.json`
- `target_physical_metric_receipt.json`
- `target_physical_metric_evaluation.json`

## Evaluation

`NOT_EVALUATED_NO_REVIEWED_TARGET_PHYSICAL_METRIC_GROUND_TRUTH`

## Visual policy

No multi-visual git commits; no final customer visual in Stage 9E.

## CLI

```bash
python -m football_analytics.cli physical integrate --output-dir ... --fixture-smoke
python -m football_analytics.cli physical evaluate --output ... --fixture-smoke
python -m football_analytics.cli physical pipeline-validate
```

## Runtime

`/home/fdoblak/workspace/physical_metric_pipeline_checks/`
