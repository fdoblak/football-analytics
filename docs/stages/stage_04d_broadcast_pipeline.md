# Stage 4D — Broadcast segment fusion, playability routing, Stage 4 closure

**Status:** CLOSED (implementation complete; commit/tag deferred to operator)

## Delivered

- Canonical `analysis_windows` contract (`schemas/data/v1/analysis_windows.json`)
  registered in schema registry (13 active contracts).
- Interval-sweep fusion (`segment_fusion.py`) with gap/conflict visibility.
- Versioned routing policy (`configs/broadcast/broadcast_routing_policy.yaml`)
  and `playability.py` eligibility routing.
- Integrate pipeline writing `analysis_windows.parquet`, `review_queue.json`,
  `pipeline_receipt.json`.
- Safety evaluator (`broadcast_evaluation.py`) with zero unsafe-FP gates on
  synthetic reviewed GT.
- CLI: `football-analytics broadcast integrate …`
- Validator: `scripts/check_broadcast_pipeline.py`
- Docs: routing guide + Stage 4 completion.

## Safety gates (synthetic)

- unsafe live-event / physical / calibration FPs = 0
- non-playable → eligible FP = 0
- manual-review recall for unknown/conflict = 1.0
- overlap = 0, unexplained gap = 0
- deterministic repeat exact

## Out of scope

- Real match videos / learned models
- Replay detector
- Stage 5 detection contracts
- Claiming player-performance analytics readiness
