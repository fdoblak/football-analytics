# Stage 16-R4-F1 / FINAL — Release integrity + consolidated delivery

## Gate

`PASS_WITH_FINDINGS — VIDEO-BACKED TECHNICAL PREVIEW CONSOLIDATED; RELEASE TREE CLEAN; VIDEO-EVENT ACCURACY NOT VALIDATED`

Tag: `single-player-analytics-technical-preview-v0.16.1` (does not move `v0.16.0`).

## Registry integrity resolution

| Prior error | Resolution |
|-------------|------------|
| `expected 3 third-party repos / found 5` | Lock already contained 5 intentional repos (incl. SoccerTrack-v2 + TeamTrack). Validator no longer hard-codes `3`; lock counts are canonical. |
| `sn_depth working tree dirty` | Untracked `__pycache__` under ZoeDepth `data/` quarantined; HEAD unchanged (`9f6636fa…`). `PYTHONDONTWRITEBYTECODE=1` in registry validator. |

Two consecutive `--verify-repos` runs: **PASS_WITH_WARNINGS**, integrity errors = 0, all locked repos clean.

## Final delivery

Canonical customer outputs live only under:

```text
artifacts/final_delivery/
```

Contents: README, summary JSON/PNG, annotated TeamTrack proof MP4, evidence + cleanup manifests, checksums.

## Real-video re-validation

TeamTrack `F_20200220_1_0330_0360` source SHA `fd42dbe6…` / size `13086969` verified before inference.

Results (CUDA YOLO11n tiled): detections **14374**, F1 **≈0.620**, target coverage **1.0**, mean IoU **≈0.675** — matches prior pilot arithmetic.
