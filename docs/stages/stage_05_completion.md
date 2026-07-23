# Stage 5 completion — Detection baseline

Stage 5 closes as a **technical detection baseline** for the single-`target_player`
product. It does **not** claim real-match football accuracy, Opta parity,
target-player identity, tracking, or events.

## Sub-stages

| ID | Deliverable | Status |
|----|-------------|--------|
| **5A** | Player/official/ball detection contracts (Arrow sidecars, taxonomy, policy, validators) | CLOSED |
| **5B** | Human detection baseline (YOLO11n person → human/unknown) | CLOSED (with findings) |
| **5C** | Ball detection baseline (YOLO11n sports ball + tiling/merge) | CLOSED (with findings) |
| **5D** | Human role classification baseline (weightless HSV/kit → roles; abstention-first) | CLOSED (with findings) |
| **5E** | Detection fusion, quality gates, Stage 5 close | CLOSED (with findings) |

## What Stage 5 is

- Canonical `detections` v1 unchanged
  (`04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6`)
- Sidecars: `detection_frame_status`, `detection_attributes`
- Fused bundle + pipeline receipt + operational quality report + sampled review
- `human detection != player`; unknown/abstention preserved

## What Stage 5 is not

- Target-player selection / identity / ReID / jersey OCR
- Multi-object tracking
- Team IDs
- Events or physical metrics
- Production football accuracy approval

## Evaluation honesty

Without reviewed football ground truth:

`NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH`

Synthetic fixtures prove pipeline integrity only.

## Open findings carried forward

- AGPL-3.0 Ultralytics (`evaluation_only`) — 5B/5C
- Human / ball / role accuracy not validated on reviewed match GT
- Kit-color role heuristics under broadcast variance
- GPU host gate may remain unverifiable in agent contexts

## Tag

`detection-baseline-v0.5.0`

## Next stage (name only)

`Aşama 6A — Çok Nesneli Takip ve Track Yaşam Döngüsü Sözleşmeleri`
