# Stage 6 completion — Tracking baseline

Stage 6 closes as a **technical multi-object tracking baseline** for the
single-`target_player` product. It does **not** claim real-match football
tracking accuracy, player identity, ReID, possession, events, or physical
metrics.

## Sub-stages

| ID | Deliverable | Status |
|----|-------------|--------|
| **6A** | Multi-object tracking & track lifecycle contracts | CLOSED |
| **6B** | Human multi-object tracking baseline & evaluation | CLOSED (with findings) |
| **6C** | Ball tracking baseline / lost-ball handling | CLOSED (with findings) |
| **6D** | Human+ball track fusion, quality gates, Stage 6 close | CLOSED (with findings) |

## What Stage 6 is

- Canonical track contracts unchanged:
  - `track_observations` v1 `9ca2f7af…`
  - `track_summaries` v1 `7b04e31d…`
  - `track_lifecycle` v1 `613cd81e…`
  - `detections` v1 `04ae8dd7…`
- Human MOT + ball MOT baselines consuming Stage 5 detection bundles
- Fused tracking bundle + pipeline receipt + operational quality + sampled review

## What Stage 6 is not

- Track ID ≠ player identity / jersey / name
- ReID or cross-shot / camera-exit identity merge
- Primary ball ≠ true-ball guarantee
- Possession / pass / shot / duel events
- Distance, speed, sprint, or other physical metrics
- Target-player selection or final performance report
- Production football tracking accuracy approval

## Evaluation honesty

Without reviewed football tracking ground truth:

`NOT_EVALUATED_NO_REVIEWED_TRACKING_GROUND_TRUTH`

Synthetic fixtures prove pipeline integrity only. Human/ball-specific
not-evaluated codes remain in provenance.

## Open findings carried forward

- No reviewed human/ball/tracking match GT
- Upstream AGPL Ultralytics detection baselines (Stage 5)
- Fragmentation under occlusion / cut / replay remains operationally gated only
- Stage 6 is a technical tracking baseline, not the final product

## Tag

`tracking-baseline-v0.6.0`

## Next stage (name only)

`Aşama 7A — ReID, Kimlik Kanıtı ve Hedef Futbolcu Sözleşmeleri`
