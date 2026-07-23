# Stage 6B — Human multi-object tracking baseline & evaluation

## 1. Purpose

Deterministic human MOT baseline that links Stage 5 `entity_type=human`
detections into `track_observations` / `track_summaries` / `track_lifecycle`
with Stage 6A lifecycle contracts.

**No ReID**, no cross-shot identity merge, no ball tracking, no team/jersey,
no events/physical metrics, no training.

## 2. Starting SHA

`387710c44783c7b9da533294defacc1444238968` (Stage 6A contracts)

## 3. Selection matrix

| Candidate | Availability | License | Determinism | CPU/4GB | Lifecycle fit | Adapter cost | Decision |
|-----------|--------------|---------|-------------|---------|---------------|--------------|----------|
| **In-repo IoU + constant-velocity + greedy assign** | Built here | Project | High | Excellent | Direct Stage 6A | Low | **SELECTED** |
| filterpy / SciPy | Present in `ai-dev` | BSD/MIT | High | Good | Would need wrappers | Medium | Available, **unused** (simplicity) |
| supervision ByteTrack | Present (`supervision` 0.29) | **AGPL** risk; deprecated toward `trackers` | Medium (FPS-oriented) | OK | Lifecycle mismatch | Medium–high | **REJECTED for 6B** |
| Ultralytics trackers (ByteTrack/BoT-SORT) | Present via ultralytics | AGPL | Medium | OK | Opaque | Medium | **REJECTED for 6B** |
| sn-tracking | `/home/fdoblak/projects/soccernet/sn-tracking` read-only | Check SoccerNet | Unknown | N/A | Future | High | **Future adapter only** |

## 4. Association & lifecycle

- Cost: `w_iou*(1−IoU) + w_motion*normalized_center_distance`
- Gate: IoU ≥ `iou_gate` **or** center distance ≤ `motion_center_gate_px`
- Assignment: deterministic greedy one-to-one; ties by `track_id` then `detection_id`
- Birth → `tentative` → `confirmed` after N associations
- Miss → `lost`; recover within `max_lost_gap_us`; terminate beyond / on shot cut / non-playable / window boundary
- Short gap: optional `predicted` observations with `physical_metric_ineligible`
- Long occlusion: **new** track; terminated never reopens; no ReID
- Role: soft consistency; `unknown` not punished; conflict → review finding

## 5. Outputs

- `track_observations.parquet`, `track_summaries.parquet`, `track_lifecycle.parquet`
- `tracking_run_receipt.json`, `tracking_evaluation.json`, `tracking_quality_report.json`
- Atomic no-overwrite; hard fail cleans partial outputs
- Receipt counts must match tables

## 6. Evaluation / GT

Without reviewed human tracking GT:

`NOT_EVALUATED_NO_REVIEWED_HUMAN_TRACKING_GROUND_TRUTH`

Synthetic metrics are labeled and must not be claimed as football accuracy.
Frozen eval fixtures are separate from development fixtures.

## 7. Fingerprints (unchanged)

| Contract | Fingerprint |
|----------|-------------|
| track_observations v1 | `9ca2f7af…` |
| track_summaries v1 | `7b04e31d…` |
| detections v1 | `04ae8dd7…` |

## 8. CLI / validator

- `football-analytics tracking humans run`
- `football-analytics tracking humans evaluate`
- `football-analytics tracking contracts validate` (kept)
- `scripts/check_human_tracking_baseline.py` → `/home/fdoblak/workspace/human_tracking_checks/`

## 9. Explicit non-claims

- Track ID ≠ player identity
- No camera/shot ReID merge
- Target player not selected
- Real match tracking accuracy not validated
- Predicted/interpolated not physical measurements

## 10. Gate / next

Gate: `PASS_WITH_FINDINGS — HUMAN TRACKING BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Next (name only): `Aşama 6C — Top Takibi Baseline, Kayıp Top Yönetimi ve Değerlendirme`
