# Stage 6C — Ball tracking baseline, lost-ball handling & evaluation

## 1. Purpose

Deterministic ball MOT baseline that links Stage 5 `entity_type=ball`
detections into `track_observations` / `track_summaries` / `track_lifecycle`
with Stage 6A lifecycle contracts and motion-first association.

**No ReID**, no possession/events, no physical km/h, no human-ball fusion
(Stage 6D), no training.

## 2. Starting SHA

`ad6df906d26fd9df9dfd2c50e007d8eae3057e36` (Stage 6B human MOT)

## 3. Association

Motion-first cost (weights sum to 1):

- Center displacement vs constant-velocity predict (primary)
- Size (area) consistency
- Detection confidence
- IoU as **support only** — IoU alone never passes the gate

Assignment: deterministic greedy one-to-one; ties by `track_id` then
`detection_id`. Image-space gates only (not physical impossibility claims).

## 4. Primary candidate & ambiguity

- Multiple tentative ball tracks allowed
- ≤1 `primary_ball_candidate` per frame
- Close scores → `ambiguous` (no primary)
- Highest confidence alone is never decisive
- JSON sidecar: `ball_primary_candidates.json` (no new Arrow contract)
- Primary is **not** a guarantee of true ball identity

## 5. Lifecycle / lost ball

- Birth → tentative → confirmed; miss → lost; long gap → terminated + **new** track
- Short gap: optional `predicted` observations (`physical_metric_ineligible`,
  `event_ineligible`); uncertainty grows with gap
- Shot cut / replay / non-playable / window boundary → terminate
- No cross-cut prediction; no ReID stitch
- Role always `unknown`

## 6. Outputs

- `track_observations.parquet`, `track_summaries.parquet`, `track_lifecycle.parquet`
- `tracking_run_receipt.json`, `tracking_evaluation.json`, `tracking_quality_report.json`
- `ball_primary_candidates.json`
- Atomic no-overwrite; hard fail cleans partial outputs; receipt recount from tables

## 7. Evaluation / GT

Without reviewed ball tracking GT:

`NOT_EVALUATED_NO_REVIEWED_BALL_TRACKING_GROUND_TRUTH`

Synthetic metrics are labeled and must not be claimed as football accuracy.

## 8. Fingerprints (unchanged)

| Contract | Fingerprint |
|----------|-------------|
| track_observations v1 | `9ca2f7af…` |
| track_summaries v1 | `7b04e31d…` |
| detections v1 | `04ae8dd7…` |
| track_lifecycle v1 | `613cd81e…` |

## 9. CLI / validator

- `football-analytics tracking ball run`
- `football-analytics tracking ball evaluate`
- `football-analytics tracking humans|contracts` (kept)
- `scripts/check_ball_tracking_baseline.py` → `/home/fdoblak/workspace/ball_tracking_checks/`

## 10. Explicit non-claims

- Primary candidate ≠ true ball identity
- Ball tracking ≠ possession / events
- Pixel motion ≠ physical speed
- Predicted ≠ detection / event / physical metric
- Long gap after terminate ≠ ReID stitch
- Real match accuracy not validated without reviewed GT

## 11. Gate / next

Gate: `PASS_WITH_FINDINGS — BALL TRACKING BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Next (name only): `Aşama 6D — İnsan ve Top Takip Birleştirme, Kalite Kapıları ve Aşama 6 Kapanışı`
