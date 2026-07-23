# Stage 5D — Human role classification baseline

## Status

**CLOSED** with findings (2026-07-23).

Gate: `PASS_WITH_FINDINGS — HUMAN ROLE CLASSIFICATION BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

## Purpose

Conservatively assign football **roles** to existing human detections using
weightless, deterministic OpenCV/NumPy kit-color features and context clustering.
This is **not** target-player identity, team classification, tracking, ReID,
jersey OCR, or events.

Critical rule: `human detection != player`. Generic person never auto-maps to
player. Weak / conflicting evidence → `unknown` / `assignment_status=abstained`.

## Role vocabulary

Canonical `RoleLabel`: `player | goalkeeper | referee | assistant_referee | staff | unknown`

**Mapping note:** user-facing / prompt “other” → canonical **`staff`** (documented in
config notes and provenance `other_maps_to=staff`).

## Contract choice

Reuses Stage 5A **`detection_attributes`** (no new Arrow contract). Role writes
set `role_source=downstream_classifier`, `role_score=null` (raw margin in
`provenance_json`). **`detections` v1 fingerprint unchanged**
(`04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6`).

Assignment statuses (distinct from detection `processing_status`):
`classified | abstained | not_eligible | skipped | failed`.

## Approach

1. Eligibility: human only, processed frames, analysis windows, crop geometry/quality.
2. Features: HSV upper/lower histograms + geometry; crop path if video provided;
   synthetic geometry+kit path for fixture tests without video.
3. Clustering: max **2** outfield kit clusters; deterministic sort; **no** `team_id` /
   team names.
4. Assignment (conservative):
   - **player**: stable outfield cluster membership + margin
   - **goalkeeper**: distinct from both clusters **and** extra evidence (lateral /
     size) — color alone never enough
   - **referee**: dark/low-sat **and** not outfield + margin — dark alone never enough
   - **staff**: rare residual (maps “other”)
   - **unknown**: default / abstain / conflict / low quality
5. Crops are **not** persisted by default.

## Outputs

Under `/home/fdoblak/workspace/human_role_checks/` run directories:

- `detection_attributes.parquet` (updated roles)
- `role_run_receipt.json`
- `role_evaluation.json` → `NOT_EVALUATED_NO_REVIEWED_HUMAN_ROLE_GROUND_TRUTH` when no reviewed GT

## CLI

```
football-analytics perception roles classify --detections --detection-attributes --output-dir [--source] [--analysis-windows] [--detection-frame-status] --config
football-analytics perception roles evaluate --predictions --ground-truth --output [--config]
```

## Findings (open)

1. **No reviewed football human-role ground truth** — real match role accuracy not claimed.
2. Weightless HSV/kit heuristics are a technical baseline only; broadcast lighting /
   occlusion / similar kits will abstain or mislabel without reviewed labels.
3. Stage 5B/5C AGPL Ultralytics findings remain for upstream detections (this stage
   adds no new weighted model).

## Out of scope

- Training / fine-tuning / new model or package download
- Team IDs, tracking, ReID, jersey OCR, events
- Stage 5E fusion / Stage 5 closure
- Claiming production role accuracy
