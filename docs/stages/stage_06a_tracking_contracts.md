# Stage 6A — Multi-object tracking contracts and lifecycle

## 1. Purpose

Define canonical multi-object tracking contracts, lifecycle rules, ID/time/bbox
semantics, request/receipt schemas, and synthetic validators so future human
and ball trackers can consume Stage 5 detection bundles safely.

**No tracker algorithm** runs in this stage (no ByteTrack / DeepSORT / TrackLab
inference). No ReID, team, identity, jersey, events, or physical metrics.

## 2. Starting SHA

`18ba78f369a73cef1ddcbc82292ee69030660c22`
(`detection-baseline-v0.5.0`)

## 3. Decision — keep track_observations / track_summaries / detections v1

Existing fingerprints are frozen:

| Contract | Fingerprint |
|----------|-------------|
| `track_observations` v1 | `9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01` |
| `track_summaries` v1 | `7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168` |
| `detections` v1 | `04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6` |

Lifecycle state is **not** added to observation rows. New Arrow sidecar:
`track_lifecycle` v1.

## 4. Observation source ↔ observation_state

| Stage 6A source | `track_observations.observation_state` | Notes |
|-----------------|----------------------------------------|-------|
| `detection_associated` | `observed` | `detection_id` required |
| `predicted` | `predicted` | `detection_id` null; `physical_metric_ineligible` |
| `interpolated` | `interpolated` | `detection_id` null; `physical_metric_ineligible` |
| `not_observed` | **prefer no row** | Absence preferred; do not invent a measured observation |

## 5. Lifecycle (sidecar)

States: `tentative` → `confirmed` → `lost` → `terminated` (policy transitions).

- Birth is `tentative` (not arbitrary `confirmed`).
- `terminated` cannot reopen.
- Lost recovery only within `max_lost_gap_us`.
- Human/ball merge forbidden; track ID ≠ player identity; no ReID merge.

## 6. Contracts / package

| Artifact | Path |
|----------|------|
| Arrow sidecar | `schemas/data/v1/track_lifecycle.json` |
| JSON | `schemas/tracking/tracking_request.schema.json` |
| JSON | `schemas/tracking/tracking_run_receipt.schema.json` |
| JSON | `schemas/tracking/tracking_evaluation.schema.json` |
| Policy | `configs/tracking/tracking_contract_policy.yaml` |
| Package | `src/football_analytics/tracking/` |
| Validator | `scripts/check_tracking_contracts.py` → `/home/fdoblak/workspace/tracking_contract_checks/` |

## 7. CLI (contract-safe only)

- `football-analytics tracking contracts validate`
- `football-analytics tracking receipt validate`

No `track run`.

## 8. Evaluator

Without reviewed tracking ground truth:

`NOT_EVALUATED_NO_REVIEWED_TRACKING_GROUND_TRUTH`

Metrics interface returns null + reason. `sn-trackeval` = future adapter only.

## 9. Explicit non-claims

- Tracking algorithm not implemented.
- Track ID is not a real player identity.
- Camera exit/re-entry sameness unproven.
- ReID / target player selection are later stages.
- Physical metrics cannot be produced from predicted/interpolated defaults.

## 10. Gate / next

Gate: `PASS — MULTI-OBJECT TRACKING CONTRACTS ACTIVE`

Next (name only): `Aşama 6B — İnsan Çok Nesneli Takip Baseline ve Değerlendirme`
