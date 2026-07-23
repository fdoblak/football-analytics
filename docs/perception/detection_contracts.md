# Detection contracts (Stage 5A)

Stage 5A defines **contracts only** for player / official / ball detection.
No detector inference, model download, or real video evaluation ships here.

## Decision: keep `detections` v1 unchanged

Canonical `schemas/data/v1/detections.json` stays as the bbox / class / score
table. It does **not** carry entity/role, frame processing status, or
preprocessing transforms. Those require sidecars:

| Concern | Contract |
|---------|----------|
| BBox + model class + confidence | `detections` @ v1 (unchanged) |
| Processed vs skipped/failed frame | `detection_frame_status` @ v1 |
| Entity type + football role | `detection_attributes` @ v1 |
| Run evidence / counts | `detection_run_receipt` (JSON schema) |
| Letterbox/stretch params | `preprocessing_transform` (JSON schema) |

## Entity vs role

```text
entity_type:  human | ball | unknown
role_label:   player | goalkeeper | referee | assistant_referee | staff | unknown
role_source:  detector_native | downstream_classifier | manual_review | imported | unknown
```

Hard rules:

- Generic `person` → `human` + `role unknown` (never auto-`player`)
- Ball forbids non-`unknown` role
- Unmapped model class → reject (default) or unknown per taxonomy policy
- Role score is **not** calibrated probability

## Frame status semantics

| `processing_status` | Meaning |
|---------------------|---------|
| `processed` | Frame ran; `detection_count > 0` |
| `processed_no_detections` | Frame ran; genuinely empty |
| `skipped` / `not_eligible` / `failed` | Not processed — do not invent empty detection rows |

Zero rows in `detections` for a frame is ambiguous without `detection_frame_status`.

## BBox / coordinates

Canonical: **xyxy**, image pixels, **half-open**, on the normalized oriented
source frame. Reject NaN/Inf/zero area. Model-input boxes must be inverse-
transformed via a fingerprinted `PreprocessingTransform` (letterbox or stretch).

## Routing

Detection requests are prepared from Stage 4 `analysis_windows` eligibility:

- tracking/identity eligible → human detection candidate
- ball_analysis ineligible → skip ball
- non-playable / full-screen graphics → skip
- live_event `unknown` does **not** by itself block visual detection
- identity-only close-up may run human detection; mark physical downstream unsafe

## Configs

- `configs/perception/detection_taxonomy.yaml` (fingerprinted)
- `configs/perception/detection_policy.yaml` (fingerprinted)

## Package

`src/football_analytics/perception/` — types, taxonomy, policy, transforms,
validation, contracts helpers.

## Validator

```bash
python scripts/check_detection_contracts.py
```

Runtime report (not git): `/home/fdoblak/workspace/detection_contract_checks/`
