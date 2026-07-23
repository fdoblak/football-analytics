# Stage 7A — ReID, identity evidence, and target-player contracts

## 1. Purpose

Define machine-readable contracts for:

- How a user describes a `target_player`
- Which tracks are target candidates
- What evidence supports “same player”
- Conflict / review / revoke rules
- When tracks may enter **customer** metrics

**Contracts only.** No ReID embedding inference, team clustering, jersey OCR,
face/biometric recognition, auto target selection, track merge, or reports.

## 2. Starting SHA

`fabaa1546f18480a8fa60cfc6dc604399dfe739d` (`tracking-baseline-v0.6.0`)

## 3. Frozen upstream fingerprints

| Contract | Fingerprint |
|----------|-------------|
| `detections` v1 | `04ae8dd7…` |
| `track_observations` v1 | `9ca2f7af…` |
| `track_summaries` v1 | `7b04e31d…` |
| `track_lifecycle` v1 | `613cd81e…` |
| `team_assignments` v1 | `759aa9b7…` |
| `jersey_observations` v1 | `aabc7642…` |

`team_assignments` / `jersey_observations` are reused **by reference** (artifact
fingerprint + evidence_type), not duplicated or mutated.

## 4. New contracts

### Arrow (registry)

- `identity_evidence` v1
- `reid_candidate_links` v1 — candidates only; **not** physical merge
- `track_identity_assignments` v1 — append-only statuses + metric eligibility

### JSON

- `schemas/identity/target_player_request.schema.json`
- `schemas/identity/identity_manual_audit.schema.json`
- `schemas/identity/identity_run_receipt.schema.json`
- `schemas/identity/identity_evaluation.schema.json`

## 5. Evidence types

`manual_track_anchor`, `appearance_similarity`, `jersey_number`,
`team_assignment`, `role_consistency`, `temporal_continuity`,
`spatial_motion_continuity`, `camera_view_suitability`, `negative_exclusion`,
`unknown`

## 6. Decision matrix (policy)

Policy: `configs/identity/identity_evidence_policy.yaml`

- Single weak / alone-insufficient cue → `candidate` (never `confirmed`)
- Jersey / team / role / appearance alone insufficient for `confirmed`
- ≥2 independent supporting types → at most `provisional` (auto-confirm forbidden)
- Scoped `manual_verified` anchor → may `confirmed` within track/time only
- Hard conflict → `rejected` / review
- `revoked` → not metric-eligible
- Cross-video auto link forbidden; face/biometric forbidden

## 7. Assignment status / metric eligibility

Statuses: `candidate|provisional|confirmed|rejected|revoked|unknown`  
Scope: `target|non_target|unknown`  
Eligibility: `eligible|provisional_only|not_eligible|not_evaluable`

Customer metrics require confirmed target + valid interval + observed tracking +
coverage + no revoke/conflict. Predicted/interpolated observations are not
customer-metric eligible.

## 8. Package / CLI / validator

| Artifact | Path |
|----------|------|
| Package | `src/football_analytics/identity/` |
| Policy | `configs/identity/identity_evidence_policy.yaml` |
| Validator | `scripts/check_identity_contracts.py` → `/home/fdoblak/workspace/identity_contract_checks/` |

CLI (contract-safe only — **no** `identity run`):

- `football-analytics identity contracts validate`
- `football-analytics identity target validate`
- `football-analytics identity receipt validate`

## 9. Evaluator

Without reviewed identity/ReID ground truth:

`NOT_EVALUATED_NO_REVIEWED_IDENTITY_GROUND_TRUTH`

Null metrics + reasons. False target attribution is the critical product error.
`sn-reid` = future adapter only.

## 10. Explicit non-claims

- No real ReID in this stage
- Track ID ≠ player identity
- Jersey / team / appearance alone insufficient
- Face recognition unused / forbidden
- Target player not auto-selected
- Final customer report not produced
- Synthetic fixtures ≠ football identity accuracy

## 11. Gate / next

Gate: `PASS — REID AND TARGET IDENTITY CONTRACTS ACTIVE`

Next (name only): `Aşama 7B — Görünüş Embedding ve Tracklet ReID Baseline`
