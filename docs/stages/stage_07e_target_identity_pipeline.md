# Stage 7E — Target-player evidence fusion + manual approval

## Gate

`PASS_WITH_FINDINGS — TARGET PLAYER IDENTITY WORKFLOW ACTIVE; STAGE 7 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Evaluation without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_TARGET_IDENTITY_GROUND_TRUTH`

## Scope

Fuse Stage 7A policy with 7B/7C/7D outputs **as evidence only**:

- Target request + manual anchors
- Appearance / ReID candidate links
- Anonymous team assignments
- Jersey observations / consensus
- Role / continuity / coverage / negative cues

Produce:

- Deterministic candidate ranking (review aid only)
- Review manifest with expected versions/hashes
- Schema-validated manual decisions + append-only audit
- `track_identity_assignments` (append-only confirm/reject/revoke)
- Metric-eligibility timeline (no real metrics)
- Fusion receipt / quality / evaluation stubs

## Confirmation rules

| Evidence | Max automatic status |
|----------|----------------------|
| Appearance alone | candidate |
| Jersey alone | candidate |
| Team alone | candidate |
| Two automatic supporting cues | provisional |
| Scoped manual decision | **confirmed** (interval only) |

- `auto_confirm=false`; face=false; cross-video auto-link=false
- Linked tracklets after a confirm stay candidate/provisional
- Simultaneous overlapping confirmed targets → hard fail + review
- Revocation is append-only; revoked intervals are not metric-eligible
- Evaluation labels must not enter decision evidence

## CLI

- `football-analytics identity target prepare-review`
- `football-analytics identity target decide`
- `football-analytics identity target resolve`
- `football-analytics identity target validate`

Validator: `scripts/check_target_identity_pipeline.py`  
Runtime: `/home/fdoblak/workspace/target_identity_checks/`  
Manual decisions stay **runtime-only** (not git).

## Limits (explicit)

- No new ReID / team / OCR models
- No face recognition / biometric identity
- No physical track merge
- No automatic confirmed targets
- No real football accuracy claims without reviewed GT
- No Stage 8A / customer metric engine

## Next

`Aşama 8A — Saha Kalibrasyonu, Homografi ve Koordinat Sözleşmeleri`
