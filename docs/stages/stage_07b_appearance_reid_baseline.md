# Stage 7B — Appearance embedding + tracklet ReID baseline

## Gate

`PASS_WITH_FINDINGS — APPEARANCE REID BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Evaluation without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_APPEARANCE_REID_GROUND_TRUTH`

## Scope

Deterministic appearance profiles from **observed-only human** crops / synthetic RGB
crops, then ReID **candidate** links + `appearance_similarity` evidence.

## Candidate matrix (selection)

| Candidate | Status | Reason |
|-----------|--------|--------|
| sn-reid / TrackLab adapter | future | SoccerNet reference-only; GPL/adapter ADR pending |
| Local pretrained ReID weights | rejected | none verified/licensed in archive |
| Torchvision feature extractor | rejected | unused; no football-tuned ReID head |
| Handcrafted HSV/Lab spatial hist + upper/lower + bounded grayscale edge/texture, L2 | **SELECTED** | deterministic, offline, no download |

## Behavior

- Fixed-dim L2-normalized vectors (default dim 88); quality-weighted aggregation
- Profiles: Arrow `tracklet_appearance_profiles` under local runtime `0600`
- Crops **not** persisted by default
- Appearance evidence alone → Stage 7A `candidate` only (`auto_confirm=false`)
- Temporal-overlap / cross-video / human-ball links forbidden
- No face recognition; no team/jersey/target confirmation; no track merge
- No Stage 7C

## Limits (explicit)

- Appearance profile is **not** biometric face identity
- Appearance similarity is **one** evidence type among many
- Tracks are **not** physically merged
- Target player is **not** confirmed in this stage
- Same-kit players are strong hard-negatives
- Real match ReID accuracy is **not** validated

## CLI

- `football-analytics identity appearance extract`
- `football-analytics identity reid candidates`
- `football-analytics identity reid evaluate`

Validator: `scripts/check_appearance_reid_baseline.py`  
Runtime: `/home/fdoblak/workspace/appearance_reid_checks/`

## Next

`Aşama 7C — Takım Görünümü Sınıflandırma ve Team Assignment Baseline`
