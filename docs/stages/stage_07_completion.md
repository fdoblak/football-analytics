# Stage 7 completion — Identity / target-player baseline

Stage 7 closes as a **technical target-player identity evidence workflow**.
It does **not** claim real-match ReID, team, jersey OCR, or identity accuracy.

## Gate

`PASS_WITH_FINDINGS — TARGET PLAYER IDENTITY WORKFLOW ACTIVE; STAGE 7 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Tag: `identity-baseline-v0.7.0`

## Sub-stages

| ID | Deliverable | Status |
|----|-------------|--------|
| **7A** | Identity evidence / assignment / policy / review-audit / metric eligibility contracts | CLOSED |
| **7B** | Appearance embedding + tracklet ReID baseline (handcrafted) | CLOSED (with findings) |
| **7C** | Anonymous team appearance clustering + `team_assignments` | CLOSED (with findings) |
| **7D** | Jersey region + OpenCV template OCR baseline | CLOSED (with findings) |
| **7E** | Evidence fusion, manual confirm, Stage 7 close | CLOSED (with findings) |

## What Stage 7 is

- Canonical identity contracts + frozen upstream FPs reused (not rewritten)
- Appearance / team / jersey cues as **supporting evidence only**
- Manual-only confirmation with scoped intervals + append-only audit
- Metric-eligibility timeline for confirmed observed coverage
- Synthetic E2E: prepare → decide → resolve → validate

## What Stage 7 is not

- Automatic confirmed target selection
- Face recognition / biometric identity
- Physical track merge from ReID links
- Real club naming / home-away from kit alone
- Customer metric / event / pitch-coordinate engine
- Production football identity accuracy approval

## Evaluation honesty

Without reviewed target-identity ground truth:

`NOT_EVALUATED_NO_REVIEWED_TARGET_IDENTITY_GROUND_TRUTH`

Critical safety metric on synthetic tests: `false_target_attribution == 0`.

## Next stage

`Aşama 8A — Saha Kalibrasyonu, Homografi ve Koordinat Sözleşmeleri`
