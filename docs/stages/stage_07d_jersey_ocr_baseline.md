# Stage 7D — Jersey region extraction + OCR baseline

## Gate

`PASS_WITH_FINDINGS — JERSEY NUMBER OCR BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Evaluation without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_JERSEY_NUMBER_GROUND_TRUTH`

## Scope

Extract bounded torso jersey-number regions from observed human tracklets and
run a **deterministic OpenCV template/shape digit matcher** (synthetic 0–9
masks). Write canonical `jersey_observations` plus supporting
`identity_evidence` (`evidence_type=jersey_number`). Optional JSON sidecars
hold tracklet consensus and region provenance.

## Selection matrix

| Candidate | Status | Reason |
|-----------|--------|--------|
| Tesseract / pytesseract | rejected | External OCR install; not selected |
| EasyOCR | rejected | Heavy DL / download risk |
| MMOCR | rejected | Heavy framework; not installed |
| SoccerNet sn-jersey | **future** | Reference-only adapter later; no Stage 7D install |
| OpenCV template/shape (synthetic 0–9) | **selected** | Offline, deterministic, no new packages |

## Eligibility

Observed human only. Exclude predicted/interpolated, ball, referee/staff,
graphics/replay. Player/goalkeeper may be candidates; unknown role is
conservative. Crops are **not** persisted by default.

## Region + OCR

- Bounded torso candidates with contrast/blur/area quality; deterministic rank
- Preprocess: grayscale, CLAHE, Otsu, light morphology, connected components
- 1–2 digits; leading zeros preserved in `raw_text` (numeric optional)
- `confidence=null`; raw score/margin in `quality_flags` / sidecar
- Status via flags: `no_region` / `no_digits` / `ambiguous` / `low_quality` /
  `not_eligible` / `failed` (plus observed clear reads)
- Partial multi-component decode abstains (false-number guard)

## Consensus + identity

Quality-weighted track vote with min observations / spread / margin.
Conflicts → ambiguous + review. Jersey evidence alone → Stage 7A **candidate**
only (`JERSEY_ALONE_INSUFFICIENT`). Team/jersey conflict → conflicting review
evidence. No target confirmation. No face. No Stage 7E.

## CLI

- `football-analytics identity jersey observe`
- `football-analytics identity jersey evaluate`

Validator: `scripts/check_jersey_ocr_baseline.py`  
Runtime: `/home/fdoblak/workspace/jersey_ocr_checks/`

## Limits (explicit)

- Jersey number is **one supporting** identity cue, not identity
- No inventing numbers when digits are absent / ambiguous
- Template/synthetic success ≠ real broadcast OCR accuracy
- Target player is **not** selected
- Face recognition is **not** used

## Next

`Aşama 7E — Hedef Futbolcu Kanıt Birleştirme, Manuel Onay ve Aşama 7 Kapanışı`
