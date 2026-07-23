# Stage 7 roadmap — ReID, identity, target player

## Goal

Attribute tracks and evidence to **one paying `target_player`** so downstream
metrics never silently report the wrong athlete.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **7A** | Identity evidence, ReID candidate links, track identity assignments, target-player request/receipt, policy, metric eligibility, review/audit, eval stubs | **CLOSED** |
| **7B** | Appearance embedding + tracklet ReID baseline (handcrafted) | **CLOSED** (with findings) |
| **7C** | Anonymous team appearance clustering + `team_assignments` baseline | **CLOSED** (with findings) |
| **7D** | Jersey region extraction + OpenCV template OCR baseline | **CLOSED** (with findings) |
| **7E** | Target-player evidence merge, manual confirm, Stage 7 closure | **CLOSED** (with findings) |

## Stage 7 status

**CLOSED** as technical identity workflow baseline (`identity-baseline-v0.7.0`).
Real football ReID / team / jersey / identity accuracy is **not** validated.

## Explicit non-goals (until later stages)

- Face recognition / biometric identity (**forbidden**)
- Auto target selection from lineup alone
- Physical track merge from ReID links
- Real club naming / home-away from kit alone
- Customer metric reports

## Dependencies

- Stage 6 tracking bundle (`track_*`)
- Stage 7B appearance profiles
- `team_assignments` / `jersey_observations` as **reference-only** evidence sources
- Manual review queue + append-only audit

## Next after Stage 7

`Aşama 8A — Saha Kalibrasyonu, Homografi ve Koordinat Sözleşmeleri`
