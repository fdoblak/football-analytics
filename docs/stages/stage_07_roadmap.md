# Stage 7 roadmap — ReID, identity, target player

## Goal

Attribute tracks and evidence to **one paying `target_player`** so downstream
metrics never silently report the wrong athlete.

## Sub-stages

| Stage | Scope | Status |
|-------|--------|--------|
| **7A** | Identity evidence, ReID candidate links, track identity assignments, target-player request/receipt, policy, metric eligibility, review/audit, eval stubs | **in-tree / contracts** |
| **7B** | Appearance embedding + tracklet ReID baseline (handcrafted) | **in-tree / baseline** |
| **7C** | Anonymous team appearance clustering + `team_assignments` baseline | **in-tree / baseline** |
| **7D+** | Jersey OCR baseline; confirmation UX; production identity run | **not started** |

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

## Next after 7C

`Aşama 7D — Forma Numarası Bölge Çıkarma, OCR Baseline ve Değerlendirme`
