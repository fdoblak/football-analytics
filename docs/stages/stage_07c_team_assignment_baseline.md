# Stage 7C — Team appearance clustering + team assignment baseline

## Gate

`PASS_WITH_FINDINGS — TEAM ASSIGNMENT BASELINE ACTIVE; REAL FOOTBALL ACCURACY NOT YET VALIDATED`

Evaluation without reviewed GT:

`NOT_EVALUATED_NO_REVIEWED_TEAM_ASSIGNMENT_GROUND_TRUTH`

## Scope

Anonymous **kit appearance** clustering from Stage 7B tracklet embeddings into
`team_a` / `team_b` / `unknown`, writing canonical `team_assignments` plus
supporting `identity_evidence` (`evidence_type=team_assignment`).

## Labels (strict)

| Label | Meaning |
|-------|---------|
| `team_a` | Anonymous cluster A (deterministic centroid fingerprint order) |
| `team_b` | Anonymous cluster B |
| `unknown` | Abstain / outlier / insufficient / not eligible |

- **Not** club names, logos, or cities
- **Not** home / away
- Cross-video auto transfer **forbidden**
- Cross-shot alignment only with strong centroid evidence (no silent swap)

## Eligibility / seeds

Seeds: human, observed, sufficient coverage/quality, role `player` only.

Excluded from seeds: referee, staff, confirmed goalkeeper, unknown role,
predicted/interpolated, low-quality crops.

Unknown-role tracklets: no seed; at most `candidate` after clusters exist.

## Role specials

| Role | Behavior |
|------|----------|
| player | `team_role=unknown`; may assign `team_a`/`team_b` |
| referee / staff | `team_id=unknown`, `team_role=official`, `not_eligible` |
| goalkeeper | `team_id=unknown`; **no** auto team from kit alone |
| unknown | candidate at most |

Never invent `goalkeeper_home` / `goalkeeper_away` from kit.

## Identity policy

Team evidence alone → Stage 7A `candidate` only (`TEAM_ALONE_INSUFFICIENT`).
No target confirmation. Team assignment ≠ player identity.

## Method

Selected: deterministic farthest-pair init + Lloyd 2-cluster on color-only
(HSV/Lab upper+lower) features from 7B embeddings. Edge/texture downweighted.

Rejected/future: learned kit classifiers, SoccerNet team spotting integration,
home/away heuristics.

## CLI

- `football-analytics identity teams classify`
- `football-analytics identity teams evaluate`

Validator: `scripts/check_team_assignment_baseline.py`  
Runtime: `/home/fdoblak/workspace/team_assignment_checks/`

## Limits (explicit)

- Anonymous clusters are **not** real team names
- Team assignment is **not** identity
- Goalkeeper team binding from kit is **not** reliable
- Target player is **not** selected
- Real match accuracy is **not** validated without reviewed GT

## Next

`Aşama 7D — Forma Numarası Bölge Çıkarma, OCR Baseline ve Değerlendirme`
