# Product Scope Freeze — football-analytics v1.0

**Status:** FROZEN as of Stage 0 (2026-07-22)

**Authority:** Stage-gate ADR-0001; changes require explicit user approval and a new ADR amendment.

Local presence of a SoccerNet or research repository does **not** imply inclusion in the v1.0 product core.

---

## 1. v1.0 core product scope (in)

The v1.0 system shall deliver an end-to-end, resume-capable pipeline producing versioned canonical artifacts and operator-facing outputs for football match clips/matches, within the hardware envelope of an RTX 3050 Laptop GPU (4 GB VRAM).

### 1.1 Pipeline and analytics capabilities

| Capability | v1.0 |
|------------|------|
| ingest | in |
| video validation | in |
| detection | in |
| tracking | in |
| team/role classification | in |
| camera calibration | in |
| pitch coordinates | in |
| canonical game state | in |
| physical player metrics | in |
| team tactical metrics | in |
| Re-ID/jersey-assisted advanced identity | in |
| ball detection/tracking | in |
| possession | in |
| player-ball interactions | in |
| pass / shot / turnover and selected events | in |
| annotated video | in |
| tactical / minimap video | in |
| match / team / player reports | in |
| FastAPI | in |
| dashboard | in |
| archive / restore | in |
| reproducibility | in |
| quality and coverage reporting | in |

### 1.2 Architectural obligations (v1.0)

- Canonical contracts first (versioned Parquet/JSON), not foreign native objects.
- External engines isolated via adapter / config / subprocess / dedicated environments.
- Stage-level resume, manifests, checksums, and coverage/quality gates.
- Unknown or low-trust results represented as `null + confidence + reason` (no fabricated values).
- Active compute in WSL workspace; durable archives on configured SSD storage once available.

### 1.3 Primary implementation home

- Application and MVP orchestration: `ai-dev` + `/home/fdoblak/projects/football-analytics`
- Heavy external baselines (e.g. TrackLab / sn-gamestate): isolated environments, comparison and optional adapters — not a hard dependency for the first detection/tracking MVP

---

## 2. Explicitly out of v1.0 core (research / optional)

These may remain cloned locally for research. They are **not** acceptance criteria for v1.0 product completion.

| Module | Classification |
|--------|----------------|
| MVFoul | v1.0 sonrası araştırma/opsiyonel |
| replay grounding | v1.0 sonrası araştırma/opsiyonel |
| dense captioning | v1.0 sonrası araştırma/opsiyonel |
| Echoes multimodal kullanım | v1.0 sonrası araştırma/opsiyonel |
| depth | v1.0 sonrası araştırma/opsiyonel |
| NVS | v1.0 sonrası araştırma/opsiyonel |
| banner replacement | v1.0 sonrası araştırma/opsiyonel |
| anticipation | v1.0 sonrası araştırma/opsiyonel |
| VQA | v1.0 sonrası araştırma/opsiyonel |

Related local repos (examples): `sn-mvfoul`, `sn-grounding`, `sn-caption`, `sn-echoes`, `sn-depth`, `sn-nvs`, `sn-banner`, plus 2026 challenge kits not listed in the v1.0 core table.

---

## 3. Delivery slices (reference only; not a license to skip gates)

| Slice | Focus |
|-------|-------|
| MVP-0 | Foundation: repo, configs, schemas, check scripts, manifests |
| MVP-1 | Ingest + detection + tracking + annotated video + archive |
| MVP-2 | Identity + calibration + spatial metrics + tactical preview |
| MVP-3 | Ball, possession, selected events |
| Productization | Reports, FastAPI, dashboard after canonical artifacts stabilize |

Each slice still requires its own stage brief, artifacts, tests, and completion document per ADR-0001.

---

## 4. Non-goals for v1.0

- Training large foundation models or full broadcast-corpus ingestion as a prerequisite.
- Guaranteeing perfect identity across an entire match without coverage reporting.
- Requiring NVS, foul VAR, captioning, or commentary fusion for core acceptance.
- Treating every cloned SoccerNet repo as production-ready.

---

## 5. Change control

To alter this freeze:

1. Open a new ADR (or amend ADR-0001 with an explicit scope delta).
2. Update this file with before/after tables.
3. Obtain explicit approval from Furkan Doblak before implementation work begins.
