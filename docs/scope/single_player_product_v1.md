# Single-Player Product Scope — v1

**Status:** Stage-bootstrap product foundation (docs/configs/schemas only)
**Authority:** Complements the Stage 0 freeze in `product_v1_scope.md`; single-player report is the **final customer output** for product delivery.
**Hardware envelope:** RTX 3050 Laptop GPU — **4 GB VRAM**
**Stage 2 closed at:** tag `foundation-v0.1.0` / HEAD `4b5089eeb8b022c95ed67a8ba1f60166b6058cdd`
**Next implementation stage:** Stage 3 — safe video ingest, probe, normalize, frame timebase (not started here)

---

## 1. Final product

Process a **full match video** and produce an **evidence-based individual performance analysis and report** for **one** selected footballer (`target_player`).

| Layer | Role |
|-------|------|
| Final customer output | Single-player report for `target_player` only |
| Intermediate context | Team, opponent, referee, ball, pitch — used only to explain the target player's actions |
| Out of final product | Standalone team report, opponent scouting pack, or match-wide dashboard as the primary deliverable |

This is **not** a general team analytics product. Team/opponent analysis is intermediate context only.

---

## 2. `target_player` contract

Every real analysis run must define exactly one `target_player` with at least:

| Field | Purpose |
|-------|---------|
| `target_player_id` | Stable project identifier for this analysis subject |
| `display_name` | Human-facing name (TR/EN report display) |
| `team_hint` | Expected team / side label |
| `jersey_number_hint` | Expected jersey number |
| `match_id` | Match / video binding |
| `reference_images` | Optional visual references for identity |
| `manual_identity_status` | Operator confirmation state |
| `identity_confidence` | Numeric or ordinal confidence for the bound identity |

Example config: `configs/product/target_player.example.yaml`.

---

## 3. Identity policy

Identity is protected by combining:

- Jersey number
- Team assignment
- Visual reference images
- Re-identification signals
- Track continuity
- Manual confirmation when needed

**Hard rule:** If identity is not sufficiently reliable, **do not** attribute another player's data to `target_player`.

Use:

```text
manual_identity_confirmation_required
```

and stop metric attribution for the uncertain interval (or the whole run, if identity cannot be established). Prefer `identity_uncertain` / `not_evaluable` over fabricated values.

---

## 4. Opta legal and provenance boundary

Official Opta data:

- May be imported **only** when the user supplies it under a valid license.
- **No** Opta scraping.
- **No** unauthorized API or data access.
- **No** presenting project-generated metrics as official Opta-branded data.
- **No** labeling video-derived estimates as “Resmî Opta verisi”.

When licensed Opta is absent, outputs must be labeled clearly as one or more of:

- `project-generated` event metrics
- `Opta-style` metrics (schema/semantics inspired by industry event taxonomies)
- `video-derived` estimates

Official Opta and video-derived values **must** use separate provenance fields. See metric registry and report schema.

---

## 5. Metric families (product intent)

The product aims to produce the 25 single-player metric families documented in:

- `docs/metrics/single_player_metric_dictionary.md` (human dictionary, version 1)
- `configs/metrics/single_player_metrics.yaml` (machine registry, version 1)

Turkish display names for reports; English `metric_id` / technical names for machines.

**Never fake `0`.** Use `not_evaluable`, `insufficient_coverage`, `identity_uncertain`, `ball_tracking_insufficient`, `calibration_insufficient`, or `event_uncertain` as appropriate.

---

## 6. Report outputs (target)

For `target_player`, the final system should produce at least:

| Output | Form |
|--------|------|
| Machine-readable report | JSON (schema: `schemas/metrics/single_player_report.schema.json`) |
| Tabular exports | CSV / Parquet |
| Human-readable | HTML |
| Printable | PDF |
| Spatial visuals | Heatmap and related figures |
| Reliability | Metric confidence / coverage table |
| Timeline | Event timeline for the target player |
| Evidence | Frame / clip / artifact references |

Each report must include: target player, match info, analysis duration, tracked duration, identity confidence, calibration coverage, ball-tracking coverage, per-metric confidence/coverage, reasons for non-evaluable metrics, model/code/schema versions, `run_id`, `git_commit`, and data provenance. No false precision.

---

## 7. Pipeline capability chain

High-level chain (architecture detail in `docs/architecture/single_player_pipeline.md`):

```text
ingest → quality/camera → calibration → detect → track → target identity
  → team/jersey/reid helpers → possession → events → metrics → QA → report
```

SoccerNet: evaluate the 19 locked repos under `/home/fdoblak/projects/soccernet` as **reference only**; integrate via adapters; never dirty original clones.

---

## 8. Agent roles (permanent)

| Role | Agent | Write access |
|------|-------|--------------|
| Executor / writer | Cursor | Project worktree (automation worktree when continuous mode is active) |
| Supervisor | Codex | **Read-only** — review, decisions, gates |

See root `AGENTS.md`.

---

## 9. Stage boundary for this bootstrap

| In scope here | Out of scope here |
|---------------|-------------------|
| Product scope docs | Stage 3 ingest / probe / normalize implementation |
| Metric dictionary + YAML registry | Metric computation code |
| Report JSON Schema | Pipeline execution beyond existing Stage 2 foundation |
| Example `target_player` config | SoccerNet clone edits |
| `AGENTS.md` permanent rules | Protected package upgrades |

Stage 2 remains closed at `foundation-v0.1.0`. Open findings stay open (API proxy 403, remote CI unverifiable, registry warnings, GPU unverifiable, Same-VHDX, RISK-029, no cache GC).

---

## 10. Change control

Material changes to this single-player product definition require:

1. Explicit user approval
2. ADR or amendment documenting the delta
3. Dictionary / registry / schema version bumps when metric semantics change
