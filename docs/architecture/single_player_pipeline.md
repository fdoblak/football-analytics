# Single-Player Pipeline Architecture

**Status:** Capability map for product foundation (not Stage 3 implementation)
**Product scope:** `docs/scope/single_player_product_v1.md`
**Hardware:** RTX 3050 Laptop — 4 GB VRAM budget for future model choices

---

## 1. Capability chain

High-level chain from match video to single-player report:

```text
ingest
  → quality / camera
  → calibration
  → detect
  → track
  → target identity
  → team / jersey / reid helpers
  → possession
  → events
  → metrics
  → QA
  → report
```

| Stage | Purpose for `target_player` |
|-------|-----------------------------|
| ingest | Safe video intake, hash, probe, normalize, frame timebase (Stage 3+) |
| quality / camera | Signal quality, camera-cut hints, usable vs unusable intervals |
| calibration | Homography / pitch plane for zones, distance, heatmap, speed |
| detect | Players, ball, and supporting classes within VRAM limits |
| track | Temporal continuity; inputs to identity and physical metrics |
| target identity | Bind one track (or track set) to `target_player`; stop if uncertain |
| team / jersey / reid helpers | Context to protect identity — not final customer products |
| possession | Who has the ball; required for passes, losses, touches |
| events | Opta-style / video-derived event candidates for the target |
| metrics | Dictionary v1 families with confidence / coverage / null reasons |
| QA | Coverage gates, identity gates, no fake zeros |
| report | JSON / CSV / Parquet / HTML / PDF / heatmap / evidence |

Team, opponent, referee, ball, and pitch exist **only** as context for the target player's evidence trail.

---

## 2. Identity before metrics

```text
identity_confidence insufficient
  → manual_identity_confirmation_required
  → do not attribute foreign tracks
  → emit identity_uncertain / not_evaluable
```

Metric computation must not “guess” the customer when identity is weak.

---

## 3. SoccerNet strategy (19 locked repos)

**Location:** `/home/fdoblak/projects/soccernet`

| Repo | Typical evaluation area |
|------|-------------------------|
| sn-tracking | Multi-object tracking |
| sn-gamestate | Game state reconstruction |
| sn-calibration | Camera / pitch calibration |
| sn-reid | Re-identification |
| sn-jersey | Jersey number |
| sn-teamspotting | Team / group spotting |
| sn-spotting / ActiveSpotting / PTS-baseline | Action / ball-action spotting |
| sn-grounding | Temporal grounding |
| sn-mvfoul | Foul / event research (optional; not v1 core) |
| sn-depth / sn-nvs | Depth / novel view (optional) |
| sn-caption / sn-echoes / sn-banner | Caption / audio / banner (optional) |
| sn-trackeval | Tracking evaluation |
| SoccerNet / SoccerNet-v3 | SDK / annotation reference |

**Evaluate before adopt.** Being cloned locally is not acceptance.

Per capability checklist:

1. Survey relevant SoccerNet repos
2. Check license (GPL caution — adapters and license compatibility ADRs required)
3. Verify lock SHA (`external_repos.lock.yaml`)
4. Confirm environment compatibility
5. Inspect I/O contracts
6. Assess 4 GB VRAM fit
7. Small synthetic / demo smoke only
8. Connect via **adapter** into `football-analytics`
9. Compare accuracy / speed / VRAM vs alternatives
10. Record accept/reject in ADR or evaluation report

---

## 4. External repo isolation

Original SoccerNet clones stay:

- clean
- locked SHA
- reference / upstream only

**Do not** dirty original clones. Integration order:

1. Adapter / config / subprocess wrapper in this project
2. Project-owned implementation when needed
3. Versioned patch (documented) if unavoidable
4. Separate project-owned fork / worktree if required — never mutate the locked original

Custom code when SoccerNet is unfit is expected; document with an ADR.

---

## 5. Agent roles on this architecture

| Agent | Role |
|-------|------|
| **Cursor** | Writer / executor — implements docs, adapters, stages in the allowed worktree |
| **Codex** | Read-only supervisor — reviews, gates, `APPROVE_CHECKPOINT` before automation commit/push |

Two agents must not write the same worktree concurrently. Permanent rules: root `AGENTS.md`.

---

## 6. Data and report contracts

| Artifact | Path |
|----------|------|
| Metric dictionary | `docs/metrics/single_player_metric_dictionary.md` |
| Metric registry | `configs/metrics/single_player_metrics.yaml` |
| Report schema | `schemas/metrics/single_player_report.schema.json` |
| Target player example | `configs/product/target_player.example.yaml` |
| Canonical tables (Stage 2C) | `schemas/data/v1/*`, `docs/data/canonical_contracts.md` |
| Stage interface / cache (Stage 2D) | `docs/development/stage_interface.md`, `docs/development/cache_design.md` |

Unknown results: `value: null` + `status` + `reason_not_evaluable` + `confidence` + `coverage`.

---

## 7. Stage boundary

This document does **not** authorize Stage 3 ingest code. Next implementation stage remains:

**Stage 3 — safe video ingest, probe, normalize, frame timebase**

Foundation closed at `foundation-v0.1.0` / `4b5089eeb8b022c95ed67a8ba1f60166b6058cdd`.
