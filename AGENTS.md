# AGENTS.md — football-analytics

Permanent instructions for **manual Cursor** development (single-stage prompts).
Product foundation checkpoint after Stage 2 (`foundation-v0.1.0` /
`4b5089eeb8b022c95ed67a8ba1f60166b6058cdd`).

---

## 1. Product scope

Final customer product: process a full match video and produce an
**evidence-based individual performance report for ONE `target_player`**.

- Team / opponent / referee / ball / pitch are **context only**.
- Final output is **not** a general team report.
- Scope docs: `docs/scope/single_player_product_v1.md`
- Metrics: `docs/metrics/single_player_metric_dictionary.md`
- Architecture: `docs/architecture/single_player_pipeline.md`

Identity uncertain → `manual_identity_confirmation_required` / `identity_uncertain`.
Never fake metric `0`; use explicit not-evaluable reasons.

Opta: no scraping; license-only official data; otherwise label
`project-generated` / `opta_style_project` / `video_derived`.

---

## 2. Manual Cursor workflow

| Role | Behavior |
|------|----------|
| **User** | Issues one explicit prompt for one stage / sub-stage |
| **Cursor** | Implements only that stage, reports results, then **stops** |

Rules:

- Do **not** call Codex CLI, Codex MCP, or any Codex reviewer.
- Do **not** start background timers, systemd services, or overnight automation.
- Do **not** invent the next stage or queue follow-up work without a new user prompt.
- After tests / commit / push / report for the requested stage: **stop and wait**.
- Continuous Cursor+Codex automation has been **removed**; do not recreate it unless the user explicitly requests it.

---

## 3. Tests and gates

Reuse Stage 2 validators and tests before claiming progress:

- `python -m unittest discover -s tests -p "test_*.py" -v`
- `python -m pytest`
- `ruff` / `black --check` / `isort --check-only` / `mypy src/football_analytics` as applicable
- `scripts/check_storage.py`, `check_registries.py`, `check_secrets.py`
- `scripts/check_runtime_foundation.py`, `check_data_contracts.py`
- `scripts/check_broadcast_contracts.py` (Stage 4A shot/camera contracts)
- `scripts/check_shot_boundary_baseline.py` / `check_camera_view_baseline.py` (4B/4C)
- `scripts/check_broadcast_pipeline.py` (Stage 4D fusion + routing)
- `scripts/check_detection_contracts.py` (Stage 5A player/official/ball detection contracts)
- `scripts/check_human_detection_baseline.py` (Stage 5B human detection baseline)
- `scripts/check_ball_detection_baseline.py` (Stage 5C ball detection baseline)
- `scripts/check_human_role_baseline.py` (Stage 5D human role classification baseline)
- `scripts/check_detection_pipeline.py` (Stage 5E detection fusion + quality gates)
- `scripts/check_tracking_contracts.py` (Stage 6A multi-object tracking contracts)
- `scripts/check_human_tracking_baseline.py` (Stage 6B human MOT baseline)
- `scripts/check_ball_tracking_baseline.py` (Stage 6C ball tracking baseline)
- `scripts/check_tracking_pipeline.py` (Stage 6D human+ball tracking fusion + quality gates)
- `scripts/check_stage_cache.py`, `check_ci_workflow.py`, `check_project.py`
- Or: `football-analytics project check --profile local --quick`

Stage 5 is closed (`detection-baseline-v0.5.0`). Stage 6 is closed
(`tracking-baseline-v0.6.0`): contracts → human/ball MOT baselines → tracking
fusion + quality gates. Do **not** start Stage 7A (ReID / identity evidence /
target-player contracts) without an explicit user prompt.

---

## 4. Git / GitHub

- No force push; no history rewrite of published commits.
- Normal `git push` only.
- Stage only explicit paths — avoid blind `git add .` / `git add -A`.
- Never commit secrets, datasets, videos, model binaries, or caches.
- See `docs/development/git_github_workflow.md`.

---

## 5. Resource and video cleanup

Automatic cleanup only when **all** are true:

- `preexisting=false`
- `automation_owned=true`
- resource ledger entry exists
- exact path/package known
- not shared by another component
- not user data

Never delete user videos, datasets, models, repos, environments, reports, or
SoccerNet original clones. Network video download default: **false**.

---

## 6. External repo isolation (SoccerNet)

- Path: `/home/fdoblak/projects/soccernet` (19 locked repos)
- Treat as locked reference; adapters only in this project
- Do not dirty original clones; do not auto-fetch/pull/update lock SHAs
- GPL caution — document license fit in ADR before deep integration
- Prefer custom project code + ADR when SoccerNet is unfit

---

## 7. Protected packages

Do not upgrade / uninstall casually (pins in `requirements/constraints-ai-dev.txt`):

- `torch`, `torchvision`, `torchaudio`
- `numpy`, `pandas`
- `opencv-python`, `opencv-python-headless`
- `ultralytics`, `SoccerNet`, `pyarrow`

---

## 8. Open findings — do not mark solved

These remain open until independently verified and closed by policy:

- GitHub API proxy 403 (Cursor agent)
- Remote CI unverifiable in agent API context
- Registry license / access warnings
- GPU host gate unverifiable (`AGENT_CONTEXT_GPU_UNVERIFIABLE`)
- Same-VHDX local archive ≠ independent backup
- RISK-029 (large-table validation / pylist memory pressure; mitigated for Stage 3D frame timeline streaming write path)
- Cache GC / automatic purge absent

---

## 9. Reports path (audit only)

| Item | Path |
|------|------|
| Windows reports (WSL) | `/mnt/c/Users/furka/Desktop/Cursor&Codex Reports` |
| Manual handoff / preservation | `/home/fdoblak/workspace/manual_handoff/` |

Network defaults: `network_video_download_allowed: false`,
`dataset_download_allowed: false`, `large_model_download_allowed: false`.

---

## 10. Stage bootstrap vs later stages

Product-scope docs/configs/schemas on `main` are accepted.
Do not modify SoccerNet repos.
Do not change protected packages.
Do not start a stage without an explicit user prompt.
Stage 4A–4D broadcast understanding baseline is in-tree when merged.
Stage 5A–5E detection baseline is in-tree when merged (`detection-baseline-v0.5.0`).
Stage 5 is **closed**. Stage 6A–6D tracking baseline is in-tree when merged
(`tracking-baseline-v0.6.0`). Stage 6 is **closed**. Stage 7A is **not**
started unless explicitly requested.
Manual Cursor flow only — no Codex/background automation.
