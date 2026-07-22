# AGENTS.md — football-analytics

Permanent instructions for Cursor (executor) and Codex (read-only supervisor).
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

## 2. Roles

| Agent | Role | Write |
|-------|------|-------|
| **Cursor** | Executor / writer | Allowed project / automation worktree only |
| **Codex** | Supervisor | **Read-only** — review, gates, structured decisions |

Do not let both agents write the same worktree concurrently.

Codex decisions (automation mode): `APPROVE_TASK`, `REQUEST_FIX`,
`APPROVE_CHECKPOINT`, `ABANDON_SAFE`, `BLOCKED`, `STOP_SCOPE_COMPLETE`,
`STOP_SAFETY`.

---

## 3. Tests and gates

Reuse Stage 2 validators and tests before claiming progress:

- `python -m unittest discover -s tests -p "test_*.py" -v`
- `python -m pytest`
- `ruff` / `black --check` / `isort --check-only` / `mypy src/football_analytics` as applicable
- `scripts/check_storage.py`, `check_registries.py`, `check_secrets.py`
- `scripts/check_runtime_foundation.py`, `check_data_contracts.py`
- `scripts/check_stage_cache.py`, `check_ci_workflow.py`, `check_project.py`
- Or: `football-analytics project check --profile local --quick`

Do not claim Stage 3 started until ingest work is explicitly in progress.

---

## 4. Git / GitHub

- No force push; no history rewrite of published commits.
- Normal `git push` only.
- In automation mode: Codex **`APPROVE_CHECKPOINT`** required before commit and before push.
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
- RISK-029 (large-table validation / pylist memory pressure)
- Cache GC / automatic purge absent

---

## 9. Continuous automation paths

| Item | Path |
|------|------|
| Automation workspace | `/home/fdoblak/workspace/agent_automation` |
| Windows reports (WSL) | `/mnt/c/Users/furka/Desktop/Cursor&Codex Reports` |
| PAUSE | `/home/fdoblak/workspace/agent_automation/PAUSE` |
| STOP | `/home/fdoblak/workspace/agent_automation/STOP` |

If `PAUSE` exists: do not start a new cycle.
If `STOP` exists: safe cleanup + session summary, then exit.
Do not delete PAUSE/STOP yourselves.

Network defaults: `network_video_download_allowed: false`,
`dataset_download_allowed: false`, `large_model_download_allowed: false`.

---

## 10. Stage bootstrap vs Stage 3

This product-scope bootstrap may add docs, configs, and schemas only.
Do not modify SoccerNet repos.
Do not change protected packages.
Do not add pipeline execution code beyond docs/configs/schemas in this bootstrap.
