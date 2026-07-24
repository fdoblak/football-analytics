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
- `scripts/check_identity_contracts.py` (Stage 7A ReID / identity evidence / target-player contracts)
- `scripts/check_appearance_reid_baseline.py` (Stage 7B appearance embedding + tracklet ReID baseline)
- `scripts/check_team_assignment_baseline.py` (Stage 7C anonymous team assignment baseline)
- `scripts/check_jersey_ocr_baseline.py` (Stage 7D jersey region + OpenCV template OCR baseline)
- `scripts/check_target_identity_pipeline.py` (Stage 7E target evidence fusion + manual approval)
- `scripts/check_calibration_contracts.py` (Stage 8A pitch calibration / homography / coordinate contracts)
- `scripts/check_pitch_feature_baseline.py` (Stage 8B pitch keypoint / line detection baseline)
- `scripts/check_homography_baseline.py` (Stage 8C homography solve + calibration segments)
- `scripts/check_pitch_projection_pipeline.py` (Stage 8D pitch projection + Stage 8 close)
- `scripts/check_physical_metric_contracts.py` (Stage 9A trajectory / physical metric contracts)
- `scripts/check_target_trajectory_baseline.py` (Stage 9B trajectory prepare + quality baseline)
- `scripts/check_human_ball_interaction_contracts.py` (Stage 10A human-ball interaction contracts)
- `scripts/check_human_ball_proximity_contact_baseline.py` (Stage 10B proximity / contact baseline)
- `scripts/check_possession_control_baseline.py` (Stage 10C possession / control baseline)
- `scripts/check_human_ball_interaction_pipeline.py` (Stage 10D fusion + Stage 10 close)
- `scripts/check_passing_contracts.py` (Stage 11A pass / reception / progression contracts)
- `scripts/check_pass_reception_baseline.py` (Stage 11B pass / reception baseline)
- `scripts/check_passing_metrics_baseline.py` (Stage 11C target passing metrics baseline)
- `scripts/check_passing_pipeline.py` (Stage 11D fusion + Stage 11 close)
- `scripts/collect_evidence.py` (small safe evidence backfill into `artifacts/evidence/`)
- `scripts/check_stage_cache.py`, `check_ci_workflow.py`, `check_project.py`
- Or: `football-analytics project check --profile local --quick`

Stage 5 is closed (`detection-baseline-v0.5.0`). Stage 6 is closed
(`tracking-baseline-v0.6.0`): contracts → human/ball MOT baselines → tracking
fusion + quality gates. Stage 7 is closed (`identity-baseline-v0.7.0`):
7A identity contracts → 7B appearance ReID → 7C anonymous team assignment →
7D jersey OCR → 7E evidence fusion + manual target workflow. Appearance / team /
jersey alone cannot confirm; confirmed requires scoped manual decision;
`auto_confirm=false`; face forbidden; real football identity accuracy not
validated. Stage 8 is closed (`calibration-baseline-v0.8.0`): 8A contracts →
8B pitch features → 8C homography/segments → 8D projected positions. Attack
direction remains unknown; ball never physical/event metric-eligible; real
football coordinate accuracy not validated. Stage 9 is closed
(`physical-metrics-baseline-v0.9.0`): 9A contracts → 9B trajectory → 9C
distance/speed/sprint → 9D heatmap/zones/activity → 9E fusion. Real football
accuracy is not validated; official Opta data was not used; final customer
visual is deferred. Stage 10 is closed: 10A contracts → 10B proximity/contact →
10C possession/control → 10D fusion. Automatic ceiling remains provisional;
nearest ≠ owner; missing ball ≠ loose/no-possession; real football interaction
accuracy is not validated. Stage 11 is closed: 11A contracts → 11B pass/reception →
11C metrics → 11D fusion. Owner change alone ≠ completed pass; cut/replay/gap → no
pass; attack direction unknown → directional metrics not_evaluable; penalty presence
≠ box touch; real football passing accuracy is not validated. Do **not** start
Stage 12 without an explicit user prompt.

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
(`tracking-baseline-v0.6.0`). Stage 6 is **closed**. Stage 7A–7E identity /
target-player baseline is in-tree when merged (`identity-baseline-v0.7.0`).
Stage 7 is **closed**. Stage 8A–8D pitch calibration / projection baseline is
in-tree when merged (`calibration-baseline-v0.8.0`). Stage 8 is **closed**.
Stage 9A–9E target-player physical metrics baseline is in-tree when merged
(`physical-metrics-baseline-v0.9.0`). Stage 9 is **closed**. Stage 10A–10D
human-ball interaction baseline is in-tree when merged (contracts → proximity →
possession → fusion). Stage 10 is **closed**. Stage 11A–11D passing / reception /
progression baseline is in-tree when merged (contracts → pass/reception → metrics →
fusion). Stage 11 is **closed**. Do **not** start Stage 12 without an explicit
user prompt.
Manual Cursor flow only — no Codex/background automation.
