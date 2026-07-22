# Current State Audit — 2026-07-22

**Gate decision:** `PASS_WITH_FINDINGS — CLOSED`

**Audit mode:** read-only (no installs, clones, fetches, downloads, mounts, or destructive git)

**Captured at:** 2026-07-22 (WSL Ubuntu 22.04.5)

**Closure revalidation at:** 2026-07-22 (GPU semantics correction + final verification)

Companion machine-readable evidence:

- `configs/environment/current_ai_dev.json`
- `configs/external/status_20260722.json`

### GPU/CUDA correction history

| Phase | Classification | Summary |
|-------|----------------|---------|
| First audit (~16:15) | Provisional / overstated | Agent saw `torch.cuda.is_available()=False`, NVML blocked, and missing `/dev/nvidia*`. Text incorrectly implied host CUDA failure risk from `/dev/nvidia*` absence. |
| Closure revalidation | **`AGENT_CONTEXT_GPU_UNVERIFIABLE`** | WSL GPU libs present under `/usr/lib/wsl/lib`; `/dev/dxg` absent in agent context; WSL `nvidia-smi` blocked; Windows `nvidia-smi.exe` present but not runnable via agent interop. Historical physical WSL tests (2026-07-10/12) remain valid evidence of RTX 3050 + CUDA True. **Not** a confirmed host GPU regression. |

Interpretation (normative):

> GPU durumu Agent çalışma bağlamından doğrulanamadı. Önceki başarılı WSL testi geçerli tarihsel kanıttır; GPU, Aşama 5 başlamadan önce gerçek inference smoke testiyle yeniden doğrulanacaktır.

Note: WSL2 GPU paravirtualization normally uses `/dev/dxg` and `/usr/lib/wsl/lib`. Missing `/dev/nvidia*` alone is **not** proof of WSL CUDA failure.

**Recommended next stage (name only):** Aşama 1 — Depolama, Veri Erişimi, Lisans ve Güvenlik

---

## 0. Pre-audit Git snapshot (user state)

| Field | Value |
|-------|-------|
| Repo | `/home/fdoblak/projects/football-analytics` |
| Branch | `main` |
| Commits | **None** (`No commits yet on main`) |
| HEAD | **N/A** (no revision) |
| Remote | **None** |
| Tags | none |
| Initial porcelain | `?? configs/` `?? external_repos.lock.yaml` `?? scripts/` `?? src/` |

Empty directories (`docs/`, `schemas/`, `tests/`, `notebooks/`, `patches/`, `workflows/`, `requirements/`) existed on disk but were not Git-tracked.

**Classification of initial tree:** user/bootstrap skeleton already present. Stage 0 only adds audit/governance artifacts under `docs/` and `configs/environment|external/` plus optional `scripts/audit_current_state.py`.

No `AGENTS.md`, `CLAUDE.md`, or `.cursor/rules` found.

---

## 1. Main repo audit

| Item | Class | Evidence |
|------|-------|----------|
| Repo root exists | VERIFIED | `/home/fdoblak/projects/football-analytics` |
| Active branch | VERIFIED | `main` (unborn / no commits) |
| HEAD full SHA | MISSING | No commits |
| Remote | MISSING | `git remote -v` empty |
| Last commit | MISSING | No history |
| Tags | MISSING | none |
| Working tree | INCONSISTENT | Untracked skeleton only; not a reproducible pinned revision |
| Top-level layout dirs | PRESENT_NOT_VERIFIED | Planned dirs exist; most empty |
| `src/football_analytics/*` | PRESENT_NOT_VERIFIED | Package dirs + empty `__init__.py` only |
| `configs/system/paths.yaml` | VERIFIED | Present; matches planned path map |
| `external_repos.lock.yaml` | VERIFIED | 19 SoccerNet entries; generated_at 2026-07-12 |
| `pyproject.toml` | MISSING | — |
| `environment.yml` | MISSING | — |
| `.gitignore` | MISSING | — |
| `README.md` | MISSING | — |
| `THIRD_PARTY_NOTICES.md` | MISSING | — |
| `LICENSE` | MISSING | — |
| `model_registry.yaml` | MISSING | Weights exist on disk but unregistered |
| `dataset_registry.yaml` | MISSING | — |
| `schemas/` | MISSING | Empty directory |
| `tests/` | MISSING | Empty; no pytest suite |
| CI / lint / type-check | MISSING | No `.github`, ruff/mypy/pre-commit configs |
| Prior setup reports in `docs/setup/` | MISSING | Directory empty (reports live on Windows Desktop copies) |
| Large binaries / secrets in repo | VERIFIED | No `>10MB` files; no `.pth/.mp4/.env` inside main repo |

---

## 2. `ai-dev` integrity

| Check | Expected (prior reports) | Actual (this audit) | Class |
|-------|--------------------------|---------------------|-------|
| Env | `ai-dev` | `ai-dev` @ `.../envs/ai-dev` | VERIFIED |
| Python | 3.10.20 | 3.10.20 | VERIFIED |
| torch | 2.11.0+cu128 | 2.11.0+cu128 | VERIFIED |
| torchvision / torchaudio | cu128 pair | match | VERIFIED |
| NumPy / pandas / OpenCV / Ultralytics | as prior | match | VERIFIED |
| FastAPI | present | 0.139.0 | VERIFIED |
| SoccerNet SDK | 0.1.62 | 0.1.62 | VERIFIED |
| `pip check` | clean | clean | VERIFIED |
| FFmpeg | 4.4.2 | 4.4.2-0ubuntu0.22.04.1 | VERIFIED |
| PyArrow | needed for Parquet contracts | **not installed** | MISSING |
| CUDA in agent context | True (historical host) | `torch.cuda.is_available()=False` | AGENT_CONTEXT_GPU_UNVERIFIABLE |
| Host GPU (historical) | RTX 3050 / 4 GB | previously verified 2026-07-10/12 | PRESENT_NOT_VERIFIED (this agent session) |
| `/usr/lib/wsl/lib` NVIDIA stack | expected | present (`nvidia-smi`, `libcuda`, driver libs) | VERIFIED |
| `/dev/dxg` in agent context | often present on healthy WSL GPU | absent here | PRESENT_NOT_VERIFIED / agent-limited |
| `/dev/nvidia*` | optional on WSL | absent | NOT_APPLICABLE as failure proof |
| `nvcc` | absent (by design) | absent | VERIFIED (expected absence) |
| opencv dual packages | both present | both 5.0.0.93 | VERIFIED |
| pytest/ruff/black/isort/mypy | not required historically | all MISSING | MISSING |
| `~/dev-check/check_env.py` | present | present; CUDA False in agent context | VERIFIED |

**Impact:** Package stack for MVP coding is largely intact. Agent-context CUDA non-availability is **not** treated as proven host regression. Mandatory GPU inference smoke is deferred to **before Stage 5**. PyArrow absence blocks canonical Parquet I/O until a later approved install (not Stage 0).

---

## 3. Storage and workspace

| Path | Class | Notes |
|------|-------|-------|
| `/mnt/d` | MISSING | Does not exist; not a mountpoint |
| `/mnt/d/football_data` | MISSING | Stage 1 blocker (not Stage 0 failure by itself) |
| `/home/fdoblak/workspace` | VERIFIED | Present |
| `runs/` `staging/` `cache/` | VERIFIED | Present, empty |
| `workspace/current` | MISSING | Symlink not created |
| `/home/fdoblak/logs` | VERIFIED | Present, empty |
| Host free space (`/`) | VERIFIED | ~926 GB available on ext4 |

No write probes or test files were created.

---

## 4. SoccerNet repositories (19/19)

All 19 expected clones exist, are non-shallow Git repos, clean working trees, and **HEAD full SHA matches** `external_repos.lock.yaml`.

| Repo | Short SHA | Disk | Lock match | Env | Smoke (this audit) |
|------|-----------|------|------------|-----|--------------------|
| sn-gamestate | 1c95834 | 196M | yes | venv 5.0G | tracklab/sn_gamestate/mmcv import + `tracklab --help` OK |
| sn-banner | f6d50b2 | 130M | yes | no dedicated | weights on disk; CLI not retested |
| sn-nvs | 1655ab1 | 395M | yes | no | blocked by missing `nvcc` |
| sn-teamspotting | 091fed2 | 4.8M | yes | conda | CLI NOT_RETESTED_SAFETY |
| sn-trackeval | 9c25232 | 2.4M | yes | conda | `trackeval` import via repo path OK |
| SoccerNet | 7446102 | 120M | yes | ai-dev pkg | SDK 0.1.62 import OK |
| sn-echoes | 7105a85 | 599M | yes | conda | `stats.py` SyntaxError |
| sn-depth | 9f6636f | 20M | yes | no | not in v1 core |
| sn-mvfoul | 502fb44 | 67M | yes | conda | NOT_RETESTED_SAFETY |
| sn-jersey | 2f43b48 | 212K | yes | no | docs-like kit |
| sn-calibration | ab38f46 | 5.7M | yes | conda | core imports OK; LFS attrs detected |
| sn-caption | c05973d | 1020K | yes | no | not in v1 core |
| sn-spotting | 9842826 | 555M | yes | no | not installed env |
| sn-reid | 621e2b0 | 92M | yes | conda | `torchreid` import FAIL |
| sn-tracking | b0bbba3 | 6.3M | yes | no | clone only |
| ActiveSpotting | 33a81cb | 700K | yes | conda | **torch missing** in env |
| PTS-baseline | af2ea82 | 24M | yes | conda | NumPy 2.x ABI warning; torch 1.11.0+cpu |
| sn-grounding | 910bf85 | 39M | yes | no | not in v1 core |
| SoccerNet-v3 | 7d483a8 | 15M | yes | no | clone only |

`git fsck` was **not re-run** in this audit.

No custom-MP4 end-to-end verification exists for any repo.

---

## 5. Third-party repositories

| Repo | Full SHA | Tag/describe | Dirty | Lock entry | Role |
|------|----------|--------------|-------|------------|------|
| tracklab | `5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2` | `v1.3.24` | clean | **MISSING** | Active in sn-gamestate venv + clone reference |
| pnlcalib | `8c87391d6f4ea40c5e4d65e61529916c7a49ce62` | `v1.0.0-58-g8c87391` | clean | **MISSING** | Calibration motor candidate |
| no-bells-just-whistles | `bd993b31c2917096c23bb8aadf148314d17f8345` | `v1.0.0-70-gbd993b3` | clean | **MISSING** | Banner/weights reference |

**Reproducibility risk:** third-party SHAs are discoverable locally but **not pinned** in `external_repos.lock.yaml`.

---

## 6. Isolated environments

| Name | Type | Python | Disk | Smoke | Blocker / note |
|------|------|--------|------|-------|----------------|
| ai-dev | conda | 3.10.20 | 8.2G | PASS | Agent CUDA unverifiable; no pyarrow |
| sn-trackeval | conda | 3.10.20 | ~700M | trackeval import PASS | — |
| sn-calibration | conda | 3.10.20 | ~720–814M | core import PASS | GD weights still needed for full use |
| sn-teamspotting | conda | 3.10.20 | 1.7G | interpreter PASS | model CLI not retested |
| sn-mvfoul | conda | 3.9.23 | 1.3G | interpreter PASS | NDA dataset |
| sn-pts-baseline | conda | 3.10.20 | 1.2G | numpy/torch warn | NumPy 2 ABI / torch 1.11 cpu |
| sn-reid | conda | 3.10.20 | 1.8G | FAIL | torchreid missing |
| sn-echoes | conda | 3.10.20 | ~155M | partial | `stats.py` syntax error |
| sn-active-spotting | conda | 3.10.20 | 643M | FAIL | torch not installed in env |
| sn-gamestate (conda) | conda | 3.9.23 | ~151M | interpreter PASS | thin base for uv/venv |
| sn-gamestate-venv | venv | 3.9.23 | 5.0G | tracklab/mmcv/sn_gamestate PASS | weights+dataset absent |

---

## 7. Models and datasets

| Asset | Status |
|-------|--------|
| `SV_kp.pth` | Present (264964645 bytes); SHA-256 `7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113` |
| `SV_lines.pth` | Present (264857893 bytes); SHA-256 `2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1` |
| Registry files | MISSING in main repo |
| Broadcast / tracking datasets | MISSING (not downloaded) |
| Demo MP4s (17) | Present inside cloned repos only — **not** user datasets |

Hashes match the short prefixes from the 2026-07-12 final install report.

---

## 8. Deviations vs prior written expectations

1. **Agent-context CUDA unverifiable** (`AGENT_CONTEXT_GPU_UNVERIFIABLE`); historical host CUDA True on 2026-07-10/12 remains standing evidence — not reclassified as host failure.
2. **Main repo had zero commits at first audit** (foundation incomplete; Stage 0 closure may add the first checkpoint only).
3. **Foundation files missing:** `pyproject.toml`, `.gitignore`, README, registries, schemas, tests → Aşama 2 scope.
4. **PyArrow missing** from `ai-dev`.
5. **Third-party not locked** despite local clones.
6. **`sn-active-spotting` env lacks torch** (weaker than prior “partially ready” narrative).
7. **Install markdown reports** are not inside `docs/setup/` (only empty dir).
8. **`/mnt/d` still absent** (known; confirmed again) → Aşama 1 blocker.
9. sn-mvfoul demo clip count observed as part of 17 total demo MP4s (14 under sn-mvfoul + 3 third-party).

---

## 9. Safety attestation

- No `sudo`, package installs, clones, fetches, pulls, checkouts, downloads, mounts, or deletes during Stage 0 work.
- No edits outside `/home/fdoblak/projects/football-analytics`.
- Pre-existing bootstrap file contents (`paths.yaml`, lock, setup scripts, empty package inits) were not mutated for audit purposes; only Stage 0 artifacts were added/updated.
- Aşama 1 not started. Push/tag not created as part of audit authoring (closure commit is separate, local-only if approved by closure checklist).
