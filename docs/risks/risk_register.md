# Risk Register — football-analytics

**Updated:** 2026-07-22 (Stage 0)

**Owner default:** Furkan Doblak unless stated otherwise

Probability / impact scale: `low` | `medium` | `high` | `critical`

---

## RISK-001 — RTX 3050 4 GB VRAM / OOM

| Field | Value |
|-------|-------|
| risk_id | RISK-001 |
| description | Detection/tracking/calibration models may exceed 4 GB VRAM, causing OOM or thrashing. |
| probability | high |
| impact | high |
| mitigation | Nano/small models; batch 1; input ≤640; FP16; sequential GPU stages; record peak VRAM. |
| trigger | CUDA OOM errors; peak VRAM near 4096 MiB |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 2+ (detection/tracking) |

## RISK-002 — `/mnt/d` mount and data loss / unavailable durable storage

| Field | Value |
|-------|-------|
| risk_id | RISK-002 |
| description | Planned SSD root `/mnt/d/football_data` does not exist; archives and large datasets have no durable mount. Risk of storing irreplaceable data only on WSL ext4 or losing data if mounts are misconfigured later. |
| probability | high |
| impact | critical |
| mitigation | Confirm Stage 1 storage design; never delete sources; checksummed archive protocol; do not invent mounts in Stage 0. |
| trigger | `/mnt/d` missing (verified 2026-07-22) |
| owner | Furkan Doblak |
| status | open — **Stage 1 blocker** |
| target_stage | Stage 1 |

## RISK-003 — Broadcast occlusion and off-camera players

| Field | Value |
|-------|-------|
| risk_id | RISK-003 |
| description | Single broadcast camera occlusions and players leaving frame create fragmented tracks and biased metrics. |
| probability | high |
| impact | high |
| mitigation | Coverage reporting; null metrics when coverage insufficient; avoid over-claiming lineup-complete analytics. |
| trigger | High ID switches; low frame coverage per track |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-1+ |

## RISK-004 — Ball small and frequently lost

| Field | Value |
|-------|-------|
| risk_id | RISK-004 |
| description | Ball detections are sparse; trajectory gaps break possession and event logic. |
| probability | high |
| impact | high |
| mitigation | Ball-specific recall focus; temporal smoothing; explicit lost-ball state; do not fabricate possession. |
| trigger | Low ball recall; long gaps in `ball_state` |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-3 |

## RISK-005 — Calibration failure under cut / pan / zoom

| Field | Value |
|-------|-------|
| risk_id | RISK-005 |
| description | Camera motion and shot changes invalidate homography; pitch metrics become wrong. |
| probability | high |
| impact | high |
| mitigation | Per-frame validity flags; reprojection error gates; invalidate game_state when calibration confidence low. |
| trigger | High reprojection error; shot-type changes |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-2 |

## RISK-006 — Re-ID false merge

| Field | Value |
|-------|-------|
| risk_id | RISK-006 |
| description | Incorrect identity merges corrupt player timelines and event attribution. |
| probability | medium |
| impact | high |
| mitigation | Conservative merge thresholds; jersey as supporting signal; measure before enabling; keep confidence/reason. |
| trigger | Identity flips; jersey conflicts |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-2 / identity stage |

## RISK-007 — Legacy external repo dependencies

| Field | Value |
|-------|-------|
| risk_id | RISK-007 |
| description | Isolated envs pin old stacks (e.g. PTS torch 1.11, ActiveSpotting without torch, sn-reid missing torchreid, sn-nvs needs nvcc). Breaks or traps engineering time. |
| probability | high |
| impact | medium |
| mitigation | Keep externals isolated; prefer adapters; pin commits; do not force into `ai-dev`. |
| trigger | Import/build failures (observed for reid, active-spotting, pts ABI warnings) |
| owner | Furkan Doblak |
| status | open |
| target_stage | ongoing |

## RISK-008 — NDA / license access barriers

| Field | Value |
|-------|-------|
| risk_id | RISK-008 |
| description | Broadcast and some task datasets require NDA/passwords; blocked downloads stall evaluation. |
| probability | high |
| impact | high |
| mitigation | Progress on golden private clips first; labels-only where allowed; track NDA status explicitly. |
| trigger | SoccerNet download auth failures |
| owner | Furkan Doblak |
| status | open |
| target_stage | dataset stages |

## RISK-009 — Misleading metrics without coverage reporting

| Field | Value |
|-------|-------|
| risk_id | RISK-009 |
| description | Distance/speed/heatmaps look plausible while based on partial tracks. |
| probability | medium |
| impact | high |
| mitigation | Mandatory coverage fields; quality gates before report publishing. |
| trigger | Metrics emitted with low coverage |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-2 |

## RISK-010 — Scope creep

| Field | Value |
|-------|-------|
| risk_id | RISK-010 |
| description | Research modules (NVS, caption, foul, echoes, etc.) pull effort away from v1.0 core. |
| probability | high |
| impact | high |
| mitigation | `docs/scope/product_v1_scope.md` freeze; ADR change control; local clone ≠ in-scope. |
| trigger | Requests to productize out-of-scope repos |
| owner | Furkan Doblak |
| status | mitigated (freeze written) / monitor |
| target_stage | Stage 0+ |

## RISK-011 — Third-party exact SHA not locked

| Field | Value |
|-------|-------|
| risk_id | RISK-011 |
| description | `tracklab`, `pnlcalib`, and `no-bells-just-whistles` lack entries in `external_repos.lock.yaml` despite local clones. |
| probability | medium |
| impact | medium |
| mitigation | Add lock entries in a later approved stage; until then record SHAs in audit JSON. |
| trigger | Stage 0 audit 2026-07-22 |
| owner | Furkan Doblak |
| status | open |
| target_stage | foundation / Stage 1 prep |

## RISK-012 — Dataset / model absence

| Field | Value |
|-------|-------|
| risk_id | RISK-012 |
| description | No broadcast datasets downloaded; most challenge weights absent; registries missing. Banner weights exist but unregistered. |
| probability | high |
| impact | high |
| mitigation | Golden clips first; registries before bulk download; user-approved downloads only. |
| trigger | Missing `dataset_registry.yaml` / empty datasets |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 1+ |

## RISK-013 — No custom MP4 end-to-end verification yet

| Field | Value |
|-------|-------|
| risk_id | RISK-013 |
| description | No repo has proven custom-MP4 e2e in this environment; prior install stopped at non-video smoke. |
| probability | high |
| impact | high |
| mitigation | Golden clip pipeline in MVP-1; forbid claiming e2e readiness before evidence. |
| trigger | Stage 0 audit confirmation |
| owner | Furkan Doblak |
| status | open |
| target_stage | MVP-1 |

## RISK-014 — Agent-context GPU/CUDA unverifiable (not confirmed host regression)

| Field | Value |
|-------|-------|
| risk_id | RISK-014 |
| description | Cursor Agent execution context cannot verify CUDA: `torch.cuda.is_available()` False, WSL `nvidia-smi` reports NVML blocked, `/dev/dxg` absent in-agent, Windows `nvidia-smi.exe` not runnable via agent interop. `/usr/lib/wsl/lib` NVIDIA stack is present. Historical physical WSL tests (2026-07-10/12) showed RTX 3050 + CUDA True. Missing `/dev/nvidia*` is not WSL failure evidence by itself. |
| probability | high for agent sessions / unknown for interactive host without retest |
| impact | high for GPU inference stages if host were actually broken; currently treated as verification gap |
| mitigation | Do not change drivers/CUDA/WSL in Stage 0–4 for this finding. Require a real GPU inference smoke test before Stage 5. Record classification `AGENT_CONTEXT_GPU_UNVERIFIABLE`. |
| trigger | Stage 0 closure revalidation 2026-07-22 |
| owner | Furkan Doblak |
| status | open — **mandatory gate before Stage 5** (does not block Stage 0 closure) |
| target_stage | before Stage 5 |

## RISK-015 — Incomplete foundation files (initial git checkpoint may exist)

| Field | Value |
|-------|-------|
| risk_id | RISK-015 |
| description | `football-analytics` lacked commits at first audit and still misses `pyproject.toml`, `.gitignore`, README, registries, schemas, and tests. Stage 0 closure may create only an initial baseline commit; foundation completeness remains Aşama 2 work. |
| probability | high (observed) |
| impact | medium |
| mitigation | Complete foundation under Aşama 2 brief; commit/tag only with approval. Do not use `v0.1.0-foundation` until Aşama 2 completes. |
| trigger | Stage 0 git audit |
| owner | Furkan Doblak |
| status | open |
| target_stage | Aşama 2 — Ana Repo Foundation ve Canonical Sözleşmeler |

## RISK-016 — WSL local storage consumes Windows C: VHDX capacity

| Field | Value |
|-------|-------|
| risk_id | RISK-016 |
| description | Active storage `/home/fdoblak/football_data` lives on WSL ext4 (`/dev/sdd`) inside the Windows C: VHDX. Large datasets/videos can fill C: unexpectedly. D: archive backend remains `planned_unverified`. |
| probability | medium |
| impact | high |
| mitigation | Free-space gate (warn below 100 GiB); per-run size telemetry; when D: is verified, checksummed migration to `/mnt/d/football_data` then switch active backend (ADR-0002). |
| trigger | Stage 1A-R1 selected `WSL_LOCAL_PRIMARY_D_UNVERIFIED` |
| owner | Furkan Doblak |
| status | open — accepted finding while local backend is active |
| target_stage | Stage 1D / operations |
