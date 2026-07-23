# Risk Register — football-analytics

**Updated:** 2026-07-22 (Stage 2D)

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
| status | open — **storage availability finding** (local backend active; D: still unverified) |
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
| description | Active storage `/home/fdoblak/football_data` and workspace runs live on WSL ext4 inside the Windows C: VHDX. Large archives/datasets can fill C:. Active archive shares this failure domain — it is a verified local archive, **not** an independent/off-device backup. D: remains `planned_unverified`. |
| probability | medium |
| impact | high |
| mitigation | Free-space gate on archive; full archive verification + restore smoke; quarantine instead of delete; checksummed migration when D: verified; release-gate real backup test on independent media. |
| trigger | Stage 1A-R1 `WSL_LOCAL_PRIMARY_D_UNVERIFIED`; Stage 1D archive policy |
| owner | Furkan Doblak |
| status | open — accepted finding while local backend is active |
| target_stage | operations / post-1D migration |

## RISK-017 — Quarantine retention without automatic purge

| Field | Value |
|-------|-------|
| risk_id | RISK-017 |
| description | Cleanup moves runs to `workspace/quarantine` with retention_days recorded but Stage 1D does not auto-purge. Quarantine can grow. |
| probability | medium |
| impact | medium |
| mitigation | Monitor quarantine size; future explicit purge tool with confirmations; never silent `rm -rf`. |
| trigger | Stage 1D cleanup design |
| owner | Furkan Doblak |
| status | open |
| target_stage | post-1D operations |

## RISK-018 — Planned D: archive still unverified

| Field | Value |
|-------|-------|
| risk_id | RISK-018 |
| description | `/mnt/d/football_data` archive path is configured as planned only. Tools must not treat it as active. |
| probability | high (current) |
| impact | high if operators assume DR coverage |
| mitigation | `independent_backup: false`; docs/ADR-0003; no `/mnt/d` writes until verified mount + checksum migration. |
| trigger | Stage 1A–1D |
| owner | Furkan Doblak |
| status | open |
| target_stage | when D: interop verified |

## RISK-019 — Config drift across runs

| Field | Value |
|-------|-------|
| risk_id | RISK-019 |
| description | Unversioned or mutable configs can make runs unreproducible. |
| probability | medium |
| impact | high |
| mitigation | Stage 2B resolved config + SHA-256 fingerprint; schema_version; immutable resolved mapping; defaults in Git. |
| trigger | Stage 2B |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2B+ |

## RISK-020 — Secret leakage in logs/records

| Field | Value |
|-------|-------|
| risk_id | RISK-020 |
| description | Tokens/passwords may appear in logs, JSON records, or CLI output. |
| probability | medium |
| impact | critical |
| mitigation | Central redaction; secret keys forbidden in config/records; JSONL redaction; no env dump; check_secrets gate. |
| trigger | Stage 2B logging/records |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | ongoing |

## RISK-021 — Environment provenance incompleteness

| Field | Value |
|-------|-------|
| risk_id | RISK-021 |
| description | Allowlisted package snapshot may omit transitive deps; GPU remains unverified in agent context. |
| probability | medium |
| impact | medium |
| mitigation | Document allowlist; GPU=`AGENT_CONTEXT_GPU_UNVERIFIABLE`; expand allowlist deliberately later. |
| trigger | Stage 2B environment record |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 2B+ |

## RISK-022 — Hash TOCTOU / mid-read mutation

| Field | Value |
|-------|-------|
| risk_id | RISK-022 |
| description | Files may change while hashing, yielding misleading digests. |
| probability | low |
| impact | medium |
| mitigation | Stat before/after during `sha256_file`; fail closed on mismatch. |
| trigger | Stage 2B hashing |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2B+ |

## RISK-023 — Partial runtime initialization

| Field | Value |
|-------|-------|
| risk_id | RISK-023 |
| description | Failed mid-init run dirs could leave partial JSON/log artifacts. |
| probability | medium |
| impact | medium |
| mitigation | Transactional init with cleanup on failure; no-overwrite; validator synthetic cleanup. |
| trigger | Stage 2B run_context |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2B+ |

## RISK-024 — GitHub API proxy finding carry-over

| Field | Value |
|-------|-------|
| risk_id | RISK-024 |
| description | Cursor Agent cannot call `api.github.com` (proxy 403). Visibility/API metadata cannot be independently verified from Agent. |
| probability | high (observed) |
| impact | low for Git sync via smart HTTPS; medium for API automation |
| mitigation | Git smart HTTPS only; user-confirmed private repo; do not claim API visibility proof. |
| trigger | Stage 2A-GH |
| owner | Furkan Doblak |
| status | open — accepted finding |
| target_stage | operations / tooling |

## RISK-025 — Schema drift across pipeline stages

| Field | Value |
|-------|-------|
| risk_id | RISK-025 |
| description | Divergent table layouts between producers/consumers break joins and analytics. |
| probability | medium |
| impact | high |
| mitigation | Stage 2C registry + strict Parquet metadata fingerprint; contract CLI/validator. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2C+ |

## RISK-026 — Semantic mismatch / unit confusion

| Field | Value |
|-------|-------|
| risk_id | RISK-026 |
| description | BBox conventions or time domains may be misinterpreted (xyxy vs xywh; video-relative vs UTC). |
| probability | medium |
| impact | high |
| mitigation | Explicit table_metadata + semantic rules; docs in canonical_contracts.md. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2C+ |

## RISK-027 — Lossy schema migration

| Field | Value |
|-------|-------|
| risk_id | RISK-027 |
| description | Migrations may drop or invent fields incorrectly. |
| probability | medium |
| impact | high |
| mitigation | Explicit edges only; receipt with hashes; detections 0→1 lossless PK/order; no overwrite. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2C+ |

## RISK-028 — Parquet/PyArrow compatibility

| Field | Value |
|-------|-------|
| risk_id | RISK-028 |
| description | Writer/reader metadata or list field naming differences can falsely fail validation. |
| probability | medium |
| impact | medium |
| mitigation | Pin pyarrow 25.0.0; type-compatible comparison; zstd + metadata stamps. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 2C+ |

## RISK-029 — Validation memory pressure

| Field | Value |
|-------|-------|
| risk_id | RISK-029 |
| description | Large tables validated via pylist conversion may use excess memory. |
| probability | medium |
| impact | medium |
| mitigation | Bounded error lists; document complexity; future Arrow-compute paths. Stage 3D adds streaming `write_contract_parquet_streaming` for the frame timeline write path (batched ParquetWriter; no full-frame pylist on timeline write). Stage 3D-F1 taxonomy work does **not** close general pylist / materialize-join pressure. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (frame timeline streaming write path); open (general semantic validation pylist + materialize metadata join) |
| target_stage | Stage 2C+/2D/3D |

## RISK-030 — Cross-table orphan data

| Field | Value |
|-------|-------|
| risk_id | RISK-030 |
| description | Orphan detections/tracks/events break analytics integrity. |
| probability | medium |
| impact | high |
| mitigation | Bundle FK validation; synthetic E2E; pipeline stages must pass validator. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2C+ |

## RISK-031 — Metadata tampering

| Field | Value |
|-------|-------|
| risk_id | RISK-031 |
| description | Parquet contract fingerprint metadata may be altered without content change. |
| probability | low |
| impact | medium |
| mitigation | Reader verifies fingerprint against registry contract; tamper tests. |
| trigger | Stage 2C |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2C+ |

## RISK-032 — Cache poisoning / corruption

| Field | Value |
|-------|-------|
| risk_id | RISK-032 |
| description | Tampered or partially written cache entries could restore incorrect artifacts into later stages. |
| probability | medium |
| impact | high |
| mitigation | Verify-on-publish/read; symlink/special/hardlink rejection; recompute cache key; quarantine corrupt entries (no permanent delete). |
| trigger | Stage 2D cache hit path |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-033 — Cache key omission

| Field | Value |
|-------|-------|
| risk_id | RISK-033 |
| description | Omitting code/config/input/contract fingerprints from the cache key yields false hits across incompatible executions. |
| probability | medium |
| impact | high |
| mitigation | Canonical key payload includes stage identity, fingerprints, ordered inputs, output contracts; unit tests for byte/schema/config diffs. |
| trigger | Stage 2D key design |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-034 — Nondeterministic stage caching

| Field | Value |
|-------|-------|
| risk_id | RISK-034 |
| description | Caching a nondeterministic stage publishes unreproducible outputs as reusable hits. |
| probability | medium |
| impact | high |
| mitigation | Publish only when `deterministic` and `cacheable` and policy/request enable cache; `force_miss` escape hatch. |
| trigger | Stage 2D execute_stage |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-035 — Concurrent publish race

| Field | Value |
|-------|-------|
| risk_id | RISK-035 |
| description | Two processes publishing the same key could corrupt the final entry or leave partial trees visible. |
| probability | medium |
| impact | high |
| mitigation | flock per key; stage into temp; atomic rename; if final exists, discard temp and re-verify winner. |
| trigger | Stage 2D concurrent publish |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-036 — Partial stage output

| Field | Value |
|-------|-------|
| risk_id | RISK-036 |
| description | Failed or interrupted stage execution leaves partial outputs that look like valid results or get cached. |
| probability | medium |
| impact | high |
| mitigation | Failures return `failed` with no publish; outputs verified before publish; restore refuses overwrite; temp cleanup on publish failure. |
| trigger | Stage 2D execution |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-037 — CI / local environment drift

| Field | Value |
|-------|-------|
| risk_id | RISK-037 |
| description | Local `ai-dev` GPU stack and CI lightweight deps diverge, hiding host-only failures or falsely failing CI. |
| probability | high |
| impact | medium |
| mitigation | Separate `requirements/ci.txt`; project-check `ci` vs `local` profiles with explicit host SKIP; local-equivalent CI commands documented. |
| trigger | Stage 2D CI |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 2D+ |

## RISK-038 — CI supply-chain action references

| Field | Value |
|-------|-------|
| risk_id | RISK-038 |
| description | Mutable or third-party GitHub Actions could inject untrusted workflow code. |
| probability | medium |
| impact | high |
| mitigation | Official actions only; full 40-hex SHA pins; `check_ci_workflow.py` allowlist + pin enforcement; `contents: read` only. |
| trigger | Stage 2D workflow |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-039 — Validator false PASS

| Field | Value |
|-------|-------|
| risk_id | RISK-039 |
| description | Host-only skips or swallowed exceptions could be reported as PASS, masking real failures. |
| probability | medium |
| impact | high |
| mitigation | Explicit SKIP records; per-check isolation to FAIL; strict mode promotes WARN; CI profile must not PASS host probes. |
| trigger | Stage 2D project check |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-040 — Validation subprocess timeout

| Field | Value |
|-------|-------|
| risk_id | RISK-040 |
| description | Hung validators inside project check can stall CI/local gates indefinitely. |
| probability | medium |
| impact | medium |
| mitigation | `shell=False`, 120s subprocess timeout, bounded capture; hung check recorded as FAIL. |
| trigger | Stage 2D project check |
| owner | Furkan Doblak |
| status | mitigated (foundation) |
| target_stage | Stage 2D+ |

## RISK-041 — Cache disk growth / no GC

| Field | Value |
|-------|-------|
| risk_id | RISK-041 |
| description | Content-addressed cache grows without bound because automatic purge/GC is disabled. |
| probability | high |
| impact | medium |
| mitigation | `automatic_purge: false` by design; publish-time size/file caps; manual operator cleanup later; document limitation. |
| trigger | Stage 2D cache use |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 3+ (policy) |

## RISK-042 — GitHub API proxy / remote CI visibility

| Field | Value |
|-------|-------|
| risk_id | RISK-042 |
| description | Cursor Agent `api.github.com` proxy 403 prevents verifying remote GitHub Actions run status; green remote CI must not be invented. |
| probability | high |
| impact | medium |
| mitigation | Classify remote CI status as `UNVERIFIABLE_AGENT_API_CONTEXT`. Rely on local-equivalent commands and the workflow YAML validator. Use Git smart HTTPS for push only. |
| trigger | Stage 2D push / CI |
| owner | Furkan Doblak |
| status | open |
| target_stage | Stage 2D+ |
