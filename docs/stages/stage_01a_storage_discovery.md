# Stage 1A — Storage Discovery and Backend Activation

| Field | Value |
|-------|-------|
| Stage | 1A / 1A-R1 |
| Date | 2026-07-22 |
| Initial gate | `BLOCKED — WINDOWS INTEROP` (preserved) |
| Resolution gate | **`PASS_WITH_FINDINGS — WSL LOCAL STORAGE ACTIVE`** |
| Classification | `WSL_LOCAL_PRIMARY_D_UNVERIFIED` |
| Active backend | `/home/fdoblak/football_data` |
| Archive backend | `/mnt/d/football_data` (`planned_unverified`) |
| Next sub-stage started? | **No** |

Companion: `configs/storage/storage_status_20260722.json` · ADR: `docs/decisions/ADR-0002-storage-backend.md`

---

## Initial attempt

### Purpose
Discover Windows `D:`, mount `/mnt/d` if safe, create `football_data`, pass probe.

### Starting checkpoint (initial)
- HEAD `24634a7eb419a9294db98d92e0d227d9601566ab`
- Working tree clean

### Result
Windows PowerShell/CMD interop failed (`UtilConnectUnix connect failed`). Classification: `D_STATUS_UNVERIFIABLE_AGENT_CONTEXT`. No mount, no `football_data` on D:, `paths.yaml` unchanged. Commit: `fbf98e5` — `Record Stage 1A storage blocker`.

That blocker commit is **preserved** (not amended/reset).

---

## Resolution attempt A1A-R1

### Starting checkpoint (R1)
| Field | Value |
|-------|-------|
| Branch | `main` |
| HEAD | `fbf98e5f0bd7062a0c9a6f591824d29734e07bbf` |
| Message | Record Stage 1A storage blocker |
| Working tree | clean |
| Git gate | PASS |

### Agent vs host identity
| Observation | Value |
|-------------|-------|
| `whoami` in Agent | `root` |
| `/proc/self/uid_map` | `0 1000 1` |
| `/proc/self/gid_map` | `0 1000 1` |
| `getent passwd fdoblak` | uid/gid **1000** |
| Interpretation | Agent uid 0 maps to host **fdoblak (1000)** |
| `chown 1000:1000` in Agent NS | `EINVAL` (expected; only uid 0 exists in map) |

Effective host ownership of newly created storage paths: **fdoblak**.

### Host terminal / Windows interop retry
Retried full-path PowerShell and CMD volume queries. Result again: **WINDOWS_INTEROP_BLOCKED** (empty stdout; same UtilConnectUnix errors). D: still **not** declared absent.

### Storage selection policy applied
**Option C** — WSL local primary while D: remains `planned_unverified`.

### Active backend
| Field | Value |
|-------|-------|
| Type | `wsl_local` |
| Root | `/home/fdoblak/football_data` |
| Filesystem | ext4 |
| Mount source | `/dev/sdd` (WSL VHDX on Windows C:) |
| Total | 1081101176832 bytes (~1007 GiB) |
| Free | 993518985216 bytes (~925 GiB) |
| Free-space warning (&lt; 100 GiB) | false |
| Writable | true |
| Probe | PASS |

### Archive backend
| Field | Value |
|-------|-------|
| Root | `/mnt/d/football_data` |
| Status | `planned_unverified` |
| `/mnt/d` created? | **No** |

### Created directories
Under `/home/fdoblak/football_data`:

- `videos/raw_matches`
- `videos/test_clips`
- `datasets`
- `results`
- `rendered_outputs`
- `reports`
- `model_archive`
- `experiments_archive`
- `backups`

No pre-existing conflict. No recursive chown on `/home/fdoblak`. No deletion of unrelated user files.

### Probe test
| Check | Result |
|-------|--------|
| write | PASS |
| read | PASS |
| sha256_match | PASS (`6e69c97e5436cde6fd128a8df81240c2cc073006e7111d1c2f3beb0ad2591892`) |
| size_match | PASS (4147 bytes) |
| cleanup | PASS (exact probe file removed) |
| parents remain | PASS |

### `paths.yaml` change
Updated minimally:

- `storage.active_backend: wsl_local`
- `storage.active_root: /home/fdoblak/football_data`
- `storage.planned_archive_root: /mnt/d/football_data`
- `storage.planned_archive_status: unverified`
- Active leaf paths retargeted under `active_root`
- Classification recorded: `WSL_LOCAL_PRIMARY_D_UNVERIFIED`

### ADR / risk
- Added `docs/decisions/ADR-0002-storage-backend.md`
- Added RISK-016 (WSL VHDX capacity / migration)

### Security attestations
- No package mutation; no downloads; no clone/fetch/pull
- External repos untouched
- No user data deleted/moved/overwritten
- No recursive permission changes on existing trees
- No `wsl.conf`/`fstab` changes
- Previous blocker commit preserved
- Aşama 1B not started

### Acceptance (R1)
Active backend exists, directories present, probe passed, config consistent, history preserved → Stage 1A closable with findings.

### Gate decision (final)
**`PASS_WITH_FINDINGS — WSL LOCAL STORAGE ACTIVE`**

### Next sub-stage (name only)
**Aşama 1B — Klasör Standardı, paths.yaml ve Storage Validator**
