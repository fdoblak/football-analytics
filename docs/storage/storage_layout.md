# Storage Layout

## Active backend (Stage 1A-R1 / 1B)

| Field | Value |
|-------|-------|
| Backend | `wsl_local` |
| Classification | `WSL_LOCAL_PRIMARY_D_UNVERIFIED` |
| Active root | `/home/fdoblak/football_data` |
| Filesystem | ext4 on `/dev/sdd` (WSL VHDX on Windows C:) |

Configured in `configs/system/paths.yaml`.

## Planned archive backend

| Field | Value |
|-------|-------|
| Root | `/mnt/d/football_data` |
| Status | `unverified` |
| Active? | **No** |

Do not write production data here until D: is verified and mounted. Migration must be checksum-validated (ADR-0002).

## Directory roles

| Path | Role |
|------|------|
| `videos/raw_matches/` | Original full match videos (immutable sources) |
| `videos/test_clips/` | Short golden/smoke clips |
| `datasets/` | Downloaded or prepared datasets (when approved) |
| `results/` | Durable run outputs / archived artifacts |
| `rendered_outputs/` | Annotated / tactical videos |
| `reports/` | Human-readable reports and summaries |
| `model_archive/` | Versioned model weights kept for the project |
| `experiments_archive/` | Frozen experiment bundles |
| `backups/` | Storage-level backups |

## Active storage vs workspace

| Area | Path | Purpose |
|------|------|---------|
| Active storage | `/home/fdoblak/football_data` | Durable project data |
| Workspace | `/home/fdoblak/workspace` | Active compute scratch (`runs`, `staging`, `cache`) |
| Validation reports | `/home/fdoblak/workspace/storage_checks` | Runtime validator JSON (not committed) |

`workspace/current` symlink is **not** created yet. That belongs to orchestration (Aşama 3).

## What goes where

- **Ham videolar** → `videos/raw_matches`
- **Test klipleri** → `videos/test_clips`
- **Datasetler** → `datasets`
- **Run sonuçları (kalıcı)** → `results` (after archive stage)
- **Render videolar** → `rendered_outputs`
- **Raporlar** → `reports`
- **Model arşivi** → `model_archive`
- **Deney arşivi** → `experiments_archive`
- **Backup** → `backups`
- **Geçici run scratch** → `/home/fdoblak/workspace/runs`

## Operator rules

- Do not manually delete or relocate user/project files under these trees without a stage brief.
- Do not treat `/mnt/d/...` as active until status becomes verified.
- When D: becomes available, migrate with checksums; then switch `active_backend` in config.

## Validator

```bash
conda activate ai-dev
cd /home/fdoblak/projects/football-analytics

# Read-only
python scripts/check_storage.py --config configs/system/paths.yaml

# Opt-in probe + JSON report
mkdir -p /home/fdoblak/workspace/storage_checks
python scripts/check_storage.py \
  --config configs/system/paths.yaml \
  --probe \
  --json-out /home/fdoblak/workspace/storage_checks/storage_validation_<timestamp>.json
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | PASS or PASS_WITH_WARNINGS |
| 1 | Validation failure |
| 2 | Config / usage error |
| 3 | Probe cleanup failure or security violation |

### Capacity thresholds (`storage_validation`)

| Threshold | Bytes | Meaning |
|-----------|-------|---------|
| `minimum_free_bytes` | 21474836480 (20 GiB) | Below → FAIL |
| `warning_free_bytes` | 107374182400 (100 GiB) | Below → WARNING |
