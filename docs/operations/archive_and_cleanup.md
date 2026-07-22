# Archive and cleanup operations

## Workspace vs archive

| Location | Role |
|---|---|
| `/home/fdoblak/workspace/runs/<run_id>` | Active working run tree |
| `/home/fdoblak/football_data/experiments_archive/<run_id>` | Verified **local** archive copy |
| `/home/fdoblak/workspace/quarantine/` | Post-cleanup holding area (not deleted) |

Archive is **copy → checksum verify → receipt**, not a move of the only copy.

## Same-VHDX limitation (important)

Active workspace and the active archive backend both live on the **WSL ext4 volume inside the Windows C: VHDX** (`failure_domain: same_wsl_vhdx`).

Therefore the archive is:

- a **verified local archive**
- **not** an independent backup
- **not** a disaster-recovery / off-device backup

`policy.independent_backup: false` is mandatory. Planned `/mnt/d/...` remains `planned_unverified` and is **not** used by these tools.

## Flow

1. **archive_run.py** (default dry-run; `--execute` to write)
2. **verify_archive.py** (read-only)
3. **cleanup_run.py** (default dry-run; `--execute` + `--confirm-run-id`)
4. **restore_run.py** (default dry-run; `--execute`)

Only `status=completed` runs with a valid `run_manifest.json` and required artifacts are archiveable.

## Manifests and checksums

- Every regular file: SHA-256 + size in `archive_manifest.json`
- Sorted relative paths; no symlinks/special files; no `..`
- `archive_manifest.json` is not listed inside its own `files[]`
- Source side: atomic `archive_receipt.json` after successful archive

## Verification

`verify_archive.py` re-hashes all files, rejects extras/missing/tamper, and confirms `independent_backup: false`.

## Restore

Restores into `runs_root/<run_id>` only if missing. Does **not** overwrite. Does **not** copy `archive_manifest.json` into the run as a normal artifact. Writes `restore_receipt.json`.

## Cleanup

- Default **dry-run** (no mutation)
- `--execute` requires exact `--confirm-run-id`
- Re-verifies archive before move
- Refuses if `workspace/current` points at the run
- **Atomic rename** to `quarantine_root/<run_id>_<timestamp>`
- Writes `quarantine_receipt.json` with `purge_status: not_performed`
- **No permanent delete** in Stage 1D; no automatic quarantine purge

## Retention

`quarantine_retention_days: 30` is recorded policy only. Purge tooling is future work.

## Planned D: migration

When `/mnt/d/football_data` is verified, migrate with checksummed copy, update `archive_policy.yaml` / ADR-0002, and only then treat external archive as a separate failure domain.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | PASS / dry-run OK |
| 1 | Integrity failure |
| 2 | Config/usage error |
| 3 | Security/path failure |

## Examples

```bash
python scripts/archive_run.py --run-id run_20260722_120000_a1d001 --dry-run
python scripts/archive_run.py --run-id run_20260722_120000_a1d001 --execute
python scripts/verify_archive.py --run-id run_20260722_120000_a1d001
python scripts/cleanup_run.py --run-id run_20260722_120000_a1d001
python scripts/cleanup_run.py --run-id run_20260722_120000_a1d001 --execute --confirm-run-id run_20260722_120000_a1d001
python scripts/restore_run.py --run-id run_20260722_120000_a1d001 --execute
```

## Security warnings

- Never archive unknown runs under `workspace/runs`
- Never point tools at `/`, `$HOME`, workspace root, or storage root as a run target
- Never use `rm -rf` or broad globs for cleanup
- Do not claim off-device backup while on the same VHDX
