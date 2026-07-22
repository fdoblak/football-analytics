# Stage 1D — Archive, Checksum, Restore, Cleanup

**Gate:** `PASS_WITH_FINDINGS — LOCAL ARCHIVE VALIDATED; STAGE 1 CLOSED`
**Date:** 2026-07-22
**Start HEAD:** `507d1c8c865e13219b8ebcc1acf33de1250be89e`

## Scripts

- `scripts/archive_run.py` (default dry-run; `--execute`)
- `scripts/verify_archive.py` (read-only)
- `scripts/restore_run.py` (default dry-run; `--execute`)
- `scripts/cleanup_run.py` (quarantine; `--execute` + `--confirm-run-id`)
- `src/football_analytics/utils/archive_safety.py`

## Schemas / policy

- `configs/system/archive_policy.yaml`
- `schemas/run_manifest.schema.json`
- `schemas/archive_manifest.schema.json`

## Tests

- `tests/archive/test_archive_workflow.py` — **42** cases PASS
- Full suite including prior stages — **102** PASS

## Real smoke chain

Fixture: `run_20260722_144322_a1d001` (`fixture_marker: stage1d_synthetic_fixture`)

| Step | Result |
|---|---|
| Archive dry-run | PASS |
| Archive execute | PASS |
| Verify | PASS |
| Receipt | present |
| Cleanup dry-run | PASS (no mutation) |
| Cleanup execute → quarantine | PASS |
| Verify after cleanup | PASS |
| Restore dry-run / execute | PASS |
| Restore hashes vs archive | MATCH |
| `independent_backup` | false |
| Fixture cleanup | exact paths removed |

## Runtime reports (not in Git)

`/home/fdoblak/workspace/archive_checks/` (`*_20260722T144322Z.json`)

## Findings (accepted)

- Same-VHDX local archive ≠ independent backup
- `/mnt/d` planned_unverified
- Quarantine purge not automatic (RISK-017)

## Docs

- `docs/operations/archive_and_cleanup.md`
- `docs/decisions/ADR-0003-archive-and-cleanup.md`
- `docs/stages/stage_01_completion.md`
- Risk register updates (RISK-016/017/018)

## Commit

`Add verified archive and safe cleanup workflow`
