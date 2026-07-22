# ADR-0003 — Archive and cleanup

**Status:** Accepted (Stage 1D)
**Date:** 2026-07-22

## Context

Stage 1 needs a safe way to retain completed experiment trees without pretending WSL-local storage is an off-device backup, and without destructive deletes.

## Decisions

1. **Copy → verify → receipt** for archive (source retained until explicit cleanup).
2. **SHA-256** per regular file; size + totals must match.
3. **Hidden temporary directory under archive_root + atomic rename** to final `<archive_root>/<run_id>`.
4. **Cleanup = quarantine move**, not permanent delete.
5. **No automatic quarantine purge** in Stage 1D.
6. **Reject symlinks and special files** in source and archive trees.
7. **Exact `--confirm-run-id`** required for cleanup execute.
8. **Document same-VHDX limitation**; `independent_backup: false`.
9. **Planned external archive** (`/mnt/d/...`) remains inactive until verified; migration will be checksummed.

## Consequences

- Operators get restore and integrity checks today.
- Disaster recovery still requires a future verified independent disk (D: or other).
- Quarantine disk use must be monitored until purge tooling exists.
