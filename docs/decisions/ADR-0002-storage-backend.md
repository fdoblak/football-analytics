# ADR-0002 — Storage Backend Selection

- **Status:** Accepted (Stage 1A-R1, 2026-07-22)
- **Context:** Stage 1A initially blocked on Windows interop; D: could not be verified from the Agent context. Project requires an active writable storage backend before later stages.

## Decision

1. **Preferred persistent/archive backend:** Windows `D:` mounted at `/mnt/d/football_data` when verified.
2. **Active backend for Stage 1A-R1:** WSL local path `/home/fdoblak/football_data` with classification `WSL_LOCAL_PRIMARY_D_UNVERIFIED`.
3. **Separation:** Active runtime/storage root is distinct from planned archive root (`/mnt/d/football_data`, status `unverified`).
4. **Reversibility:** When D: is later verified and safely mounted, migrate with checksum validation; then switch `active_backend` without rewriting history.
5. **No silent redirection:** Unverified `/mnt/d` must not be treated as an active D: volume.

## Consequences

- WSL local storage consumes space inside the Windows C: WSL VHDX (`ext4` on `/dev/sdd`).
- Free-space gates and per-run size telemetry are required (see RISK-016).
- Stage 1B may validate paths against the active root; archive backend remains planned until D: verification succeeds.

## References

- `configs/system/paths.yaml`
- `configs/storage/storage_status_20260722.json`
- `docs/stages/stage_01a_storage_discovery.md`
