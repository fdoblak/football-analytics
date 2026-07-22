# ADR-0004 — Runtime identity and provenance

**Status:** Accepted
**Date:** 2026-07-22
**Stage:** 2B

## Context

Stages 0–2A established storage, registries, packaging, and private GitHub sync.
Before pipelines write artifacts, the project needs deterministic run identity,
config resolution, hashing, secret-safe logging, and environment provenance —
without importing GPU stacks or dumping secrets.

## Decisions

1. **UTC compact timestamp + cryptographic hex suffix** for run IDs
   Sortable identity without coordinating a central counter; `secrets.token_hex`
   avoids global RNG state. Injectable clock/suffix for tests.

2. **SHA-256** for fingerprints and file/tree digests
   Ubiquitous, collision-resistant enough for integrity manifests; matches Stage 1
   archive checksum policy.

3. **Canonical JSON** (`sort_keys`, compact separators, `allow_nan=False`)
   Stable fingerprints independent of dict insertion order; rejects non-JSON numbers.

4. **No full environment dump**
   Only allowlisted package versions and sanitized git metadata. Prevents token
   leakage via `os.environ`.

5. **No heavy package import for provenance**
   Read versions with `importlib.metadata`; keep GPU classification as
   `AGENT_CONTEXT_GPU_UNVERIFIABLE` without `torch` import.

6. **Immutable resolved config**
   `MappingProxyType` deep freeze reduces accidental mutation after merge.

7. **JSON Lines logging**
   One JSON object per line for machine parsing; human console remains separate.

## Alternatives considered

- UUID-only run IDs — less sortable / harder to eyeball chronology
- MessagePack fingerprints — less human-auditable
- Dumping `pip freeze` into Git — rejected (runtime evidence stays outside Git)
- Shared `logging.py` module name — would shadow stdlib; used `structured_logging.py`

## Consequences

- Archive policy pattern extended to accept Stage 2B IDs alongside Stage 1 fixtures
- Foundation CLI helpers added without pipeline commands
- Synthetic validation lives under workspace `foundation_checks/` only
