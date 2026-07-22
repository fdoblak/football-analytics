# ADR-0006 — Stage interface, content-addressed cache, and CI

**Status:** Accepted
**Date:** 2026-07-22
**Stage:** 2D

## Context

Stages 2A–2C delivered packaging, runtime provenance, and canonical Arrow
contracts. Foundation closure needs a stable single-stage execution contract,
repeatable local artifact reuse, and a least-privilege CI gate — without
introducing a full orchestrator or trusting mutable GitHub Action tags.

## Decision

1. **Explicit Stage interface** — `Stage` protocol with immutable
   `StageIdentity` / `StageRequest` / `StageResult` / `ArtifactRef`; in-process
   `StageRegistry` only; **no DAG** in Stage 2D.
2. **Content-addressed local cache** — SHA-256 keys over stage code, config/
   compatibility fingerprints, and ordered input artifact identities; layout
   `v1/sha256/<ab>/<rest>/`; atomic publish + `fcntl.flock`; verify on
   publish/read; quarantine corrupt entries; **no automatic purge**.
3. **Least-privilege SHA-pinned CI** — `permissions: contents: read`; official
   `actions/checkout` and `actions/setup-python` pinned to full commit SHAs;
   lightweight `requirements/ci.txt` (no GPU/external repos); YAML safety
   validator without GitHub API calls.

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Airflow / heavy orchestrator | Premature; DAG/scheduling out of foundation scope |
| Mutable Action tags (`@v6`) | Supply-chain drift; pins required |
| Pandas as cache/index layer | Not content-addressed; contracts already Arrow/Parquet |
| Automatic cache purge/GC | Unsafe without retention policy; defer past 2D |

## Consequences

- Synthetic `SyntheticEchoStage` proves miss→hit without product stages
- Project check profiles separate host-only SKIPs from CI PASS
- Remote CI run status may remain `UNVERIFIABLE_AGENT_API_CONTEXT` under
  `api.github.com` proxy 403; local-equivalent + workflow validator are the
  closure evidence
- Disk growth without GC is an accepted open finding (RISK-041)
- Next stage (ingest) consumes this interface; does not redesign it
