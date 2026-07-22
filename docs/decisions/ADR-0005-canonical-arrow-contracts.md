# ADR-0005 — Canonical Arrow contracts

**Status:** Accepted
**Date:** 2026-07-22
**Stage:** 2C

## Context

Stage 2B established run identity and provenance. Pipelines need stable tabular
contracts before detection/tracking stages write artifacts.

## Decisions

1. **PyArrow** as the in-memory canonical table representation
2. **Parquet** as the on-disk interchange format (zstd, metadata-stamped)
3. **JSON specs as single source of truth** compiled to Arrow schemas
4. **Strict schema validation** (no implicit casts; order matters)
5. **Explicit migration graph** with receipts; no silent upgrades/downgrades
6. **SHA-256 fingerprints** over normalized contract specs
7. **Pandas is not the canonical layer** (may appear at edges later, not here)
8. **Contracts isolated from model/API code** under `football_analytics.data`

## Alternatives

- Hand-maintained Python schemas — drift risk
- Avro/ORC — weaker ecosystem fit for this stack
- Automatic schema evolution — unsafe for analytics PK/FK semantics

## Consequences

Nine v1 contracts + detections v0 fixture; CLI/validator; synthetic E2E only.
