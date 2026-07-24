"""Deterministic fingerprint helpers for Stage 15 hardening checks."""

from __future__ import annotations

from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.types import ArtifactRef, ContractRef, StageIdentity


def assert_deterministic_fingerprint(payload: dict[str, Any], *, repeats: int = 2) -> str:
    """Compute the same canonical fingerprint `repeats` times or raise."""
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    first = hash_canonical_json(payload)
    for _ in range(repeats - 1):
        again = hash_canonical_json(payload)
        if again != first:
            raise AssertionError("fingerprint nondeterministic")
    if len(first) != 64:
        raise AssertionError("fingerprint must be 64 hex chars")
    return first


def assert_deterministic_cache_key(*, repeats: int = 2) -> str:
    """Ensure compute_cache_key is stable across repeats with a synthetic stage."""
    stage = StageIdentity(
        name="hardening_probe_stage",
        version=1,
        code_fingerprint="a" * 64,
        input_contracts=(ContractRef(name="probe_in", version=1),),
        output_contracts=(ContractRef(name="probe_out", version=1),),
        deterministic=True,
        cacheable=True,
    )
    ref = ArtifactRef(
        logical_name="probe_input",
        relative_path="probe/input.json",
        media_type="application/json",
        size_bytes=12,
        sha256="b" * 64,
        contract_name="probe_in",
        contract_version=1,
        schema_fingerprint="c" * 64,
    )
    first = compute_cache_key(
        stage=stage,
        config_fingerprint="d" * 64,
        compatibility_fingerprint="e" * 64,
        inputs={"probe_input": ref},
    )
    for _ in range(max(1, repeats - 1)):
        again = compute_cache_key(
            stage=stage,
            config_fingerprint="d" * 64,
            compatibility_fingerprint="e" * 64,
            inputs={"probe_input": ref},
        )
        if again != first:
            raise AssertionError("cache key nondeterministic")
    return first
