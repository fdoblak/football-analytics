"""Canonical content-addressed cache key computation (Stage 2D)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.pipeline.exceptions import CacheError
from football_analytics.pipeline.types import ArtifactRef, StageIdentity

CACHE_KEY_VERSION = 1
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _require_hex64(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CacheError(f"{label} must be 64 lowercase hex chars")
    return value


def compute_cache_key(
    *,
    stage: StageIdentity,
    config_fingerprint: str,
    compatibility_fingerprint: str,
    inputs: Mapping[str, ArtifactRef],
) -> str:
    """Compute a 64-hex cache key from stage identity, fingerprints, and inputs.

    MUST include: cache_key_version, stage name/version/code_fingerprint/deterministic/
    cacheable, config_fingerprint, compatibility_fingerprint, ordered inputs (sha256, size,
    contract fields, media_type, relative_path), and output_contracts.

    MUST NOT include: absolute paths, username, hostname, timestamps, secrets,
    working_directory, output_directory, run_id, requested_at_utc.
    """
    if not isinstance(stage, StageIdentity):
        raise CacheError("stage must be StageIdentity")
    cfg = _require_hex64(config_fingerprint, label="config_fingerprint")
    compat = _require_hex64(compatibility_fingerprint, label="compatibility_fingerprint")
    if not isinstance(inputs, Mapping):
        raise CacheError("inputs must be a mapping")
    ordered_inputs: list[dict[str, object]] = []
    for name in sorted(inputs.keys()):
        ref = inputs[name]
        if not isinstance(ref, ArtifactRef):
            raise CacheError("input values must be ArtifactRef")
        ordered_inputs.append(
            {
                "logical_name": name,
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
                "contract_name": ref.contract_name,
                "contract_version": ref.contract_version,
                "schema_fingerprint": ref.schema_fingerprint,
                "media_type": ref.media_type,
                "relative_path": ref.relative_path,
            }
        )

    payload = {
        "cache_key_version": CACHE_KEY_VERSION,
        "stage": {
            "name": stage.name,
            "version": stage.version,
            "code_fingerprint": stage.code_fingerprint,
            "deterministic": stage.deterministic,
            "cacheable": stage.cacheable,
            "output_contracts": [c.to_dict() for c in stage.output_contracts],
        },
        "config_fingerprint": cfg,
        "compatibility_fingerprint": compat,
        "inputs": ordered_inputs,
    }
    return hash_canonical_json(payload)
