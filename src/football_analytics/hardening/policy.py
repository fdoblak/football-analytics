"""Load and fingerprint Stage 15 hardening policy YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json

_MAX_POLICY_BYTES = 128 * 1024
_DEFAULT_REL = Path("configs/system/hardening_policy.yaml")


class HardeningPolicyError(ValueError):
    """Invalid hardening policy."""


@dataclass(frozen=True)
class HardeningPolicy:
    """Typed view over configs/system/hardening_policy.yaml."""

    raw: dict[str, Any]
    schema_version: int

    @property
    def max_pylist_rows(self) -> int:
        return int(self.raw["materialize"]["max_pylist_rows"])

    @property
    def allow_unbounded_pylist(self) -> bool:
        return bool(self.raw["materialize"]["allow_unbounded_pylist"])

    @property
    def chunk_rows(self) -> int:
        return int(self.raw["materialize"]["chunk_rows"])

    @property
    def cache_gc_mode(self) -> str:
        return str(self.raw["cache_gc"]["default_mode"])

    @property
    def permanent_delete_by_default(self) -> bool:
        return bool(self.raw["cache_gc"]["permanent_delete_by_default"])

    @property
    def automatic_purge(self) -> bool:
        return bool(self.raw["cache_gc"]["automatic_purge"])

    @property
    def min_free_pipeline_bytes(self) -> int:
        return int(self.raw["disk"]["minimum_free_bytes_pipeline"])

    @property
    def gpu_classification_default(self) -> str:
        return str(self.raw["gpu"]["classification_default"])

    @property
    def rtx3050_profile(self) -> dict[str, Any]:
        return dict(self.raw["gpu"]["rtx3050_4gb"])

    @property
    def network_defaults(self) -> dict[str, Any]:
        return dict(self.raw["network"])

    @property
    def gate_hint(self) -> str:
        return " ".join(str(self.raw.get("gate_hint", "")).split())


def default_hardening_policy_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / _DEFAULT_REL


def load_hardening_policy(path: Path | None = None) -> HardeningPolicy:
    """Load and validate hardening policy."""
    target = Path(path) if path is not None else default_hardening_policy_path()
    if not target.is_file() or target.is_symlink():
        raise HardeningPolicyError("hardening policy must be a regular file")
    if target.stat().st_size > _MAX_POLICY_BYTES:
        raise HardeningPolicyError("hardening policy file too large")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HardeningPolicyError("hardening policy root must be a mapping")
    required = {
        "schema_version",
        "materialize",
        "cache_gc",
        "disk",
        "gpu",
        "network",
        "artifacts",
        "licensing",
        "storage",
        "performance",
        "concurrency",
        "recovery",
        "ci_parity",
    }
    missing = required - set(raw)
    if missing:
        raise HardeningPolicyError(f"hardening policy missing fields: {sorted(missing)}")
    if int(raw["schema_version"]) != 1:
        raise HardeningPolicyError("unsupported hardening policy schema_version")
    if bool(raw["cache_gc"].get("automatic_purge")):
        raise HardeningPolicyError("automatic_purge must remain false")
    if bool(raw["cache_gc"].get("permanent_delete_by_default")):
        raise HardeningPolicyError("permanent_delete_by_default must remain false")
    if bool(raw["storage"].get("pretend_mnt_d_exists")):
        raise HardeningPolicyError("pretend_mnt_d_exists must remain false")
    if bool(raw["licensing"].get("invent_legal_approval")):
        raise HardeningPolicyError("invent_legal_approval must remain false")
    if bool(raw["materialize"].get("allow_unbounded_pylist")):
        raise HardeningPolicyError("allow_unbounded_pylist must remain false")
    return HardeningPolicy(raw=raw, schema_version=1)


def hardening_policy_fingerprint(policy: HardeningPolicy) -> str:
    """Deterministic SHA-256 of canonical policy content."""
    return hash_canonical_json(policy.raw)
