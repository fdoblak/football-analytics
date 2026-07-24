"""Strict loader for Stage 9E physical metric pipeline config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "schema_version",
        "config_version",
        "pipeline_id",
        "pipeline_version",
        "stage",
        "metric_origin",
        "definition_style",
        "primary_sample_layer",
        "require_confirmed_identity",
        "forbid_revoked_identity",
        "forbid_provisional_identity",
        "attack_direction",
        "integrity",
        "quality_gate",
        "coverage",
        "overall_status_rules",
        "output_policy",
        "forbidden",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class PipelineConfigError(ValueError):
    """Physical metric pipeline config failure."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def default_pipeline_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "physical_metric_pipeline.yaml"


def load_pipeline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_pipeline_config_path(project_root=project_root)
    if p.is_symlink():
        raise PipelineConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise PipelineConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise PipelineConfigError("config root must be mapping")
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise PipelineConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise PipelineConfigError("unsupported config_version")
    if str(data.get("stage")) != "9E":
        raise PipelineConfigError("stage must be 9E")
    if str(data.get("attack_direction")) != "unknown":
        raise PipelineConfigError("attack_direction must be unknown")
    if data.get("overwrite_allowed") is not False:
        raise PipelineConfigError("overwrite_allowed must be false")
    if data.get("require_confirmed_identity") is not True:
        raise PipelineConfigError("require_confirmed_identity must be true")
    if data.get("output_policy", {}).get("write_visuals_to_git") is not False:
        raise PipelineConfigError("write_visuals_to_git must be false")
    if data.get("output_policy", {}).get("write_final_customer_visual") is not False:
        raise PipelineConfigError("write_final_customer_visual must be false")
    for key in (
        "events",
        "possession",
        "box_touch",
        "full_match_extrapolation",
        "auto_confirm_identity",
        "official_opta_claim",
        "final_customer_visual",
        "real_accuracy_claim",
    ):
        if data["forbidden"].get(key) is not True:
            raise PipelineConfigError(f"forbidden.{key} must be true")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def pipeline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "PipelineConfigError",
    "default_pipeline_config_path",
    "load_pipeline_config",
    "pipeline_config_fingerprint",
]
