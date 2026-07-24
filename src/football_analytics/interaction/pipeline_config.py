"""Strict loader for Stage 10D human-ball interaction pipeline config."""

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
        "automatic_ceiling",
        "nearest_player_is_not_possession",
        "no_real_event_metrics",
        "no_automatic_confirmed",
        "separate_operational_from_event_accuracy",
        "stages",
        "inputs",
        "integrity",
        "quality_gate",
        "output_policy",
        "forbidden",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class InteractionPipelineConfigError(ValueError):
    """Interaction pipeline config failure."""


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


def default_interaction_pipeline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "interaction" / "human_ball_interaction_pipeline.yaml"


def load_interaction_pipeline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_interaction_pipeline_path(project_root=project_root)
    if p.is_symlink():
        raise InteractionPipelineConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise InteractionPipelineConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise InteractionPipelineConfigError("config root must be mapping")
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise InteractionPipelineConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise InteractionPipelineConfigError("unsupported config_version")
    if str(data.get("stage")) != "10D":
        raise InteractionPipelineConfigError("stage must be 10D")
    if str(data.get("automatic_ceiling")) != "provisional":
        raise InteractionPipelineConfigError("automatic_ceiling must be provisional")
    if data.get("no_real_event_metrics") is not True:
        raise InteractionPipelineConfigError("no_real_event_metrics must be true")
    if data.get("overwrite_allowed") is not False:
        raise InteractionPipelineConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise InteractionPipelineConfigError("network_sources_allowed must be false")
    return _deep_freeze(data)


def interaction_pipeline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "InteractionPipelineConfigError",
    "default_interaction_pipeline_path",
    "load_interaction_pipeline_config",
    "interaction_pipeline_config_fingerprint",
]
