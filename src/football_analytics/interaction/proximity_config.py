"""Strict loader for Stage 10B proximity / contact-candidate baseline config."""

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
        "single_frame_is_not_contact",
        "no_real_event_metrics",
        "no_automatic_confirmed",
        "input_eligibility",
        "proximity",
        "contact",
        "lifecycle",
        "coverage",
        "output_policy",
        "forbidden",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class ProximityConfigError(ValueError):
    """Proximity/contact baseline config failure."""


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


def default_proximity_baseline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "interaction" / "human_ball_proximity_contact_baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ProximityConfigError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ProximityConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ProximityConfigError("config root must be mapping")
    return data


def load_proximity_baseline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_proximity_baseline_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise ProximityConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise ProximityConfigError("unsupported config_version")
    if str(data.get("stage")) != "10B":
        raise ProximityConfigError("stage must be 10B")
    if str(data.get("metric_origin")) != "project_generated":
        raise ProximityConfigError("metric_origin must be project_generated")
    if str(data.get("automatic_ceiling")) != "provisional":
        raise ProximityConfigError("automatic_ceiling must be provisional")
    if data.get("nearest_player_is_not_possession") is not True:
        raise ProximityConfigError("nearest_player_is_not_possession must be true")
    if data.get("no_real_event_metrics") is not True:
        raise ProximityConfigError("no_real_event_metrics must be true")
    if data.get("overwrite_allowed") is not False:
        raise ProximityConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise ProximityConfigError("network_sources_allowed must be false")
    return _deep_freeze(data)


def proximity_baseline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "ProximityConfigError",
    "default_proximity_baseline_path",
    "load_proximity_baseline_config",
    "proximity_baseline_config_fingerprint",
]
