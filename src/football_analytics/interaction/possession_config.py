"""Strict loader for Stage 10C possession / control baseline config."""

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
        "contact_is_not_controlled_possession",
        "no_real_event_metrics",
        "no_automatic_confirmed",
        "missing_ball_is_not_no_possession",
        "missing_ball_is_not_loose",
        "input_eligibility",
        "possession",
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


class PossessionConfigError(ValueError):
    """Possession/control baseline config failure."""


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


def default_possession_baseline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "interaction" / "possession_control_baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise PossessionConfigError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise PossessionConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise PossessionConfigError("config root must be mapping")
    return data


def load_possession_baseline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_possession_baseline_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise PossessionConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise PossessionConfigError("unsupported config_version")
    if str(data.get("stage")) != "10C":
        raise PossessionConfigError("stage must be 10C")
    if str(data.get("metric_origin")) != "project_generated":
        raise PossessionConfigError("metric_origin must be project_generated")
    if str(data.get("automatic_ceiling")) != "provisional":
        raise PossessionConfigError("automatic_ceiling must be provisional")
    if data.get("nearest_player_is_not_possession") is not True:
        raise PossessionConfigError("nearest_player_is_not_possession must be true")
    if data.get("no_real_event_metrics") is not True:
        raise PossessionConfigError("no_real_event_metrics must be true")
    if data.get("missing_ball_is_not_no_possession") is not True:
        raise PossessionConfigError("missing_ball_is_not_no_possession must be true")
    if data.get("missing_ball_is_not_loose") is not True:
        raise PossessionConfigError("missing_ball_is_not_loose must be true")
    if data.get("overwrite_allowed") is not False:
        raise PossessionConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise PossessionConfigError("network_sources_allowed must be false")
    poss = data.get("possession") or {}
    if str(poss.get("max_state")) == "confirmed":
        raise PossessionConfigError("possession.max_state cannot be confirmed")
    return _deep_freeze(data)


def possession_baseline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "PossessionConfigError",
    "default_possession_baseline_path",
    "load_possession_baseline_config",
    "possession_baseline_config_fingerprint",
]
