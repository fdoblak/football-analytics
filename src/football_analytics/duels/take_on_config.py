"""Stage 12B take-on baseline config loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.duels.types import DEFINITION_STYLE, METRIC_ORIGIN, DuelsError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024


class TakeOnConfigError(DuelsError):
    """Take-on baseline config failure."""


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
    return value


def default_take_on_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "duels" / "take_on_baseline.yaml"


def load_take_on_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_take_on_config_path(project_root=project_root)
    if p.is_symlink():
        raise TakeOnConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise TakeOnConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise TakeOnConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise TakeOnConfigError("schema_version mismatch")
    if data.get("metric_origin") != METRIC_ORIGIN:
        raise TakeOnConfigError("metric_origin must be project_generated")
    if data.get("definition_style") != DEFINITION_STYLE:
        raise TakeOnConfigError("definition_style must be opta_style_metric_definition")
    if data.get("automatic_ceiling") != "provisional":
        raise TakeOnConfigError("automatic_ceiling must be provisional")
    if data.get("nearby_opponent_alone_is_not_take_on") is not True:
        raise TakeOnConfigError("nearby_opponent_alone_is_not_take_on must be true")
    return _deep_freeze(data)


def take_on_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "TakeOnConfigError",
    "default_take_on_config_path",
    "load_take_on_config",
    "take_on_config_fingerprint",
]
