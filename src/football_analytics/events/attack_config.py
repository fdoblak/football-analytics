"""13B config loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.events.types import EventsContractError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024


class AttackConfigError(EventsContractError):
    """Config failure."""


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


def default_attack_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "events" / "attack_direction_resolver.yaml"


def load_attack_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_attack_config_path(project_root=project_root)
    if p.is_symlink():
        raise AttackConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise AttackConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise AttackConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise AttackConfigError("schema_version mismatch")

    if data.get("invent_forbidden") is not True:
        raise AttackConfigError("invent_forbidden must be true")
    if data.get("conflict_yields") != "unknown":
        raise AttackConfigError("conflict_yields must be unknown")
    if data.get("never_invent_team_names") is not True:
        raise AttackConfigError("never_invent_team_names must be true")

    return _deep_freeze(data)


def attack_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "AttackConfigError",
    "default_attack_config_path",
    "load_attack_config",
    "attack_config_fingerprint",
]
