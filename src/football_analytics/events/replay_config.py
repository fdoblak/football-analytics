"""Replay candidate baseline config (Stage 13A)."""

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


class ReplayConfigError(EventsContractError):
    """Replay config failure."""


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


def default_replay_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "events" / "replay_candidate_baseline.yaml"


def load_replay_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_replay_config_path(project_root=project_root)
    if p.is_symlink():
        raise ReplayConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ReplayConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ReplayConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise ReplayConfigError("schema_version mismatch")
    if data.get("never_invent_live_when_uncertain") is not True:
        raise ReplayConfigError("never_invent_live_when_uncertain must be true")
    return _deep_freeze(data)


def replay_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "ReplayConfigError",
    "default_replay_config_path",
    "load_replay_config",
    "replay_config_fingerprint",
]
