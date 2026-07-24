"""Stage 11B pass/reception baseline config loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.passing.types import DEFINITION_STYLE, METRIC_ORIGIN, PassingError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024


class PassConfigError(PassingError):
    """Pass/reception baseline config failure."""


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


def default_pass_reception_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "passing" / "pass_reception_baseline.yaml"


def load_pass_reception_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_pass_reception_config_path(project_root=project_root)
    if p.is_symlink():
        raise PassConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise PassConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise PassConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise PassConfigError("schema_version mismatch")
    if data.get("metric_origin") != METRIC_ORIGIN:
        raise PassConfigError("metric_origin must be project_generated")
    if data.get("definition_style") != DEFINITION_STYLE:
        raise PassConfigError("definition_style must be opta_style_metric_definition")
    if data.get("automatic_ceiling") != "provisional":
        raise PassConfigError("automatic_ceiling must be provisional")
    if data.get("owner_change_alone_is_not_completed_pass") is not True:
        raise PassConfigError("owner_change_alone_is_not_completed_pass must be true")
    if data.get("cut_replay_gap_no_pass") is not True:
        raise PassConfigError("cut_replay_gap_no_pass must be true")
    return _deep_freeze(data)


def pass_reception_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "PassConfigError",
    "default_pass_reception_config_path",
    "load_pass_reception_config",
    "pass_reception_config_fingerprint",
]
