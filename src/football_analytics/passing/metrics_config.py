"""Stage 11C passing metrics baseline config loader."""

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


class MetricsConfigError(PassingError):
    """Passing metrics config failure."""


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


def default_metrics_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "passing" / "passing_metrics_baseline.yaml"


def load_metrics_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_metrics_config_path(project_root=project_root)
    if p.is_symlink():
        raise MetricsConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise MetricsConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise MetricsConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise MetricsConfigError("schema_version mismatch")
    if data.get("metric_origin") != METRIC_ORIGIN:
        raise MetricsConfigError("metric_origin must be project_generated")
    if data.get("definition_style") != DEFINITION_STYLE:
        raise MetricsConfigError("definition_style must be opta_style_metric_definition")
    attack = data.get("attack_direction") or {}
    if attack.get("invent_forbidden") is not True:
        raise MetricsConfigError("attack_direction.invent_forbidden must be true")
    if attack.get("conflict_yields") != "unknown":
        raise MetricsConfigError("attack_direction.conflict_yields must be unknown")
    return _deep_freeze(data)


def metrics_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "MetricsConfigError",
    "default_metrics_config_path",
    "load_metrics_config",
    "metrics_config_fingerprint",
]
