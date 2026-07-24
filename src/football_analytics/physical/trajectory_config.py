"""Strict loader for Stage 9B target trajectory baseline config."""

from __future__ import annotations

import math
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
        "input_eligibility",
        "pitch",
        "segment_split",
        "quality_filter",
        "resample",
        "output_policy",
        "customer_metrics_forbidden",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "attack_direction",
        "notes",
    }
)


class TrajectoryConfigError(ValueError):
    """Target trajectory baseline config failure."""


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


def default_trajectory_baseline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "target_trajectory_baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise TrajectoryConfigError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise TrajectoryConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise TrajectoryConfigError("config root must be mapping")
    return data


def load_trajectory_baseline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_trajectory_baseline_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise TrajectoryConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise TrajectoryConfigError("unsupported config_version")
    if str(data.get("stage")) != "9B":
        raise TrajectoryConfigError("stage must be 9B")
    if data.get("overwrite_allowed") is not False:
        raise TrajectoryConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise TrajectoryConfigError("network_sources_allowed must be false")
    if str(data.get("attack_direction")) != "unknown":
        raise TrajectoryConfigError("attack_direction must be unknown")
    qf = data["quality_filter"]
    if qf.get("enabled") is not True:
        raise TrajectoryConfigError("quality_filter.enabled must be true")
    if str(qf.get("speed_check_provenance")) != "quality_gate_only":
        raise TrajectoryConfigError("speed_check_provenance must be quality_gate_only")
    rs = data["resample"]
    if rs.get("enabled") is not True:
        raise TrajectoryConfigError("resample.enabled must be true")
    if rs.get("endpoint_policy") != "no_extrapolate":
        raise TrajectoryConfigError("resample must not extrapolate")
    forbidden = data["customer_metrics_forbidden"]
    for key in (
        "distance",
        "speed",
        "max_speed",
        "sprint_count",
        "sprint_distance",
        "heatmap",
        "activity_score",
        "events",
    ):
        if forbidden.get(key) is not True:
            raise TrajectoryConfigError(f"customer_metrics_forbidden.{key} must be true")
    pitch = data["pitch"]
    for k in ("length_m", "width_m"):
        v = float(pitch[k])
        if not math.isfinite(v) or v <= 0:
            raise TrajectoryConfigError(f"invalid pitch.{k}")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def trajectory_baseline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "TrajectoryConfigError",
    "default_trajectory_baseline_path",
    "load_trajectory_baseline_config",
    "trajectory_baseline_config_fingerprint",
]
