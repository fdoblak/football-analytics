"""Strict loader for Stage 9C distance / speed / sprint baseline config."""

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
        "metric_origin",
        "definition_style",
        "primary_sample_layer",
        "diagnostic_sample_layers",
        "input_eligibility",
        "pitch",
        "distance",
        "speed",
        "sprint",
        "coverage",
        "output_policy",
        "forbidden",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "attack_direction",
        "notes",
    }
)


class MotionConfigError(ValueError):
    """Distance/speed/sprint baseline config failure."""


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


def default_motion_baseline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "distance_speed_sprint_baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise MotionConfigError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise MotionConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise MotionConfigError("config root must be mapping")
    return data


def load_motion_baseline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_motion_baseline_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise MotionConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise MotionConfigError("unsupported config_version")
    if str(data.get("stage")) != "9C":
        raise MotionConfigError("stage must be 9C")
    if str(data.get("metric_origin")) != "project_generated":
        raise MotionConfigError("metric_origin must be project_generated")
    if str(data.get("definition_style")) != "opta_style_metric_definition":
        raise MotionConfigError("definition_style must be opta_style_metric_definition")
    if str(data.get("primary_sample_layer")) != "filtered":
        raise MotionConfigError("primary_sample_layer must be filtered")
    if data.get("overwrite_allowed") is not False:
        raise MotionConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise MotionConfigError("network_sources_allowed must be false")
    if str(data.get("attack_direction")) != "unknown":
        raise MotionConfigError("attack_direction must be unknown")
    if data.get("coverage", {}).get("extrapolate_uncovered_time") is not False:
        raise MotionConfigError("coverage.extrapolate_uncovered_time must be false")
    sprint = data["sprint"]
    if float(sprint["entry_speed_mps"]) <= float(sprint["exit_speed_mps"]):
        raise MotionConfigError("sprint entry_speed must exceed exit_speed (hysteresis)")
    if sprint.get("not_official_opta") is not True:
        raise MotionConfigError("sprint.not_official_opta must be true")
    forbidden = data["forbidden"]
    for key in (
        "heatmap",
        "activity_score",
        "events",
        "pixel_distance",
        "coverage_extrapolation",
        "official_opta_claim",
    ):
        if forbidden.get(key) is not True:
            raise MotionConfigError(f"forbidden.{key} must be true")
    pitch = data["pitch"]
    for k in ("length_m", "width_m"):
        v = float(pitch[k])
        if not math.isfinite(v) or v <= 0:
            raise MotionConfigError(f"invalid pitch.{k}")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def motion_baseline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "MotionConfigError",
    "default_motion_baseline_path",
    "load_motion_baseline_config",
    "motion_baseline_config_fingerprint",
]
