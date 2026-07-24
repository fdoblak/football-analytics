"""Strict loader for Stage 9D heatmap / zones / activity baseline config."""

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
        "heatmap",
        "zones",
        "activity",
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


class SpatialConfigError(ValueError):
    """Heatmap/zones/activity baseline config failure."""


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


def default_spatial_baseline_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "physical" / "heatmap_activity_baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise SpatialConfigError(f"symlink rejected: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise SpatialConfigError("config too large")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise SpatialConfigError("config root must be mapping")
    return data


def load_spatial_baseline_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_spatial_baseline_path(project_root=project_root)
    data = _load_yaml(p)
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise SpatialConfigError(f"missing keys: {sorted(missing)}")
    if int(data.get("config_version", -1)) != CONFIG_VERSION:
        raise SpatialConfigError("unsupported config_version")
    if str(data.get("stage")) != "9D":
        raise SpatialConfigError("stage must be 9D")
    if str(data.get("metric_origin")) != "project_generated":
        raise SpatialConfigError("metric_origin must be project_generated")
    if str(data.get("primary_sample_layer")) != "filtered":
        raise SpatialConfigError("primary_sample_layer must be filtered")
    if str(data.get("attack_direction")) != "unknown":
        raise SpatialConfigError("attack_direction must be unknown")
    if data.get("overwrite_allowed") is not False:
        raise SpatialConfigError("overwrite_allowed must be false")
    if data.get("network_sources_allowed") is not False:
        raise SpatialConfigError("network_sources_allowed must be false")
    if data.get("coverage", {}).get("extrapolate_uncovered_time") is not False:
        raise SpatialConfigError("coverage.extrapolate_uncovered_time must be false")
    if data.get("coverage", {}).get("missing_coverage_is_not_inactive") is not True:
        raise SpatialConfigError("missing_coverage_is_not_inactive must be true")
    if data.get("output_policy", {}).get("write_visuals_to_git") is not False:
        raise SpatialConfigError("write_visuals_to_git must be false")
    zones = data["zones"]
    if zones.get("attack_relative_names_forbidden") is not True:
        raise SpatialConfigError("attack_relative_names_forbidden must be true")
    if zones.get("penalty", {}).get("forbid_touch_possession_claim") is not True:
        raise SpatialConfigError("penalty.forbid_touch_possession_claim must be true")
    act = data["activity"]
    sprint_align = float(act.get("sprint_entry_mps_align", 7.0))
    spr_min = float(act["classes"]["sprinting"]["min_mps"])
    if abs(spr_min - sprint_align) > 1e-9:
        raise SpatialConfigError("sprinting.min_mps must align with sprint_entry_mps_align")
    hm = data["heatmap"]
    if str(hm.get("weighting")) != "time_weighted":
        raise SpatialConfigError("heatmap.weighting must be time_weighted")
    forbidden = data["forbidden"]
    for key in (
        "events",
        "pass_dribble_duel",
        "ball_touch_in_penalty",
        "possession",
        "attack_direction_invention",
        "coverage_as_inactive",
        "coverage_extrapolation",
        "official_opta_claim",
        "final_customer_visual",
    ):
        if forbidden.get(key) is not True:
            raise SpatialConfigError(f"forbidden.{key} must be true")
    pitch = data["pitch"]
    for k in ("length_m", "width_m"):
        v = float(pitch[k])
        if not math.isfinite(v) or v <= 0:
            raise SpatialConfigError(f"invalid pitch.{k}")
    return MappingProxyType(_deep_freeze(data))  # type: ignore[arg-type]


def spatial_baseline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "CONFIG_VERSION",
    "SpatialConfigError",
    "default_spatial_baseline_path",
    "load_spatial_baseline_config",
    "spatial_baseline_config_fingerprint",
]
