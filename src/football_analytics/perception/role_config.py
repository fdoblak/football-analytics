"""Strict loader for Stage 5D human role baseline config."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024
REQUIRED_TOP = frozenset(
    {
        "config_version",
        "classifier_id",
        "classifier_version",
        "eligibility",
        "features",
        "clustering",
        "thresholds",
        "output_policy",
        "maximum_frames_per_run",
        "runtime_root",
        "taxonomy_path",
        "policy_path",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class RoleConfigError(ValueError):
    """Human role baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoleConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoleConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise RoleConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RoleConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoleConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise RoleConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise RoleConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise RoleConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise RoleConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RoleConfigError(f"{label} must be a non-empty string")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RoleConfigError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise RoleConfigError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise RoleConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise RoleConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise RoleConfigError(f"config_version must be {CONFIG_VERSION}")

    elig = dict(_require_mapping(raw["eligibility"], label="eligibility"))
    entities = _require_str_list(elig["entity_types"], label="eligibility.entity_types")
    if entities != ["human"]:
        raise RoleConfigError("eligibility.entity_types must be exactly [human]")
    if (
        _require_bool(
            elig["require_processed_frames"], label="eligibility.require_processed_frames"
        )
        is not True
    ):
        raise RoleConfigError("eligibility.require_processed_frames must be true")
    if (
        _require_bool(elig["require_analysis_window"], label="eligibility.require_analysis_window")
        is not True
    ):
        raise RoleConfigError("eligibility.require_analysis_window must be true")
    play = _require_str_list(elig["playability_allowed"], label="eligibility.playability_allowed")
    track = _require_str_list(
        elig["tracking_eligibility_allowed"], label="eligibility.tracking_eligibility_allowed"
    )
    min_area = _require_float(
        elig["min_crop_area_px"], label="eligibility.min_crop_area_px", minimum=1.0
    )
    min_w = _require_float(elig["min_crop_width"], label="eligibility.min_crop_width", minimum=1.0)
    min_h = _require_float(
        elig["min_crop_height"], label="eligibility.min_crop_height", minimum=1.0
    )
    min_ar = _require_float(
        elig["min_aspect_ratio"], label="eligibility.min_aspect_ratio", minimum=0.01
    )
    max_ar = _require_float(
        elig["max_aspect_ratio"], label="eligibility.max_aspect_ratio", minimum=0.01
    )
    if min_ar > max_ar:
        raise RoleConfigError("eligibility aspect ratio min/max inconsistent")
    min_q = _require_float(
        elig["min_crop_quality"], label="eligibility.min_crop_quality", minimum=0.0, maximum=1.0
    )

    feats = dict(_require_mapping(raw["features"], label="features"))
    h_bins = _require_int(feats["hsv_h_bins"], label="features.hsv_h_bins", minimum=4, maximum=64)
    s_bins = _require_int(feats["hsv_s_bins"], label="features.hsv_s_bins", minimum=2, maximum=32)
    v_bins = _require_int(feats["hsv_v_bins"], label="features.hsv_v_bins", minimum=2, maximum=32)
    upper = _require_float(
        feats["upper_body_fraction"], label="features.upper_body_fraction", minimum=0.1, maximum=0.9
    )
    lower = _require_float(
        feats["lower_body_fraction"], label="features.lower_body_fraction", minimum=0.1, maximum=0.9
    )
    if abs((upper + lower) - 1.0) > 1e-6:
        raise RoleConfigError("features upper+lower body fractions must sum to 1")
    if _require_bool(feats["persist_crops"], label="features.persist_crops") is not False:
        raise RoleConfigError("features.persist_crops must be false")

    clus = dict(_require_mapping(raw["clustering"], label="clustering"))
    max_out = _require_int(
        clus["max_outfield_clusters"],
        label="clustering.max_outfield_clusters",
        minimum=1,
        maximum=2,
    )
    if max_out != 2:
        raise RoleConfigError("clustering.max_outfield_clusters must be 2")
    min_cs = _require_int(clus["min_cluster_size"], label="clustering.min_cluster_size", minimum=1)
    color_thr = _require_float(
        clus["color_distance_threshold"],
        label="clustering.color_distance_threshold",
        minimum=0.0,
        maximum=2.0,
    )
    stab = _require_float(
        clus["min_cluster_stability"],
        label="clustering.min_cluster_stability",
        minimum=0.0,
        maximum=1.0,
    )

    thr = dict(_require_mapping(raw["thresholds"], label="thresholds"))
    thr_keys = (
        "player_cluster_margin",
        "player_min_quality",
        "goalkeeper_color_margin",
        "goalkeeper_lateral_edge",
        "goalkeeper_size_zscore",
        "referee_max_saturation",
        "referee_max_value",
        "referee_color_margin",
        "staff_max_fraction",
        "abstain_margin",
        "conflict_margin",
    )
    cleaned_thr: dict[str, Any] = {}
    for key in thr_keys:
        cleaned_thr[key] = _require_float(thr[key], label=f"thresholds.{key}", minimum=0.0)
    cleaned_thr["goalkeeper_require_extra_evidence"] = _require_bool(
        thr["goalkeeper_require_extra_evidence"],
        label="thresholds.goalkeeper_require_extra_evidence",
    )
    if cleaned_thr["goalkeeper_require_extra_evidence"] is not True:
        raise RoleConfigError("thresholds.goalkeeper_require_extra_evidence must be true")
    cleaned_thr["referee_require_non_outfield"] = _require_bool(
        thr["referee_require_non_outfield"], label="thresholds.referee_require_non_outfield"
    )
    if cleaned_thr["referee_require_non_outfield"] is not True:
        raise RoleConfigError("thresholds.referee_require_non_outfield must be true")
    for key in (
        "player_cluster_margin",
        "player_min_quality",
        "goalkeeper_color_margin",
        "goalkeeper_lateral_edge",
        "referee_max_saturation",
        "referee_max_value",
        "referee_color_margin",
        "staff_max_fraction",
        "abstain_margin",
        "conflict_margin",
    ):
        if cleaned_thr[key] > 1.0:
            raise RoleConfigError(f"thresholds.{key} must be <= 1.0")

    out = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if out.get("atomic_writes") is not True:
        raise RoleConfigError("output_policy.atomic_writes must be true")
    if out.get("overwrite_allowed") is not False:
        raise RoleConfigError("output_policy.overwrite_allowed must be false")
    if out.get("role_score_null") is not True:
        raise RoleConfigError("output_policy.role_score_null must be true")
    if out.get("role_source") != "downstream_classifier":
        raise RoleConfigError("output_policy.role_source must be downstream_classifier")
    if out.get("persist_crops") is not False:
        raise RoleConfigError("output_policy.persist_crops must be false")

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/home/fdoblak/workspace/human_role_checks"):
        raise RoleConfigError(
            "runtime_root must be under /home/fdoblak/workspace/human_role_checks"
        )

    if raw["overwrite_allowed"] is not False:
        raise RoleConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise RoleConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise RoleConfigError("network_sources_allowed must be false")

    notes = raw["notes"]
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise RoleConfigError("notes must be a list of strings")
    notes_joined = " ".join(notes).lower()
    if "staff" not in notes_joined or "other" not in notes_joined:
        raise RoleConfigError('notes must document that user "other" maps to staff')

    return {
        "config_version": CONFIG_VERSION,
        "classifier_id": _require_str(raw["classifier_id"], label="classifier_id"),
        "classifier_version": _require_str(raw["classifier_version"], label="classifier_version"),
        "eligibility": {
            "entity_types": entities,
            "require_processed_frames": True,
            "require_analysis_window": True,
            "playability_allowed": play,
            "tracking_eligibility_allowed": track,
            "min_crop_area_px": min_area,
            "min_crop_width": min_w,
            "min_crop_height": min_h,
            "min_aspect_ratio": min_ar,
            "max_aspect_ratio": max_ar,
            "min_crop_quality": min_q,
        },
        "features": {
            "hsv_h_bins": h_bins,
            "hsv_s_bins": s_bins,
            "hsv_v_bins": v_bins,
            "upper_body_fraction": upper,
            "lower_body_fraction": lower,
            "persist_crops": False,
        },
        "clustering": {
            "max_outfield_clusters": 2,
            "min_cluster_size": min_cs,
            "color_distance_threshold": color_thr,
            "min_cluster_stability": stab,
        },
        "thresholds": cleaned_thr,
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "write_evaluation_json": bool(out.get("write_evaluation_json", True)),
            "role_score_null": True,
            "role_source": "downstream_classifier",
            "persist_crops": False,
        },
        "maximum_frames_per_run": _require_int(
            raw["maximum_frames_per_run"],
            label="maximum_frames_per_run",
            minimum=1,
            maximum=100000,
        ),
        "runtime_root": runtime_root,
        "taxonomy_path": _require_str(raw["taxonomy_path"], label="taxonomy_path"),
        "policy_path": _require_str(raw["policy_path"], label="policy_path"),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "notes": list(notes),
    }


def load_human_role_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise RoleConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise RoleConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RoleConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise RoleConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def human_role_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_human_role_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "perception" / "human_role_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "RoleConfigError",
    "load_human_role_config",
    "human_role_config_fingerprint",
    "default_human_role_config_path",
]
