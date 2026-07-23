"""Strict loader for Stage 4B shot boundary baseline config."""

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
SUPPORTED_TRANSITION_TYPES = frozenset({"hard_cut", "dissolve", "fade", "unknown"})
REQUIRED_TOP = frozenset(
    {
        "config_version",
        "analysis_width",
        "analysis_height",
        "feature_weights",
        "hard_cut_threshold",
        "gradual",
        "flash_suppression",
        "peak_suppression",
        "minimum_shot_duration_us",
        "boundary_merge_tolerance_us",
        "supported_transition_types",
        "decode",
        "evaluation",
        "deterministic_seed",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "resource_limits",
    }
)


class ShotConfigError(ValueError):
    """Shot boundary baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShotConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShotConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise ShotConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ShotConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShotConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise ShotConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise ShotConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise ShotConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ShotConfigError(f"{label} must be a bool")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise ShotConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise ShotConfigError(f"unknown top-level keys: {sorted(unknown)}")

    if int(raw["config_version"]) != CONFIG_VERSION:
        raise ShotConfigError(f"config_version must be {CONFIG_VERSION}")

    width = _require_int(raw["analysis_width"], label="analysis_width", minimum=16, maximum=640)
    height = _require_int(raw["analysis_height"], label="analysis_height", minimum=16, maximum=360)

    weights = dict(_require_mapping(raw["feature_weights"], label="feature_weights"))
    for key in ("luma", "hist", "edge"):
        if key not in weights:
            raise ShotConfigError(f"feature_weights.{key} required")
        _require_float(weights[key], label=f"feature_weights.{key}", minimum=0.0, maximum=1.0)
    weight_sum = float(weights["luma"]) + float(weights["hist"]) + float(weights["edge"])
    if abs(weight_sum - 1.0) > 1e-6:
        raise ShotConfigError("feature_weights must sum to 1.0")

    hard = _require_float(
        raw["hard_cut_threshold"], label="hard_cut_threshold", minimum=0.05, maximum=1.0
    )

    gradual = dict(_require_mapping(raw["gradual"], label="gradual"))
    for key in (
        "elevated_mean_threshold",
        "peak_sharpness_max",
        "window_frames",
        "min_elevated_frames",
    ):
        if key not in gradual:
            raise ShotConfigError(f"gradual.{key} required")
    _require_float(
        gradual["elevated_mean_threshold"],
        label="gradual.elevated_mean_threshold",
        minimum=0.01,
        maximum=1.0,
    )
    _require_float(
        gradual["peak_sharpness_max"],
        label="gradual.peak_sharpness_max",
        minimum=0.01,
        maximum=1.0,
    )
    _require_int(gradual["window_frames"], label="gradual.window_frames", minimum=3, maximum=60)
    _require_int(
        gradual["min_elevated_frames"], label="gradual.min_elevated_frames", minimum=2, maximum=60
    )

    flash = dict(_require_mapping(raw["flash_suppression"], label="flash_suppression"))
    for key in ("max_duration_us", "intensity_min", "baseline_similarity_max"):
        if key not in flash:
            raise ShotConfigError(f"flash_suppression.{key} required")
    _require_int(flash["max_duration_us"], label="flash_suppression.max_duration_us", minimum=1)
    _require_float(
        flash["intensity_min"], label="flash_suppression.intensity_min", minimum=0.05, maximum=1.0
    )
    _require_float(
        flash["baseline_similarity_max"],
        label="flash_suppression.baseline_similarity_max",
        minimum=0.0,
        maximum=1.0,
    )

    peak = dict(_require_mapping(raw["peak_suppression"], label="peak_suppression"))
    if "enabled" not in peak:
        raise ShotConfigError("peak_suppression.enabled required")
    _require_bool(peak["enabled"], label="peak_suppression.enabled")

    _require_int(
        raw["minimum_shot_duration_us"],
        label="minimum_shot_duration_us",
        minimum=0,
        maximum=10_000_000,
    )
    _require_int(
        raw["boundary_merge_tolerance_us"],
        label="boundary_merge_tolerance_us",
        minimum=0,
        maximum=5_000_000,
    )

    transitions = raw["supported_transition_types"]
    if not isinstance(transitions, list) or not transitions:
        raise ShotConfigError("supported_transition_types must be a non-empty list")
    for t in transitions:
        if t not in SUPPORTED_TRANSITION_TYPES:
            raise ShotConfigError(f"unsupported transition type: {t}")

    decode = dict(_require_mapping(raw["decode"], label="decode"))
    if decode.get("backend") != "opencv":
        raise ShotConfigError("decode.backend must be opencv")
    _require_int(decode["max_frames"], label="decode.max_frames", minimum=1, maximum=1_000_000)
    _require_float(
        decode["timeout_seconds"], label="decode.timeout_seconds", minimum=1.0, maximum=3600.0
    )

    evaluation = dict(_require_mapping(raw["evaluation"], label="evaluation"))
    _require_int(
        evaluation["matching_tolerance_us"],
        label="evaluation.matching_tolerance_us",
        minimum=0,
        maximum=5_000_000,
    )

    _require_int(raw["deterministic_seed"], label="deterministic_seed", minimum=0)

    runtime_root = raw["runtime_root"]
    if not isinstance(runtime_root, str) or not runtime_root.startswith("/"):
        raise ShotConfigError("runtime_root must be an absolute path string")

    if raw["overwrite_allowed"] is not False:
        raise ShotConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise ShotConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise ShotConfigError("network_sources_allowed must be false")

    limits = dict(_require_mapping(raw["resource_limits"], label="resource_limits"))
    _require_int(limits["max_frames"], label="resource_limits.max_frames", minimum=1)
    _require_float(limits["timeout_seconds"], label="resource_limits.timeout_seconds", minimum=1.0)

    return {
        "config_version": CONFIG_VERSION,
        "analysis_width": width,
        "analysis_height": height,
        "feature_weights": {
            "luma": float(weights["luma"]),
            "hist": float(weights["hist"]),
            "edge": float(weights["edge"]),
        },
        "hard_cut_threshold": hard,
        "gradual": {
            "elevated_mean_threshold": float(gradual["elevated_mean_threshold"]),
            "peak_sharpness_max": float(gradual["peak_sharpness_max"]),
            "window_frames": int(gradual["window_frames"]),
            "min_elevated_frames": int(gradual["min_elevated_frames"]),
        },
        "flash_suppression": {
            "max_duration_us": int(flash["max_duration_us"]),
            "intensity_min": float(flash["intensity_min"]),
            "baseline_similarity_max": float(flash["baseline_similarity_max"]),
        },
        "peak_suppression": {"enabled": bool(peak["enabled"])},
        "minimum_shot_duration_us": int(raw["minimum_shot_duration_us"]),
        "boundary_merge_tolerance_us": int(raw["boundary_merge_tolerance_us"]),
        "supported_transition_types": list(transitions),
        "decode": {
            "backend": "opencv",
            "max_frames": int(decode["max_frames"]),
            "timeout_seconds": float(decode["timeout_seconds"]),
        },
        "evaluation": {
            "matching_tolerance_us": int(evaluation["matching_tolerance_us"]),
        },
        "deterministic_seed": int(raw["deterministic_seed"]),
        "runtime_root": str(runtime_root),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "resource_limits": {
            "max_frames": int(limits["max_frames"]),
            "timeout_seconds": float(limits["timeout_seconds"]),
        },
    }


def load_shot_boundary_config(path: Path | str) -> Mapping[str, Any]:
    """Load and strictly validate shot boundary baseline YAML."""
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ShotConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ShotConfigError("config exceeds maximum byte size")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ShotConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise ShotConfigError("config root must be a mapping")
    validated = _validate_config(data)
    return _deep_freeze(validated)


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def shot_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 of canonical JSON config payload."""
    return hash_canonical_json(_deep_unfreeze(config))


def default_shot_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "broadcast" / "shot_boundary_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "SUPPORTED_TRANSITION_TYPES",
    "ShotConfigError",
    "load_shot_boundary_config",
    "shot_config_fingerprint",
    "default_shot_config_path",
]
