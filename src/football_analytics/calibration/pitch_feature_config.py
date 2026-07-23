"""Strict loader for Stage 8B pitch feature detection baseline config."""

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
        "adapter_id",
        "adapter_version",
        "adapter_choice",
        "kp_model_registry_id",
        "lines_model_registry_id",
        "hrnet_kp_module_path",
        "hrnet_lines_module_path",
        "hrnet_kp_config_path",
        "hrnet_lines_config_path",
        "image_size",
        "color_order",
        "normalize_mean",
        "normalize_std",
        "tensor_dtype",
        "to_tensor_range",
        "num_joints_kp",
        "num_joints_lines",
        "kp_channels_used",
        "lines_channels_used",
        "drop_background_channel",
        "peak_decode_scale",
        "kp_max_peaks",
        "kp_min_peak_distance",
        "lines_max_peaks",
        "lines_min_peak_distance",
        "kp_score_threshold",
        "line_score_threshold",
        "minimum_line_length_px",
        "duplicate_keypoint_distance_px",
        "duplicate_line_endpoint_distance_px",
        "max_features_per_frame",
        "device_policy",
        "precision_policy",
        "batch_size",
        "maximum_frames_per_run",
        "timeout_seconds",
        "resource_limits",
        "routing",
        "output_policy",
        "review_sampling",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "auto_homography",
        "license_note",
        "coordinate_space_note",
    }
)


class PitchFeatureConfigError(ValueError):
    """Pitch feature baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PitchFeatureConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PitchFeatureConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PitchFeatureConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise PitchFeatureConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PitchFeatureConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise PitchFeatureConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise PitchFeatureConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise PitchFeatureConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PitchFeatureConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PitchFeatureConfigError(f"{label} must be a non-empty string")
    return value


def _require_abs_path(value: Any, *, label: str) -> str:
    s = _require_str(value, label=label)
    if not s.startswith("/"):
        raise PitchFeatureConfigError(f"{label} must be absolute")
    return s


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise PitchFeatureConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise PitchFeatureConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise PitchFeatureConfigError(f"config_version must be {CONFIG_VERSION}")

    image_size = raw["image_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise PitchFeatureConfigError("image_size must be [width, height]")
    w = _require_int(image_size[0], label="image_size[0]", minimum=32, maximum=4096)
    h = _require_int(image_size[1], label="image_size[1]", minimum=32, maximum=4096)
    if (w, h) != (960, 540):
        raise PitchFeatureConfigError("image_size must be [960, 540] for Stage 8B SV models")

    color_order = _require_str(raw["color_order"], label="color_order").lower()
    if color_order != "rgb":
        raise PitchFeatureConfigError("color_order must be rgb")
    if raw["normalize_mean"] is not None or raw["normalize_std"] is not None:
        raise PitchFeatureConfigError("normalize_mean/std must be null (no mean/std)")
    if _require_str(raw["tensor_dtype"], label="tensor_dtype") != "float32":
        raise PitchFeatureConfigError("tensor_dtype must be float32")
    tr = raw["to_tensor_range"]
    if not isinstance(tr, list) or len(tr) != 2:
        raise PitchFeatureConfigError("to_tensor_range must be [lo, hi]")
    lo = _require_float(tr[0], label="to_tensor_range[0]")
    hi = _require_float(tr[1], label="to_tensor_range[1]")
    if (lo, hi) != (0.0, 1.0):
        raise PitchFeatureConfigError("to_tensor_range must be [0.0, 1.0]")

    if _require_int(raw["num_joints_kp"], label="num_joints_kp") != 58:
        raise PitchFeatureConfigError("num_joints_kp must be 58")
    if _require_int(raw["num_joints_lines"], label="num_joints_lines") != 24:
        raise PitchFeatureConfigError("num_joints_lines must be 24")
    if _require_int(raw["kp_channels_used"], label="kp_channels_used") != 57:
        raise PitchFeatureConfigError("kp_channels_used must be 57")
    if _require_int(raw["lines_channels_used"], label="lines_channels_used") != 23:
        raise PitchFeatureConfigError("lines_channels_used must be 23")
    if raw["drop_background_channel"] is not True:
        raise PitchFeatureConfigError("drop_background_channel must be true")
    if _require_int(raw["peak_decode_scale"], label="peak_decode_scale") != 2:
        raise PitchFeatureConfigError("peak_decode_scale must be 2")
    if _require_int(raw["batch_size"], label="batch_size", minimum=1, maximum=1) != 1:
        raise PitchFeatureConfigError("batch_size must be 1")

    device_policy = _require_str(raw["device_policy"], label="device_policy")
    if device_policy not in {"prefer_cuda_else_cpu", "cpu_only", "cuda_required"}:
        raise PitchFeatureConfigError("unsupported device_policy")
    precision_policy = _require_str(raw["precision_policy"], label="precision_policy")
    if precision_policy not in {"fp32_only", "fp16_on_cuda_else_fp32"}:
        raise PitchFeatureConfigError("unsupported precision_policy")

    limits = dict(_require_mapping(raw["resource_limits"], label="resource_limits"))
    _require_float(
        limits["decode_timeout_seconds"],
        label="resource_limits.decode_timeout_seconds",
        minimum=1.0,
    )
    _require_int(limits["max_concurrency"], label="resource_limits.max_concurrency", minimum=1)

    routing = dict(_require_mapping(raw["routing"], label="routing"))
    if routing.get("require_calibration_eligible") is not True:
        raise PitchFeatureConfigError("routing.require_calibration_eligible must be true")
    if routing.get("auto_homography") is not False:
        raise PitchFeatureConfigError("routing.auto_homography must be false")
    _require_bool(
        routing.get("reject_graphics_replay_closeup", True),
        label="routing.reject_graphics_replay_closeup",
    )

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if output.get("atomic_writes") is not True:
        raise PitchFeatureConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise PitchFeatureConfigError("output_policy.overwrite_allowed must be false")
    if output.get("confidence_always_null") is not True:
        raise PitchFeatureConfigError("output_policy.confidence_always_null must be true")

    review = dict(_require_mapping(raw["review_sampling"], label="review_sampling"))
    _require_bool(review.get("enabled", False), label="review_sampling.enabled")
    _require_int(review.get("max_samples", 0), label="review_sampling.max_samples", minimum=0)

    runtime_root = _require_abs_path(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/home/fdoblak/workspace/"):
        raise PitchFeatureConfigError("runtime_root must be under /home/fdoblak/workspace/")

    if raw["overwrite_allowed"] is not False:
        raise PitchFeatureConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise PitchFeatureConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise PitchFeatureConfigError("network_sources_allowed must be false")
    if raw["auto_homography"] is not False:
        raise PitchFeatureConfigError("auto_homography must be false")

    return {
        "config_version": CONFIG_VERSION,
        "adapter_id": _require_str(raw["adapter_id"], label="adapter_id"),
        "adapter_version": _require_str(raw["adapter_version"], label="adapter_version"),
        "adapter_choice": _require_str(raw["adapter_choice"], label="adapter_choice"),
        "kp_model_registry_id": _require_str(
            raw["kp_model_registry_id"], label="kp_model_registry_id"
        ),
        "lines_model_registry_id": _require_str(
            raw["lines_model_registry_id"], label="lines_model_registry_id"
        ),
        "hrnet_kp_module_path": _require_abs_path(
            raw["hrnet_kp_module_path"], label="hrnet_kp_module_path"
        ),
        "hrnet_lines_module_path": _require_abs_path(
            raw["hrnet_lines_module_path"], label="hrnet_lines_module_path"
        ),
        "hrnet_kp_config_path": _require_abs_path(
            raw["hrnet_kp_config_path"], label="hrnet_kp_config_path"
        ),
        "hrnet_lines_config_path": _require_abs_path(
            raw["hrnet_lines_config_path"], label="hrnet_lines_config_path"
        ),
        "image_size": [w, h],
        "color_order": "rgb",
        "normalize_mean": None,
        "normalize_std": None,
        "tensor_dtype": "float32",
        "to_tensor_range": [0.0, 1.0],
        "num_joints_kp": 58,
        "num_joints_lines": 24,
        "kp_channels_used": 57,
        "lines_channels_used": 23,
        "drop_background_channel": True,
        "peak_decode_scale": 2,
        "kp_max_peaks": _require_int(raw["kp_max_peaks"], label="kp_max_peaks", minimum=1),
        "kp_min_peak_distance": _require_int(
            raw["kp_min_peak_distance"], label="kp_min_peak_distance", minimum=1
        ),
        "lines_max_peaks": _require_int(raw["lines_max_peaks"], label="lines_max_peaks", minimum=1),
        "lines_min_peak_distance": _require_int(
            raw["lines_min_peak_distance"], label="lines_min_peak_distance", minimum=1
        ),
        "kp_score_threshold": _require_float(
            raw["kp_score_threshold"], label="kp_score_threshold", minimum=0.0, maximum=1.0
        ),
        "line_score_threshold": _require_float(
            raw["line_score_threshold"],
            label="line_score_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_line_length_px": _require_float(
            raw["minimum_line_length_px"], label="minimum_line_length_px", minimum=1.0
        ),
        "duplicate_keypoint_distance_px": _require_float(
            raw["duplicate_keypoint_distance_px"],
            label="duplicate_keypoint_distance_px",
            minimum=0.0,
        ),
        "duplicate_line_endpoint_distance_px": _require_float(
            raw["duplicate_line_endpoint_distance_px"],
            label="duplicate_line_endpoint_distance_px",
            minimum=0.0,
        ),
        "max_features_per_frame": _require_int(
            raw["max_features_per_frame"], label="max_features_per_frame", minimum=1
        ),
        "device_policy": device_policy,
        "precision_policy": precision_policy,
        "batch_size": 1,
        "maximum_frames_per_run": _require_int(
            raw["maximum_frames_per_run"],
            label="maximum_frames_per_run",
            minimum=1,
            maximum=100,
        ),
        "timeout_seconds": _require_float(
            raw["timeout_seconds"], label="timeout_seconds", minimum=1.0
        ),
        "resource_limits": {
            "decode_timeout_seconds": float(limits["decode_timeout_seconds"]),
            "max_concurrency": int(limits["max_concurrency"]),
        },
        "routing": {
            "require_calibration_eligible": True,
            "reject_graphics_replay_closeup": bool(
                routing.get("reject_graphics_replay_closeup", True)
            ),
            "auto_homography": False,
        },
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "write_evaluation_json": bool(output.get("write_evaluation_json", True)),
            "write_quality_json": bool(output.get("write_quality_json", True)),
            "confidence_always_null": True,
        },
        "review_sampling": {
            "enabled": bool(review.get("enabled", False)),
            "max_samples": int(review.get("max_samples", 0)),
        },
        "runtime_root": runtime_root,
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "auto_homography": False,
        "license_note": _require_str(raw["license_note"], label="license_note"),
        "coordinate_space_note": _require_str(
            raw["coordinate_space_note"], label="coordinate_space_note"
        ),
    }


def load_pitch_feature_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise PitchFeatureConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise PitchFeatureConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PitchFeatureConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise PitchFeatureConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def pitch_feature_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_pitch_feature_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "calibration" / "pitch_feature_baseline.yaml"


def unfreeze_pitch_feature_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable deep copy of a frozen config mapping."""
    return _deep_unfreeze(config)


__all__ = [
    "CONFIG_VERSION",
    "PitchFeatureConfigError",
    "load_pitch_feature_config",
    "pitch_feature_config_fingerprint",
    "default_pitch_feature_config_path",
    "unfreeze_pitch_feature_config",
]
