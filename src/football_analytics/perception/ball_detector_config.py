"""Strict loader for Stage 5C ball detector baseline config."""

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
ALLOWED_INFERENCE_MODES = frozenset({"full_frame", "tiled", "hybrid"})
REQUIRED_TOP = frozenset(
    {
        "config_version",
        "model_registry_id",
        "adapter_id",
        "adapter_version",
        "sports_ball_class",
        "inference_mode",
        "input_size",
        "tiling",
        "device_policy",
        "precision_policy",
        "batch_size",
        "confidence_threshold",
        "nms_iou",
        "merge_iou",
        "class_aware_nms",
        "filters",
        "routing",
        "maximum_frames_per_run",
        "timeout_seconds",
        "resource_limits",
        "output_policy",
        "evaluation_iou_thresholds",
        "runtime_root",
        "taxonomy_path",
        "policy_path",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "coordinate_space_note",
    }
)


class BallDetectorConfigError(ValueError):
    """Ball detector baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BallDetectorConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BallDetectorConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise BallDetectorConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise BallDetectorConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BallDetectorConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise BallDetectorConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise BallDetectorConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise BallDetectorConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BallDetectorConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BallDetectorConfigError(f"{label} must be a non-empty string")
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
        raise BallDetectorConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise BallDetectorConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise BallDetectorConfigError(f"config_version must be {CONFIG_VERSION}")

    sbc = dict(_require_mapping(raw["sports_ball_class"], label="sports_ball_class"))
    class_ids = sbc.get("class_ids")
    if not isinstance(class_ids, list) or not class_ids:
        raise BallDetectorConfigError("sports_ball_class.class_ids must be non-empty int list")
    for cid in class_ids:
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise BallDetectorConfigError("sports_ball_class.class_ids entries must be int")
    class_names = sbc.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise BallDetectorConfigError("sports_ball_class.class_names must be non-empty list")
    for name in class_names:
        if not isinstance(name, str) or not name:
            raise BallDetectorConfigError("sports_ball_class.class_names entries must be strings")
    if "sports ball" not in class_names and "sports_ball" not in class_names:
        raise BallDetectorConfigError(
            'sports_ball_class.class_names must include "sports ball" or sports_ball'
        )

    mode = _require_str(raw["inference_mode"], label="inference_mode")
    if mode not in ALLOWED_INFERENCE_MODES:
        raise BallDetectorConfigError(
            f"inference_mode must be one of {sorted(ALLOWED_INFERENCE_MODES)}"
        )

    tiling = dict(_require_mapping(raw["tiling"], label="tiling"))
    for key in ("tile_width", "tile_height", "overlap_x", "overlap_y", "max_tiles"):
        if key not in tiling:
            raise BallDetectorConfigError(f"tiling.{key} required")
    tile_w = _require_int(tiling["tile_width"], label="tiling.tile_width", minimum=16)
    tile_h = _require_int(tiling["tile_height"], label="tiling.tile_height", minimum=16)
    overlap_x = _require_int(tiling["overlap_x"], label="tiling.overlap_x", minimum=0)
    overlap_y = _require_int(tiling["overlap_y"], label="tiling.overlap_y", minimum=0)
    max_tiles = _require_int(tiling["max_tiles"], label="tiling.max_tiles", minimum=1, maximum=256)
    if overlap_x >= tile_w or overlap_y >= tile_h:
        raise BallDetectorConfigError("tiling overlap must be < tile size")

    input_size = _require_int(raw["input_size"], label="input_size", minimum=32, maximum=1280)
    device_policy = _require_str(raw["device_policy"], label="device_policy")
    if device_policy not in {"prefer_cuda_else_cpu", "cpu_only", "cuda_required"}:
        raise BallDetectorConfigError("unsupported device_policy")
    precision_policy = _require_str(raw["precision_policy"], label="precision_policy")
    if precision_policy not in {"fp16_on_cuda_else_fp32", "fp32_only"}:
        raise BallDetectorConfigError("unsupported precision_policy")
    if _require_int(raw["batch_size"], label="batch_size", minimum=1, maximum=1) != 1:
        raise BallDetectorConfigError("batch_size must be 1")

    conf = _require_float(
        raw["confidence_threshold"], label="confidence_threshold", minimum=0.0, maximum=1.0
    )
    nms = _require_float(raw["nms_iou"], label="nms_iou", minimum=0.0, maximum=1.0)
    merge = _require_float(raw["merge_iou"], label="merge_iou", minimum=0.0, maximum=1.0)
    class_aware = _require_bool(raw["class_aware_nms"], label="class_aware_nms")
    if class_aware is not True:
        raise BallDetectorConfigError("class_aware_nms must be true")

    filters = dict(_require_mapping(raw["filters"], label="filters"))
    min_w = _require_float(filters["min_width"], label="filters.min_width", minimum=1.0)
    max_w = _require_float(filters["max_width"], label="filters.max_width", minimum=1.0)
    min_h = _require_float(filters["min_height"], label="filters.min_height", minimum=1.0)
    max_h = _require_float(filters["max_height"], label="filters.max_height", minimum=1.0)
    min_af = _require_float(
        filters["min_area_fraction"], label="filters.min_area_fraction", minimum=0.0, maximum=1.0
    )
    max_af = _require_float(
        filters["max_area_fraction"], label="filters.max_area_fraction", minimum=0.0, maximum=1.0
    )
    max_ar = _require_float(
        filters["max_aspect_ratio"], label="filters.max_aspect_ratio", minimum=1.0
    )
    if min_w > max_w or min_h > max_h or min_af > max_af:
        raise BallDetectorConfigError("filters min/max inconsistent")

    routing = dict(_require_mapping(raw["routing"], label="routing"))
    if _require_bool(routing.get("detect_ball", False), label="routing.detect_ball") is not True:
        raise BallDetectorConfigError("routing.detect_ball must be true for Stage 5C")
    if _require_bool(routing.get("emit_human", True), label="routing.emit_human") is not False:
        raise BallDetectorConfigError("routing.emit_human must be false for Stage 5C")
    skip_identity = _require_bool(
        routing.get("skip_identity_only", True), label="routing.skip_identity_only"
    )
    if skip_identity is not True:
        raise BallDetectorConfigError("routing.skip_identity_only must be true by default")
    ref = routing.get("policy_ref")
    if ref is not None and not isinstance(ref, str):
        raise BallDetectorConfigError("routing.policy_ref must be string or null")
    require_ball_el = routing.get("ball_requires", ["eligible", "conditionally_eligible"])
    if not isinstance(require_ball_el, list) or not require_ball_el:
        raise BallDetectorConfigError("routing.ball_requires must be non-empty list")

    max_frames = _require_int(
        raw["maximum_frames_per_run"], label="maximum_frames_per_run", minimum=1, maximum=100000
    )
    timeout = _require_float(raw["timeout_seconds"], label="timeout_seconds", minimum=1.0)

    limits = dict(_require_mapping(raw["resource_limits"], label="resource_limits"))
    _require_int(
        limits["max_detections_per_frame"],
        label="resource_limits.max_detections_per_frame",
        minimum=1,
    )
    _require_float(
        limits["decode_timeout_seconds"],
        label="resource_limits.decode_timeout_seconds",
        minimum=1.0,
    )

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if output.get("atomic_writes") is not True:
        raise BallDetectorConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise BallDetectorConfigError("output_policy.overwrite_allowed must be false")
    if output.get("emit_human_detections") is not False:
        raise BallDetectorConfigError("output_policy.emit_human_detections must be false")

    ious = raw["evaluation_iou_thresholds"]
    if not isinstance(ious, list) or not ious:
        raise BallDetectorConfigError("evaluation_iou_thresholds must be non-empty list")
    cleaned_ious: list[float] = []
    for i, v in enumerate(ious):
        cleaned_ious.append(
            _require_float(v, label=f"evaluation_iou_thresholds[{i}]", minimum=0.0, maximum=1.0)
        )

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/"):
        raise BallDetectorConfigError("runtime_root must be absolute")
    if not runtime_root.startswith("/home/fdoblak/workspace/ball_detection_checks"):
        raise BallDetectorConfigError(
            "runtime_root must be under /home/fdoblak/workspace/ball_detection_checks"
        )
    taxonomy_path = _require_str(raw["taxonomy_path"], label="taxonomy_path")
    policy_path = _require_str(raw["policy_path"], label="policy_path")

    if raw["overwrite_allowed"] is not False:
        raise BallDetectorConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise BallDetectorConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise BallDetectorConfigError("network_sources_allowed must be false")

    return {
        "config_version": CONFIG_VERSION,
        "model_registry_id": _require_str(raw["model_registry_id"], label="model_registry_id"),
        "adapter_id": _require_str(raw["adapter_id"], label="adapter_id"),
        "adapter_version": _require_str(raw["adapter_version"], label="adapter_version"),
        "sports_ball_class": {
            "class_ids": [int(x) for x in class_ids],
            "class_names": [str(x) for x in class_names],
        },
        "inference_mode": mode,
        "input_size": input_size,
        "tiling": {
            "tile_width": tile_w,
            "tile_height": tile_h,
            "overlap_x": overlap_x,
            "overlap_y": overlap_y,
            "max_tiles": max_tiles,
        },
        "device_policy": device_policy,
        "precision_policy": precision_policy,
        "batch_size": 1,
        "confidence_threshold": conf,
        "nms_iou": nms,
        "merge_iou": merge,
        "class_aware_nms": True,
        "filters": {
            "min_width": min_w,
            "max_width": max_w,
            "min_height": min_h,
            "max_height": max_h,
            "min_area_fraction": min_af,
            "max_area_fraction": max_af,
            "max_aspect_ratio": max_ar,
        },
        "routing": {
            "detect_ball": True,
            "emit_human": False,
            "skip_identity_only": True,
            "ball_requires": [str(x) for x in require_ball_el],
            "policy_ref": None if ref is None else str(ref),
        },
        "maximum_frames_per_run": max_frames,
        "timeout_seconds": timeout,
        "resource_limits": {
            "max_detections_per_frame": int(limits["max_detections_per_frame"]),
            "decode_timeout_seconds": float(limits["decode_timeout_seconds"]),
        },
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "emit_human_detections": False,
            "write_evaluation_json": bool(output.get("write_evaluation_json", True)),
        },
        "evaluation_iou_thresholds": cleaned_ious,
        "runtime_root": runtime_root,
        "taxonomy_path": taxonomy_path,
        "policy_path": policy_path,
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "coordinate_space_note": _require_str(
            raw["coordinate_space_note"], label="coordinate_space_note"
        ),
    }


def load_ball_detector_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise BallDetectorConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise BallDetectorConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise BallDetectorConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise BallDetectorConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def ball_detector_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_ball_detector_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "perception" / "ball_detector_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "ALLOWED_INFERENCE_MODES",
    "BallDetectorConfigError",
    "load_ball_detector_config",
    "ball_detector_config_fingerprint",
    "default_ball_detector_config_path",
]
