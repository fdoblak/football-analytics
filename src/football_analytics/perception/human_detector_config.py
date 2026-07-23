"""Strict loader for Stage 5B human detector baseline config."""

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
        "model_registry_id",
        "adapter_id",
        "adapter_version",
        "person_class",
        "input_size",
        "device_policy",
        "precision_policy",
        "batch_size",
        "confidence_threshold",
        "nms_iou",
        "class_aware_nms",
        "minimum_bbox_area",
        "maximum_aspect_ratio",
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


class HumanDetectorConfigError(ValueError):
    """Human detector baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanDetectorConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HumanDetectorConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise HumanDetectorConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HumanDetectorConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanDetectorConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise HumanDetectorConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise HumanDetectorConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise HumanDetectorConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise HumanDetectorConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HumanDetectorConfigError(f"{label} must be a non-empty string")
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
        raise HumanDetectorConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise HumanDetectorConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise HumanDetectorConfigError(f"config_version must be {CONFIG_VERSION}")

    person = dict(_require_mapping(raw["person_class"], label="person_class"))
    class_ids = person.get("class_ids")
    if not isinstance(class_ids, list) or not class_ids:
        raise HumanDetectorConfigError("person_class.class_ids must be non-empty int list")
    for cid in class_ids:
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise HumanDetectorConfigError("person_class.class_ids entries must be int")
    class_names = person.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise HumanDetectorConfigError("person_class.class_names must be non-empty list")
    for name in class_names:
        if not isinstance(name, str) or not name:
            raise HumanDetectorConfigError("person_class.class_names entries must be strings")

    input_size = _require_int(raw["input_size"], label="input_size", minimum=32, maximum=1280)
    device_policy = _require_str(raw["device_policy"], label="device_policy")
    if device_policy not in {"prefer_cuda_else_cpu", "cpu_only", "cuda_required"}:
        raise HumanDetectorConfigError("unsupported device_policy")
    precision_policy = _require_str(raw["precision_policy"], label="precision_policy")
    if precision_policy not in {"fp16_on_cuda_else_fp32", "fp32_only"}:
        raise HumanDetectorConfigError("unsupported precision_policy")
    if _require_int(raw["batch_size"], label="batch_size", minimum=1, maximum=1) != 1:
        raise HumanDetectorConfigError("batch_size must be 1")

    conf = _require_float(
        raw["confidence_threshold"], label="confidence_threshold", minimum=0.0, maximum=1.0
    )
    nms = _require_float(raw["nms_iou"], label="nms_iou", minimum=0.0, maximum=1.0)
    class_aware = _require_bool(raw["class_aware_nms"], label="class_aware_nms")
    if class_aware is not True:
        raise HumanDetectorConfigError("class_aware_nms must be true")

    min_area = _require_float(raw["minimum_bbox_area"], label="minimum_bbox_area", minimum=1.0)
    max_ar = _require_float(raw["maximum_aspect_ratio"], label="maximum_aspect_ratio", minimum=1.0)

    routing = dict(_require_mapping(raw["routing"], label="routing"))
    _require_bool(routing.get("detect_ball", True), label="routing.detect_ball")
    if routing.get("detect_ball") is not False:
        raise HumanDetectorConfigError("routing.detect_ball must be false for Stage 5B")
    ref = routing.get("policy_ref")
    if ref is not None and not isinstance(ref, str):
        raise HumanDetectorConfigError("routing.policy_ref must be string or null")

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
        raise HumanDetectorConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise HumanDetectorConfigError("output_policy.overwrite_allowed must be false")
    if output.get("emit_ball_detections") is not False:
        raise HumanDetectorConfigError("output_policy.emit_ball_detections must be false")

    ious = raw["evaluation_iou_thresholds"]
    if not isinstance(ious, list) or not ious:
        raise HumanDetectorConfigError("evaluation_iou_thresholds must be non-empty list")
    cleaned_ious: list[float] = []
    for i, v in enumerate(ious):
        cleaned_ious.append(
            _require_float(v, label=f"evaluation_iou_thresholds[{i}]", minimum=0.0, maximum=1.0)
        )

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/"):
        raise HumanDetectorConfigError("runtime_root must be absolute")
    taxonomy_path = _require_str(raw["taxonomy_path"], label="taxonomy_path")
    policy_path = _require_str(raw["policy_path"], label="policy_path")

    if raw["overwrite_allowed"] is not False:
        raise HumanDetectorConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise HumanDetectorConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise HumanDetectorConfigError("network_sources_allowed must be false")

    return {
        "config_version": CONFIG_VERSION,
        "model_registry_id": _require_str(raw["model_registry_id"], label="model_registry_id"),
        "adapter_id": _require_str(raw["adapter_id"], label="adapter_id"),
        "adapter_version": _require_str(raw["adapter_version"], label="adapter_version"),
        "person_class": {
            "class_ids": [int(x) for x in class_ids],
            "class_names": [str(x).lower() for x in class_names],
        },
        "input_size": input_size,
        "device_policy": device_policy,
        "precision_policy": precision_policy,
        "batch_size": 1,
        "confidence_threshold": conf,
        "nms_iou": nms,
        "class_aware_nms": True,
        "minimum_bbox_area": min_area,
        "maximum_aspect_ratio": max_ar,
        "routing": {
            "detect_ball": False,
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
            "emit_ball_detections": False,
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


def load_human_detector_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise HumanDetectorConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise HumanDetectorConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HumanDetectorConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise HumanDetectorConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def human_detector_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_human_detector_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "perception" / "human_detector_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "HumanDetectorConfigError",
    "load_human_detector_config",
    "human_detector_config_fingerprint",
    "default_human_detector_config_path",
]
