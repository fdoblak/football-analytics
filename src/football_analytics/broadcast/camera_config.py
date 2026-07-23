"""Strict loader for Stage 4C camera-view classification baseline config."""

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

SUPPORTED_VIEW = frozenset({"main_broadcast", "player_isolation", "graphics", "unknown"})
SUPPORTED_FRAMING = frozenset({"wide", "medium", "close_up", "unknown"})
SUPPORTED_MOTION = frozenset({"static", "pan", "zoom", "compound", "unstable", "unknown"})
SUPPORTED_GRAPHICS = frozenset(
    {"none", "partial_overlay", "dominant_overlay", "full_screen", "unknown"}
)
SUPPORTED_PLAYABILITY = frozenset({"playable", "partially_playable", "non_playable", "uncertain"})
ALWAYS_UNKNOWN = frozenset({"camera_position", "replay_status"})

REQUIRED_TOP = frozenset(
    {
        "config_version",
        "sampling",
        "analysis_width",
        "analysis_height",
        "feature_version",
        "pitch_hsv",
        "skin_hsv",
        "thresholds",
        "weights",
        "optical_flow",
        "min_coverage",
        "supported_axes",
        "always_unknown_axes",
        "suitability_rule_ids",
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


class CameraConfigError(ValueError):
    """Camera-view baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CameraConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise CameraConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise CameraConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise CameraConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise CameraConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise CameraConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise CameraConfigError(f"{label} must be a bool")
    return value


def _require_str_list(
    value: Any, *, label: str, allowed: frozenset[str] | None = None
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CameraConfigError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise CameraConfigError(f"{label} entries must be non-empty strings")
        if allowed is not None and item not in allowed:
            raise CameraConfigError(f"{label} unsupported class: {item}")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_hsv(block: Mapping[str, Any], *, label: str, keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        if key not in block:
            raise CameraConfigError(f"{label}.{key} required")
        out[key] = _require_int(block[key], label=f"{label}.{key}", minimum=0, maximum=255)
    return out


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise CameraConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise CameraConfigError(f"unknown top-level keys: {sorted(unknown)}")

    if int(raw["config_version"]) != CONFIG_VERSION:
        raise CameraConfigError(f"config_version must be {CONFIG_VERSION}")

    sampling = dict(_require_mapping(raw["sampling"], label="sampling"))
    for key in ("samples_per_shot", "edge_exclude_fraction", "min_samples", "max_samples"):
        if key not in sampling:
            raise CameraConfigError(f"sampling.{key} required")
    samples = _require_int(
        sampling["samples_per_shot"], label="sampling.samples_per_shot", minimum=1
    )
    edge = _require_float(
        sampling["edge_exclude_fraction"],
        label="sampling.edge_exclude_fraction",
        minimum=0.0,
        maximum=0.45,
    )
    min_s = _require_int(sampling["min_samples"], label="sampling.min_samples", minimum=1)
    max_s = _require_int(sampling["max_samples"], label="sampling.max_samples", minimum=1)
    if min_s > max_s:
        raise CameraConfigError("sampling.min_samples must be <= max_samples")
    if samples < min_s or samples > max_s:
        raise CameraConfigError("sampling.samples_per_shot must be within min/max")

    width = _require_int(raw["analysis_width"], label="analysis_width", minimum=16, maximum=640)
    height = _require_int(raw["analysis_height"], label="analysis_height", minimum=16, maximum=360)
    feature_version = _require_int(raw["feature_version"], label="feature_version", minimum=1)

    pitch = dict(_require_mapping(raw["pitch_hsv"], label="pitch_hsv"))
    pitch_out = _validate_hsv(
        pitch,
        label="pitch_hsv",
        keys=(
            "h_min",
            "h_max",
            "s_min",
            "v_min",
            "alt_h_min",
            "alt_h_max",
            "alt_s_min",
            "alt_v_min",
        ),
    )
    skin = dict(_require_mapping(raw["skin_hsv"], label="skin_hsv"))
    skin_out = _validate_hsv(
        skin,
        label="skin_hsv",
        keys=("h_min", "h_max", "s_min", "v_min", "v_max"),
    )

    thresholds = dict(_require_mapping(raw["thresholds"], label="thresholds"))
    for block_name in ("view", "framing", "motion", "graphics", "aggregation", "abstention"):
        if block_name not in thresholds:
            raise CameraConfigError(f"thresholds.{block_name} required")
        dict(_require_mapping(thresholds[block_name], label=f"thresholds.{block_name}"))

    weights = dict(_require_mapping(raw["weights"], label="weights"))
    for axis in ("view", "framing", "motion", "graphics"):
        if axis not in weights:
            raise CameraConfigError(f"weights.{axis} required")
        wblock = dict(_require_mapping(weights[axis], label=f"weights.{axis}"))
        for k, v in wblock.items():
            _require_float(v, label=f"weights.{axis}.{k}", minimum=0.0, maximum=1.0)

    flow = dict(_require_mapping(raw["optical_flow"], label="optical_flow"))
    _require_bool(flow.get("enabled"), label="optical_flow.enabled")
    for key in ("pyr_scale", "poly_sigma"):
        if key not in flow:
            raise CameraConfigError(f"optical_flow.{key} required")
        _require_float(flow[key], label=f"optical_flow.{key}", minimum=0.01)
    for key in ("levels", "winsize", "iterations", "poly_n"):
        if key not in flow:
            raise CameraConfigError(f"optical_flow.{key} required")
        _require_int(flow[key], label=f"optical_flow.{key}", minimum=1)

    min_coverage = _require_float(
        raw["min_coverage"], label="min_coverage", minimum=0.0, maximum=1.0
    )

    axes = dict(_require_mapping(raw["supported_axes"], label="supported_axes"))
    supported = {
        "view_family": _require_str_list(
            axes.get("view_family"), label="supported_axes.view_family", allowed=SUPPORTED_VIEW
        ),
        "framing_scale": _require_str_list(
            axes.get("framing_scale"),
            label="supported_axes.framing_scale",
            allowed=SUPPORTED_FRAMING,
        ),
        "camera_motion": _require_str_list(
            axes.get("camera_motion"),
            label="supported_axes.camera_motion",
            allowed=SUPPORTED_MOTION,
        ),
        "graphics_status": _require_str_list(
            axes.get("graphics_status"),
            label="supported_axes.graphics_status",
            allowed=SUPPORTED_GRAPHICS,
        ),
        "playability": _require_str_list(
            axes.get("playability"),
            label="supported_axes.playability",
            allowed=SUPPORTED_PLAYABILITY,
        ),
    }

    always_unknown = _require_str_list(
        raw["always_unknown_axes"], label="always_unknown_axes", allowed=ALWAYS_UNKNOWN
    )
    if set(always_unknown) != ALWAYS_UNKNOWN:
        raise CameraConfigError(
            "always_unknown_axes must be exactly camera_position, replay_status"
        )

    rules = raw["suitability_rule_ids"]
    if not isinstance(rules, list) or not rules:
        raise CameraConfigError("suitability_rule_ids must be a non-empty list")
    rule_ids = []
    for r in rules:
        if not isinstance(r, str) or not r:
            raise CameraConfigError("suitability_rule_ids entries must be non-empty strings")
        rule_ids.append(r)

    decode = dict(_require_mapping(raw["decode"], label="decode"))
    if decode.get("backend") != "opencv":
        raise CameraConfigError("decode.backend must be opencv")
    _require_int(decode["max_frames"], label="decode.max_frames", minimum=1, maximum=1_000_000)
    _require_float(
        decode["timeout_seconds"], label="decode.timeout_seconds", minimum=1.0, maximum=3600.0
    )

    evaluation = dict(_require_mapping(raw["evaluation"], label="evaluation"))
    for key in (
        "view_framing_macro_f1_min",
        "graphics_macro_f1_min",
        "motion_macro_f1_min",
        "playability_macro_f1_min",
        "unsafe_playable_fp_rate_max",
        "ood_abstention_rate_min",
    ):
        if key not in evaluation:
            raise CameraConfigError(f"evaluation.{key} required")
        _require_float(evaluation[key], label=f"evaluation.{key}", minimum=0.0, maximum=1.0)

    _require_int(raw["deterministic_seed"], label="deterministic_seed", minimum=0)

    runtime_root = raw["runtime_root"]
    if not isinstance(runtime_root, str) or not runtime_root.startswith("/"):
        raise CameraConfigError("runtime_root must be an absolute path string")

    if raw["overwrite_allowed"] is not False:
        raise CameraConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise CameraConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise CameraConfigError("network_sources_allowed must be false")

    limits = dict(_require_mapping(raw["resource_limits"], label="resource_limits"))
    _require_int(limits["max_frames"], label="resource_limits.max_frames", minimum=1)
    _require_float(limits["timeout_seconds"], label="resource_limits.timeout_seconds", minimum=1.0)
    _require_int(limits["max_shots"], label="resource_limits.max_shots", minimum=1)

    # Deep-copy nested threshold/weight maps with float coercion for fingerprint stability.
    def _num_map(block: Mapping[str, Any], *, label: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in block.items():
            out[str(k)] = _require_float(v, label=f"{label}.{k}")
        return out

    return {
        "config_version": CONFIG_VERSION,
        "sampling": {
            "samples_per_shot": samples,
            "edge_exclude_fraction": edge,
            "min_samples": min_s,
            "max_samples": max_s,
        },
        "analysis_width": width,
        "analysis_height": height,
        "feature_version": feature_version,
        "pitch_hsv": pitch_out,
        "skin_hsv": skin_out,
        "thresholds": {
            "view": _num_map(thresholds["view"], label="thresholds.view"),
            "framing": _num_map(thresholds["framing"], label="thresholds.framing"),
            "motion": _num_map(thresholds["motion"], label="thresholds.motion"),
            "graphics": _num_map(thresholds["graphics"], label="thresholds.graphics"),
            "aggregation": _num_map(thresholds["aggregation"], label="thresholds.aggregation"),
            "abstention": _num_map(thresholds["abstention"], label="thresholds.abstention"),
        },
        "weights": {
            "view": _num_map(weights["view"], label="weights.view"),
            "framing": _num_map(weights["framing"], label="weights.framing"),
            "motion": _num_map(weights["motion"], label="weights.motion"),
            "graphics": _num_map(weights["graphics"], label="weights.graphics"),
        },
        "optical_flow": {
            "enabled": bool(flow["enabled"]),
            "pyr_scale": float(flow["pyr_scale"]),
            "levels": int(flow["levels"]),
            "winsize": int(flow["winsize"]),
            "iterations": int(flow["iterations"]),
            "poly_n": int(flow["poly_n"]),
            "poly_sigma": float(flow["poly_sigma"]),
        },
        "min_coverage": min_coverage,
        "supported_axes": supported,
        "always_unknown_axes": sorted(ALWAYS_UNKNOWN),
        "suitability_rule_ids": list(rule_ids),
        "decode": {
            "backend": "opencv",
            "max_frames": int(decode["max_frames"]),
            "timeout_seconds": float(decode["timeout_seconds"]),
        },
        "evaluation": {
            "view_framing_macro_f1_min": float(evaluation["view_framing_macro_f1_min"]),
            "graphics_macro_f1_min": float(evaluation["graphics_macro_f1_min"]),
            "motion_macro_f1_min": float(evaluation["motion_macro_f1_min"]),
            "playability_macro_f1_min": float(evaluation["playability_macro_f1_min"]),
            "unsafe_playable_fp_rate_max": float(evaluation["unsafe_playable_fp_rate_max"]),
            "ood_abstention_rate_min": float(evaluation["ood_abstention_rate_min"]),
        },
        "deterministic_seed": int(raw["deterministic_seed"]),
        "runtime_root": str(runtime_root),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "resource_limits": {
            "max_frames": int(limits["max_frames"]),
            "timeout_seconds": float(limits["timeout_seconds"]),
            "max_shots": int(limits["max_shots"]),
        },
    }


def load_camera_view_config(path: Path | str) -> Mapping[str, Any]:
    """Load and strictly validate camera-view baseline YAML."""
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise CameraConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise CameraConfigError("config exceeds maximum byte size")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CameraConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise CameraConfigError("config root must be a mapping")
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


def camera_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 of canonical JSON config payload."""
    return hash_canonical_json(_deep_unfreeze(config))


def default_camera_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "broadcast" / "camera_view_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "SUPPORTED_VIEW",
    "SUPPORTED_FRAMING",
    "SUPPORTED_MOTION",
    "SUPPORTED_GRAPHICS",
    "SUPPORTED_PLAYABILITY",
    "ALWAYS_UNKNOWN",
    "CameraConfigError",
    "load_camera_view_config",
    "camera_config_fingerprint",
    "default_camera_config_path",
]
