"""Strict loader for Stage 6B human tracking baseline config."""

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
        "tracker_id",
        "tracker_version",
        "tracker_algorithm",
        "association_method",
        "association_version",
        "association",
        "lifecycle",
        "boundaries",
        "role",
        "entity_filter",
        "output_policy",
        "safety_limits",
        "runtime_root",
        "policy_path",
        "deterministic_seed",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "notes",
    }
)


class HumanTrackingConfigError(ValueError):
    """Human tracking baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanTrackingConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HumanTrackingConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise HumanTrackingConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HumanTrackingConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanTrackingConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise HumanTrackingConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise HumanTrackingConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise HumanTrackingConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise HumanTrackingConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HumanTrackingConfigError(f"{label} must be a non-empty string")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise HumanTrackingConfigError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise HumanTrackingConfigError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise HumanTrackingConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise HumanTrackingConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise HumanTrackingConfigError(f"config_version must be {CONFIG_VERSION}")

    algo = _require_str(raw["tracker_algorithm"], label="tracker_algorithm")
    if algo != "iou_constant_velocity_v1":
        raise HumanTrackingConfigError("unsupported tracker_algorithm")
    method = _require_str(raw["association_method"], label="association_method")
    if method != "iou_constant_velocity":
        raise HumanTrackingConfigError("unsupported association_method")

    assoc = dict(_require_mapping(raw["association"], label="association"))
    iou_gate = _require_float(
        assoc["iou_gate"], label="association.iou_gate", minimum=0.0, maximum=1.0
    )
    motion_gate = _require_float(
        assoc["motion_center_gate_px"],
        label="association.motion_center_gate_px",
        minimum=0.0,
    )
    iw = _require_float(assoc["cost_iou_weight"], label="association.cost_iou_weight", minimum=0.0)
    mw = _require_float(
        assoc["cost_motion_weight"], label="association.cost_motion_weight", minimum=0.0
    )
    if abs((iw + mw) - 1.0) > 1e-9:
        raise HumanTrackingConfigError("association cost weights must sum to 1.0")
    min_conf = _require_float(
        assoc["min_confidence"], label="association.min_confidence", minimum=0.0, maximum=1.0
    )
    tie = _require_str(assoc["tie_break"], label="association.tie_break")
    if tie != "track_id_then_detection_id":
        raise HumanTrackingConfigError("unsupported tie_break")

    life = dict(_require_mapping(raw["lifecycle"], label="lifecycle"))
    confirm = _require_int(
        life["confirmation_observation_threshold"],
        label="lifecycle.confirmation_observation_threshold",
        minimum=1,
    )
    max_lost = _require_int(life["max_lost_gap_us"], label="lifecycle.max_lost_gap_us", minimum=0)
    max_pred = _require_int(
        life["max_prediction_gap_us"], label="lifecycle.max_prediction_gap_us", minimum=0
    )
    emit_pred = _require_bool(
        life["emit_predicted_observations"], label="lifecycle.emit_predicted_observations"
    )
    if life.get("reopen_terminated") is not False:
        raise HumanTrackingConfigError("lifecycle.reopen_terminated must be false")
    tent_term = _require_int(
        life["tentative_miss_terminate_us"],
        label="lifecycle.tentative_miss_terminate_us",
        minimum=0,
    )

    bounds = dict(_require_mapping(raw["boundaries"], label="boundaries"))
    for key in (
        "terminate_on_shot_cut",
        "terminate_on_non_playable",
        "terminate_on_window_boundary",
        "terminate_on_ineligible_tracking",
        "no_cross_shot_continuation",
    ):
        _require_bool(bounds[key], label=f"boundaries.{key}")
    if bounds["no_cross_shot_continuation"] is not True:
        raise HumanTrackingConfigError("no_cross_shot_continuation must be true")

    role = dict(_require_mapping(raw["role"], label="role"))
    for key in ("soft_consistency", "unknown_not_punished", "conflict_requires_review"):
        _require_bool(role[key], label=f"role.{key}")
    if role["unknown_not_punished"] is not True:
        raise HumanTrackingConfigError("role.unknown_not_punished must be true")

    entity = dict(_require_mapping(raw["entity_filter"], label="entity_filter"))
    allowed = _require_str_list(entity["allowed_entity_types"], label="allowed_entity_types")
    if allowed != ["human"]:
        raise HumanTrackingConfigError("allowed_entity_types must be [human]")
    reject = _require_str_list(entity["reject_entity_types"], label="reject_entity_types")
    if "ball" not in reject:
        raise HumanTrackingConfigError("reject_entity_types must include ball")
    human_names = [
        s.lower() for s in _require_str_list(entity["human_class_names"], label="human_class_names")
    ]
    reject_names = [
        s.lower()
        for s in _require_str_list(entity["reject_class_names"], label="reject_class_names")
    ]

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if output.get("atomic_writes") is not True:
        raise HumanTrackingConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise HumanTrackingConfigError("output_policy.overwrite_allowed must be false")
    emit_pred_out = _require_bool(output["emit_predicted"], label="output_policy.emit_predicted")
    emit_interp = _require_bool(
        output["emit_interpolated"], label="output_policy.emit_interpolated"
    )
    if emit_interp is not False:
        raise HumanTrackingConfigError("emit_interpolated must be false in Stage 6B")

    limits = dict(_require_mapping(raw["safety_limits"], label="safety_limits"))
    max_tracks = _require_int(
        limits["max_tracks_per_video"], label="max_tracks_per_video", minimum=1
    )
    max_obs = _require_int(
        limits["max_observations_per_track"], label="max_observations_per_track", minimum=1
    )
    max_frames = _require_int(limits["max_frames_per_run"], label="max_frames_per_run", minimum=1)
    timeout = _require_float(limits["timeout_seconds"], label="timeout_seconds", minimum=1.0)

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/"):
        raise HumanTrackingConfigError("runtime_root must be absolute")
    if raw["overwrite_allowed"] is not False:
        raise HumanTrackingConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise HumanTrackingConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise HumanTrackingConfigError("network_sources_allowed must be false")
    notes = raw["notes"]
    if not isinstance(notes, list):
        raise HumanTrackingConfigError("notes must be a list")

    return {
        "config_version": CONFIG_VERSION,
        "tracker_id": _require_str(raw["tracker_id"], label="tracker_id"),
        "tracker_version": _require_str(raw["tracker_version"], label="tracker_version"),
        "tracker_algorithm": algo,
        "association_method": method,
        "association_version": _require_str(
            raw["association_version"], label="association_version"
        ),
        "association": {
            "iou_gate": iou_gate,
            "motion_center_gate_px": motion_gate,
            "cost_iou_weight": iw,
            "cost_motion_weight": mw,
            "min_confidence": min_conf,
            "tie_break": tie,
        },
        "lifecycle": {
            "confirmation_observation_threshold": confirm,
            "max_lost_gap_us": max_lost,
            "max_prediction_gap_us": max_pred,
            "emit_predicted_observations": emit_pred,
            "reopen_terminated": False,
            "tentative_miss_terminate_us": tent_term,
        },
        "boundaries": {
            "terminate_on_shot_cut": bool(bounds["terminate_on_shot_cut"]),
            "terminate_on_non_playable": bool(bounds["terminate_on_non_playable"]),
            "terminate_on_window_boundary": bool(bounds["terminate_on_window_boundary"]),
            "terminate_on_ineligible_tracking": bool(bounds["terminate_on_ineligible_tracking"]),
            "no_cross_shot_continuation": True,
        },
        "role": {
            "soft_consistency": bool(role["soft_consistency"]),
            "unknown_not_punished": True,
            "conflict_requires_review": bool(role["conflict_requires_review"]),
        },
        "entity_filter": {
            "allowed_entity_types": ["human"],
            "reject_entity_types": list(reject),
            "human_class_names": human_names,
            "reject_class_names": reject_names,
        },
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "emit_predicted": emit_pred_out,
            "emit_interpolated": False,
        },
        "safety_limits": {
            "max_tracks_per_video": max_tracks,
            "max_observations_per_track": max_obs,
            "max_frames_per_run": max_frames,
            "timeout_seconds": timeout,
        },
        "runtime_root": runtime_root,
        "policy_path": _require_str(raw["policy_path"], label="policy_path"),
        "deterministic_seed": _require_int(
            raw["deterministic_seed"], label="deterministic_seed", minimum=0
        ),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "notes": [str(n) for n in notes],
    }


def load_human_tracking_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise HumanTrackingConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise HumanTrackingConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HumanTrackingConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise HumanTrackingConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def human_tracking_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_human_tracking_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "tracking" / "human_tracking_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "HumanTrackingConfigError",
    "load_human_tracking_config",
    "human_tracking_config_fingerprint",
    "default_human_tracking_config_path",
]
