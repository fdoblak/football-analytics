"""Strict loader for Stage 6C ball tracking baseline config."""

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
        "primary_candidate",
        "lifecycle",
        "boundaries",
        "role",
        "entity_filter",
        "prediction",
        "output_policy",
        "review",
        "frame_geometry",
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


class BallTrackingConfigError(ValueError):
    """Ball tracking baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BallTrackingConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BallTrackingConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise BallTrackingConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise BallTrackingConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BallTrackingConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise BallTrackingConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise BallTrackingConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise BallTrackingConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BallTrackingConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BallTrackingConfigError(f"{label} must be a non-empty string")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BallTrackingConfigError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise BallTrackingConfigError(f"{label} entries must be non-empty strings")
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
        raise BallTrackingConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise BallTrackingConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise BallTrackingConfigError(f"config_version must be {CONFIG_VERSION}")

    algo = _require_str(raw["tracker_algorithm"], label="tracker_algorithm")
    if algo != "motion_constant_velocity_v1":
        raise BallTrackingConfigError("unsupported tracker_algorithm")
    method = _require_str(raw["association_method"], label="association_method")
    if method != "motion_first_constant_velocity":
        raise BallTrackingConfigError("unsupported association_method")

    assoc = dict(_require_mapping(raw["association"], label="association"))
    motion_gate = _require_float(
        assoc["motion_center_gate_px"],
        label="association.motion_center_gate_px",
        minimum=0.0,
    )
    motion_scale = _require_float(
        assoc["motion_gate_scale_per_us"],
        label="association.motion_gate_scale_per_us",
        minimum=0.0,
    )
    size_gate = _require_float(
        assoc["size_ratio_gate"], label="association.size_ratio_gate", minimum=1.0
    )
    iou_support = _require_float(
        assoc["iou_support_min"],
        label="association.iou_support_min",
        minimum=0.0,
        maximum=1.0,
    )
    w_m = _require_float(
        assoc["cost_motion_weight"], label="association.cost_motion_weight", minimum=0.0
    )
    w_s = _require_float(
        assoc["cost_size_weight"], label="association.cost_size_weight", minimum=0.0
    )
    w_c = _require_float(
        assoc["cost_confidence_weight"],
        label="association.cost_confidence_weight",
        minimum=0.0,
    )
    w_i = _require_float(assoc["cost_iou_weight"], label="association.cost_iou_weight", minimum=0.0)
    if abs((w_m + w_s + w_c + w_i) - 1.0) > 1e-9:
        raise BallTrackingConfigError("association cost weights must sum to 1.0")
    if w_m < w_i:
        raise BallTrackingConfigError("motion weight must be >= IoU weight (motion-first)")
    min_conf = _require_float(
        assoc["min_confidence"], label="association.min_confidence", minimum=0.0, maximum=1.0
    )
    require_motion = _require_bool(assoc["require_motion_gate"], label="require_motion_gate")
    if require_motion is not True:
        raise BallTrackingConfigError("require_motion_gate must be true")
    tie = _require_str(assoc["tie_break"], label="association.tie_break")
    if tie != "track_id_then_detection_id":
        raise BallTrackingConfigError("unsupported tie_break")

    primary = dict(_require_mapping(raw["primary_candidate"], label="primary_candidate"))
    max_primary = _require_int(
        primary["max_primary_per_frame"], label="max_primary_per_frame", minimum=1, maximum=1
    )
    score_margin = _require_float(
        primary["score_margin"], label="score_margin", minimum=0.0, maximum=1.0
    )
    amb_margin = _require_float(
        primary["ambiguity_margin"], label="ambiguity_margin", minimum=0.0, maximum=1.0
    )
    prefer_confirmed = _require_bool(primary["prefer_confirmed"], label="prefer_confirmed")

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
        raise BallTrackingConfigError("lifecycle.reopen_terminated must be false")
    tent_term = _require_int(
        life["tentative_miss_terminate_us"],
        label="lifecycle.tentative_miss_terminate_us",
        minimum=0,
    )
    weak_term = _require_int(
        life["weak_candidate_terminate_us"],
        label="lifecycle.weak_candidate_terminate_us",
        minimum=0,
    )

    bounds = dict(_require_mapping(raw["boundaries"], label="boundaries"))
    for key in (
        "terminate_on_shot_cut",
        "terminate_on_non_playable",
        "terminate_on_window_boundary",
        "terminate_on_ineligible_tracking",
        "no_cross_shot_continuation",
        "no_cross_cut_prediction",
    ):
        _require_bool(bounds[key], label=f"boundaries.{key}")
    if bounds["no_cross_shot_continuation"] is not True:
        raise BallTrackingConfigError("no_cross_shot_continuation must be true")
    if bounds["no_cross_cut_prediction"] is not True:
        raise BallTrackingConfigError("no_cross_cut_prediction must be true")

    role = dict(_require_mapping(raw["role"], label="role"))
    if role.get("always_unknown") is not True:
        raise BallTrackingConfigError("role.always_unknown must be true")

    entity = dict(_require_mapping(raw["entity_filter"], label="entity_filter"))
    allowed = _require_str_list(entity["allowed_entity_types"], label="allowed_entity_types")
    if allowed != ["ball"]:
        raise BallTrackingConfigError("allowed_entity_types must be [ball]")
    reject = _require_str_list(entity["reject_entity_types"], label="reject_entity_types")
    if "human" not in reject:
        raise BallTrackingConfigError("reject_entity_types must include human")
    ball_names = [
        s.lower() for s in _require_str_list(entity["ball_class_names"], label="ball_class_names")
    ]
    reject_names = [
        s.lower()
        for s in _require_str_list(entity["reject_class_names"], label="reject_class_names")
    ]

    pred = dict(_require_mapping(raw["prediction"], label="prediction"))
    if pred.get("physical_metric_eligible") is not False:
        raise BallTrackingConfigError("prediction.physical_metric_eligible must be false")
    if pred.get("event_eligible") is not False:
        raise BallTrackingConfigError("prediction.event_eligible must be false")
    unc = _require_bool(pred["uncertainty_grows_with_gap"], label="uncertainty_grows_with_gap")

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if output.get("atomic_writes") is not True:
        raise BallTrackingConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise BallTrackingConfigError("output_policy.overwrite_allowed must be false")
    emit_pred_out = _require_bool(output["emit_predicted"], label="output_policy.emit_predicted")
    emit_interp = _require_bool(
        output["emit_interpolated"], label="output_policy.emit_interpolated"
    )
    if emit_interp is not False:
        raise BallTrackingConfigError("emit_interpolated must be false in Stage 6C")
    emit_sidecar = _require_bool(
        output["emit_primary_sidecar"], label="output_policy.emit_primary_sidecar"
    )

    review = dict(_require_mapping(raw["review"], label="review"))
    for key in ("sample_ambiguous", "sample_invalid_jump", "no_spam_empty_frames"):
        _require_bool(review[key], label=f"review.{key}")
    max_rev = _require_int(
        review["max_review_samples_per_run"], label="max_review_samples_per_run", minimum=0
    )

    geom = dict(_require_mapping(raw["frame_geometry"], label="frame_geometry"))
    fw = _require_int(geom["frame_width"], label="frame_width", minimum=1)
    fh = _require_int(geom["frame_height"], label="frame_height", minimum=1)
    edge = _require_float(geom["edge_margin_px"], label="edge_margin_px", minimum=0.0)
    inv_jump = _require_float(geom["invalid_jump_px"], label="invalid_jump_px", minimum=0.0)

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
        raise BallTrackingConfigError("runtime_root must be absolute")
    if raw["overwrite_allowed"] is not False:
        raise BallTrackingConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise BallTrackingConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise BallTrackingConfigError("network_sources_allowed must be false")
    notes = raw["notes"]
    if not isinstance(notes, list):
        raise BallTrackingConfigError("notes must be a list")

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
            "motion_center_gate_px": motion_gate,
            "motion_gate_scale_per_us": motion_scale,
            "size_ratio_gate": size_gate,
            "iou_support_min": iou_support,
            "cost_motion_weight": w_m,
            "cost_size_weight": w_s,
            "cost_confidence_weight": w_c,
            "cost_iou_weight": w_i,
            "min_confidence": min_conf,
            "require_motion_gate": True,
            "tie_break": tie,
        },
        "primary_candidate": {
            "max_primary_per_frame": max_primary,
            "score_margin": score_margin,
            "ambiguity_margin": amb_margin,
            "prefer_confirmed": prefer_confirmed,
        },
        "lifecycle": {
            "confirmation_observation_threshold": confirm,
            "max_lost_gap_us": max_lost,
            "max_prediction_gap_us": max_pred,
            "emit_predicted_observations": emit_pred,
            "reopen_terminated": False,
            "tentative_miss_terminate_us": tent_term,
            "weak_candidate_terminate_us": weak_term,
        },
        "boundaries": {
            "terminate_on_shot_cut": bool(bounds["terminate_on_shot_cut"]),
            "terminate_on_non_playable": bool(bounds["terminate_on_non_playable"]),
            "terminate_on_window_boundary": bool(bounds["terminate_on_window_boundary"]),
            "terminate_on_ineligible_tracking": bool(bounds["terminate_on_ineligible_tracking"]),
            "no_cross_shot_continuation": True,
            "no_cross_cut_prediction": True,
        },
        "role": {"always_unknown": True},
        "entity_filter": {
            "allowed_entity_types": ["ball"],
            "reject_entity_types": list(reject),
            "ball_class_names": ball_names,
            "reject_class_names": reject_names,
        },
        "prediction": {
            "physical_metric_eligible": False,
            "event_eligible": False,
            "uncertainty_grows_with_gap": unc,
        },
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "emit_predicted": emit_pred_out,
            "emit_interpolated": False,
            "emit_primary_sidecar": emit_sidecar,
        },
        "review": {
            "sample_ambiguous": bool(review["sample_ambiguous"]),
            "sample_invalid_jump": bool(review["sample_invalid_jump"]),
            "max_review_samples_per_run": max_rev,
            "no_spam_empty_frames": bool(review["no_spam_empty_frames"]),
        },
        "frame_geometry": {
            "frame_width": fw,
            "frame_height": fh,
            "edge_margin_px": edge,
            "invalid_jump_px": inv_jump,
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


def load_ball_tracking_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise BallTrackingConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise BallTrackingConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise BallTrackingConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise BallTrackingConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def ball_tracking_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_ball_tracking_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "tracking" / "ball_tracking_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "BallTrackingConfigError",
    "load_ball_tracking_config",
    "ball_tracking_config_fingerprint",
    "default_ball_tracking_config_path",
]
