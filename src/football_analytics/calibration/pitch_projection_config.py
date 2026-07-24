"""Strict loader for Stage 8D pitch projection pipeline config."""

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
        "pipeline_id",
        "pipeline_version",
        "direction",
        "human_source",
        "ball_source",
        "projection",
        "uncertainty",
        "segment_selection",
        "eligibility",
        "pitch",
        "attack_direction",
        "output_policy",
        "review_sampling",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "compute_physical_metrics",
        "compute_events",
        "notes",
    }
)


class PitchProjectionConfigError(ValueError):
    """Pitch projection pipeline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PitchProjectionConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PitchProjectionConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PitchProjectionConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise PitchProjectionConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool):
        raise PitchProjectionConfigError(f"{label} must be a number")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise PitchProjectionConfigError(f"{label} must be a number") from exc
    if not isinstance(value, (int, float)):
        raise PitchProjectionConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise PitchProjectionConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise PitchProjectionConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise PitchProjectionConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PitchProjectionConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PitchProjectionConfigError(f"{label} must be a non-empty string")
    return value


def _require_abs_path(value: Any, *, label: str) -> str:
    s = _require_str(value, label=label)
    p = Path(s)
    if not p.is_absolute():
        raise PitchProjectionConfigError(f"{label} must be absolute")
    return str(p)


def _freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def default_pitch_projection_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "calibration" / "pitch_projection_pipeline.yaml"


def load_pitch_projection_config(path: Path | str) -> Mapping[str, Any]:
    p = Path(path)
    if p.is_symlink():
        raise PitchProjectionConfigError(f"symlink rejected: {p}")
    if not p.is_file():
        raise PitchProjectionConfigError(f"config missing: {p}")
    size = p.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise PitchProjectionConfigError("config too large")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PitchProjectionConfigError("config root must be a mapping")
    missing = REQUIRED_TOP - set(raw)
    if missing:
        raise PitchProjectionConfigError(f"missing keys: {sorted(missing)}")

    version = _require_int(raw["config_version"], label="config_version", minimum=1)
    if version != CONFIG_VERSION:
        raise PitchProjectionConfigError(f"unsupported config_version: {version}")

    direction = _require_str(raw["direction"], label="direction")
    if direction != "image_to_pitch":
        raise PitchProjectionConfigError("direction must be image_to_pitch")
    attack = _require_str(raw["attack_direction"], label="attack_direction")
    if attack != "unknown":
        raise PitchProjectionConfigError("attack_direction must remain unknown in Stage 8D")

    human = dict(_require_mapping(raw["human_source"], label="human_source"))
    human["point_type"] = _require_str(human["point_type"], label="human_source.point_type")
    if human["point_type"] != "bbox_bottom_centre":
        raise PitchProjectionConfigError("human_source.point_type must be bbox_bottom_centre")
    human["require_observed"] = _require_bool(
        human["require_observed"], label="human_source.require_observed"
    )
    for key in (
        "min_bbox_width_px",
        "min_bbox_height_px",
        "max_aspect_ratio",
        "frame_edge_margin_px",
    ):
        human[key] = _require_float(human[key], label=f"human_source.{key}", minimum=0.0)
    human["truncated_uncertainty_boost_m"] = _require_float(
        human["truncated_uncertainty_boost_m"],
        label="human_source.truncated_uncertainty_boost_m",
        minimum=0.0,
    )
    human["pose_foot_model"] = _require_bool(
        human["pose_foot_model"], label="human_source.pose_foot_model"
    )
    if human["pose_foot_model"] is True:
        raise PitchProjectionConfigError("human_source.pose_foot_model must be false in Stage 8D")

    ball = dict(_require_mapping(raw["ball_source"], label="ball_source"))
    ball["point_type"] = _require_str(ball["point_type"], label="ball_source.point_type")
    if ball["point_type"] != "bbox_centre":
        raise PitchProjectionConfigError("ball_source.point_type must be bbox_centre")
    ball["require_observed"] = _require_bool(
        ball["require_observed"], label="ball_source.require_observed"
    )
    for key in (
        "min_bbox_width_px",
        "min_bbox_height_px",
        "max_aspect_ratio",
        "frame_edge_margin_px",
    ):
        ball[key] = _require_float(ball[key], label=f"ball_source.{key}", minimum=0.0)
    ball["ambiguous_primary_review"] = _require_bool(
        ball["ambiguous_primary_review"], label="ball_source.ambiguous_primary_review"
    )
    ball["airborne_status"] = _require_str(
        ball["airborne_status"], label="ball_source.airborne_status"
    )
    if ball["airborne_status"] != "unknown":
        raise PitchProjectionConfigError("ball_source.airborne_status must be unknown in Stage 8D")
    ball["physical_metric_eligible"] = _require_bool(
        ball["physical_metric_eligible"], label="ball_source.physical_metric_eligible"
    )
    ball["event_metric_eligible"] = _require_bool(
        ball["event_metric_eligible"], label="ball_source.event_metric_eligible"
    )
    if ball["physical_metric_eligible"] is True or ball["event_metric_eligible"] is True:
        raise PitchProjectionConfigError(
            "ball physical/event metric eligibility must be false in Stage 8D"
        )

    proj = dict(_require_mapping(raw["projection"], label="projection"))
    for key in (
        "homogeneous_w_epsilon",
        "pitch_bound_tolerance_m",
        "round_trip_tolerance_px",
        "round_trip_tolerance_m",
    ):
        proj[key] = _require_float(proj[key], label=f"projection.{key}", minimum=0.0)
    for key in (
        "outside_coverage_is_extrapolated",
        "clamp_pitch_coordinates",
        "use_image_to_pitch_only",
        "forbid_h_inv_for_projection",
    ):
        proj[key] = _require_bool(proj[key], label=f"projection.{key}")
    if proj["clamp_pitch_coordinates"] is True:
        raise PitchProjectionConfigError("projection.clamp_pitch_coordinates must be false")
    if proj["use_image_to_pitch_only"] is not True:
        raise PitchProjectionConfigError("projection.use_image_to_pitch_only must be true")
    if proj["forbid_h_inv_for_projection"] is not True:
        raise PitchProjectionConfigError("projection.forbid_h_inv_for_projection must be true")

    unc = dict(_require_mapping(raw["uncertainty"], label="uncertainty"))
    for key in (
        "max_uncertainty_m",
        "base_from_reprojection_scale",
        "coverage_distance_scale_m",
        "truncation_boost_m",
        "ambiguity_boost_m",
    ):
        unc[key] = _require_float(unc[key], label=f"uncertainty.{key}", minimum=0.0)
    unc["unknown_null_reason"] = _require_str(
        unc["unknown_null_reason"], label="uncertainty.unknown_null_reason"
    )

    segs = dict(_require_mapping(raw["segment_selection"], label="segment_selection"))
    segs["require_validity_status"] = _require_str(
        segs["require_validity_status"], label="segment_selection.require_validity_status"
    )
    if segs["require_validity_status"] != "valid":
        raise PitchProjectionConfigError("segment_selection.require_validity_status must be valid")
    for key in (
        "require_physical_metric_eligible",
        "allow_interpolated",
        "overlap_is_hard_conflict",
        "degraded_uncertain_not_physical",
        "verify_inverse_round_trip",
    ):
        segs[key] = _require_bool(segs[key], label=f"segment_selection.{key}")
    if segs["allow_interpolated"] is True:
        raise PitchProjectionConfigError("segment_selection.allow_interpolated must be false")
    if segs["require_physical_metric_eligible"] is not True:
        raise PitchProjectionConfigError(
            "segment_selection.require_physical_metric_eligible must be true"
        )

    elig = dict(_require_mapping(raw["eligibility"], label="eligibility"))
    for key in (
        "predicted_observation_physical_metric_eligible",
        "interpolated_observation_physical_metric_eligible",
        "extrapolated_physical_metric_eligible",
        "ball_physical_metric_eligible",
        "ball_event_metric_eligible",
        "require_playable_window",
        "require_confirmed_target_for_customer_metric",
        "provisional_target_customer_metric_eligible",
        "revoked_target_customer_metric_eligible",
    ):
        elig[key] = _require_bool(elig[key], label=f"eligibility.{key}")
    for key in (
        "predicted_observation_physical_metric_eligible",
        "interpolated_observation_physical_metric_eligible",
        "extrapolated_physical_metric_eligible",
        "ball_physical_metric_eligible",
        "ball_event_metric_eligible",
        "provisional_target_customer_metric_eligible",
        "revoked_target_customer_metric_eligible",
    ):
        if elig[key] is True:
            raise PitchProjectionConfigError(f"eligibility.{key} must be false in Stage 8D")

    pitch = dict(_require_mapping(raw["pitch"], label="pitch"))
    pitch["length_m"] = _require_float(pitch["length_m"], label="pitch.length_m", minimum=1.0)
    pitch["width_m"] = _require_float(pitch["width_m"], label="pitch.width_m", minimum=1.0)
    pitch["real_size_known"] = _require_bool(
        pitch["real_size_known"], label="pitch.real_size_known"
    )

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    output["atomic_writes"] = _require_bool(output["atomic_writes"], label="atomic_writes")
    if output["atomic_writes"] is not True:
        raise PitchProjectionConfigError("output_policy.atomic_writes must be true")
    output["overwrite_allowed"] = _require_bool(
        output["overwrite_allowed"], label="output_policy.overwrite_allowed"
    )
    if output["overwrite_allowed"] is True:
        raise PitchProjectionConfigError("output_policy.overwrite_allowed must be false")
    for key in (
        "write_projected_positions",
        "write_evaluation_json",
        "write_quality_json",
        "write_review_queue",
    ):
        output[key] = _require_bool(output[key], label=f"output_policy.{key}")
    if output["write_projected_positions"] is not True:
        raise PitchProjectionConfigError("write_projected_positions must be true in Stage 8D")

    review = dict(_require_mapping(raw["review_sampling"], label="review_sampling"))
    review["enabled"] = _require_bool(review["enabled"], label="review_sampling.enabled")
    review["max_samples"] = _require_int(
        review["max_samples"], label="review_sampling.max_samples", minimum=0
    )

    if raw["overwrite_allowed"] is not False:
        raise PitchProjectionConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise PitchProjectionConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise PitchProjectionConfigError("network_sources_allowed must be false")
    if raw["compute_physical_metrics"] is not False:
        raise PitchProjectionConfigError("compute_physical_metrics must be false")
    if raw["compute_events"] is not False:
        raise PitchProjectionConfigError("compute_events must be false")

    out: dict[str, Any] = {
        "config_version": version,
        "pipeline_id": _require_str(raw["pipeline_id"], label="pipeline_id"),
        "pipeline_version": _require_str(raw["pipeline_version"], label="pipeline_version"),
        "direction": direction,
        "human_source": human,
        "ball_source": ball,
        "projection": proj,
        "uncertainty": unc,
        "segment_selection": segs,
        "eligibility": elig,
        "pitch": pitch,
        "attack_direction": attack,
        "output_policy": output,
        "review_sampling": review,
        "runtime_root": _require_abs_path(raw["runtime_root"], label="runtime_root"),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "compute_physical_metrics": False,
        "compute_events": False,
        "notes": _require_str(raw["notes"], label="notes"),
    }
    return _freeze(out)


def unfreeze_pitch_projection_config(config: Mapping[str, Any]) -> dict[str, Any]:
    def _u(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: _u(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return [_u(v) for v in obj]
        return obj

    return _u(config)


def pitch_projection_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(unfreeze_pitch_projection_config(config))


__all__ = [
    "CONFIG_VERSION",
    "PitchProjectionConfigError",
    "default_pitch_projection_config_path",
    "load_pitch_projection_config",
    "unfreeze_pitch_projection_config",
    "pitch_projection_config_fingerprint",
]
