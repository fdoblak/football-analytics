"""Strict loader for Stage 8C homography baseline config."""

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
        "solver_id",
        "solver_version",
        "method",
        "direction",
        "correspondence",
        "solver",
        "quality",
        "segments",
        "pitch",
        "attack_direction",
        "output_policy",
        "review_sampling",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "auto_project_positions",
        "notes",
    }
)


class HomographyConfigError(ValueError):
    """Homography baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HomographyConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HomographyConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise HomographyConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HomographyConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool):
        raise HomographyConfigError(f"{label} must be a number")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise HomographyConfigError(f"{label} must be a number") from exc
    if not isinstance(value, (int, float)):
        raise HomographyConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise HomographyConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise HomographyConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise HomographyConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise HomographyConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HomographyConfigError(f"{label} must be a non-empty string")
    return value


def _require_abs_path(value: Any, *, label: str) -> str:
    s = _require_str(value, label=label)
    p = Path(s)
    if not p.is_absolute():
        raise HomographyConfigError(f"{label} must be absolute")
    return str(p)


def _freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def default_homography_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "calibration" / "homography_baseline.yaml"


def load_homography_config(path: Path | str) -> Mapping[str, Any]:
    p = Path(path)
    if p.is_symlink():
        raise HomographyConfigError(f"symlink rejected: {p}")
    if not p.is_file():
        raise HomographyConfigError(f"config missing: {p}")
    size = p.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise HomographyConfigError("config too large")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise HomographyConfigError("config root must be a mapping")
    missing = REQUIRED_TOP - set(raw)
    if missing:
        raise HomographyConfigError(f"missing keys: {sorted(missing)}")

    version = _require_int(raw["config_version"], label="config_version", minimum=1)
    if version != CONFIG_VERSION:
        raise HomographyConfigError(f"unsupported config_version: {version}")

    method = _require_str(raw["method"], label="method")
    if method not in {"dlt_normalized", "ransac_opencv"}:
        raise HomographyConfigError(f"unsupported method: {method}")
    direction = _require_str(raw["direction"], label="direction")
    if direction != "image_to_pitch":
        raise HomographyConfigError("direction must be image_to_pitch")
    attack = _require_str(raw["attack_direction"], label="attack_direction")
    if attack != "unknown":
        raise HomographyConfigError("attack_direction must remain unknown in Stage 8C")

    corr = dict(_require_mapping(raw["correspondence"], label="correspondence"))
    for key in (
        "min_keypoint_score",
        "min_line_score",
        "min_intersection_score",
        "duplicate_image_distance_px",
        "duplicate_pitch_distance_m",
    ):
        corr[key] = _require_float(corr[key], label=f"correspondence.{key}", minimum=0.0)
    corr["reject_unknown_canonical"] = _require_bool(
        corr["reject_unknown_canonical"], label="correspondence.reject_unknown_canonical"
    )
    corr["reject_unsuitable"] = _require_bool(
        corr["reject_unsuitable"], label="correspondence.reject_unsuitable"
    )
    corr["allow_marginal"] = _require_bool(
        corr["allow_marginal"], label="correspondence.allow_marginal"
    )
    corr["max_correspondences_per_frame"] = _require_int(
        corr["max_correspondences_per_frame"],
        label="correspondence.max_correspondences_per_frame",
        minimum=4,
    )
    corr["min_feature_diversity"] = _require_int(
        corr["min_feature_diversity"], label="correspondence.min_feature_diversity", minimum=1
    )
    li = dict(_require_mapping(corr["line_intersection"], label="correspondence.line_intersection"))
    li["enabled"] = _require_bool(li["enabled"], label="line_intersection.enabled")
    li["min_abs_sin_angle"] = _require_float(
        li["min_abs_sin_angle"],
        label="line_intersection.min_abs_sin_angle",
        minimum=0.0,
        maximum=1.0,
    )
    li["min_segment_length_px"] = _require_float(
        li["min_segment_length_px"], label="line_intersection.min_segment_length_px", minimum=0.0
    )
    li["max_endpoint_gap_px"] = _require_float(
        li["max_endpoint_gap_px"], label="line_intersection.max_endpoint_gap_px", minimum=0.0
    )
    li["require_both_mapped"] = _require_bool(
        li["require_both_mapped"], label="line_intersection.require_both_mapped"
    )
    corr["line_intersection"] = li

    solver = dict(_require_mapping(raw["solver"], label="solver"))
    solver["min_correspondences"] = _require_int(
        solver["min_correspondences"], label="solver.min_correspondences", minimum=4
    )
    for key in (
        "ransac_reproj_threshold_px",
        "ransac_confidence",
        "max_condition_number",
        "min_abs_determinant",
        "max_mean_reprojection_error_px",
        "max_median_reprojection_error_px",
        "max_max_reprojection_error_px",
        "max_mean_pitch_error_m",
        "round_trip_tolerance_px",
        "round_trip_tolerance_m",
        "min_inlier_ratio",
        "min_coverage_hull_fraction",
        "pitch_bound_tolerance_m",
        "max_extrapolation_ratio",
    ):
        solver[key] = _require_float(solver[key], label=f"solver.{key}", minimum=0.0)
    solver["ransac_max_iters"] = _require_int(
        solver["ransac_max_iters"], label="solver.ransac_max_iters", minimum=1
    )
    solver["ransac_seed"] = _require_int(
        solver["ransac_seed"], label="solver.ransac_seed", minimum=0
    )
    solver["min_inlier_count"] = _require_int(
        solver["min_inlier_count"], label="solver.min_inlier_count", minimum=4
    )
    solver["use_normalized_dlt"] = _require_bool(
        solver["use_normalized_dlt"], label="solver.use_normalized_dlt"
    )
    solver["reject_mirrored"] = _require_bool(
        solver["reject_mirrored"], label="solver.reject_mirrored"
    )

    quality = dict(_require_mapping(raw["quality"], label="quality"))
    pelig = quality.get("physical_eligible_qualities")
    if not isinstance(pelig, list) or not all(isinstance(x, str) for x in pelig):
        raise HomographyConfigError("quality.physical_eligible_qualities must be list[str]")
    quality["physical_eligible_qualities"] = list(pelig)
    for key in (
        "degraded_physical_eligible",
        "uncertain_physical_eligible",
        "interpolated_physical_eligible",
    ):
        quality[key] = _require_bool(quality[key], label=f"quality.{key}")
        if quality[key] is True:
            raise HomographyConfigError(f"quality.{key} must be false in Stage 8C policy")
    for band in ("valid", "degraded", "uncertain"):
        b = dict(_require_mapping(quality[band], label=f"quality.{band}"))
        for k in ("max_mean_reproj_px", "min_inlier_ratio", "min_coverage", "max_condition"):
            b[k] = _require_float(b[k], label=f"quality.{band}.{k}", minimum=0.0)
        quality[band] = b

    segments = dict(_require_mapping(raw["segments"], label="segments"))
    if segments.get("silent_gap_fill") is True:
        raise HomographyConfigError("segments.silent_gap_fill must be false")
    segments["silent_gap_fill"] = False
    segments["min_support_frames"] = _require_int(
        segments["min_support_frames"], label="segments.min_support_frames", minimum=1
    )
    segments["min_duration_us"] = _require_int(
        segments["min_duration_us"], label="segments.min_duration_us", minimum=1
    )
    for key in ("drift_mean_test_point_m", "drift_max_test_point_m"):
        segments[key] = _require_float(segments[key], label=f"segments.{key}", minimum=0.0)
    for key in (
        "terminate_on_shot_cut",
        "new_segment_on_pan_zoom",
        "allow_interpolated",
        "interpolation_physical_eligible",
        "overlap_is_hard_conflict",
        "medoid_by_pitch_test_points",
    ):
        segments[key] = _require_bool(segments[key], label=f"segments.{key}")
    if segments["interpolation_physical_eligible"] is True:
        raise HomographyConfigError("segments.interpolation_physical_eligible must be false")
    segments["representative_selection"] = _require_str(
        segments["representative_selection"], label="segments.representative_selection"
    )
    if segments["representative_selection"] not in {"medoid", "best_supported"}:
        raise HomographyConfigError("segments.representative_selection unsupported")
    segments["max_gap_us_within_segment"] = _require_int(
        segments["max_gap_us_within_segment"],
        label="segments.max_gap_us_within_segment",
        minimum=0,
    )

    pitch = dict(_require_mapping(raw["pitch"], label="pitch"))
    pitch["length_m"] = _require_float(pitch["length_m"], label="pitch.length_m", minimum=1.0)
    pitch["width_m"] = _require_float(pitch["width_m"], label="pitch.width_m", minimum=1.0)
    pitch["real_size_known"] = _require_bool(
        pitch["real_size_known"], label="pitch.real_size_known"
    )

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    output["atomic_writes"] = _require_bool(output["atomic_writes"], label="atomic_writes")
    if output["atomic_writes"] is not True:
        raise HomographyConfigError("output_policy.atomic_writes must be true")
    output["overwrite_allowed"] = _require_bool(
        output["overwrite_allowed"], label="output_policy.overwrite_allowed"
    )
    if output["overwrite_allowed"] is True:
        raise HomographyConfigError("output_policy.overwrite_allowed must be false")
    for key in (
        "write_calibrations",
        "write_segments",
        "write_evaluation_json",
        "write_quality_json",
        "write_projected_positions",
    ):
        output[key] = _require_bool(output[key], label=f"output_policy.{key}")
    if output["write_projected_positions"] is True:
        raise HomographyConfigError("write_projected_positions must be false in Stage 8C")

    review = dict(_require_mapping(raw["review_sampling"], label="review_sampling"))
    review["enabled"] = _require_bool(review["enabled"], label="review_sampling.enabled")
    review["max_samples"] = _require_int(
        review["max_samples"], label="review_sampling.max_samples", minimum=0
    )

    if raw["overwrite_allowed"] is not False:
        raise HomographyConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise HomographyConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise HomographyConfigError("network_sources_allowed must be false")
    if raw["auto_project_positions"] is not False:
        raise HomographyConfigError("auto_project_positions must be false")

    out: dict[str, Any] = {
        "config_version": version,
        "solver_id": _require_str(raw["solver_id"], label="solver_id"),
        "solver_version": _require_str(raw["solver_version"], label="solver_version"),
        "method": method,
        "direction": direction,
        "correspondence": corr,
        "solver": solver,
        "quality": quality,
        "segments": segments,
        "pitch": pitch,
        "attack_direction": attack,
        "output_policy": output,
        "review_sampling": review,
        "runtime_root": _require_abs_path(raw["runtime_root"], label="runtime_root"),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "auto_project_positions": False,
        "notes": _require_str(raw["notes"], label="notes"),
    }
    return _freeze(out)


def unfreeze_homography_config(config: Mapping[str, Any]) -> dict[str, Any]:
    def _u(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: _u(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return [_u(v) for v in obj]
        return obj

    return _u(config)


def homography_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(unfreeze_homography_config(config))


__all__ = [
    "CONFIG_VERSION",
    "HomographyConfigError",
    "default_homography_config_path",
    "load_homography_config",
    "unfreeze_homography_config",
    "homography_config_fingerprint",
]
