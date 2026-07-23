"""Strict loader for Stage 7B appearance ReID baseline config."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.identity.appearance_descriptor import expected_embedding_dim

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

REQUIRED_TOP = frozenset(
    {
        "config_version",
        "extractor_id",
        "extractor_version",
        "extractor_type",
        "descriptor",
        "sampling",
        "aggregation",
        "matching",
        "review",
        "output_policy",
        "safety_limits",
        "runtime_root",
        "identity_policy_path",
        "deterministic_seed",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "selection_matrix",
        "notes",
    }
)


class AppearanceReidConfigError(ValueError):
    """Appearance ReID baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AppearanceReidConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppearanceReidConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise AppearanceReidConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise AppearanceReidConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AppearanceReidConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise AppearanceReidConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise AppearanceReidConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise AppearanceReidConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise AppearanceReidConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AppearanceReidConfigError(f"{label} must be a non-empty string")
    return value


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
        raise AppearanceReidConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise AppearanceReidConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise AppearanceReidConfigError(f"config_version must be {CONFIG_VERSION}")

    extractor_type = _require_str(raw["extractor_type"], label="extractor_type")
    if extractor_type != "handcrafted":
        raise AppearanceReidConfigError("Stage 7B extractor_type must be handcrafted")

    desc = dict(_require_mapping(raw["descriptor"], label="descriptor"))
    for key in (
        "hsv_h_bins",
        "hsv_s_bins",
        "hsv_v_bins",
        "lab_l_bins",
        "lab_a_bins",
        "lab_b_bins",
        "edge_bins",
        "texture_bins",
        "embedding_dim",
    ):
        desc[key] = _require_int(desc[key], label=f"descriptor.{key}", minimum=1)
    desc["upper_body_fraction"] = _require_float(
        desc["upper_body_fraction"],
        label="descriptor.upper_body_fraction",
        minimum=0.1,
        maximum=0.9,
    )
    expected_embedding_dim(desc)

    sampling = dict(_require_mapping(raw["sampling"], label="sampling"))
    sampling["max_samples_per_track"] = _require_int(
        sampling["max_samples_per_track"], label="max_samples_per_track", minimum=1
    )
    sampling["min_samples_for_profile"] = _require_int(
        sampling["min_samples_for_profile"], label="min_samples_for_profile", minimum=1
    )
    sampling["min_bbox_area_px"] = _require_float(
        sampling["min_bbox_area_px"], label="min_bbox_area_px", minimum=1.0
    )
    sampling["min_crop_side_px"] = _require_int(
        sampling["min_crop_side_px"], label="min_crop_side_px", minimum=1
    )
    sampling["min_crop_quality"] = _require_float(
        sampling["min_crop_quality"], label="min_crop_quality", minimum=0.0, maximum=1.0
    )
    sampling["temporal_stride_frames"] = _require_int(
        sampling["temporal_stride_frames"], label="temporal_stride_frames", minimum=1
    )
    sampling["max_samples_per_temporal_bin"] = _require_int(
        sampling["max_samples_per_temporal_bin"],
        label="max_samples_per_temporal_bin",
        minimum=1,
    )
    sampling["temporal_bin_count"] = _require_int(
        sampling["temporal_bin_count"], label="temporal_bin_count", minimum=1
    )
    for key in ("observed_only", "human_only", "persist_crops", "debug_crop_output"):
        sampling[key] = _require_bool(sampling[key], label=f"sampling.{key}")
    if sampling["observed_only"] is not True:
        raise AppearanceReidConfigError("sampling.observed_only must be true")
    if sampling["human_only"] is not True:
        raise AppearanceReidConfigError("sampling.human_only must be true")
    if sampling["persist_crops"] is not False:
        raise AppearanceReidConfigError("persist_crops must be false")
    if sampling["debug_crop_output"] is not False:
        raise AppearanceReidConfigError("debug_crop_output must be false in Stage 7B")

    agg = dict(_require_mapping(raw["aggregation"], label="aggregation"))
    method = _require_str(agg["method"], label="aggregation.method")
    if method not in {"quality_weighted_mean", "median"}:
        raise AppearanceReidConfigError("unsupported aggregation.method")
    agg["method"] = method
    agg["outlier_cosine_reject"] = _require_float(
        agg["outlier_cosine_reject"],
        label="outlier_cosine_reject",
        minimum=0.0,
        maximum=1.0,
    )
    agg["min_coverage"] = _require_float(
        agg["min_coverage"], label="min_coverage", minimum=0.0, maximum=1.0
    )
    agg["single_crop_strong_forbidden"] = _require_bool(
        agg["single_crop_strong_forbidden"], label="single_crop_strong_forbidden"
    )
    if agg["single_crop_strong_forbidden"] is not True:
        raise AppearanceReidConfigError("single_crop_strong_forbidden must be true")

    matching = dict(_require_mapping(raw["matching"], label="matching"))
    for key in ("similarity_threshold", "ambiguity_margin", "reject_below"):
        matching[key] = _require_float(
            matching[key], label=f"matching.{key}", minimum=0.0, maximum=1.0
        )
    matching["candidate_cap_per_track"] = _require_int(
        matching["candidate_cap_per_track"], label="candidate_cap_per_track", minimum=1
    )
    for key in (
        "mutual_nearest",
        "temporal_overlap_forbidden",
        "cross_video_forbidden",
        "human_ball_forbidden",
        "auto_confirm",
        "face_regions_use",
    ):
        matching[key] = _require_bool(matching[key], label=f"matching.{key}")
    if matching["auto_confirm"] is not False:
        raise AppearanceReidConfigError("matching.auto_confirm must be false")
    if matching["face_regions_use"] is not False:
        raise AppearanceReidConfigError("matching.face_regions_use must be false")
    if matching["cross_video_forbidden"] is not True:
        raise AppearanceReidConfigError("cross_video_forbidden must be true")
    if matching["human_ball_forbidden"] is not True:
        raise AppearanceReidConfigError("human_ball_forbidden must be true")
    if matching["temporal_overlap_forbidden"] is not True:
        raise AppearanceReidConfigError("temporal_overlap_forbidden must be true")
    if matching["reject_below"] > matching["similarity_threshold"]:
        raise AppearanceReidConfigError("reject_below must be <= similarity_threshold")

    review = dict(_require_mapping(raw["review"], label="review"))
    review["sample_ambiguous"] = _require_bool(
        review["sample_ambiguous"], label="review.sample_ambiguous"
    )
    review["sample_same_kit_flag"] = _require_bool(
        review["sample_same_kit_flag"], label="review.sample_same_kit_flag"
    )
    review["max_review_items"] = _require_int(
        review["max_review_items"], label="max_review_items", minimum=1
    )

    output = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if output.get("atomic_writes") is not True:
        raise AppearanceReidConfigError("output_policy.atomic_writes must be true")
    if output.get("overwrite_allowed") is not False:
        raise AppearanceReidConfigError("output_policy.overwrite_allowed must be false")
    # YAML may parse 0o600 as int 384
    mode = output.get("chmod_mode", 0o600)
    if isinstance(mode, str):
        mode = int(mode, 0)
    output["chmod_mode"] = _require_int(mode, label="chmod_mode", minimum=0o600, maximum=0o600)
    output["atomic_writes"] = True
    output["overwrite_allowed"] = False

    limits = dict(_require_mapping(raw["safety_limits"], label="safety_limits"))
    limits["max_tracks_per_video"] = _require_int(
        limits["max_tracks_per_video"], label="max_tracks_per_video", minimum=1
    )
    limits["max_profiles_per_run"] = _require_int(
        limits["max_profiles_per_run"], label="max_profiles_per_run", minimum=1
    )
    limits["max_candidates_per_run"] = _require_int(
        limits["max_candidates_per_run"], label="max_candidates_per_run", minimum=1
    )
    limits["timeout_seconds"] = _require_float(
        limits["timeout_seconds"], label="timeout_seconds", minimum=1.0
    )

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/"):
        raise AppearanceReidConfigError("runtime_root must be absolute")
    if raw["overwrite_allowed"] is not False:
        raise AppearanceReidConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise AppearanceReidConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise AppearanceReidConfigError("network_sources_allowed must be false")

    matrix = raw["selection_matrix"]
    if not isinstance(matrix, list) or not matrix:
        raise AppearanceReidConfigError("selection_matrix must be a non-empty list")
    selected = [
        m
        for m in matrix
        if isinstance(m, Mapping) and str(m.get("status", "")).lower() == "selected"
    ]
    if len(selected) != 1:
        raise AppearanceReidConfigError("selection_matrix must have exactly one selected")
    if "handcrafted" not in str(selected[0].get("candidate", "")).lower():
        raise AppearanceReidConfigError("selected candidate must be handcrafted descriptor")

    notes = raw["notes"]
    if not isinstance(notes, list):
        raise AppearanceReidConfigError("notes must be a list")

    return {
        "config_version": CONFIG_VERSION,
        "extractor_id": _require_str(raw["extractor_id"], label="extractor_id"),
        "extractor_version": _require_str(raw["extractor_version"], label="extractor_version"),
        "extractor_type": extractor_type,
        "descriptor": desc,
        "sampling": sampling,
        "aggregation": agg,
        "matching": matching,
        "review": review,
        "output_policy": output,
        "safety_limits": limits,
        "runtime_root": runtime_root,
        "identity_policy_path": _require_str(
            raw["identity_policy_path"], label="identity_policy_path"
        ),
        "deterministic_seed": _require_int(
            raw["deterministic_seed"], label="deterministic_seed", minimum=0
        ),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "selection_matrix": [_deep_unfreeze(m) for m in matrix],
        "notes": [str(n) for n in notes],
    }


def load_appearance_reid_config(path: Path | str) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise AppearanceReidConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise AppearanceReidConfigError("config size out of bounds")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AppearanceReidConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise AppearanceReidConfigError("config root must be a mapping")
    return _deep_freeze(_validate_config(data))


def appearance_reid_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_appearance_reid_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "identity" / "appearance_reid_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "AppearanceReidConfigError",
    "load_appearance_reid_config",
    "appearance_reid_config_fingerprint",
    "default_appearance_reid_config_path",
]
