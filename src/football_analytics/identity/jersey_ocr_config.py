"""Strict loader for Stage 7D jersey OCR baseline config."""

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
        "producer",
        "producer_version",
        "method_id",
        "method_type",
        "identity_policy_path",
        "eligibility",
        "region",
        "preprocessing",
        "ocr",
        "consensus",
        "review",
        "assignment",
        "output_policy",
        "safety_limits",
        "runtime_root",
        "deterministic_seed",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
        "selection_matrix",
        "notes",
    }
)


class JerseyOcrConfigError(ValueError):
    """Jersey OCR baseline config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JerseyOcrConfigError(f"{label} must be a mapping")
    return value


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JerseyOcrConfigError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise JerseyOcrConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise JerseyOcrConfigError(f"{label} must be <= {maximum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JerseyOcrConfigError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise JerseyOcrConfigError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise JerseyOcrConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise JerseyOcrConfigError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise JerseyOcrConfigError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise JerseyOcrConfigError(f"{label} must be a non-empty string")
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
        raise JerseyOcrConfigError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise JerseyOcrConfigError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise JerseyOcrConfigError(f"config_version must be {CONFIG_VERSION}")
    if _require_str(raw["method_type"], label="method_type") != "template_ocr":
        raise JerseyOcrConfigError("method_type must be template_ocr")

    elig = dict(_require_mapping(raw["eligibility"], label="eligibility"))
    for key in (
        "human_only",
        "observed_only",
        "exclude_predicted",
        "exclude_ball",
        "unknown_role_conservative",
        "graphics_replay_forbidden",
    ):
        elig[key] = _require_bool(elig[key], label=f"eligibility.{key}")
    for key in (
        "min_bbox_width",
        "min_bbox_height",
        "min_bbox_area",
        "max_samples_per_track",
        "sample_stride",
    ):
        elig[key] = _require_int(elig[key], label=f"eligibility.{key}", minimum=1)
    elig["min_coverage"] = _require_float(
        elig["min_coverage"], label="eligibility.min_coverage", minimum=0.0, maximum=1.0
    )
    for list_key in ("exclude_roles", "allow_roles"):
        vals = list(elig.get(list_key) or [])
        if not all(isinstance(x, str) and x for x in vals):
            raise JerseyOcrConfigError(f"eligibility.{list_key} must be strings")
        elig[list_key] = vals
    if not elig["exclude_predicted"] or not elig["observed_only"]:
        raise JerseyOcrConfigError("observed_only/exclude_predicted must be true")

    region = dict(_require_mapping(raw["region"], label="region"))
    for key in (
        "torso_y0_frac",
        "torso_y1_frac",
        "torso_x0_frac",
        "torso_x1_frac",
        "min_contrast",
        "max_blur_laplacian_var",
        "min_blur_laplacian_var",
    ):
        region[key] = _require_float(region[key], label=f"region.{key}", minimum=0.0)
    for key in ("min_region_width", "min_region_height", "min_region_area"):
        region[key] = _require_int(region[key], label=f"region.{key}", minimum=1)
    region["orientation_claim"] = _require_str(
        region["orientation_claim"], label="region.orientation_claim"
    )
    if region["orientation_claim"] not in {"candidate", "unknown", "not_suitable"}:
        raise JerseyOcrConfigError("orientation_claim must be candidate|unknown|not_suitable")
    region["persist_crops"] = _require_bool(region["persist_crops"], label="region.persist_crops")
    if region["persist_crops"]:
        raise JerseyOcrConfigError("persist_crops must be false in Stage 7D")
    if not (0.0 <= region["torso_y0_frac"] < region["torso_y1_frac"] <= 1.0):
        raise JerseyOcrConfigError("invalid torso y fractions")
    if not (0.0 <= region["torso_x0_frac"] < region["torso_x1_frac"] <= 1.0):
        raise JerseyOcrConfigError("invalid torso x fractions")

    prep = dict(_require_mapping(raw["preprocessing"], label="preprocessing"))
    for key in ("grayscale", "clahe"):
        prep[key] = _require_bool(prep[key], label=f"preprocessing.{key}")
    prep["clahe_clip"] = _require_float(prep["clahe_clip"], label="clahe_clip", minimum=0.1)
    prep["clahe_tile"] = _require_int(prep["clahe_tile"], label="clahe_tile", minimum=2)
    prep["threshold_mode"] = _require_str(prep["threshold_mode"], label="threshold_mode")
    if prep["threshold_mode"] not in {"otsu", "adaptive", "fixed"}:
        raise JerseyOcrConfigError("threshold_mode must be otsu|adaptive|fixed")
    for key in (
        "morph_open_ksize",
        "morph_close_ksize",
        "min_component_area",
        "max_digits",
        "min_digits",
    ):
        prep[key] = _require_int(prep[key], label=f"preprocessing.{key}", minimum=0)
    prep["max_component_area_frac"] = _require_float(
        prep["max_component_area_frac"], label="max_component_area_frac", minimum=0.0, maximum=1.0
    )
    prep["min_aspect"] = _require_float(prep["min_aspect"], label="min_aspect", minimum=0.0)
    prep["max_aspect"] = _require_float(prep["max_aspect"], label="max_aspect", minimum=0.0)
    if prep["max_digits"] != 2 or prep["min_digits"] < 1:
        raise JerseyOcrConfigError("Stage 7D requires 1-2 digits (max_digits=2)")

    ocr = dict(_require_mapping(raw["ocr"], label="ocr"))
    ocr["template_version"] = _require_str(ocr["template_version"], label="template_version")
    size = list(ocr["digit_match_size"])
    if len(size) != 2:
        raise JerseyOcrConfigError("digit_match_size must be [w,h]")
    ocr["digit_match_size"] = [
        _require_int(size[0], label="digit_match_size[0]", minimum=8),
        _require_int(size[1], label="digit_match_size[1]", minimum=8),
    ]
    for key in ("min_digit_score", "min_number_score", "ambiguity_margin", "similar_digit_margin"):
        ocr[key] = _require_float(ocr[key], label=f"ocr.{key}", minimum=0.0, maximum=1.0)
    ocr["allow_leading_zero"] = _require_bool(
        ocr["allow_leading_zero"], label="ocr.allow_leading_zero"
    )
    ocr["max_normalized_number"] = _require_int(
        ocr["max_normalized_number"], label="max_normalized_number", minimum=0, maximum=99
    )
    ocr["source"] = _require_str(ocr["source"], label="ocr.source")
    if ocr["source"] not in {"model", "rule", "manual"}:
        raise JerseyOcrConfigError("ocr.source must be model|rule|manual")

    cons = dict(_require_mapping(raw["consensus"], label="consensus"))
    for key in ("min_observations", "min_temporal_spread_frames"):
        cons[key] = _require_int(cons[key], label=f"consensus.{key}", minimum=1)
    cons["min_winning_margin"] = _require_float(
        cons["min_winning_margin"], label="min_winning_margin", minimum=0.0
    )
    cons["quality_weight_power"] = _require_float(
        cons["quality_weight_power"], label="quality_weight_power", minimum=0.0
    )
    cons["conflict_on_switch"] = _require_bool(
        cons["conflict_on_switch"], label="conflict_on_switch"
    )
    cons["cross_track_forbidden"] = _require_bool(
        cons["cross_track_forbidden"], label="cross_track_forbidden"
    )
    if not cons["cross_track_forbidden"]:
        raise JerseyOcrConfigError("cross_track_forbidden must be true")

    review = dict(_require_mapping(raw["review"], label="review"))
    for key in ("sample_ambiguous", "sample_conflict", "sample_low_quality"):
        review[key] = _require_bool(review[key], label=f"review.{key}")
    review["max_review_items"] = _require_int(
        review["max_review_items"], label="max_review_items", minimum=0
    )

    asn = dict(_require_mapping(raw["assignment"], label="assignment"))
    for key in ("auto_confirm_identity", "auto_target_confirmation", "persist_crops"):
        asn[key] = _require_bool(asn[key], label=f"assignment.{key}")
        if asn[key]:
            raise JerseyOcrConfigError(f"assignment.{key} must be false in Stage 7D")

    out_pol = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    out_pol["atomic_writes"] = _require_bool(out_pol["atomic_writes"], label="atomic_writes")
    out_pol["overwrite_allowed"] = _require_bool(
        out_pol["overwrite_allowed"], label="overwrite_allowed"
    )
    if out_pol["overwrite_allowed"]:
        raise JerseyOcrConfigError("overwrite_allowed must be false")
    mode = out_pol["chmod_mode"]
    if isinstance(mode, str):
        mode = int(mode, 0)
    out_pol["chmod_mode"] = _require_int(mode, label="chmod_mode", minimum=0)
    out_pol["write_consensus_sidecar"] = _require_bool(
        out_pol["write_consensus_sidecar"], label="write_consensus_sidecar"
    )
    out_pol["write_region_provenance"] = _require_bool(
        out_pol["write_region_provenance"], label="write_region_provenance"
    )

    safety = dict(_require_mapping(raw["safety_limits"], label="safety_limits"))
    safety["max_tracks_per_video"] = _require_int(
        safety["max_tracks_per_video"], label="max_tracks_per_video", minimum=1
    )
    safety["max_observations_per_run"] = _require_int(
        safety["max_observations_per_run"], label="max_observations_per_run", minimum=1
    )
    safety["timeout_seconds"] = _require_float(
        safety["timeout_seconds"], label="timeout_seconds", minimum=1.0
    )

    matrix = list(raw["selection_matrix"])
    if not matrix:
        raise JerseyOcrConfigError("selection_matrix must be non-empty")
    selected = [m for m in matrix if str(m.get("status", "")).lower() == "selected"]
    if len(selected) != 1:
        raise JerseyOcrConfigError("exactly one selection_matrix entry must be selected")
    cand = str(selected[0].get("candidate", "")).lower()
    if "opencv" not in cand or "template" not in cand:
        raise JerseyOcrConfigError("selected method must be OpenCV template/shape matcher")
    future = [m for m in matrix if "sn-jersey" in str(m.get("candidate", "")).lower()]
    if future and str(future[0].get("status", "")).lower() != "future":
        raise JerseyOcrConfigError("sn-jersey must remain future adapter only")

    if _require_bool(raw["overwrite_allowed"], label="overwrite_allowed"):
        raise JerseyOcrConfigError("overwrite_allowed must be false")
    if _require_bool(raw["symlinks_allowed"], label="symlinks_allowed"):
        raise JerseyOcrConfigError("symlinks_allowed must be false")
    if _require_bool(raw["network_sources_allowed"], label="network_sources_allowed"):
        raise JerseyOcrConfigError("network_sources_allowed must be false")

    return {
        "config_version": CONFIG_VERSION,
        "producer": _require_str(raw["producer"], label="producer"),
        "producer_version": _require_str(raw["producer_version"], label="producer_version"),
        "method_id": _require_str(raw["method_id"], label="method_id"),
        "method_type": "template_ocr",
        "identity_policy_path": _require_str(
            raw["identity_policy_path"], label="identity_policy_path"
        ),
        "eligibility": elig,
        "region": region,
        "preprocessing": prep,
        "ocr": ocr,
        "consensus": cons,
        "review": review,
        "assignment": asn,
        "output_policy": out_pol,
        "safety_limits": safety,
        "runtime_root": _require_str(raw["runtime_root"], label="runtime_root"),
        "deterministic_seed": _require_int(
            raw["deterministic_seed"], label="deterministic_seed", minimum=0
        ),
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "selection_matrix": matrix,
        "notes": list(raw["notes"]),
    }


def load_jersey_ocr_config(path: Path | str) -> Mapping[str, Any]:
    p = Path(path)
    if p.is_symlink():
        raise JerseyOcrConfigError(f"symlink rejected: {p}")
    if not p.is_file():
        raise JerseyOcrConfigError(f"config not found: {p}")
    if p.stat().st_size > MAX_CONFIG_BYTES:
        raise JerseyOcrConfigError("config too large")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise JerseyOcrConfigError("config root must be mapping")
    validated = _validate_config(raw)
    return _deep_freeze(validated)


def jersey_ocr_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


def default_jersey_ocr_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "identity" / "jersey_ocr_baseline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "JerseyOcrConfigError",
    "load_jersey_ocr_config",
    "jersey_ocr_config_fingerprint",
    "default_jersey_ocr_config_path",
]
