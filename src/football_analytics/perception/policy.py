"""Strict detection policy loader (Stage 5A)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.perception.types import ERROR_CODES, PerceptionContractError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

ELIGIBILITY_VALUES = frozenset({"eligible", "conditionally_eligible", "ineligible", "unknown"})
REQUIRED_TOP = frozenset(
    {
        "policy_version",
        "config_version",
        "thresholds",
        "nms",
        "bbox",
        "preprocessing",
        "routing",
        "empty_frame_semantics",
        "error_codes",
        "skip_codes",
        "resource_limits",
        "provenance_requirements",
        "notes",
    }
)


class PolicyError(PerceptionContractError):
    """Detection policy config failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{label} must be a mapping")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{label} must be a bool")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{label} must be a number")
    f = float(value)
    if not (f == f) or f in (float("inf"), float("-inf")):  # noqa: PLR0124
        raise PolicyError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise PolicyError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise PolicyError(f"{label} must be <= {maximum}")
    return f


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PolicyError(f"{label} must be >= {minimum}")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PolicyError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise PolicyError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise PolicyError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise PolicyError(f"unknown top-level keys: {sorted(unknown)}")
    if int(raw["config_version"]) != CONFIG_VERSION:
        raise PolicyError(f"config_version must be {CONFIG_VERSION}")

    thresholds = dict(_require_mapping(raw["thresholds"], label="thresholds"))
    _require_float(
        thresholds["default_score_min"],
        label="thresholds.default_score_min",
        minimum=0.0,
        maximum=1.0,
    )
    per_class = dict(
        _require_mapping(thresholds["per_class_score_min"], label="thresholds.per_class_score_min")
    )
    for k, v in per_class.items():
        _require_float(v, label=f"thresholds.per_class_score_min.{k}", minimum=0.0, maximum=1.0)
    _require_bool(
        thresholds["calibrated_confidence_separate"],
        label="thresholds.calibrated_confidence_separate",
    )

    nms = dict(_require_mapping(raw["nms"], label="nms"))
    _require_float(nms["iou_threshold"], label="nms.iou_threshold", minimum=0.0, maximum=1.0)
    mode = _require_str(nms["mode"], label="nms.mode")
    if mode not in {"class_aware", "class_agnostic"}:
        raise PolicyError("nms.mode must be class_aware|class_agnostic")
    if nms.get("keep_suppressed_in_canonical") is not False:
        raise PolicyError("nms.keep_suppressed_in_canonical must be false")
    _require_bool(nms["record_pre_post_counts"], label="nms.record_pre_post_counts")

    bbox = dict(_require_mapping(raw["bbox"], label="bbox"))
    if bbox.get("format") != "xyxy" or bbox.get("bounds") != "half_open":
        raise PolicyError("bbox must be xyxy half_open")
    for key in (
        "reject_nan",
        "reject_inf",
        "reject_zero_area",
        "reject_negative_area",
        "clip_to_frame",
        "require_inverse_from_model_space",
        "center_or_foot_not_canonical",
    ):
        _require_bool(bbox[key], label=f"bbox.{key}")

    prep = dict(_require_mapping(raw["preprocessing"], label="preprocessing"))
    modes = _require_str_list(
        prep["allowed_resize_modes"], label="preprocessing.allowed_resize_modes"
    )
    if set(modes) != {"letterbox", "stretch"}:
        raise PolicyError("preprocessing.allowed_resize_modes must be letterbox+stretch")
    default_mode = _require_str(
        prep["default_resize_mode"], label="preprocessing.default_resize_mode"
    )
    if default_mode not in modes:
        raise PolicyError("default_resize_mode not in allowed_resize_modes")
    _require_int(
        prep["default_model_input_width"],
        label="preprocessing.default_model_input_width",
        minimum=1,
    )
    _require_int(
        prep["default_model_input_height"],
        label="preprocessing.default_model_input_height",
        minimum=1,
    )
    _require_float(
        prep["roundtrip_tolerance_px"],
        label="preprocessing.roundtrip_tolerance_px",
        minimum=0.0,
    )
    if prep.get("forbid_arbitrary_callables") is not True:
        raise PolicyError("forbid_arbitrary_callables must be true")

    routing = dict(_require_mapping(raw["routing"], label="routing"))
    for key in ("skip_on_playability", "skip_on_graphics"):
        _require_str_list(routing[key], label=f"routing.{key}")
    identity = dict(
        _require_mapping(routing["identity_only_closeup"], label="routing.identity_only_closeup")
    )
    _require_bool(
        identity["allow_human_detection"],
        label="routing.identity_only_closeup.allow_human_detection",
    )
    _require_bool(
        identity["mark_downstream_physical_unsafe"],
        label="routing.identity_only_closeup.mark_downstream_physical_unsafe",
    )
    _require_bool(
        routing["live_event_unknown_blocks_visual_detection"],
        label="routing.live_event_unknown_blocks_visual_detection",
    )
    if routing.get("record_processing_status") is not True:
        raise PolicyError("routing.record_processing_status must be true")

    empty = dict(_require_mapping(raw["empty_frame_semantics"], label="empty_frame_semantics"))
    if empty.get("processed_empty_status") != "processed_no_detections":
        raise PolicyError("processed_empty_status must be processed_no_detections")
    if empty.get("zero_counts_only_when_processed") is not True:
        raise PolicyError("zero_counts_only_when_processed must be true")
    if empty.get("no_fake_zeros_for_skipped_failed") is not True:
        raise PolicyError("no_fake_zeros_for_skipped_failed must be true")

    error_codes = _require_str_list(raw["error_codes"], label="error_codes")
    if set(error_codes) != ERROR_CODES:
        raise PolicyError("error_codes must match canonical ERROR_CODES exactly")
    skip_codes = _require_str_list(raw["skip_codes"], label="skip_codes")
    if not set(skip_codes).issubset(ERROR_CODES):
        raise PolicyError("skip_codes must be subset of error_codes")

    limits = dict(_require_mapping(raw["resource_limits"], label="resource_limits"))
    for key in (
        "max_detections_per_frame",
        "max_frames_per_run",
        "max_receipt_warnings",
        "max_config_bytes",
    ):
        _require_int(limits[key], label=f"resource_limits.{key}", minimum=1)

    prov = dict(_require_mapping(raw["provenance_requirements"], label="provenance_requirements"))
    for key in (
        "require_detector_id",
        "require_config_fingerprint",
        "require_taxonomy_version",
        "require_transform_fingerprint",
    ):
        if prov.get(key) is not True:
            raise PolicyError(f"provenance_requirements.{key} must be true")
    if prov.get("invent_model_sha_when_missing") is not False:
        raise PolicyError("invent_model_sha_when_missing must be false")

    notes = raw["notes"]
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise PolicyError("notes must be a list of strings")

    # Preserve nested structures after light validation.
    return {
        "policy_version": _require_str(raw["policy_version"], label="policy_version"),
        "config_version": CONFIG_VERSION,
        "thresholds": thresholds,
        "nms": nms,
        "bbox": bbox,
        "preprocessing": prep,
        "routing": routing,
        "empty_frame_semantics": empty,
        "error_codes": error_codes,
        "skip_codes": skip_codes,
        "resource_limits": limits,
        "provenance_requirements": prov,
        "notes": list(notes),
    }


def default_policy_path(*, project_root: Path | None = None) -> Path:
    if project_root is None:
        from football_analytics.data.registry import default_project_root

        project_root = default_project_root()
    return project_root / "configs" / "perception" / "detection_policy.yaml"


def load_detection_policy(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    cfg_path = path or default_policy_path(project_root=project_root)
    if cfg_path.is_symlink():
        raise PolicyError(f"symlink rejected: {cfg_path}")
    if not cfg_path.is_file():
        raise PolicyError(f"policy missing: {cfg_path}")
    size = cfg_path.stat().st_size
    if size <= 0 or size > MAX_CONFIG_BYTES:
        raise PolicyError(f"policy size out of bounds: {size}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PolicyError("policy root must be a mapping")
    return _deep_freeze(_validate_policy(raw))


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(policy))


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_unfreeze(v) for v in value]
    return value


def resolve_frame_routing(
    window: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any],
    detect_ball: bool = True,
) -> dict[str, Any]:
    """Decide human/ball eligibility from an analysis_window row (or None)."""
    if window is None:
        return {
            "eligibility": "ineligible",
            "process_human": False,
            "process_ball": False,
            "processing_status": "not_eligible",
            "skip_reason": "FRAME_NOT_ELIGIBLE",
            "error_code": "FRAME_NOT_ELIGIBLE",
        }

    playability = str(window.get("playability", "unknown"))
    graphics = str(window.get("graphics_status", "unknown"))
    tracking = str(window.get("tracking_eligibility", "unknown"))
    identity = str(window.get("identity_eligibility", "unknown"))
    ball_el = str(window.get("ball_analysis_eligibility", "unknown"))

    if playability in set(policy["routing"]["skip_on_playability"]):
        return {
            "eligibility": "ineligible",
            "process_human": False,
            "process_ball": False,
            "processing_status": "not_eligible",
            "skip_reason": "FRAME_NOT_ELIGIBLE",
            "error_code": "FRAME_NOT_ELIGIBLE",
        }
    if graphics in set(policy["routing"]["skip_on_graphics"]):
        return {
            "eligibility": "ineligible",
            "process_human": False,
            "process_ball": False,
            "processing_status": "not_eligible",
            "skip_reason": "FRAME_NOT_ELIGIBLE",
            "error_code": "FRAME_NOT_ELIGIBLE",
        }

    human_ok = tracking in {"eligible", "conditionally_eligible"} or identity in {
        "eligible",
        "conditionally_eligible",
    }
    ball_ok = detect_ball and ball_el in {"eligible", "conditionally_eligible"}
    ball_skip = "BALL_ANALYSIS_NOT_ELIGIBLE" if detect_ball and ball_el == "ineligible" else None

    if playability == "uncertain" and not human_ok:
        return {
            "eligibility": "unknown",
            "process_human": False,
            "process_ball": False,
            "processing_status": "skipped",
            "skip_reason": "UNKNOWN_PLAYABILITY",
            "error_code": "UNKNOWN_PLAYABILITY",
        }

    if not human_ok and not ball_ok:
        code = ball_skip or "FRAME_NOT_ELIGIBLE"
        return {
            "eligibility": "ineligible",
            "process_human": False,
            "process_ball": False,
            "processing_status": "not_eligible",
            "skip_reason": code,
            "error_code": code,
        }

    eligibility = "eligible" if human_ok else "conditionally_eligible"
    return {
        "eligibility": eligibility,
        "process_human": human_ok,
        "process_ball": ball_ok,
        "processing_status": "processed",  # caller may still choose processed_no_detections
        "skip_reason": None if ball_ok or not detect_ball else ball_skip,
        "error_code": None,
        "identity_only": identity in {"eligible", "conditionally_eligible"}
        and tracking not in {"eligible", "conditionally_eligible"},
    }


__all__ = [
    "CONFIG_VERSION",
    "PolicyError",
    "default_policy_path",
    "load_detection_policy",
    "policy_fingerprint",
    "resolve_frame_routing",
    "ELIGIBILITY_VALUES",
]
