"""Stage 5E detection pipeline config loader (fingerprinted, strict)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json

CONFIG_VERSION = 1


class DetectionPipelineConfigError(ValueError):
    """Invalid detection pipeline config."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectionPipelineConfigError(f"{label} must be a mapping")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetectionPipelineConfigError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise DetectionPipelineConfigError(f"{label} must be a bool")
    return value


def _require_float(value: Any, *, label: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DetectionPipelineConfigError(f"{label} must be a number")
    f = float(value)
    if f < minimum or f > maximum:
        raise DetectionPipelineConfigError(f"{label} out of range [{minimum}, {maximum}]")
    return f


def _require_int(value: Any, *, label: str, minimum: int = 0, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DetectionPipelineConfigError(f"{label} must be an int")
    if value < minimum or value > maximum:
        raise DetectionPipelineConfigError(f"{label} out of range")
    return value


def load_detection_pipeline_config(path: Path | str) -> dict[str, Any]:
    """Load and strictly validate detection pipeline YAML."""
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise DetectionPipelineConfigError(f"config missing or symlink: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DetectionPipelineConfigError("config root must be a mapping")
    if int(raw.get("config_version", -1)) != CONFIG_VERSION:
        raise DetectionPipelineConfigError(f"config_version must be {CONFIG_VERSION}")

    align = dict(_require_mapping(raw["alignment"], label="alignment"))
    for key in (
        "require_run_id_match",
        "require_video_id_match",
        "require_frames_ref_match",
        "require_source_sha_match",
        "fail_on_missing_receipt",
    ):
        align[key] = _require_bool(align[key], label=f"alignment.{key}")

    fusion = dict(_require_mapping(raw["fusion"], label="fusion"))
    for key in (
        "remap_ball_detection_ids",
        "prefer_processed_over_skipped",
        "preserve_not_eligible",
        "eligibility_conflict_fails",
    ):
        fusion[key] = _require_bool(fusion[key], label=f"fusion.{key}")
    if fusion.get("cross_class_nms") is not False:
        raise DetectionPipelineConfigError("fusion.cross_class_nms must be false")
    fusion["cross_class_nms"] = False

    thr = dict(_require_mapping(raw["quality_thresholds"], label="quality_thresholds"))
    cleaned_thr = {
        "max_failed_frame_rate": _require_float(
            thr["max_failed_frame_rate"], label="max_failed_frame_rate"
        ),
        "max_dangling_fk": _require_int(thr["max_dangling_fk"], label="max_dangling_fk"),
        "max_duplicate_detection_keys": _require_int(
            thr["max_duplicate_detection_keys"], label="max_duplicate_detection_keys"
        ),
        "max_invalid_bbox": _require_int(thr["max_invalid_bbox"], label="max_invalid_bbox"),
        "max_receipt_mismatch": _require_int(
            thr["max_receipt_mismatch"], label="max_receipt_mismatch"
        ),
        "min_eligible_processing_coverage": _require_float(
            thr["min_eligible_processing_coverage"],
            label="min_eligible_processing_coverage",
        ),
        "role_abstention_finding_rate": _require_float(
            thr["role_abstention_finding_rate"], label="role_abstention_finding_rate"
        ),
    }

    review = dict(_require_mapping(raw["review_policy"], label="review_policy"))
    cleaned_review = {
        "sample_unknown_roles": _require_bool(
            review["sample_unknown_roles"], label="sample_unknown_roles"
        ),
        "max_unknown_review_items": _require_int(
            review["max_unknown_review_items"], label="max_unknown_review_items", minimum=0
        ),
        "unknown_sample_stride": _require_int(
            review["unknown_sample_stride"], label="unknown_sample_stride", minimum=1
        ),
        "review_on_receipt_mismatch": _require_bool(
            review["review_on_receipt_mismatch"], label="review_on_receipt_mismatch"
        ),
        "review_on_invalid_bbox": _require_bool(
            review["review_on_invalid_bbox"], label="review_on_invalid_bbox"
        ),
        "review_on_duplicate": _require_bool(
            review["review_on_duplicate"], label="review_on_duplicate"
        ),
        "review_on_long_no_detection": _require_bool(
            review["review_on_long_no_detection"], label="review_on_long_no_detection"
        ),
        "long_no_detection_frames": _require_int(
            review["long_no_detection_frames"], label="long_no_detection_frames", minimum=1
        ),
        "do_not_spam_every_unknown": _require_bool(
            review["do_not_spam_every_unknown"], label="do_not_spam_every_unknown"
        ),
    }

    out = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if out.get("atomic_writes") is not True:
        raise DetectionPipelineConfigError("output_policy.atomic_writes must be true")
    if out.get("overwrite_allowed") is not False:
        raise DetectionPipelineConfigError("output_policy.overwrite_allowed must be false")

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/home/fdoblak/workspace/detection_pipeline_checks"):
        raise DetectionPipelineConfigError(
            "runtime_root must be under /home/fdoblak/workspace/detection_pipeline_checks"
        )

    if raw["overwrite_allowed"] is not False:
        raise DetectionPipelineConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise DetectionPipelineConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise DetectionPipelineConfigError("network_sources_allowed must be false")

    notes = raw.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise DetectionPipelineConfigError("notes must be a list of strings")

    return {
        "config_version": CONFIG_VERSION,
        "pipeline_id": _require_str(raw["pipeline_id"], label="pipeline_id"),
        "pipeline_version": _require_str(raw["pipeline_version"], label="pipeline_version"),
        "alignment": align,
        "fusion": fusion,
        "quality_thresholds": cleaned_thr,
        "review_policy": cleaned_review,
        "output_policy": {
            "atomic_writes": True,
            "overwrite_allowed": False,
            "write_quality_report": bool(out.get("write_quality_report", True)),
            "write_review_queue": bool(out.get("write_review_queue", True)),
            "write_pipeline_receipt": bool(out.get("write_pipeline_receipt", True)),
        },
        "maximum_frames_per_run": _require_int(
            raw["maximum_frames_per_run"],
            label="maximum_frames_per_run",
            minimum=1,
            maximum=100000,
        ),
        "runtime_root": runtime_root,
        "overwrite_allowed": False,
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "notes": list(notes),
    }


def detection_pipeline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(dict(config))


def default_detection_pipeline_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "perception" / "detection_pipeline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "DetectionPipelineConfigError",
    "load_detection_pipeline_config",
    "detection_pipeline_config_fingerprint",
    "default_detection_pipeline_config_path",
]
