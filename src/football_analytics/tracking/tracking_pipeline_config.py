"""Stage 6D tracking pipeline config loader (fingerprinted, strict)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json

CONFIG_VERSION = 1


class TrackingPipelineConfigError(ValueError):
    """Invalid tracking pipeline config."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrackingPipelineConfigError(f"{label} must be a mapping")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrackingPipelineConfigError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrackingPipelineConfigError(f"{label} must be a bool")
    return value


def _require_float(value: Any, *, label: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrackingPipelineConfigError(f"{label} must be a number")
    f = float(value)
    if f < minimum or f > maximum:
        raise TrackingPipelineConfigError(f"{label} out of range [{minimum}, {maximum}]")
    return f


def _require_int(value: Any, *, label: str, minimum: int = 0, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrackingPipelineConfigError(f"{label} must be an int")
    if value < minimum or value > maximum:
        raise TrackingPipelineConfigError(f"{label} out of range")
    return value


def load_tracking_pipeline_config(path: Path | str) -> dict[str, Any]:
    """Load and strictly validate tracking pipeline YAML."""
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise TrackingPipelineConfigError(f"config missing or symlink: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TrackingPipelineConfigError("config root must be a mapping")
    if int(raw.get("config_version", -1)) != CONFIG_VERSION:
        raise TrackingPipelineConfigError(f"config_version must be {CONFIG_VERSION}")

    align = dict(_require_mapping(raw["alignment"], label="alignment"))
    for key in (
        "require_run_id_match",
        "require_video_id_match",
        "require_source_sha_match",
        "require_timeline_fingerprint_match",
        "require_detection_fingerprint_match",
        "require_analysis_window_fingerprint_match",
        "fail_on_missing_receipt",
    ):
        align[key] = _require_bool(align[key], label=f"alignment.{key}")

    fusion = dict(_require_mapping(raw["fusion"], label="fusion"))
    for key in (
        "namespace_ball_tracks",
        "remap_ball_track_ids",
        "preserve_predicted_flags",
        "preserve_unknown_roles",
        "do_not_upgrade_ambiguous_ball",
        "reject_cross_cut_continuation",
        "reject_terminated_reopen",
        "reject_duplicate_detection_assignment",
        "no_human_ball_relationship_table",
    ):
        fusion[key] = _require_bool(fusion[key], label=f"fusion.{key}")
    if fusion["no_human_ball_relationship_table"] is not True:
        raise TrackingPipelineConfigError("fusion.no_human_ball_relationship_table must be true")

    thr = dict(_require_mapping(raw["quality_thresholds"], label="quality_thresholds"))
    cleaned_thr = {
        "min_eligible_tracking_coverage": _require_float(
            thr["min_eligible_tracking_coverage"], label="min_eligible_tracking_coverage"
        ),
        "max_failed_frame_rate": _require_float(
            thr["max_failed_frame_rate"], label="max_failed_frame_rate"
        ),
        "max_dangling_fk": _require_int(thr["max_dangling_fk"], label="max_dangling_fk"),
        "max_duplicate_keys": _require_int(thr["max_duplicate_keys"], label="max_duplicate_keys"),
        "max_invalid_bbox": _require_int(thr["max_invalid_bbox"], label="max_invalid_bbox"),
        "max_receipt_mismatch": _require_int(
            thr["max_receipt_mismatch"], label="max_receipt_mismatch"
        ),
        "max_cross_cut_violations": _require_int(
            thr["max_cross_cut_violations"], label="max_cross_cut_violations"
        ),
        "max_terminated_reopen": _require_int(
            thr["max_terminated_reopen"], label="max_terminated_reopen"
        ),
        "max_predicted_ratio": _require_float(
            thr["max_predicted_ratio"], label="max_predicted_ratio"
        ),
        "fragmentation_finding_rate": _require_float(
            thr["fragmentation_finding_rate"], label="fragmentation_finding_rate"
        ),
        "role_abstention_finding_rate": _require_float(
            thr["role_abstention_finding_rate"], label="role_abstention_finding_rate"
        ),
        "ball_ambiguity_finding_rate": _require_float(
            thr["ball_ambiguity_finding_rate"], label="ball_ambiguity_finding_rate"
        ),
    }

    review = dict(_require_mapping(raw["review_policy"], label="review_policy"))
    cleaned_review = {
        "sample_role_abstention": _require_bool(
            review["sample_role_abstention"], label="sample_role_abstention"
        ),
        "max_role_review_items": _require_int(
            review["max_role_review_items"], label="max_role_review_items", minimum=0
        ),
        "role_sample_stride": _require_int(
            review["role_sample_stride"], label="role_sample_stride", minimum=1
        ),
        "sample_fragmentation": _require_bool(
            review["sample_fragmentation"], label="sample_fragmentation"
        ),
        "max_fragmentation_review_items": _require_int(
            review["max_fragmentation_review_items"],
            label="max_fragmentation_review_items",
            minimum=0,
        ),
        "sample_ball_ambiguity": _require_bool(
            review["sample_ball_ambiguity"], label="sample_ball_ambiguity"
        ),
        "max_ambiguity_review_items": _require_int(
            review["max_ambiguity_review_items"], label="max_ambiguity_review_items", minimum=0
        ),
        "sample_long_no_ball": _require_bool(
            review["sample_long_no_ball"], label="sample_long_no_ball"
        ),
        "long_no_ball_frames": _require_int(
            review["long_no_ball_frames"], label="long_no_ball_frames", minimum=1
        ),
        "sample_high_predicted": _require_bool(
            review["sample_high_predicted"], label="sample_high_predicted"
        ),
        "review_on_receipt_mismatch": _require_bool(
            review["review_on_receipt_mismatch"], label="review_on_receipt_mismatch"
        ),
        "review_on_fk_duplicate": _require_bool(
            review["review_on_fk_duplicate"], label="review_on_fk_duplicate"
        ),
        "review_on_cross_cut": _require_bool(
            review["review_on_cross_cut"], label="review_on_cross_cut"
        ),
        "do_not_spam_empty_frames": _require_bool(
            review["do_not_spam_empty_frames"], label="do_not_spam_empty_frames"
        ),
    }

    out = dict(_require_mapping(raw["output_policy"], label="output_policy"))
    if out.get("atomic_writes") is not True:
        raise TrackingPipelineConfigError("output_policy.atomic_writes must be true")
    if out.get("overwrite_allowed") is not False:
        raise TrackingPipelineConfigError("output_policy.overwrite_allowed must be false")

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    if not runtime_root.startswith("/home/fdoblak/workspace/tracking_pipeline_checks"):
        raise TrackingPipelineConfigError(
            "runtime_root must be under /home/fdoblak/workspace/tracking_pipeline_checks"
        )

    if raw["overwrite_allowed"] is not False:
        raise TrackingPipelineConfigError("overwrite_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise TrackingPipelineConfigError("symlinks_allowed must be false")
    if raw["network_sources_allowed"] is not False:
        raise TrackingPipelineConfigError("network_sources_allowed must be false")

    notes = raw.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise TrackingPipelineConfigError("notes must be a list of strings")

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
            "write_bundle_manifest": bool(out.get("write_bundle_manifest", True)),
            "emit_primary_sidecar": bool(out.get("emit_primary_sidecar", True)),
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


def tracking_pipeline_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(dict(config))


def default_tracking_pipeline_config_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "tracking" / "tracking_pipeline.yaml"


__all__ = [
    "CONFIG_VERSION",
    "TrackingPipelineConfigError",
    "load_tracking_pipeline_config",
    "tracking_pipeline_config_fingerprint",
    "default_tracking_pipeline_config_path",
]
