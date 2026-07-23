"""Calibration request/receipt builders for contract fixtures only (Stage 8A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.calibration.contracts import (
    load_calibration_json_schema,
    validate_against_json_schema,
)
from football_analytics.calibration.evaluation import NOT_EVALUATED_CALIBRATION
from football_analytics.calibration.types import CalibrationContractError, MappingStatus


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    coordinate_system_fingerprint: str,
    pitch_template_fingerprint: str,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    request_id: str = "cal_req_01",
    output_root: str = "/home/fdoblak/workspace/calibration_contract_checks",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": None,
        "timeline_ref": None,
        "pitch_length_m": pitch_length_m,
        "pitch_width_m": pitch_width_m,
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "feature_source_ref": None,
        "model_ref": None,
        "policy_fingerprint": policy_fingerprint,
        "coordinate_system_fingerprint": coordinate_system_fingerprint,
        "output_root": output_root,
        "no_overwrite": True,
        "cache_enabled": False,
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "8A",
            "label": "synthetic_contract_fixture",
            "notes": "no_sv_inference",
            "no_sv_inference": True,
        },
    }


def recount_projection_counts(projections: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {m.value: 0 for m in MappingStatus}
    for p in projections:
        st = str(p["mapping_status"])
        counts[st] = counts.get(st, 0) + 1
    counts["total"] = len(projections)
    return counts


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    coordinate_system_fingerprint: str,
    pitch_template_fingerprint: str,
    pitch_length_m: float,
    pitch_width_m: float,
    features: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    projections: Sequence[Mapping[str, Any]] | None = None,
    request_id: str = "cal_req_01",
    receipt_id: str = "cal_receipt_01",
    status: str = "succeeded",
    review_count: int = 0,
    real_pitch_size_known: bool = False,
    correspondence_accepted: int = 0,
    correspondence_rejected: int = 0,
    gap_us: int = 0,
    overlap_conflicts: int = 0,
    degenerate: int = 0,
    singular: int = 0,
    ill_conditioned: int = 0,
    mirrored: int = 0,
    mean_reprojection_error_px: float | None = None,
) -> dict[str, Any]:
    projections = list(projections or [])
    feat_counts: dict[str, int] = {}
    for f in features:
        ft = str(f["feature_type"])
        feat_counts[ft] = feat_counts.get(ft, 0) + 1
    solved = sum(1 for s in segments if s.get("validity_status") == "valid")
    failed = sum(1 for s in segments if s.get("validity_status") == "invalid")
    invalid = failed
    valid_us = sum(
        max(0, int(s["end_time_us"]) - int(s["start_time_us"]))
        for s in segments
        if s.get("validity_status") == "valid"
    )
    eligible = sum(1 for p in projections if p.get("physical_metric_eligibility") == "eligible")
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "run_id": run_id,
        "video_id": video_id,
        "request_id": request_id,
        "config_fingerprint": policy_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "coordinate_system_fingerprint": coordinate_system_fingerprint,
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "pitch_length_m": float(pitch_length_m),
        "pitch_width_m": float(pitch_width_m),
        "real_pitch_size_known": bool(real_pitch_size_known),
        "feature_counts": feat_counts,
        "correspondence_counts": {
            "accepted": correspondence_accepted,
            "rejected": correspondence_rejected,
            "total": correspondence_accepted + correspondence_rejected,
        },
        "calibration_counts": {
            "solved": solved,
            "failed": failed,
            "invalid": invalid,
            "total": len(segments),
        },
        "segment_coverage": {
            "valid_us": valid_us,
            "gap_us": gap_us,
            "overlap_conflicts": overlap_conflicts,
        },
        "homography_quality": {
            "degenerate": degenerate,
            "singular": singular,
            "ill_conditioned": ill_conditioned,
            "mirrored": mirrored,
            "mean_reprojection_error_px": mean_reprojection_error_px,
        },
        "projection_counts": recount_projection_counts(projections),
        "physical_metric_eligible_count": eligible,
        "review_count": review_count,
        "ground_truth_evaluation_status": NOT_EVALUATED_CALIBRATION,
        "output_artifacts": {
            "calibration_segments": {
                "path": "calibration_segments.parquet",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now(),
        "status": status,
        "warnings": [] if real_pitch_size_known else ["real_pitch_size_unknown"],
        "errors": [],
        "provenance": {
            "stage": "8A",
            "label": "synthetic_contract_fixture",
            "notes": "no_sv_inference",
            "no_sv_inference": True,
            "attack_direction": "unknown",
        },
    }


def validate_request_payload(payload: Mapping[str, Any]) -> None:
    schema = load_calibration_json_schema("calibration_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise CalibrationContractError(f"request schema validation failed: {exc}") from exc


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_calibration_json_schema("calibration_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise CalibrationContractError(f"receipt schema validation failed: {exc}") from exc


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "recount_projection_counts",
    "validate_request_payload",
    "validate_receipt_payload",
]
