"""Calibration segment lifecycle — shot-cut terminate; no silent fill."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.calibration.types import (
    CONTRACT_VERSION,
    CalibrationContractError,
    CameraMotion,
    ValidityStatus,
)

HALF_OPEN = True  # [start_us, end_us)


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open [start, end) overlap test."""
    return a_start < b_end and b_start < a_end


def validate_segment_interval(start_us: int, end_us: int) -> None:
    if start_us < 0 or end_us < 0:
        raise CalibrationContractError("segment times must be >= 0")
    if end_us <= start_us:
        raise CalibrationContractError("segment interval must be half-open with end > start")


def terminate_on_shot_cut(
    *,
    segment: Mapping[str, Any],
    cut_time_us: int,
    boundary_reason: str = "SHOT_CUT_TERMINATE",
) -> dict[str, Any]:
    """Return a terminated copy of segment ending at cut_time_us (exclusive)."""
    out = dict(segment)
    start = int(out["start_time_us"])
    if cut_time_us <= start:
        raise CalibrationContractError("shot cut at or before segment start")
    out["end_time_us"] = int(cut_time_us)
    out["boundary_reason"] = boundary_reason
    out["next_segment_id"] = None
    return out


def apply_camera_motion_policy(
    motion: str,
    *,
    previous_motion: str | None = None,
) -> str:
    """Return validity guidance for camera motion change."""
    if motion not in {m.value for m in CameraMotion}:
        raise CalibrationContractError(f"unknown camera_motion: {motion}")
    if (
        previous_motion is not None
        and previous_motion != motion
        and motion
        in {
            CameraMotion.PAN.value,
            CameraMotion.ZOOM.value,
            CameraMotion.PAN_ZOOM.value,
        }
    ):
        return "require_new_or_invalid"
    return "ok"


def camera_view_eligibility(camera_view: str) -> str:
    if camera_view == "replay":
        return ValidityStatus.NOT_ELIGIBLE.value
    if camera_view == "unknown":
        return ValidityStatus.ABSTAIN.value
    return ValidityStatus.VALID.value


def find_segment_overlaps(segments: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    conflicts: list[tuple[str, str]] = []
    rows = list(segments)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if str(a.get("run_id")) != str(b.get("run_id")):
                continue
            if str(a.get("video_id")) != str(b.get("video_id")):
                continue
            if intervals_overlap(
                int(a["start_time_us"]),
                int(a["end_time_us"]),
                int(b["start_time_us"]),
                int(b["end_time_us"]),
            ):
                conflicts.append((str(a["segment_id"]), str(b["segment_id"])))
    return conflicts


def find_calibration_gaps(
    segments: Sequence[Mapping[str, Any]],
    *,
    timeline_start_us: int,
    timeline_end_us: int,
) -> list[tuple[int, int]]:
    """Return uncovered half-open gaps; never silently filled."""
    validate_segment_interval(timeline_start_us, timeline_end_us)
    valid = [
        (int(s["start_time_us"]), int(s["end_time_us"]))
        for s in segments
        if s.get("validity_status") == ValidityStatus.VALID.value
    ]
    valid.sort()
    gaps: list[tuple[int, int]] = []
    cursor = timeline_start_us
    for start, end in valid:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < timeline_end_us:
        gaps.append((cursor, timeline_end_us))
    return gaps


def assert_no_silent_gap_fill(policy: Mapping[str, Any]) -> None:
    segs = policy.get("segments")
    if isinstance(segs, Mapping) and segs.get("silent_gap_fill") is True:
        raise CalibrationContractError("SILENT_GAP_FILL_FORBIDDEN")


def segment_row(
    *,
    run_id: str,
    video_id: str,
    segment_id: str,
    calibration_id: int,
    start_time_us: int,
    end_time_us: int,
    source_frame_index: int,
    homography_image_to_pitch: Sequence[float] | None,
    pitch_length_m: float,
    pitch_width_m: float,
    pitch_template_fingerprint: str,
    validity_status: str = "valid",
    camera_view: str = "main",
    camera_motion: str = "static",
    is_interpolated: bool = False,
    physical_metric_eligible: bool = True,
    boundary_reason: str = "none",
    start_frame_index: int | None = None,
    correspondence_count: int = 4,
    inlier_count: int = 4,
    solver_method: str = "dlt_numpy",
    solver_version: str = "1",
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    validate_segment_interval(start_time_us, end_time_us)
    if is_interpolated:
        physical_metric_eligible = False
    row: dict[str, Any] = {
        "run_id": run_id,
        "video_id": video_id,
        "segment_id": segment_id,
        "calibration_id": int(calibration_id),
        "start_time_us": int(start_time_us),
        "end_time_us": int(end_time_us),
        "start_frame_index": int(
            start_frame_index if start_frame_index is not None else source_frame_index
        ),
        "end_frame_index": extra.pop("end_frame_index", None),
        "source_frame_index": int(source_frame_index),
        "shot_segment_id": extra.pop("shot_segment_id", None),
        "analysis_window_id": extra.pop("analysis_window_id", None),
        "camera_view": camera_view,
        "camera_motion": camera_motion,
        "validity_status": validity_status,
        "homography_image_to_pitch": (
            list(homography_image_to_pitch) if homography_image_to_pitch is not None else None
        ),
        "homography_pitch_to_image": extra.pop("homography_pitch_to_image", None),
        "condition_number": extra.pop("condition_number", None),
        "determinant": extra.pop("determinant", None),
        "correspondence_count": int(correspondence_count),
        "inlier_count": int(inlier_count),
        "inlier_ratio": extra.pop("inlier_ratio", 1.0),
        "mean_reprojection_error_px": extra.pop("mean_reprojection_error_px", None),
        "coverage_hull_area_fraction": extra.pop("coverage_hull_area_fraction", None),
        "pitch_length_m": float(pitch_length_m),
        "pitch_width_m": float(pitch_width_m),
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "solver_method": solver_method,
        "solver_version": solver_version,
        "is_interpolated": bool(is_interpolated),
        "reuse_policy": extra.pop("reuse_policy", "none"),
        "boundary_reason": boundary_reason,
        "previous_segment_id": extra.pop("previous_segment_id", None),
        "next_segment_id": extra.pop("next_segment_id", None),
        "physical_metric_eligible": bool(physical_metric_eligible),
        "manual_review_required": bool(extra.pop("manual_review_required", False)),
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "provenance_json": extra.pop("provenance_json", None),
        "contract_version": CONTRACT_VERSION,
    }
    if extra:
        raise CalibrationContractError(f"unexpected segment fields: {sorted(extra)}")
    return row


__all__ = [
    "HALF_OPEN",
    "intervals_overlap",
    "validate_segment_interval",
    "terminate_on_shot_cut",
    "apply_camera_motion_policy",
    "camera_view_eligibility",
    "find_segment_overlaps",
    "find_calibration_gaps",
    "assert_no_silent_gap_fill",
    "segment_row",
]
