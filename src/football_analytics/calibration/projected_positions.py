"""Projected positions — mapping status + physical-metric eligibility (no metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.calibration.coordinates import (
    IMAGE_FRAME,
    PITCH_FRAME,
    ball_centre_from_bbox,
    human_footpoint_from_bbox,
    pitch_point_in_bounds,
)
from football_analytics.calibration.homography import apply_homography, matrix_from_row_major
from football_analytics.calibration.types import (
    CONTRACT_VERSION,
    CalibrationContractError,
    MappingStatus,
    ObservationSource,
    PhysicalMetricEligibility,
    SourcePointType,
)


def projection_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    video_time_us: int,
    projection_id: str,
    entity_type: str,
    observation_source: str,
    source_point_type: str,
    image_x: float,
    image_y: float,
    mapping_status: str,
    physical_metric_eligibility: str,
    calibration_quality: str = "unknown",
    pitch_x_m: float | None = None,
    pitch_y_m: float | None = None,
    calibration_id: int | None = None,
    segment_id: str | None = None,
    track_id: int | None = None,
    detection_id: int | None = None,
    in_bounds: bool | None = None,
    is_extrapolated: bool = False,
    uncertainty_m: float | None = None,
    manual_review_required: bool = False,
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
    provenance_json: str | None = None,
    coordinate_frame_id: str = IMAGE_FRAME,
    pitch_coordinate_frame_id: str = PITCH_FRAME,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "video_time_us": int(video_time_us),
        "projection_id": projection_id,
        "entity_type": entity_type,
        "track_id": track_id,
        "detection_id": detection_id,
        "observation_source": observation_source,
        "source_point_type": source_point_type,
        "image_x": float(image_x),
        "image_y": float(image_y),
        "coordinate_frame_id": coordinate_frame_id,
        "pitch_x_m": pitch_x_m,
        "pitch_y_m": pitch_y_m,
        "pitch_coordinate_frame_id": pitch_coordinate_frame_id,
        "calibration_id": calibration_id,
        "segment_id": segment_id,
        "mapping_status": mapping_status,
        "in_bounds": in_bounds,
        "is_extrapolated": bool(is_extrapolated),
        "calibration_quality": calibration_quality,
        "uncertainty_m": uncertainty_m,
        "physical_metric_eligibility": physical_metric_eligibility,
        "manual_review_required": bool(manual_review_required),
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "provenance_json": provenance_json,
        "contract_version": CONTRACT_VERSION,
    }


def compute_physical_metric_eligibility(
    *,
    observation_source: str,
    mapping_status: str,
    is_extrapolated: bool,
    calibration_valid: bool,
    calibration_interpolated: bool = False,
    calibration_uncertain: bool = False,
    in_bounds: bool | None,
    entity_type: str,
    airborne_ball: bool = False,
    identity_confirmed_if_required: bool = True,
    uncertainty_m: float | None = None,
    max_uncertainty_m: float | None = None,
) -> tuple[str, list[str]]:
    """Eligibility rules only — does not compute physical metrics."""
    reasons: list[str] = []
    if observation_source in {
        ObservationSource.PREDICTED.value,
        ObservationSource.INTERPOLATED.value,
        ObservationSource.NOT_OBSERVED.value,
    }:
        reasons.append("PREDICTED_OR_INTERPOLATED_SOURCE")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if not calibration_valid:
        reasons.append("NOT_CALIBRATED")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if calibration_interpolated or calibration_uncertain:
        reasons.append("UNCERTAIN_OR_INTERPOLATED_CALIBRATION")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if is_extrapolated or mapping_status == MappingStatus.EXTRAPOLATED.value:
        reasons.append("EXTRAPOLATED_NOT_METRIC_ELIGIBLE")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if mapping_status in {
        MappingStatus.OUTSIDE_PITCH.value,
        MappingStatus.NOT_CALIBRATED.value,
        MappingStatus.NOT_ELIGIBLE.value,
        MappingStatus.FAILED.value,
        MappingStatus.UNCERTAIN.value,
    }:
        reasons.append(f"MAPPING_STATUS_{mapping_status.upper()}")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if in_bounds is False:
        reasons.append("OUTSIDE_PITCH")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if entity_type == "ball" and airborne_ball:
        reasons.append("AIRBORNE_BALL_NOT_METRIC_ELIGIBLE")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if not identity_confirmed_if_required:
        reasons.append("IDENTITY_NOT_CONFIRMED")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if (
        uncertainty_m is not None
        and max_uncertainty_m is not None
        and uncertainty_m > max_uncertainty_m
    ):
        reasons.append("UNCERTAINTY_ABOVE_THRESHOLD")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    if mapping_status != MappingStatus.MAPPED.value:
        reasons.append("NOT_MAPPED")
        return PhysicalMetricEligibility.NOT_ELIGIBLE.value, reasons
    return PhysicalMetricEligibility.ELIGIBLE.value, reasons


def project_point(
    *,
    image_x: float,
    image_y: float,
    H_row_major: Sequence[float],
    pitch_length_m: float,
    pitch_width_m: float,
    tolerance_m: float = 0.5,
) -> tuple[float, float, str, bool, bool]:
    """Return pitch_x, pitch_y, mapping_status, in_bounds, is_extrapolated."""
    H = matrix_from_row_major(H_row_major)
    mapped = apply_homography(H, [(image_x, image_y)])[0]
    x_m, y_m = float(mapped[0]), float(mapped[1])
    strict = pitch_point_in_bounds(
        x_m, y_m, length_m=pitch_length_m, width_m=pitch_width_m, tolerance_m=0.0
    )
    soft = pitch_point_in_bounds(
        x_m, y_m, length_m=pitch_length_m, width_m=pitch_width_m, tolerance_m=tolerance_m
    )
    if strict:
        return x_m, y_m, MappingStatus.MAPPED.value, True, False
    if soft:
        return x_m, y_m, MappingStatus.EXTRAPOLATED.value, False, True
    return x_m, y_m, MappingStatus.OUTSIDE_PITCH.value, False, True


def source_point_for_entity(
    entity_type: str,
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float, str]:
    if entity_type == "human":
        pt = human_footpoint_from_bbox(bbox_xyxy)
        return pt.x_px, pt.y_px, SourcePointType.BBOX_BOTTOM_CENTRE.value
    if entity_type == "ball":
        pt = ball_centre_from_bbox(bbox_xyxy)
        return pt.x_px, pt.y_px, SourcePointType.BBOX_CENTRE.value
    raise CalibrationContractError(f"unsupported entity_type for source point: {entity_type}")


def validate_projection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("mapping_status") not in {m.value for m in MappingStatus}:
        raise CalibrationContractError("invalid mapping_status")
    if out.get("physical_metric_eligibility") not in {e.value for e in PhysicalMetricEligibility}:
        raise CalibrationContractError("invalid physical_metric_eligibility")
    if out.get("observation_source") not in {o.value for o in ObservationSource}:
        raise CalibrationContractError("invalid observation_source")
    # Enforce predicted/interpolated never eligible.
    if (
        out["observation_source"]
        in {
            ObservationSource.PREDICTED.value,
            ObservationSource.INTERPOLATED.value,
        }
        and out["physical_metric_eligibility"] == PhysicalMetricEligibility.ELIGIBLE.value
    ):
        raise CalibrationContractError("PREDICTED_NOT_METRIC_ELIGIBLE")
    if out.get("is_extrapolated") and out["physical_metric_eligibility"] == (
        PhysicalMetricEligibility.ELIGIBLE.value
    ):
        raise CalibrationContractError("EXTRAPOLATED_NOT_METRIC_ELIGIBLE")
    return out


__all__ = [
    "projection_row",
    "compute_physical_metric_eligibility",
    "project_point",
    "source_point_for_entity",
    "validate_projection_row",
]
