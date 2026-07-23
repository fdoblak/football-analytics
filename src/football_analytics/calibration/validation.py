"""Calibration bundle validation: segments, features, projections, fingerprints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from football_analytics.calibration.features import validate_feature_rows
from football_analytics.calibration.projected_positions import validate_projection_row
from football_analytics.calibration.segments import find_calibration_gaps, find_segment_overlaps
from football_analytics.calibration.types import ValidityStatus


@dataclass
class CalibrationValidationResult:
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = "FAIL"

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _rows(table: Any | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    if isinstance(table, list):
        return [dict(r) for r in table]
    raise TypeError("expected pyarrow.Table or list of mappings")


def validate_calibration_bundle(
    *,
    calibration_features: Any | None = None,
    calibration_segments: Any | None = None,
    projected_positions: Any | None = None,
    calibrations: Any | None = None,
    frames: Any | None = None,
    policy: Mapping[str, Any] | None = None,
    pitch_template_fingerprint: str | None = None,
    receipt: Mapping[str, Any] | None = None,
    timeline_start_us: int | None = None,
    timeline_end_us: int | None = None,
) -> CalibrationValidationResult:
    result = CalibrationValidationResult()
    _ = calibrations

    try:
        features = validate_feature_rows(_rows(calibration_features))
    except Exception as exc:  # noqa: BLE001
        result.err(f"features: {exc}")
        features = _rows(calibration_features)

    segments = _rows(calibration_segments)
    for seg in segments:
        if int(seg.get("end_time_us", 0)) <= int(seg.get("start_time_us", 0)):
            result.err(f"segment {seg.get('segment_id')} invalid interval")
        if pitch_template_fingerprint and seg.get("pitch_template_fingerprint") not in (
            None,
            pitch_template_fingerprint,
        ):
            result.err(f"segment {seg.get('segment_id')} pitch template fingerprint mismatch")
        if seg.get("is_interpolated") and seg.get("physical_metric_eligible"):
            result.err(f"segment {seg.get('segment_id')} interpolated cannot be metric-eligible")
        if (
            seg.get("validity_status") == ValidityStatus.VALID.value
            and seg.get("homography_image_to_pitch") is None
        ):
            result.err(f"segment {seg.get('segment_id')} valid but missing H")

    overlaps = find_segment_overlaps(segments)
    for a, b in overlaps:
        result.err(f"SEGMENT_OVERLAP_CONFLICT: {a} vs {b}")

    if timeline_start_us is not None and timeline_end_us is not None:
        gaps = find_calibration_gaps(
            segments, timeline_start_us=timeline_start_us, timeline_end_us=timeline_end_us
        )
        if gaps and policy is not None:
            segs_pol = policy.get("segments") if isinstance(policy, Mapping) else None
            if isinstance(segs_pol, Mapping) and segs_pol.get("silent_gap_fill") is True:
                result.err("SILENT_GAP_FILL_FORBIDDEN")
            else:
                result.warn(f"calibration gaps (not silently filled): {gaps}")

    projections = _rows(projected_positions)
    for row in projections:
        try:
            validate_projection_row(row)
        except Exception as exc:  # noqa: BLE001
            result.err(f"projection {row.get('projection_id')}: {exc}")

    # Soft FK: features/projections → frames when provided
    frame_keys: set[tuple[str, str, int]] = set()
    for fr in _rows(frames):
        frame_keys.add((str(fr["run_id"]), str(fr["video_id"]), int(fr["frame_index"])))
    if frame_keys:
        for feat in features:
            key = (str(feat["run_id"]), str(feat["video_id"]), int(feat["frame_index"]))
            if key not in frame_keys:
                result.err(f"feature dangling frame FK: {feat.get('feature_id')}")
        for proj in projections:
            key = (str(proj["run_id"]), str(proj["video_id"]), int(proj["frame_index"]))
            if key not in frame_keys:
                result.err(f"projection dangling frame FK: {proj.get('projection_id')}")

    if receipt is not None:
        gt = receipt.get("ground_truth_evaluation_status")
        if gt is None:
            result.err("receipt missing ground_truth_evaluation_status")

    return result


__all__ = [
    "CalibrationValidationResult",
    "validate_calibration_bundle",
]
