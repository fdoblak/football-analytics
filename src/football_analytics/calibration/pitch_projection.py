"""Stage 8D image→pitch projection core (no physical metrics / events)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.calibration.coordinates import (
    IMAGE_FRAME,
    PITCH_FRAME,
    ball_centre_from_bbox,
    human_footpoint_from_bbox,
    pitch_point_in_bounds,
)
from football_analytics.calibration.homography import (
    HomographyError,
    apply_homography,
    invert_homography,
    matrix_from_row_major,
)
from football_analytics.calibration.projected_positions import (
    compute_physical_metric_eligibility,
    projection_row,
    validate_projection_row,
)
from football_analytics.calibration.segments import intervals_overlap
from football_analytics.calibration.types import (
    CalibrationContractError,
    MappingStatus,
    ObservationSource,
    PhysicalMetricEligibility,
    SourcePointType,
    ValidityStatus,
)


class PitchProjectionError(CalibrationContractError):
    """Pitch projection pipeline failure."""


@dataclass(frozen=True)
class SegmentSelection:
    status: str  # ok | not_calibrated | conflict | invalid_matrix
    segment: Mapping[str, Any] | None
    reason_codes: tuple[str, ...]
    conflict_segment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourcePointResult:
    ok: bool
    image_x: float | None
    image_y: float | None
    source_point_type: str
    truncated: bool
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionGeometry:
    mapping_status: str
    pitch_x_m: float | None
    pitch_y_m: float | None
    in_bounds: bool | None
    is_extrapolated: bool
    round_trip_error_px: float | None
    round_trip_error_m: float | None
    outside_coverage: bool
    reason_codes: tuple[str, ...]
    w: float | None = None


def observation_source_from_state(state: str) -> str:
    if state == "observed":
        return ObservationSource.DETECTION_ASSOCIATED.value
    if state == "predicted":
        return ObservationSource.PREDICTED.value
    if state == "interpolated":
        return ObservationSource.INTERPOLATED.value
    if state == "not_observed":
        return ObservationSource.NOT_OBSERVED.value
    if state == "synthetic":
        return ObservationSource.SYNTHETIC.value
    raise PitchProjectionError(f"unsupported observation_state: {state}")


def entity_type_from_observation(row: Mapping[str, Any]) -> str:
    if "entity_type" in row and row["entity_type"] is not None:
        et = str(row["entity_type"])
        if et in {"human", "ball"}:
            return et
    class_id = int(row.get("class_id", -1))
    # COCO-ish: 0 person, 32 sports ball (tracking fixtures).
    if class_id == 0:
        return "human"
    if class_id == 32:
        return "ball"
    raise PitchProjectionError(f"cannot resolve entity_type from observation class_id={class_id}")


def _point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Ray casting; boundary counts as inside."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        # On-edge check (approx).
        on_line = abs((yi - yj) * (x - xi) - (xi - xj) * (y - yi)) < 1e-6
        in_bbox = (
            min(xi, xj) - 1e-6 <= x <= max(xi, xj) + 1e-6
            and min(yi, yj) - 1e-6 <= y <= max(yi, yj) + 1e-6
        )
        if on_line and in_bbox:
            return True
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _convex_hull(pts: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return [(float(p[0]), float(p[1])) for p in pts]
    p = arr[np.lexsort((arr[:, 1], arr[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: list[np.ndarray] = []
    for pt in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    return [(float(h[0]), float(h[1])) for h in hull]


def coverage_distance_px(x: float, y: float, hull: Sequence[Sequence[float]]) -> tuple[bool, float]:
    """Return (inside, distance_outside_px). Distance 0 when inside."""
    if len(hull) < 3:
        return False, float("inf")
    if _point_in_polygon(x, y, hull):
        return True, 0.0
    # Distance to nearest edge.
    best = float("inf")
    n = len(hull)
    for i in range(n):
        x1, y1 = float(hull[i][0]), float(hull[i][1])
        x2, y2 = float(hull[(i + 1) % n][0]), float(hull[(i + 1) % n][1])
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            d = math.hypot(x - x1, y - y1)
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        best = min(best, d)
    return False, best


def select_segment_for_time(
    segments: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    video_id: str,
    video_time_us: int,
    config: Mapping[str, Any],
    pitch_template_fingerprint: str | None = None,
) -> SegmentSelection:
    """Select unique valid physical-eligible segment covering video_time_us."""
    covering: list[dict[str, Any]] = []
    for s in segments:
        if str(s.get("run_id")) != run_id or str(s.get("video_id")) != video_id:
            continue
        start = int(s["start_time_us"])
        end = int(s["end_time_us"])
        if not (start <= video_time_us < end):
            continue
        covering.append(dict(s))

    if not covering:
        return SegmentSelection(
            status="not_calibrated",
            segment=None,
            reason_codes=("NOT_CALIBRATED", "NO_SEGMENT_COVERING_TIME"),
        )

    # Hard conflict among physical-eligible valid segments.
    physical: list[dict[str, Any]] = []
    for s in covering:
        if s.get("validity_status") != ValidityStatus.VALID.value:
            continue
        if config["segment_selection"]["require_physical_metric_eligible"] and not s.get(
            "physical_metric_eligible"
        ):
            continue
        if not config["segment_selection"]["allow_interpolated"] and s.get("is_interpolated"):
            continue
        if pitch_template_fingerprint is not None and str(
            s.get("pitch_template_fingerprint")
        ) != str(pitch_template_fingerprint):
            continue
        physical.append(s)

    if len(physical) > 1 and config["segment_selection"]["overlap_is_hard_conflict"]:
        ids = tuple(sorted(str(s["segment_id"]) for s in physical))
        return SegmentSelection(
            status="conflict",
            segment=None,
            reason_codes=("SEGMENT_OVERLAP_CONFLICT",),
            conflict_segment_ids=ids,
        )

    if len(physical) == 1:
        seg = physical[0]
        H = seg.get("homography_image_to_pitch")
        if H is None or len(list(H)) != 9:
            return SegmentSelection(
                status="invalid_matrix",
                segment=seg,
                reason_codes=("SINGULAR_HOMOGRAPHY", "MISSING_IMAGE_TO_PITCH"),
            )
        return SegmentSelection(status="ok", segment=seg, reason_codes=())

    # Covered only by non-physical segments (degraded/uncertain/etc.).
    return SegmentSelection(
        status="not_calibrated",
        segment=None,
        reason_codes=("NOT_CALIBRATED", "NO_PHYSICAL_ELIGIBLE_SEGMENT"),
    )


def extract_source_point(
    *,
    entity_type: str,
    bbox_xyxy: tuple[float, float, float, float],
    observation_source: str,
    config: Mapping[str, Any],
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> SourcePointResult:
    reasons: list[str] = []
    flags: list[str] = []
    src_cfg = config["human_source"] if entity_type == "human" else config["ball_source"]
    if src_cfg["require_observed"] and observation_source in {
        ObservationSource.PREDICTED.value,
        ObservationSource.INTERPOLATED.value,
        ObservationSource.NOT_OBSERVED.value,
    }:
        # Still compute geometry for row writing, but mark not ok for eligibility path.
        reasons.append("PREDICTED_OR_INTERPOLATED_SOURCE")

    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or w <= 0 or h <= 0:
        return SourcePointResult(
            ok=False,
            image_x=None,
            image_y=None,
            source_point_type=(
                SourcePointType.BBOX_BOTTOM_CENTRE.value
                if entity_type == "human"
                else SourcePointType.BBOX_CENTRE.value
            ),
            truncated=False,
            reason_codes=("INVALID_BBOX",),
            quality_flags=("invalid_bbox",),
        )
    if w < float(src_cfg["min_bbox_width_px"]) or h < float(src_cfg["min_bbox_height_px"]):
        return SourcePointResult(
            ok=False,
            image_x=None,
            image_y=None,
            source_point_type=(
                SourcePointType.BBOX_BOTTOM_CENTRE.value
                if entity_type == "human"
                else SourcePointType.BBOX_CENTRE.value
            ),
            truncated=False,
            reason_codes=("BBOX_TOO_SMALL",),
            quality_flags=("bbox_too_small",),
        )
    aspect = max(w / h, h / w)
    if aspect > float(src_cfg["max_aspect_ratio"]):
        return SourcePointResult(
            ok=False,
            image_x=None,
            image_y=None,
            source_point_type=(
                SourcePointType.BBOX_BOTTOM_CENTRE.value
                if entity_type == "human"
                else SourcePointType.BBOX_CENTRE.value
            ),
            truncated=False,
            reason_codes=("BBOX_ASPECT_REJECTED",),
            quality_flags=("bbox_aspect_rejected",),
        )

    if entity_type == "human":
        pt = human_footpoint_from_bbox(bbox_xyxy)
        sp = SourcePointType.BBOX_BOTTOM_CENTRE.value
        flags.append("footpoint_approx_no_pose_model")
    elif entity_type == "ball":
        pt = ball_centre_from_bbox(bbox_xyxy)
        sp = SourcePointType.BBOX_CENTRE.value
        flags.append("ball_centre_image_plane_only")
        flags.append("airborne_status_unknown")
    else:
        raise PitchProjectionError(f"unsupported entity_type: {entity_type}")

    margin = float(src_cfg["frame_edge_margin_px"])
    truncated = False
    if (
        frame_width is not None
        and frame_height is not None
        and (
            x1 < margin
            or y1 < margin
            or x2 > float(frame_width) - margin
            or y2 > float(frame_height) - margin
        )
    ):
        truncated = True
        flags.append("frame_edge_truncated")
        reasons.append("FRAME_EDGE_TRUNCATION")

    return SourcePointResult(
        ok=True,
        image_x=float(pt.x_px),
        image_y=float(pt.y_px),
        source_point_type=sp,
        truncated=truncated,
        reason_codes=tuple(reasons),
        quality_flags=tuple(flags),
    )


def apply_image_to_pitch_projection(
    *,
    image_x: float,
    image_y: float,
    H_row_major: Sequence[float],
    H_inv_row_major: Sequence[float] | None,
    pitch_length_m: float,
    pitch_width_m: float,
    config: Mapping[str, Any],
    coverage_hull_image: Sequence[Sequence[float]] | None = None,
) -> ProjectionGeometry:
    """Project with image_to_pitch only; never use H_inv for the primary map."""
    reasons: list[str] = []
    pcfg = config["projection"]
    if not pcfg["use_image_to_pitch_only"] or not pcfg["forbid_h_inv_for_projection"]:
        raise PitchProjectionError("image_to_pitch-only policy violated")

    H = matrix_from_row_major(H_row_major)
    # Homogeneous w check before dehomogenization.
    vec = H @ np.array([image_x, image_y, 1.0], dtype=np.float64)
    w = float(vec[2])
    w_eps = float(pcfg["homogeneous_w_epsilon"])
    if not math.isfinite(w) or abs(w) < w_eps:
        return ProjectionGeometry(
            mapping_status=MappingStatus.FAILED.value,
            pitch_x_m=None,
            pitch_y_m=None,
            in_bounds=None,
            is_extrapolated=False,
            round_trip_error_px=None,
            round_trip_error_m=None,
            outside_coverage=False,
            reason_codes=("HOMOGENEOUS_W_SINGULARITY",),
            w=w if math.isfinite(w) else None,
        )
    try:
        mapped = apply_homography(H, [(image_x, image_y)])[0]
    except HomographyError as exc:
        return ProjectionGeometry(
            mapping_status=MappingStatus.FAILED.value,
            pitch_x_m=None,
            pitch_y_m=None,
            in_bounds=None,
            is_extrapolated=False,
            round_trip_error_px=None,
            round_trip_error_m=None,
            outside_coverage=False,
            reason_codes=("HOMOGRAPHY_APPLY_FAILED", str(exc)[:64]),
            w=w,
        )
    x_m, y_m = float(mapped[0]), float(mapped[1])
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        return ProjectionGeometry(
            mapping_status=MappingStatus.FAILED.value,
            pitch_x_m=None,
            pitch_y_m=None,
            in_bounds=None,
            is_extrapolated=False,
            round_trip_error_px=None,
            round_trip_error_m=None,
            outside_coverage=False,
            reason_codes=("NON_FINITE_PITCH_COORDS",),
            w=w,
        )
    if pcfg["clamp_pitch_coordinates"]:
        raise PitchProjectionError("pitch clamp forbidden")

    # Round-trip via inverse (validation only — not the projection path).
    rt_px: float | None = None
    rt_m: float | None = None
    try:
        if H_inv_row_major is not None and len(list(H_inv_row_major)) == 9:
            H_inv = matrix_from_row_major(H_inv_row_major)
        else:
            H_inv = invert_homography(H)
        back = apply_homography(H_inv, [(x_m, y_m)])[0]
        rt_px = float(math.hypot(float(back[0]) - image_x, float(back[1]) - image_y))
        # Pitch-space round trip: image→pitch→image→pitch.
        again = apply_homography(H, [(float(back[0]), float(back[1]))])[0]
        rt_m = float(math.hypot(float(again[0]) - x_m, float(again[1]) - y_m))
        if rt_px > float(pcfg["round_trip_tolerance_px"]) or rt_m > float(
            pcfg["round_trip_tolerance_m"]
        ):
            reasons.append("ROUND_TRIP_FAILURE")
            return ProjectionGeometry(
                mapping_status=MappingStatus.FAILED.value,
                pitch_x_m=x_m,
                pitch_y_m=y_m,
                in_bounds=None,
                is_extrapolated=False,
                round_trip_error_px=rt_px,
                round_trip_error_m=rt_m,
                outside_coverage=False,
                reason_codes=tuple(reasons),
                w=w,
            )
    except HomographyError:
        reasons.append("ROUND_TRIP_FAILURE")
        return ProjectionGeometry(
            mapping_status=MappingStatus.FAILED.value,
            pitch_x_m=x_m,
            pitch_y_m=y_m,
            in_bounds=None,
            is_extrapolated=False,
            round_trip_error_px=None,
            round_trip_error_m=None,
            outside_coverage=False,
            reason_codes=tuple(reasons),
            w=w,
        )

    outside_coverage = False
    if coverage_hull_image is not None and len(coverage_hull_image) >= 3:
        hull = _convex_hull(coverage_hull_image)
        inside, _dist = coverage_distance_px(image_x, image_y, hull)
        if not inside:
            outside_coverage = True
            reasons.append("OUTSIDE_COVERAGE_HULL")

    tol = float(pcfg["pitch_bound_tolerance_m"])
    strict = pitch_point_in_bounds(
        x_m, y_m, length_m=pitch_length_m, width_m=pitch_width_m, tolerance_m=0.0
    )
    soft = pitch_point_in_bounds(
        x_m, y_m, length_m=pitch_length_m, width_m=pitch_width_m, tolerance_m=tol
    )

    if outside_coverage and pcfg["outside_coverage_is_extrapolated"]:
        # Even if pitch-in-bounds, coverage outside → extrapolated.
        return ProjectionGeometry(
            mapping_status=MappingStatus.EXTRAPOLATED.value,
            pitch_x_m=x_m,
            pitch_y_m=y_m,
            in_bounds=strict,
            is_extrapolated=True,
            round_trip_error_px=rt_px,
            round_trip_error_m=rt_m,
            outside_coverage=True,
            reason_codes=tuple(reasons),
            w=w,
        )
    if strict:
        return ProjectionGeometry(
            mapping_status=MappingStatus.MAPPED.value,
            pitch_x_m=x_m,
            pitch_y_m=y_m,
            in_bounds=True,
            is_extrapolated=False,
            round_trip_error_px=rt_px,
            round_trip_error_m=rt_m,
            outside_coverage=False,
            reason_codes=tuple(reasons),
            w=w,
        )
    if soft:
        reasons.append("PITCH_SOFT_BOUNDS")
        return ProjectionGeometry(
            mapping_status=MappingStatus.EXTRAPOLATED.value,
            pitch_x_m=x_m,
            pitch_y_m=y_m,
            in_bounds=False,
            is_extrapolated=True,
            round_trip_error_px=rt_px,
            round_trip_error_m=rt_m,
            outside_coverage=outside_coverage,
            reason_codes=tuple(reasons),
            w=w,
        )
    reasons.append("OUTSIDE_PITCH")
    return ProjectionGeometry(
        mapping_status=MappingStatus.OUTSIDE_PITCH.value,
        pitch_x_m=x_m,
        pitch_y_m=y_m,
        in_bounds=False,
        is_extrapolated=True,
        round_trip_error_px=rt_px,
        round_trip_error_m=rt_m,
        outside_coverage=outside_coverage,
        reason_codes=tuple(reasons),
        w=w,
    )


def estimate_uncertainty_m(
    *,
    segment: Mapping[str, Any] | None,
    geometry: ProjectionGeometry | None,
    truncated: bool,
    ambiguous_ball: bool,
    config: Mapping[str, Any],
    coverage_dist_px: float | None = None,
) -> tuple[float | None, list[str]]:
    unc = config["uncertainty"]
    reasons: list[str] = []
    if segment is None or geometry is None:
        return None, [str(unc["unknown_null_reason"])]
    parts: list[float] = []
    reproj = segment.get("mean_reprojection_error_px")
    if reproj is not None and math.isfinite(float(reproj)):
        parts.append(float(reproj) * float(unc["base_from_reprojection_scale"]))
    if geometry.round_trip_error_m is not None and math.isfinite(geometry.round_trip_error_m):
        parts.append(float(geometry.round_trip_error_m))
    if coverage_dist_px is not None and math.isfinite(coverage_dist_px) and coverage_dist_px > 0:
        parts.append(coverage_dist_px * float(unc["coverage_distance_scale_m"]))
    if truncated:
        parts.append(float(unc["truncation_boost_m"]))
        reasons.append("TRUNCATION_UNCERTAINTY")
    if ambiguous_ball:
        parts.append(float(unc["ambiguity_boost_m"]))
        reasons.append("BALL_AMBIGUITY_UNCERTAINTY")
    if not parts:
        return None, [str(unc["unknown_null_reason"])]
    value = float(sum(parts))
    if not math.isfinite(value) or value < 0:
        return None, [str(unc["unknown_null_reason"])]
    return value, reasons


def target_customer_metric_eligible(
    *,
    track_id: int | None,
    frame_index: int,
    eligibility_timeline: Mapping[str, Any] | None,
    human_physical_eligible: bool,
    config: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Customer physical-metric eligibility requires confirmed target interval + geometry."""
    reasons: list[str] = []
    if not human_physical_eligible:
        return False, ["HUMAN_NOT_PHYSICAL_ELIGIBLE"]
    if not config["eligibility"]["require_confirmed_target_for_customer_metric"]:
        return human_physical_eligible, ["CONFIRMED_TARGET_NOT_REQUIRED"]
    if eligibility_timeline is None:
        return False, ["NO_IDENTITY_ELIGIBILITY_TIMELINE"]
    if track_id is None:
        return False, ["NO_TRACK_ID"]
    matched = False
    for iv in eligibility_timeline.get("intervals") or []:
        if int(iv["track_id"]) != int(track_id):
            continue
        if not (int(iv["start_frame_index"]) <= frame_index <= int(iv["end_frame_index"])):
            continue
        matched = True
        status = str(iv.get("assignment_status", ""))
        elig = str(iv.get("eligibility", ""))
        if status == "revoked" or elig == "not_eligible":
            return False, ["REVOKED_OR_NOT_ELIGIBLE_TARGET"]
        if (status == "provisional" or elig == "provisional_only") and not config["eligibility"][
            "provisional_target_customer_metric_eligible"
        ]:
            return False, ["PROVISIONAL_TARGET_NOT_CUSTOMER_ELIGIBLE"]
        if elig == "eligible" and status == "confirmed":
            return True, ["CONFIRMED_TARGET_INTERVAL"]
        return False, [f"TARGET_ELIGIBILITY_{elig.upper()}"]
    if not matched:
        return False, ["NO_TARGET_INTERVAL_FOR_TRACK_FRAME"]
    return False, reasons


def is_playable_at_time(
    analysis_windows: Sequence[Mapping[str, Any]] | None,
    *,
    video_time_us: int,
    require: bool,
) -> tuple[bool, list[str]]:
    if not require:
        return True, []
    if not analysis_windows:
        # No windows provided → treat as playable for synthetic fixtures.
        return True, ["NO_ANALYSIS_WINDOWS_ASSUME_PLAYABLE"]
    for w in analysis_windows:
        start = int(w.get("start_time_us", w.get("start_frame_index", 0)))
        end = int(w.get("end_time_us", w.get("end_frame_index", start) + 1))
        # Support both time-us and frame-index style windows.
        if "start_time_us" in w:
            if not (start <= video_time_us < end):
                continue
        else:
            # Caller should pass frame-based check separately; skip time mismatch.
            continue
        playability = str(w.get("playability_status", w.get("playability", "playable")))
        replay = str(w.get("replay_status", "live"))
        if replay in {"replay", "replay_transition"}:
            return False, ["REPLAY_WINDOW_NOT_PLAYABLE"]
        if playability in {"playable", "partially_playable"}:
            return True, []
        return False, [f"WINDOW_NOT_PLAYABLE_{playability.upper()}"]
    return False, ["NO_COVERING_ANALYSIS_WINDOW"]


def is_playable_at_frame(
    analysis_windows: Sequence[Mapping[str, Any]] | None,
    *,
    frame_index: int,
    require: bool,
) -> tuple[bool, list[str]]:
    if not require:
        return True, []
    if not analysis_windows:
        return True, ["NO_ANALYSIS_WINDOWS_ASSUME_PLAYABLE"]
    for w in analysis_windows:
        if "start_frame_index" not in w:
            continue
        start = int(w["start_frame_index"])
        end = int(w.get("end_frame_index", start))
        if not (start <= frame_index <= end):
            continue
        playability = str(w.get("playability_status", w.get("playability", "playable")))
        replay = str(w.get("replay_status", "live"))
        camera = str(w.get("camera_view", "main"))
        if replay in {"replay", "replay_transition"} or camera == "replay":
            return False, ["REPLAY_WINDOW_NOT_PLAYABLE"]
        if playability in {"playable", "partially_playable"}:
            return True, []
        return False, [f"WINDOW_NOT_PLAYABLE_{playability.upper()}"]
    return True, ["NO_FRAME_WINDOW_ASSUME_PLAYABLE"]


def calibration_quality_label(segment: Mapping[str, Any] | None) -> str:
    if segment is None:
        return "missing"
    vs = str(segment.get("validity_status", "unknown"))
    if vs == "valid":
        return "good"
    if vs == "uncertain":
        return "poor"
    if vs in {"invalid", "conflict"}:
        return "invalid"
    if vs == "not_eligible":
        return "marginal"
    return "unknown"


def build_projection_for_observation(
    *,
    observation: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    run_id: str,
    video_id: str,
    video_time_us: int,
    projection_id: str,
    pitch_template_fingerprint: str,
    coverage_hulls: Mapping[str, Sequence[Sequence[float]]] | None = None,
    eligibility_timeline: Mapping[str, Any] | None = None,
    analysis_windows: Sequence[Mapping[str, Any]] | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
    ambiguous_ball: bool = False,
    force_conflict_fail: bool = True,
) -> dict[str, Any]:
    """Build one projected_positions row (+ sidecar eligibility fields in provenance)."""
    entity = entity_type_from_observation(observation)
    obs_state = str(observation.get("observation_state", "observed"))
    obs_source = observation_source_from_state(obs_state)
    bbox = (
        float(observation["bbox_x1"]),
        float(observation["bbox_y1"]),
        float(observation["bbox_x2"]),
        float(observation["bbox_y2"]),
    )
    src = extract_source_point(
        entity_type=entity,
        bbox_xyxy=bbox,
        observation_source=obs_source,
        config=config,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    reasons: list[str] = list(src.reason_codes)
    flags: list[str] = list(src.quality_flags)
    track_id = observation.get("track_id")
    detection_id = observation.get("detection_id")
    frame_index = int(observation["frame_index"])

    sel = select_segment_for_time(
        segments,
        run_id=run_id,
        video_id=video_id,
        video_time_us=video_time_us,
        config=config,
        pitch_template_fingerprint=pitch_template_fingerprint,
    )
    if sel.status == "conflict":
        if force_conflict_fail:
            raise PitchProjectionError(
                f"SEGMENT_OVERLAP_CONFLICT:{','.join(sel.conflict_segment_ids)}"
            )
        reasons.extend(sel.reason_codes)
        row = projection_row(
            run_id=run_id,
            video_id=video_id,
            frame_index=frame_index,
            video_time_us=video_time_us,
            projection_id=projection_id,
            entity_type=entity,
            observation_source=obs_source,
            source_point_type=src.source_point_type,
            image_x=float(src.image_x or 0.0),
            image_y=float(src.image_y or 0.0),
            mapping_status=MappingStatus.FAILED.value,
            physical_metric_eligibility=PhysicalMetricEligibility.NOT_ELIGIBLE.value,
            calibration_quality="invalid",
            track_id=int(track_id) if track_id is not None else None,
            detection_id=int(detection_id) if detection_id is not None else None,
            manual_review_required=True,
            reason_codes=reasons + ["SEGMENT_OVERLAP_CONFLICT"],
            quality_flags=flags + ["segment_conflict"],
        )
        return validate_projection_row(row)

    if not src.ok or src.image_x is None or src.image_y is None:
        row = projection_row(
            run_id=run_id,
            video_id=video_id,
            frame_index=frame_index,
            video_time_us=video_time_us,
            projection_id=projection_id,
            entity_type=entity,
            observation_source=obs_source,
            source_point_type=src.source_point_type,
            image_x=0.0,
            image_y=0.0,
            mapping_status=MappingStatus.FAILED.value,
            physical_metric_eligibility=PhysicalMetricEligibility.NOT_ELIGIBLE.value,
            calibration_quality=calibration_quality_label(sel.segment),
            track_id=int(track_id) if track_id is not None else None,
            detection_id=int(detection_id) if detection_id is not None else None,
            reason_codes=reasons or ["SOURCE_POINT_FAILED"],
            quality_flags=flags,
        )
        return validate_projection_row(row)

    if sel.status != "ok" or sel.segment is None:
        reasons.extend(sel.reason_codes)
        row = projection_row(
            run_id=run_id,
            video_id=video_id,
            frame_index=frame_index,
            video_time_us=video_time_us,
            projection_id=projection_id,
            entity_type=entity,
            observation_source=obs_source,
            source_point_type=src.source_point_type,
            image_x=float(src.image_x),
            image_y=float(src.image_y),
            mapping_status=MappingStatus.NOT_CALIBRATED.value,
            physical_metric_eligibility=PhysicalMetricEligibility.NOT_ELIGIBLE.value,
            calibration_quality="missing",
            track_id=int(track_id) if track_id is not None else None,
            detection_id=int(detection_id) if detection_id is not None else None,
            reason_codes=reasons,
            quality_flags=flags,
        )
        return validate_projection_row(row)

    seg = sel.segment
    hull = None
    if coverage_hulls is not None:
        hull = coverage_hulls.get(str(seg["segment_id"]))
    geom = apply_image_to_pitch_projection(
        image_x=float(src.image_x),
        image_y=float(src.image_y),
        H_row_major=list(seg["homography_image_to_pitch"]),
        H_inv_row_major=(
            list(seg["homography_pitch_to_image"])
            if seg.get("homography_pitch_to_image") is not None
            else None
        ),
        pitch_length_m=float(seg["pitch_length_m"]),
        pitch_width_m=float(seg["pitch_width_m"]),
        config=config,
        coverage_hull_image=hull,
    )
    reasons.extend(geom.reason_codes)

    cov_dist = None
    if hull is not None and len(hull) >= 3:
        _inside, cov_dist = coverage_distance_px(float(src.image_x), float(src.image_y), hull)
        cov_dist = cov_dist if not _inside else 0.0

    unc_m, unc_reasons = estimate_uncertainty_m(
        segment=seg,
        geometry=geom,
        truncated=src.truncated,
        ambiguous_ball=ambiguous_ball and entity == "ball",
        config=config,
        coverage_dist_px=cov_dist,
    )
    reasons.extend(unc_reasons)

    mapping_status = geom.mapping_status
    if (
        unc_m is not None
        and unc_m > float(config["uncertainty"]["max_uncertainty_m"])
        and mapping_status == MappingStatus.MAPPED.value
    ):
        mapping_status = MappingStatus.UNCERTAIN.value
        reasons.append("UNCERTAINTY_ABOVE_THRESHOLD")
    if ambiguous_ball and entity == "ball":
        reasons.append("BALL_AMBIGUOUS_PRIMARY")
        flags.append("ambiguous_primary_ball")
        if mapping_status == MappingStatus.MAPPED.value:
            mapping_status = MappingStatus.UNCERTAIN.value

    playable, play_reasons = is_playable_at_frame(
        analysis_windows,
        frame_index=frame_index,
        require=bool(config["eligibility"]["require_playable_window"]),
    )
    reasons.extend(play_reasons)

    # Ball: always not physical / not event eligible.
    event_metric_eligible = False
    if entity == "ball":
        phys = PhysicalMetricEligibility.NOT_ELIGIBLE.value
        reasons.append("BALL_NOT_PHYSICAL_METRIC_ELIGIBLE")
        reasons.append("BALL_NOT_EVENT_METRIC_ELIGIBLE")
        reasons.append("AIRBORNE_STATUS_UNKNOWN")
        flags.append("ball_physical_metric_eligible=false")
        flags.append("ball_event_metric_eligible=false")
    else:
        phys, phys_reasons = compute_physical_metric_eligibility(
            observation_source=obs_source,
            mapping_status=mapping_status,
            is_extrapolated=geom.is_extrapolated,
            calibration_valid=bool(seg.get("physical_metric_eligible"))
            and str(seg.get("validity_status")) == ValidityStatus.VALID.value,
            calibration_interpolated=bool(seg.get("is_interpolated")),
            calibration_uncertain=False,
            in_bounds=geom.in_bounds,
            entity_type=entity,
            airborne_ball=False,
            identity_confirmed_if_required=True,
            uncertainty_m=unc_m,
            max_uncertainty_m=float(config["uncertainty"]["max_uncertainty_m"]),
        )
        reasons.extend(phys_reasons)
        if not playable and phys == PhysicalMetricEligibility.ELIGIBLE.value:
            phys = PhysicalMetricEligibility.NOT_ELIGIBLE.value
            reasons.append("NOT_PLAYABLE_WINDOW")

    human_phys_ok = entity == "human" and phys == PhysicalMetricEligibility.ELIGIBLE.value
    target_ok, target_reasons = target_customer_metric_eligible(
        track_id=int(track_id) if track_id is not None else None,
        frame_index=frame_index,
        eligibility_timeline=eligibility_timeline,
        human_physical_eligible=human_phys_ok,
        config=config,
    )
    reasons.extend(target_reasons)
    if target_ok:
        flags.append("target_customer_metric_eligible")
    else:
        flags.append("target_customer_metric_ineligible")

    provenance = {
        "stage": "8D",
        "direction": "image_to_pitch",
        "used_h_inv_for_projection": False,
        "pose_foot_model": False,
        "airborne_status": "unknown" if entity == "ball" else None,
        "event_metric_eligible": bool(event_metric_eligible),
        "target_customer_metric_eligible": bool(target_ok),
        "round_trip_error_px": geom.round_trip_error_px,
        "round_trip_error_m": geom.round_trip_error_m,
        "outside_coverage": geom.outside_coverage,
        "homogeneous_w": geom.w,
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "segment_id": str(seg["segment_id"]),
        "no_trajectory_smoothing": True,
        "no_physical_metrics": True,
        "no_events": True,
        "attack_direction": "unknown",
    }
    row = projection_row(
        run_id=run_id,
        video_id=video_id,
        frame_index=frame_index,
        video_time_us=video_time_us,
        projection_id=projection_id,
        entity_type=entity,
        observation_source=obs_source,
        source_point_type=src.source_point_type,
        image_x=float(src.image_x),
        image_y=float(src.image_y),
        mapping_status=mapping_status,
        physical_metric_eligibility=phys,
        calibration_quality=calibration_quality_label(seg),
        pitch_x_m=geom.pitch_x_m,
        pitch_y_m=geom.pitch_y_m,
        calibration_id=int(seg["calibration_id"]),
        segment_id=str(seg["segment_id"]),
        track_id=int(track_id) if track_id is not None else None,
        detection_id=int(detection_id) if detection_id is not None else None,
        in_bounds=geom.in_bounds,
        is_extrapolated=geom.is_extrapolated,
        uncertainty_m=unc_m,
        manual_review_required=bool(
            mapping_status
            in {
                MappingStatus.UNCERTAIN.value,
                MappingStatus.FAILED.value,
                MappingStatus.EXTRAPOLATED.value,
            }
            or ambiguous_ball
            or src.truncated
        ),
        reason_codes=sorted(set(reasons)),
        quality_flags=sorted(set(flags)),
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        coordinate_frame_id=IMAGE_FRAME,
        pitch_coordinate_frame_id=PITCH_FRAME,
    )
    return validate_projection_row(row)


def assert_fingerprints_aligned(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    for k in keys:
        if k not in expected:
            continue
        if str(expected[k]) != str(actual.get(k)):
            raise PitchProjectionError(f"FINGERPRINT_MISMATCH:{k}")


def find_duplicate_projection_keys(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    dups: list[tuple[Any, ...]] = []
    for r in rows:
        key = (
            str(r["run_id"]),
            str(r["video_id"]),
            int(r["frame_index"]),
            str(r["projection_id"]),
        )
        if key in seen:
            dups.append(key)
        seen.add(key)
    return dups


def covering_segment_conflicts(
    segments: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    video_id: str,
) -> list[tuple[str, str]]:
    physical = [
        s
        for s in segments
        if str(s.get("run_id")) == run_id
        and str(s.get("video_id")) == video_id
        and s.get("validity_status") == ValidityStatus.VALID.value
        and s.get("physical_metric_eligible")
        and not s.get("is_interpolated")
    ]
    conflicts: list[tuple[str, str]] = []
    for i in range(len(physical)):
        for j in range(i + 1, len(physical)):
            a, b = physical[i], physical[j]
            if intervals_overlap(
                int(a["start_time_us"]),
                int(a["end_time_us"]),
                int(b["start_time_us"]),
                int(b["end_time_us"]),
            ):
                conflicts.append((str(a["segment_id"]), str(b["segment_id"])))
    return conflicts


__all__ = [
    "PitchProjectionError",
    "SegmentSelection",
    "SourcePointResult",
    "ProjectionGeometry",
    "observation_source_from_state",
    "entity_type_from_observation",
    "select_segment_for_time",
    "extract_source_point",
    "apply_image_to_pitch_projection",
    "estimate_uncertainty_m",
    "target_customer_metric_eligible",
    "is_playable_at_time",
    "is_playable_at_frame",
    "calibration_quality_label",
    "build_projection_for_observation",
    "assert_fingerprints_aligned",
    "find_duplicate_projection_keys",
    "covering_segment_conflicts",
    "coverage_distance_px",
]
