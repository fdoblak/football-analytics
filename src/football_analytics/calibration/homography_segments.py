"""Calibration segment builder: shot-cut / drift / gap / medoid (Stage 8C)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.calibration.homography import (
    apply_homography,
    invert_homography,
    matrix_from_row_major,
)
from football_analytics.calibration.homography_solve import HomographyQuality
from football_analytics.calibration.pitch_template import PitchTemplate
from football_analytics.calibration.segments import (
    find_calibration_gaps,
    find_segment_overlaps,
    segment_row,
    terminate_on_shot_cut,
)
from football_analytics.calibration.types import CalibrationContractError, ValidityStatus

# Canonical pitch test points for medoid / drift (metres).
DEFAULT_TEST_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (105.0, 0.0),
    (105.0, 68.0),
    (0.0, 68.0),
    (52.5, 34.0),
    (11.0, 34.0),
    (94.0, 34.0),
)


@dataclass(frozen=True)
class FrameCalibrationCandidate:
    frame_index: int
    video_time_us: int
    calibration_id: int
    quality: str
    H_row_major: tuple[float, ...] | None
    H_inv_row_major: tuple[float, ...] | None
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float | None
    mean_reprojection_error_px: float | None
    condition_number: float | None
    determinant: float | None
    coverage_hull_fraction: float | None
    solver_method: str
    solver_version: str
    physical_mapping_eligible: bool
    shot_segment_id: str | None = None
    analysis_window_id: str | None = None
    camera_view: str = "main"
    camera_motion: str = "static"
    reason_codes: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()


@dataclass
class SegmentBuildResult:
    segments: list[dict[str, Any]]
    gaps: list[tuple[int, int]]
    overlaps: list[tuple[str, str]]
    stats: dict[str, int]
    review_required: list[str]


def pitch_test_points(template: PitchTemplate | None = None) -> list[tuple[float, float]]:
    if template is None:
        return list(DEFAULT_TEST_POINTS)
    L, W = float(template.length_m), float(template.width_m)
    return [
        (0.0, 0.0),
        (L, 0.0),
        (L, W),
        (0.0, W),
        (L / 2.0, W / 2.0),
        (11.0, W / 2.0),
        (L - 11.0, W / 2.0),
    ]


def row_major_with_inverse(
    H_row_major: Sequence[float] | None,
    H_inv_row_major: Sequence[float] | None = None,
) -> tuple[list[float] | None, list[float] | None]:
    """Return (H, H_inv) as 9-float lists.

    When H is present, always fill a valid inverse (never None into Arrow
    fixed_size_list fields — cast fails on null list values). Prefer a
    provided inverse; otherwise compute via invert_homography.
    """
    if H_row_major is None:
        return None, None
    h_list = [float(x) for x in H_row_major]
    if len(h_list) != 9:
        raise CalibrationContractError("homography must have 9 values")
    if H_inv_row_major is not None:
        inv_list = [float(x) for x in H_inv_row_major]
        if len(inv_list) != 9:
            raise CalibrationContractError("inverse homography must have 9 values")
        return h_list, inv_list
    H_inv = invert_homography(matrix_from_row_major(h_list))
    return h_list, [float(x) for x in H_inv.reshape(9)]


def projection_distance(
    H_a: Sequence[float],
    H_b: Sequence[float],
    test_points: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Mean/max distance between pitch→image projections of test points under two H."""
    Ha = matrix_from_row_major(H_a)
    Hb = matrix_from_row_major(H_b)
    Ha_inv = np.linalg.inv(Ha)
    Hb_inv = np.linalg.inv(Hb)
    pts = np.asarray(test_points, dtype=np.float64)
    img_a = apply_homography(Ha_inv, pts)
    img_b = apply_homography(Hb_inv, pts)
    d = np.linalg.norm(img_a - img_b, axis=1)
    return float(np.mean(d)), float(np.max(d))


def select_medoid_candidate(
    candidates: Sequence[FrameCalibrationCandidate],
    *,
    test_points: Sequence[Sequence[float]],
) -> FrameCalibrationCandidate:
    """Medoid by pitch test-point projection distance — NOT element-wise H average."""
    usable = [c for c in candidates if c.H_row_major is not None]
    if not usable:
        raise CalibrationContractError("no candidates for medoid")
    if len(usable) == 1:
        return usable[0]
    best = usable[0]
    best_cost = float("inf")
    for i, cand in enumerate(usable):
        assert cand.H_row_major is not None
        total = 0.0
        for j, other in enumerate(usable):
            if i == j:
                continue
            assert other.H_row_major is not None
            mean_d, _ = projection_distance(cand.H_row_major, other.H_row_major, test_points)
            total += mean_d
        # Tie-break: prefer more inliers, then lower frame index.
        cost = total - 1e-6 * cand.inlier_count + 1e-9 * cand.frame_index
        if cost < best_cost:
            best_cost = cost
            best = cand
    return best


def _quality_to_validity(quality: str) -> str:
    if quality == HomographyQuality.VALID.value:
        return ValidityStatus.VALID.value
    if quality == HomographyQuality.UNCERTAIN.value:
        return ValidityStatus.UNCERTAIN.value
    if quality in {HomographyQuality.INVALID.value, HomographyQuality.NOT_AVAILABLE.value}:
        return ValidityStatus.INVALID.value
    if quality == HomographyQuality.DEGRADED.value:
        return ValidityStatus.UNCERTAIN.value
    return ValidityStatus.ABSTAIN.value


def _is_segmentable(c: FrameCalibrationCandidate, *, config: Mapping[str, Any]) -> bool:
    if c.H_row_major is None:
        return False
    if c.quality == HomographyQuality.VALID.value:
        return True
    # Degraded/uncertain may form segments but never physical-eligible.
    if c.quality in {HomographyQuality.DEGRADED.value, HomographyQuality.UNCERTAIN.value}:
        return True
    _ = config
    return False


def build_calibration_segments(
    candidates: Sequence[FrameCalibrationCandidate],
    *,
    run_id: str,
    video_id: str,
    config: Mapping[str, Any],
    pitch_template_fingerprint: str,
    pitch_length_m: float,
    pitch_width_m: float,
    shot_cuts_us: Sequence[int] | None = None,
    timeline_start_us: int | None = None,
    timeline_end_us: int | None = None,
    test_points: Sequence[Sequence[float]] | None = None,
) -> SegmentBuildResult:
    """Build half-open segments; shot-cut terminates; drift opens new; no silent gap fill."""
    segs_cfg = config["segments"]
    if segs_cfg.get("silent_gap_fill") is True:
        raise CalibrationContractError("SILENT_GAP_FILL_FORBIDDEN")
    cuts = sorted({int(x) for x in (shot_cuts_us or [])})
    pts = list(test_points) if test_points is not None else list(DEFAULT_TEST_POINTS)
    ordered = sorted(candidates, key=lambda c: (c.video_time_us, c.frame_index))

    stats = {
        "input_frames": len(ordered),
        "segmentable_frames": 0,
        "segments": 0,
        "cut_terminations": 0,
        "drift_splits": 0,
        "pan_zoom_splits": 0,
        "gaps": 0,
        "overlaps": 0,
        "physical_eligible_segments": 0,
        "review_required": 0,
    }
    review: list[str] = []

    # Partition by shot intervals using cuts.
    cut_set = cuts
    groups: list[list[FrameCalibrationCandidate]] = []
    current: list[FrameCalibrationCandidate] = []
    active_shot_end: int | None = cut_set[0] if cut_set else None
    cut_idx = 0
    for cand in ordered:
        while active_shot_end is not None and cand.video_time_us >= active_shot_end:
            if current:
                groups.append(current)
                current = []
                stats["cut_terminations"] += 1
            cut_idx += 1
            active_shot_end = cut_set[cut_idx] if cut_idx < len(cut_set) else None
        if _is_segmentable(cand, config=config):
            stats["segmentable_frames"] += 1
            current.append(cand)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)

    segments: list[dict[str, Any]] = []
    seg_counter = 0
    prev_seg_id: str | None = None

    for group in groups:
        # Split group by drift / pan-zoom / large temporal gap.
        buckets: list[list[FrameCalibrationCandidate]] = []
        bucket: list[FrameCalibrationCandidate] = []
        for cand in group:
            if not bucket:
                bucket = [cand]
                continue
            prev = bucket[-1]
            gap = int(cand.video_time_us) - int(prev.video_time_us)
            if gap > int(segs_cfg["max_gap_us_within_segment"]):
                buckets.append(bucket)
                bucket = [cand]
                continue
            if (
                segs_cfg["new_segment_on_pan_zoom"]
                and cand.camera_motion in {"pan", "zoom", "pan_zoom"}
                and prev.camera_motion != cand.camera_motion
            ):
                buckets.append(bucket)
                bucket = [cand]
                stats["pan_zoom_splits"] += 1
                continue
            if prev.H_row_major is not None and cand.H_row_major is not None:
                mean_d, max_d = projection_distance(prev.H_row_major, cand.H_row_major, pts)
                if mean_d > float(segs_cfg["drift_mean_test_point_m"]) or max_d > float(
                    segs_cfg["drift_max_test_point_m"]
                ):
                    buckets.append(bucket)
                    bucket = [cand]
                    stats["drift_splits"] += 1
                    continue
            bucket.append(cand)
        if bucket:
            buckets.append(bucket)

        for buck in buckets:
            if len(buck) < int(segs_cfg["min_support_frames"]):
                continue
            start_us = int(buck[0].video_time_us)
            # End exclusive: next frame time or +1 us after last.
            if len(buck) >= 2:
                # Extend to midpoint toward next gap; use last+1 as minimum.
                end_us = int(buck[-1].video_time_us) + 1
                # Prefer span covering last sample with min duration.
                end_us = max(end_us, start_us + int(segs_cfg["min_duration_us"]))
            else:
                end_us = start_us + max(1, int(segs_cfg["min_duration_us"]))

            # If a shot cut falls inside, terminate.
            boundary = "none"
            for cut in cuts:
                if start_us < cut < end_us:
                    end_us = cut
                    boundary = "SHOT_CUT_TERMINATE"
                    stats["cut_terminations"] += 1
                    break

            if end_us <= start_us:
                continue

            medoid = select_medoid_candidate(buck, test_points=pts)
            validity = _quality_to_validity(medoid.quality)
            is_interp = False
            physical = (
                bool(medoid.physical_mapping_eligible) and validity == ValidityStatus.VALID.value
            )
            if medoid.quality in {
                HomographyQuality.DEGRADED.value,
                HomographyQuality.UNCERTAIN.value,
            }:
                physical = False
            if is_interp:
                physical = False

            seg_counter += 1
            seg_id = f"cseg_{seg_counter:04d}"
            reasons = list(medoid.reason_codes)
            flags = list(medoid.quality_flags)
            flags.append(f"support_frames:{len(buck)}")
            flags.append(f"representative:{medoid.frame_index}")
            if boundary != "none":
                reasons.append(boundary)

            h_list, hinv_list = row_major_with_inverse(
                medoid.H_row_major, medoid.H_inv_row_major
            )
            row = segment_row(
                run_id=run_id,
                video_id=video_id,
                segment_id=seg_id,
                calibration_id=int(medoid.calibration_id),
                start_time_us=start_us,
                end_time_us=end_us,
                source_frame_index=int(medoid.frame_index),
                start_frame_index=int(buck[0].frame_index),
                end_frame_index=int(buck[-1].frame_index),
                homography_image_to_pitch=h_list,
                homography_pitch_to_image=hinv_list,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                pitch_template_fingerprint=pitch_template_fingerprint,
                validity_status=validity,
                camera_view=medoid.camera_view,
                camera_motion=medoid.camera_motion,
                is_interpolated=is_interp,
                physical_metric_eligible=physical,
                boundary_reason=boundary if boundary != "none" else "none",
                correspondence_count=medoid.correspondence_count,
                inlier_count=medoid.inlier_count,
                inlier_ratio=medoid.inlier_ratio if medoid.inlier_ratio is not None else 0.0,
                mean_reprojection_error_px=medoid.mean_reprojection_error_px,
                condition_number=medoid.condition_number,
                determinant=medoid.determinant,
                coverage_hull_area_fraction=medoid.coverage_hull_fraction,
                solver_method=medoid.solver_method,
                solver_version=medoid.solver_version,
                shot_segment_id=medoid.shot_segment_id,
                analysis_window_id=medoid.analysis_window_id,
                previous_segment_id=prev_seg_id,
                reason_codes=reasons,
                quality_flags=flags,
                provenance_json=json.dumps(
                    {
                        "representative_frame_index": medoid.frame_index,
                        "support_frame_indices": [c.frame_index for c in buck],
                        "selection": segs_cfg["representative_selection"],
                        "attack_direction": "unknown",
                    },
                    sort_keys=True,
                ),
            )
            if physical:
                stats["physical_eligible_segments"] += 1
            if validity == ValidityStatus.CONFLICT.value:
                review.append(seg_id)
            segments.append(row)
            if prev_seg_id is not None:
                # Link next_segment_id on previous.
                for s in segments:
                    if s["segment_id"] == prev_seg_id:
                        s["next_segment_id"] = seg_id
                        break
            prev_seg_id = seg_id
            stats["segments"] += 1

    overlaps = find_segment_overlaps(segments)
    if overlaps and segs_cfg["overlap_is_hard_conflict"]:
        for a, b in overlaps:
            for s in segments:
                if s["segment_id"] in {a, b}:
                    s["validity_status"] = ValidityStatus.CONFLICT.value
                    s["physical_metric_eligible"] = False
                    s["manual_review_required"] = True
                    codes = list(s.get("reason_codes") or [])
                    if "SEGMENT_OVERLAP_CONFLICT" not in codes:
                        codes.append("SEGMENT_OVERLAP_CONFLICT")
                    s["reason_codes"] = codes
                    review.append(s["segment_id"])
        stats["overlaps"] = len(overlaps)
        stats["review_required"] = len(set(review))

    t0 = (
        timeline_start_us
        if timeline_start_us is not None
        else (min((s["start_time_us"] for s in segments), default=0))
    )
    t1 = (
        timeline_end_us
        if timeline_end_us is not None
        else (max((s["end_time_us"] for s in segments), default=t0 + 1))
    )
    gaps = find_calibration_gaps(segments, timeline_start_us=t0, timeline_end_us=max(t1, t0 + 1))
    stats["gaps"] = len(gaps)

    return SegmentBuildResult(
        segments=segments,
        gaps=gaps,
        overlaps=overlaps,
        stats=stats,
        review_required=sorted(set(review)),
    )


def apply_shot_cut_to_segments(
    segments: Sequence[Mapping[str, Any]], *, cut_time_us: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        start = int(seg["start_time_us"])
        end = int(seg["end_time_us"])
        if start < cut_time_us < end:
            out.append(terminate_on_shot_cut(segment=seg, cut_time_us=cut_time_us))
        elif end <= cut_time_us or start >= cut_time_us:
            out.append(dict(seg))
        else:
            out.append(dict(seg))
    return out


__all__ = [
    "DEFAULT_TEST_POINTS",
    "FrameCalibrationCandidate",
    "SegmentBuildResult",
    "pitch_test_points",
    "row_major_with_inverse",
    "projection_distance",
    "select_medoid_candidate",
    "build_calibration_segments",
    "apply_shot_cut_to_segments",
]
