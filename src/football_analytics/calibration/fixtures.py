"""Synthetic calibration contract fixtures (Stage 8A — no real accuracy claims)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from football_analytics.calibration.features import feature_row
from football_analytics.calibration.homography import (
    apply_homography,
    identity_homography,
    invert_homography,
    scale_translate_homography,
    solve_homography,
)
from football_analytics.calibration.pitch_template import (
    build_pitch_template,
    pitch_template_fingerprint,
)
from football_analytics.calibration.projected_positions import (
    compute_physical_metric_eligibility,
    project_point,
    projection_row,
    source_point_for_entity,
)
from football_analytics.calibration.segments import segment_row
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {"run_id": generate_run_id(), "video_id": "video_cal_01"}


def default_template():
    return build_pitch_template()


def known_perspective_H() -> np.ndarray:
    """A mild perspective image→pitch map (synthetic)."""
    return np.array(
        [
            [0.12, 0.01, 5.0],
            [0.02, 0.11, 3.0],
            [0.0002, 0.0001, 1.0],
        ],
        dtype=np.float64,
    )


def pitch_sample_points(template=None) -> list[tuple[float, float]]:
    t = template or default_template()
    L, W = t.length_m, t.width_m
    return [
        (0.0, 0.0),
        (L, 0.0),
        (L, W),
        (0.0, W),
        (L / 2.0, W / 2.0),
        (11.0, W / 2.0),
    ]


def image_points_from_H(
    H: np.ndarray, pitch_pts: Sequence[Sequence[float]]
) -> list[tuple[float, float]]:
    H_inv = invert_homography(H)
    mapped = apply_homography(H_inv, pitch_pts)
    return [(float(p[0]), float(p[1])) for p in mapped]


def correspondences_for_H(
    H: np.ndarray | None = None, *, n: int = 4
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    H = H if H is not None else known_perspective_H()
    pitch = pitch_sample_points()[:n]
    image = image_points_from_H(H, pitch)
    return image, pitch


def rotation_homography(theta_rad: float) -> np.ndarray:
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    # Rotate about pitch centre-ish in homogeneous form then translate.
    R = np.array([[c, -s, 40.0], [s, c, 20.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def mirrored_homography() -> np.ndarray:
    # Reflect x axis (mirror).
    return np.array([[-1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def singular_matrix_row_major() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def ill_conditioned_matrix_row_major() -> list[float]:
    # Near-singular: second row ≈ first * 1e-10
    return [1.0, 0.0, 0.0, 1e-12, 1e-12, 0.0, 0.0, 0.0, 1.0]


def valid_feature_bundle(run_id: str, video_id: str) -> list[dict[str, Any]]:
    img, pitch = correspondences_for_H(n=4)
    rows = []
    for i, ((ix, iy), (px, py)) in enumerate(zip(img, pitch, strict=True)):
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=0,
                video_time_us=0,
                feature_id=f"kp_{i}",
                feature_type="keypoint",
                image_x=ix,
                image_y=iy,
                canonical_pitch_feature_id=f"pitch_pt_{i}",
                status="matched",
                source="synthetic",
            )
        )
        _ = (px, py)
    return rows


def valid_segment(
    run_id: str,
    video_id: str,
    *,
    segment_id: str = "seg_01",
    start_us: int = 0,
    end_us: int = 1_000_000,
    H: np.ndarray | None = None,
) -> dict[str, Any]:
    template = default_template()
    fp = pitch_template_fingerprint(template)
    H = H if H is not None else known_perspective_H()
    img, pitch = correspondences_for_H(H, n=4)
    result = solve_homography(
        img, pitch, pitch_length_m=template.length_m, pitch_width_m=template.width_m
    )
    return segment_row(
        run_id=run_id,
        video_id=video_id,
        segment_id=segment_id,
        calibration_id=0,
        start_time_us=start_us,
        end_time_us=end_us,
        source_frame_index=0,
        homography_image_to_pitch=result.matrix_row_major(),
        homography_pitch_to_image=result.inverse_row_major(),
        pitch_length_m=template.length_m,
        pitch_width_m=template.width_m,
        pitch_template_fingerprint=fp,
        condition_number=result.condition_number,
        determinant=result.determinant,
        mean_reprojection_error_px=result.mean_reprojection_error_px,
        correspondence_count=result.correspondence_count,
        inlier_count=result.inlier_count,
        inlier_ratio=1.0,
        coverage_hull_area_fraction=0.2,
        physical_metric_eligible=True,
    )


def projected_from_track(
    run_id: str,
    video_id: str,
    *,
    entity_type: str,
    bbox: tuple[float, float, float, float],
    H_row_major: Sequence[float],
    observation_source: str = "detection_associated",
    projection_id: str = "proj_01",
    airborne_ball: bool = False,
    frame_index: int = 0,
    video_time_us: int = 0,
    calibration_id: int = 0,
    segment_id: str = "seg_01",
) -> dict[str, Any]:
    template = default_template()
    ix, iy, spt = source_point_for_entity(entity_type, bbox)
    x_m, y_m, status, in_bounds, extrapolated = project_point(
        image_x=ix,
        image_y=iy,
        H_row_major=H_row_major,
        pitch_length_m=template.length_m,
        pitch_width_m=template.width_m,
    )
    elig, reasons = compute_physical_metric_eligibility(
        observation_source=observation_source,
        mapping_status=status,
        is_extrapolated=extrapolated,
        calibration_valid=True,
        in_bounds=in_bounds,
        entity_type=entity_type,
        airborne_ball=airborne_ball,
    )
    return projection_row(
        run_id=run_id,
        video_id=video_id,
        frame_index=frame_index,
        video_time_us=video_time_us,
        projection_id=projection_id,
        entity_type=entity_type,
        track_id=0,
        detection_id=0 if observation_source == "detection_associated" else None,
        observation_source=observation_source,
        source_point_type=spt,
        image_x=ix,
        image_y=iy,
        pitch_x_m=x_m,
        pitch_y_m=y_m,
        calibration_id=calibration_id,
        segment_id=segment_id,
        mapping_status=status,
        in_bounds=in_bounds,
        is_extrapolated=extrapolated,
        calibration_quality="good",
        physical_metric_eligibility=elig,
        reason_codes=reasons,
    )


def e2e_bundle() -> dict[str, Any]:
    ids = base_ids()
    run_id, video_id = ids["run_id"], ids["video_id"]
    template = default_template()
    H = known_perspective_H()
    img, pitch = correspondences_for_H(H, n=4)
    solved = solve_homography(
        img, pitch, pitch_length_m=template.length_m, pitch_width_m=template.width_m
    )
    features = valid_feature_bundle(run_id, video_id)
    seg = valid_segment(run_id, video_id, H=H)
    human = projected_from_track(
        run_id,
        video_id,
        entity_type="human",
        bbox=(100.0, 100.0, 140.0, 200.0),
        H_row_major=solved.matrix_row_major(),
        projection_id="proj_human",
    )
    ball = projected_from_track(
        run_id,
        video_id,
        entity_type="ball",
        bbox=(200.0, 150.0, 220.0, 170.0),
        H_row_major=solved.matrix_row_major(),
        projection_id="proj_ball",
    )
    return {
        "run_id": run_id,
        "video_id": video_id,
        "template": template,
        "template_fp": pitch_template_fingerprint(template),
        "H": solved,
        "image_points": img,
        "pitch_points": pitch,
        "features": features,
        "segments": [seg],
        "projections": [human, ball],
        "calibration_features": _cast("calibration_features", features),
        "calibration_segments": _cast("calibration_segments", [seg]),
        "projected_positions": _cast("projected_positions", [human, ball]),
    }


def overlapping_segments_bundle(run_id: str, video_id: str) -> list[dict[str, Any]]:
    a = valid_segment(run_id, video_id, segment_id="seg_a", start_us=0, end_us=1_000_000)
    b = valid_segment(run_id, video_id, segment_id="seg_b", start_us=500_000, end_us=1_500_000)
    return [a, b]


def gap_segments_bundle(run_id: str, video_id: str) -> list[dict[str, Any]]:
    a = valid_segment(run_id, video_id, segment_id="seg_a", start_us=0, end_us=500_000)
    b = valid_segment(run_id, video_id, segment_id="seg_b", start_us=1_000_000, end_us=1_500_000)
    return [a, b]


__all__ = [
    "base_ids",
    "default_template",
    "known_perspective_H",
    "pitch_sample_points",
    "image_points_from_H",
    "correspondences_for_H",
    "rotation_homography",
    "mirrored_homography",
    "singular_matrix_row_major",
    "ill_conditioned_matrix_row_major",
    "identity_homography",
    "scale_translate_homography",
    "valid_feature_bundle",
    "valid_segment",
    "projected_from_track",
    "e2e_bundle",
    "overlapping_segments_bundle",
    "gap_segments_bundle",
]
