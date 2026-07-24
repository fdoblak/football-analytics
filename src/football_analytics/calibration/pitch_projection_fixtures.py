"""Synthetic fixtures for Stage 8D pitch projection (no real accuracy claims)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from football_analytics.calibration.fixtures import known_perspective_H
from football_analytics.calibration.homography import (
    apply_homography,
    identity_homography,
    invert_homography,
)
from football_analytics.calibration.pitch_template import (
    build_pitch_template,
    pitch_template_fingerprint,
)
from football_analytics.calibration.segments import segment_row
from football_analytics.core.run_id import generate_run_id
from football_analytics.identity.target_eligibility_timeline import build_eligibility_timeline

RUNTIME_ROOT = "/home/fdoblak/workspace/pitch_projection_checks"


def assert_runtime_root() -> None:
    from pathlib import Path

    root = Path(RUNTIME_ROOT)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir():
        raise RuntimeError(f"runtime root missing: {root}")


def identity_H() -> np.ndarray:
    return identity_homography()


def perspective_H() -> np.ndarray:
    return known_perspective_H()


def singular_w_H() -> np.ndarray:
    """Homography that maps a chosen image point near w≈0."""
    # Third row nearly orthogonal to (x,y,1) for point (50, 50).
    return np.array(
        [
            [0.1, 0.0, 10.0],
            [0.0, 0.1, 5.0],
            [0.01, 0.01, -1.0],  # w = 0.01*x + 0.01*y - 1 → at (50,50) w=0
        ],
        dtype=np.float64,
    )


def row_major(H: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(H, dtype=np.float64).reshape(9)]


def inverse_row_major(H: np.ndarray) -> list[float]:
    return row_major(invert_homography(H))


def coverage_hull_for_H(
    H: np.ndarray,
    pitch_pts: Sequence[Sequence[float]] | None = None,
) -> list[tuple[float, float]]:
    template = build_pitch_template()
    L, W = template.length_m, template.width_m
    pts = (
        list(pitch_pts)
        if pitch_pts is not None
        else [
            (10.0, 10.0),
            (L - 10.0, 10.0),
            (L - 10.0, W - 10.0),
            (10.0, W - 10.0),
        ]
    )
    H_inv = invert_homography(H)
    img = apply_homography(H_inv, pts)
    return [(float(p[0]), float(p[1])) for p in img]


def make_segment(
    *,
    run_id: str,
    video_id: str,
    segment_id: str,
    calibration_id: int,
    start_time_us: int,
    end_time_us: int,
    H: np.ndarray,
    validity_status: str = "valid",
    physical_metric_eligible: bool = True,
    is_interpolated: bool = False,
    mean_reprojection_error_px: float = 1.0,
    coverage_hull_area_fraction: float = 0.2,
    template_fp: str | None = None,
    source_frame_index: int = 0,
) -> dict[str, Any]:
    template = build_pitch_template()
    t_fp = template_fp or pitch_template_fingerprint(template)
    return segment_row(
        run_id=run_id,
        video_id=video_id,
        segment_id=segment_id,
        calibration_id=calibration_id,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        source_frame_index=source_frame_index,
        homography_image_to_pitch=row_major(H),
        pitch_length_m=template.length_m,
        pitch_width_m=template.width_m,
        pitch_template_fingerprint=t_fp,
        validity_status=validity_status,
        physical_metric_eligible=physical_metric_eligible and validity_status == "valid",
        is_interpolated=is_interpolated,
        mean_reprojection_error_px=mean_reprojection_error_px,
        coverage_hull_area_fraction=coverage_hull_area_fraction,
        homography_pitch_to_image=inverse_row_major(H),
        correspondence_count=4,
        inlier_count=4,
        inlier_ratio=1.0,
        condition_number=10.0,
        determinant=1.0,
    )


def obs_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    track_id: int,
    entity_type: str,
    bbox: tuple[float, float, float, float],
    observation_state: str = "observed",
    detection_id: int | None = 0,
    class_id: int | None = None,
) -> dict[str, Any]:
    cid = class_id if class_id is not None else (0 if entity_type == "human" else 32)
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "track_id": track_id,
        "detection_id": detection_id,
        "class_id": cid,
        "entity_type": entity_type,
        "confidence": 0.9 if observation_state == "observed" else None,
        "bbox_x1": float(bbox[0]),
        "bbox_y1": float(bbox[1]),
        "bbox_x2": float(bbox[2]),
        "bbox_y2": float(bbox[3]),
        "observation_state": observation_state,
        "model_id": "synthetic",
        "quality_flags": [],
    }


def pitch_point_to_image(H: np.ndarray, x_m: float, y_m: float) -> tuple[float, float]:
    H_inv = invert_homography(H)
    p = apply_homography(H_inv, [(x_m, y_m)])[0]
    return float(p[0]), float(p[1])


def human_bbox_for_footpoint(
    H: np.ndarray, x_m: float, y_m: float, *, height_px: float = 80.0, width_px: float = 30.0
) -> tuple[float, float, float, float]:
    ix, iy = pitch_point_to_image(H, x_m, y_m)
    x1 = ix - width_px / 2.0
    x2 = ix + width_px / 2.0
    y2 = iy
    y1 = iy - height_px
    return (x1, y1, x2, y2)


def ball_bbox_for_centre(
    H: np.ndarray, x_m: float, y_m: float, *, size_px: float = 8.0
) -> tuple[float, float, float, float]:
    ix, iy = pitch_point_to_image(H, x_m, y_m)
    half = size_px / 2.0
    return (ix - half, iy - half, ix + half, iy + half)


def base_bundle(
    *,
    H: np.ndarray | None = None,
    run_id: str | None = None,
    video_id: str = "video_proj_01",
) -> dict[str, Any]:
    H = H if H is not None else perspective_H()
    rid = run_id or generate_run_id()
    template = build_pitch_template()
    t_fp = pitch_template_fingerprint(template)
    seg = make_segment(
        run_id=rid,
        video_id=video_id,
        segment_id="seg_main",
        calibration_id=1,
        start_time_us=0,
        end_time_us=1_000_000,
        H=H,
        template_fp=t_fp,
    )
    hull = coverage_hull_for_H(H)
    human_bbox = human_bbox_for_footpoint(H, 52.5, 34.0)
    ball_bbox = ball_bbox_for_centre(H, 60.0, 30.0)
    observations = [
        obs_row(
            run_id=rid,
            video_id=video_id,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox,
        ),
        obs_row(
            run_id=rid,
            video_id=video_id,
            frame_index=0,
            track_id=10,
            entity_type="ball",
            bbox=ball_bbox,
            detection_id=1,
        ),
    ]
    frame_times = {0: 0, 1: 40_000, 2: 80_000, 3: 120_000, 4: 160_000}
    timeline = build_eligibility_timeline(
        [
            {
                "assignment_id": "asg_confirmed",
                "track_id": 1,
                "assignment_status": "confirmed",
                "target_scope": "target",
                "start_frame_index": 0,
                "end_frame_index": 2,
            },
            {
                "assignment_id": "asg_provisional",
                "track_id": 2,
                "assignment_status": "provisional",
                "target_scope": "target",
                "start_frame_index": 0,
                "end_frame_index": 2,
            },
            {
                "assignment_id": "asg_revoked",
                "track_id": 3,
                "assignment_status": "revoked",
                "target_scope": "target",
                "start_frame_index": 0,
                "end_frame_index": 2,
            },
        ],
        timeline_id="tl_proj_01",
        run_id=rid,
        video_id=video_id,
        target_player_id="target_01",
    )
    return {
        "run_id": rid,
        "video_id": video_id,
        "H": H,
        "segments": [seg],
        "observations": observations,
        "coverage_hulls": {"seg_main": hull},
        "frame_times": frame_times,
        "eligibility_timeline": timeline,
        "pitch_template_fingerprint": t_fp,
        "fingerprints": {
            "run_id": rid,
            "video_id": video_id,
            "source_video_sha": "a" * 64,
            "frame_timeline": "b" * 64,
            "tracking_bundle": "c" * 64,
            "pitch_template": t_fp,
            "calibration_artifact": "d" * 64,
            "coordinate_frame": "source_image",
        },
        "analysis_windows": [
            {
                "start_frame_index": 0,
                "end_frame_index": 10,
                "playability_status": "playable",
                "replay_status": "live",
                "camera_view": "main",
            }
        ],
        "frame_width": 1280.0,
        "frame_height": 720.0,
    }


def scenario_observations(bundle: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Named observation sets covering prompt §14 scenarios."""
    rid, vid = bundle["run_id"], bundle["video_id"]
    H = bundle["H"]
    H_id = identity_H()
    out: dict[str, list[dict[str, Any]]] = {}
    out["identity_footpoint"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H_id, 40.0, 20.0),
        )
    ]
    out["perspective_human"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 52.5, 34.0),
        )
    ]
    out["ball_centre"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=10,
            entity_type="ball",
            bbox=ball_bbox_for_centre(H, 60.0, 30.0),
            detection_id=1,
        )
    ]
    out["multiple_tracks"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 40.0, 20.0),
        ),
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=2,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 70.0, 40.0),
            detection_id=2,
        ),
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=10,
            entity_type="ball",
            bbox=ball_bbox_for_centre(H, 55.0, 33.0),
            detection_id=3,
        ),
    ]
    out["predicted_human"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=1,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 52.5, 34.0),
            observation_state="predicted",
            detection_id=None,
        )
    ]
    out["predicted_ball"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=1,
            track_id=10,
            entity_type="ball",
            bbox=ball_bbox_for_centre(H, 60.0, 30.0),
            observation_state="predicted",
            detection_id=None,
        )
    ]
    out["confirmed_target"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 52.5, 34.0),
        )
    ]
    out["provisional_target"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=2,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 52.5, 34.0),
            detection_id=4,
        )
    ]
    out["revoked_target"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=3,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 52.5, 34.0),
            detection_id=5,
        )
    ]
    # Outside pitch: place footpoint far outside via image coords that map outside.
    out["outside_pitch"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=(-200.0, -200.0, -160.0, -120.0),
        )
    ]
    # Truncated near frame edge.
    out["truncated"] = [
        obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=(0.0, 640.0, 30.0, 720.0),
        )
    ]
    return out


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "identity_H",
    "perspective_H",
    "singular_w_H",
    "row_major",
    "inverse_row_major",
    "coverage_hull_for_H",
    "make_segment",
    "obs_row",
    "pitch_point_to_image",
    "human_bbox_for_footpoint",
    "ball_bbox_for_centre",
    "base_bundle",
    "scenario_observations",
]
