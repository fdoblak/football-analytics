"""Synthetic known-H fixtures for Stage 8C (not football accuracy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from football_analytics.calibration.correspondence import Correspondence
from football_analytics.calibration.features import feature_row
from football_analytics.calibration.fixtures import (
    correspondences_for_H,
    image_points_from_H,
    known_perspective_H,
    mirrored_homography,
    rotation_homography,
)
from football_analytics.calibration.homography import (
    apply_homography,
    identity_homography,
    invert_homography,
    normalize_homography,
    scale_translate_homography,
)
from football_analytics.calibration.pitch_template import (
    build_pitch_template,
    pitch_template_fingerprint,
)

RUNTIME_ROOT = Path("/home/fdoblak/workspace/homography_checks")

# Map synthetic pitch sample index → canonical template feature ids.
SAMPLE_CANONICAL_IDS: tuple[str, ...] = (
    "corner_a_left",
    "corner_b_left",
    "corner_b_right",
    "corner_a_right",
    "centre_spot",
    "penalty_spot_a",
)


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)


def known_H_bundle(kind: str = "perspective") -> dict[str, Any]:
    if kind == "identity":
        H = identity_homography()
    elif kind == "scale_translate":
        H = scale_translate_homography(scale_x=0.15, scale_y=0.12, tx=10.0, ty=5.0)
    elif kind == "rotation":
        H = normalize_homography(rotation_homography(0.2))
    elif kind == "mirrored":
        H = mirrored_homography()
    else:
        H = known_perspective_H()
    return {"kind": kind, "H": H, "H_inv": invert_homography(H)}


def synthetic_feature_rows_for_H(
    H: np.ndarray,
    *,
    run_id: str,
    video_id: str,
    frame_index: int = 0,
    video_time_us: int = 0,
    n: int = 4,
    noise_px: float = 0.0,
    seed: int = 0,
    include_unknown: bool = False,
    score: float = 0.9,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    template = build_pitch_template()
    pitch_pts = []
    for fid in SAMPLE_CANONICAL_IDS[:n]:
        pt = next(p for p in template.keypoints if p.feature_id == fid)
        pitch_pts.append((pt.x_m, pt.y_m))
    img = image_points_from_H(H, pitch_pts)
    rows: list[dict[str, Any]] = []
    for i, ((ix, iy), fid) in enumerate(zip(img, SAMPLE_CANONICAL_IDS[:n], strict=True)):
        nx = ix + float(rng.normal(0.0, noise_px)) if noise_px else ix
        ny = iy + float(rng.normal(0.0, noise_px)) if noise_px else iy
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                feature_id=f"kp_{frame_index}_{i}",
                feature_type="keypoint",
                image_x=nx,
                image_y=ny,
                canonical_pitch_feature_id=fid,
                score=score,
                status="matched",
                suitability="suitable",
                source="synthetic",
            )
        )
    if include_unknown:
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                feature_id=f"kp_{frame_index}_unk",
                feature_type="keypoint",
                image_x=50.0,
                image_y=50.0,
                canonical_pitch_feature_id=None,
                score=0.99,
                status="detected",
                suitability="unknown",
                source="synthetic",
            )
        )
    return rows


def synthetic_line_features_for_intersections(
    H: np.ndarray,
    *,
    run_id: str,
    video_id: str,
    frame_index: int = 0,
    video_time_us: int = 0,
    score: float = 0.8,
) -> list[dict[str, Any]]:
    """Four mapped lines that intersect at known corners in image space."""
    template = build_pitch_template()
    H_inv = invert_homography(H)
    line_defs = [
        ("touchline_left", "tl"),
        ("touchline_right", "tr"),
        ("goalline_a", "ga"),
        ("goalline_b", "gb"),
    ]
    rows: list[dict[str, Any]] = []
    for canon, tag in line_defs:
        ln = next(x for x in template.lines if x.feature_id == canon)
        p1 = apply_inv(H_inv, ln.x1_m, ln.y1_m)
        p2 = apply_inv(H_inv, ln.x2_m, ln.y2_m)
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                feature_id=f"ln_{frame_index}_{tag}",
                feature_type="line",
                line_x1=p1[0],
                line_y1=p1[1],
                line_x2=p2[0],
                line_y2=p2[1],
                canonical_pitch_feature_id=canon,
                score=score,
                status="matched",
                suitability="suitable",
                source="synthetic",
            )
        )
    return rows


def apply_inv(H_inv: np.ndarray, x: float, y: float) -> tuple[float, float]:
    out = apply_homography(H_inv, [(x, y)])[0]
    return float(out[0]), float(out[1])


def outlier_correspondences(
    H: np.ndarray, *, n_inliers: int = 6, n_outliers: int = 3, seed: int = 1
) -> list[Correspondence]:
    rng = np.random.default_rng(seed)
    template = build_pitch_template()
    ids = SAMPLE_CANONICAL_IDS[:n_inliers]
    pitch = []
    for fid in ids:
        pt = next(p for p in template.keypoints if p.feature_id == fid)
        pitch.append((pt.x_m, pt.y_m))
    img = image_points_from_H(H, pitch)
    items: list[Correspondence] = []
    for i, ((ix, iy), fid, (px, py)) in enumerate(zip(img, ids, pitch, strict=True)):
        items.append(
            Correspondence(
                correspondence_id=f"in_{i}",
                source_type="keypoint",
                feature_ids=(f"f{i}",),
                canonical_pitch_feature_id=fid,
                image_x=ix,
                image_y=iy,
                pitch_x_m=px,
                pitch_y_m=py,
                score=0.9,
                quality=None,
            )
        )
    for j in range(n_outliers):
        items.append(
            Correspondence(
                correspondence_id=f"out_{j}",
                source_type="keypoint",
                feature_ids=(f"o{j}",),
                canonical_pitch_feature_id=f"outlier_pitch_{j}",
                image_x=float(rng.uniform(0, 400)),
                image_y=float(rng.uniform(0, 300)),
                pitch_x_m=float(rng.uniform(0, 105)),
                pitch_y_m=float(rng.uniform(0, 68)),
                score=0.5,
                quality=None,
            )
        )
    # Outlier pitch ids must exist for array conversion only — solver uses points directly.
    # Remap outlier canonical to unused corners by using centre offsets as distinct points.
    return items


def multi_frame_stable_features(
    H: np.ndarray,
    *,
    run_id: str,
    video_id: str,
    n_frames: int = 5,
    dt_us: int = 40_000,
    drift_px: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fi in range(n_frames):
        # Mild optional drift via translation in image before inverse — approximate by
        # adding drift to image points after projection.
        base = synthetic_feature_rows_for_H(
            H,
            run_id=run_id,
            video_id=video_id,
            frame_index=fi,
            video_time_us=fi * dt_us,
            n=4,
            noise_px=0.0,
            seed=fi,
        )
        if drift_px:
            for r in base:
                if r.get("image_x") is not None:
                    r["image_x"] = float(r["image_x"]) + drift_px * fi
                    r["image_y"] = float(r["image_y"]) + 0.5 * drift_px * fi
        rows.extend(base)
    return rows


def collinear_feature_rows(*, run_id: str, video_id: str) -> list[dict[str, Any]]:
    # Three collinear pitch points along touchline — insufficient span.
    return [
        feature_row(
            run_id=run_id,
            video_id=video_id,
            frame_index=0,
            video_time_us=0,
            feature_id=f"col_{i}",
            feature_type="keypoint",
            image_x=10.0 + 20.0 * i,
            image_y=10.0,
            canonical_pitch_feature_id=fid,
            score=0.9,
            status="matched",
            source="synthetic",
        )
        for i, fid in enumerate(("corner_a_left", "halfway_left", "corner_b_left", "centre_spot"))
    ]


def insufficient_feature_rows(*, run_id: str, video_id: str) -> list[dict[str, Any]]:
    return synthetic_feature_rows_for_H(
        known_perspective_H(), run_id=run_id, video_id=video_id, n=3
    )


def duplicate_feature_rows(*, run_id: str, video_id: str) -> list[dict[str, Any]]:
    rows = synthetic_feature_rows_for_H(
        known_perspective_H(), run_id=run_id, video_id=video_id, n=4, score=0.7
    )
    # Duplicate canonical with higher score — ranking should keep one.
    dup = dict(rows[0])
    dup["feature_id"] = "kp_dup"
    dup["score"] = 0.95
    dup["image_x"] = float(dup["image_x"]) + 0.5
    rows.append(dup)
    return rows


def template_meta() -> dict[str, Any]:
    t = build_pitch_template()
    return {
        "template": t,
        "fingerprint": pitch_template_fingerprint(t),
        "length_m": t.length_m,
        "width_m": t.width_m,
    }


__all__ = [
    "RUNTIME_ROOT",
    "SAMPLE_CANONICAL_IDS",
    "assert_runtime_root",
    "known_H_bundle",
    "synthetic_feature_rows_for_H",
    "synthetic_line_features_for_intersections",
    "outlier_correspondences",
    "multi_frame_stable_features",
    "collinear_feature_rows",
    "insufficient_feature_rows",
    "duplicate_feature_rows",
    "template_meta",
    "correspondences_for_H",
    "known_perspective_H",
    "identity_homography",
    "scale_translate_homography",
    "rotation_homography",
    "mirrored_homography",
]
