"""Frame-level homography solve with quality classes (Stage 8C).

Wraps Stage 8A DLT helpers and optional OpenCV RANSAC. Does not mutate
calibrations schema fingerprint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from football_analytics.calibration.correspondence import (
    Correspondence,
    correspondences_to_arrays,
)
from football_analytics.calibration.homography import (
    HOMOGRAPHY_DIRECTION,
    apply_homography,
    condition_number,
    detect_mirror,
    invert_homography,
    normalize_homography,
    solve_homography,
)
from football_analytics.calibration.types import HomographyError

SOLVER_VERSION = "1"


class HomographyQuality(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class FrameHomographySolution:
    status: str  # solved | failed | not_available
    quality: HomographyQuality
    H: np.ndarray | None
    H_inv: np.ndarray | None
    direction: str
    solver_method: str
    solver_version: str
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float | None
    inlier_mask: tuple[bool, ...]
    residuals_px: tuple[float, ...]
    mean_reprojection_error_px: float | None
    median_reprojection_error_px: float | None
    max_reprojection_error_px: float | None
    mean_pitch_error_m: float | None
    mean_round_trip_error_px: float | None
    mean_round_trip_error_m: float | None
    condition_number: float | None
    determinant: float | None
    coverage_hull_fraction: float | None
    canonical_diversity: int
    keypoint_support: int
    line_intersection_support: int
    is_mirrored: bool
    extrapolation_ratio: float | None
    physical_mapping_eligible: bool
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]

    def matrix_row_major(self) -> list[float] | None:
        if self.H is None:
            return None
        return [float(x) for x in self.H.reshape(9)]

    def inverse_row_major(self) -> list[float] | None:
        if self.H_inv is None:
            return None
        return [float(x) for x in self.H_inv.reshape(9)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "quality": self.quality.value,
            "homography_image_to_pitch": self.matrix_row_major(),
            "homography_pitch_to_image": self.inverse_row_major(),
            "direction": self.direction,
            "solver_method": self.solver_method,
            "solver_version": self.solver_version,
            "correspondence_count": self.correspondence_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "inlier_mask": list(self.inlier_mask),
            "residuals_px": list(self.residuals_px),
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "mean_pitch_error_m": self.mean_pitch_error_m,
            "mean_round_trip_error_px": self.mean_round_trip_error_px,
            "mean_round_trip_error_m": self.mean_round_trip_error_m,
            "condition_number": self.condition_number,
            "determinant": self.determinant,
            "coverage_hull_fraction": self.coverage_hull_fraction,
            "canonical_diversity": self.canonical_diversity,
            "keypoint_support": self.keypoint_support,
            "line_intersection_support": self.line_intersection_support,
            "is_mirrored": self.is_mirrored,
            "extrapolation_ratio": self.extrapolation_ratio,
            "physical_mapping_eligible": self.physical_mapping_eligible,
            "reason_codes": list(self.reason_codes),
            "quality_flags": list(self.quality_flags),
        }


def _empty(
    *,
    quality: HomographyQuality,
    reason: str,
    method: str,
    n: int = 0,
) -> FrameHomographySolution:
    return FrameHomographySolution(
        status="not_available" if quality == HomographyQuality.NOT_AVAILABLE else "failed",
        quality=quality,
        H=None,
        H_inv=None,
        direction=HOMOGRAPHY_DIRECTION,
        solver_method=method,
        solver_version=SOLVER_VERSION,
        correspondence_count=n,
        inlier_count=0,
        inlier_ratio=None,
        inlier_mask=tuple(),
        residuals_px=tuple(),
        mean_reprojection_error_px=None,
        median_reprojection_error_px=None,
        max_reprojection_error_px=None,
        mean_pitch_error_m=None,
        mean_round_trip_error_px=None,
        mean_round_trip_error_m=None,
        condition_number=None,
        determinant=None,
        coverage_hull_fraction=None,
        canonical_diversity=0,
        keypoint_support=0,
        line_intersection_support=0,
        is_mirrored=False,
        extrapolation_ratio=None,
        physical_mapping_eligible=False,
        reason_codes=(reason,),
        quality_flags=(reason,),
    )


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization; returns normalized pts and 3x3 T."""
    c = pts.mean(axis=0)
    centered = pts - c
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_dist < 1e-12:
        raise HomographyError("DEGENERATE_NORMALIZATION")
    s = math.sqrt(2.0) / mean_dist
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    normed = (T @ homo.T).T[:, :2]
    return normed, T


def normalized_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_n, T_src = _normalize_points(src)
    dst_n, T_dst = _normalize_points(dst)
    n = src_n.shape[0]
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vh = np.linalg.svd(A)
    H_n = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return normalize_homography(H)


def _convex_hull_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    # Monotone chain.
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

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
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def _coverage_fraction(
    image_pts: np.ndarray, *, width: float | None, height: float | None
) -> float:
    area = _convex_hull_area(image_pts)
    if width is None or height is None or width <= 0 or height <= 0:
        # Relative to bbox of points.
        bbox = float(np.ptp(image_pts[:, 0]) * np.ptp(image_pts[:, 1]))
        return float(area / max(bbox, 1e-9)) if bbox > 0 else 0.0
    return float(area / (width * height))


def _ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold: float,
    max_iters: int,
    confidence: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    src32 = np.asarray(src, dtype=np.float32)
    dst32 = np.asarray(dst, dtype=np.float32)
    # OpenCV RNG is seeded via setRNGSeed for determinism.
    cv2.setRNGSeed(int(seed))
    H, mask = cv2.findHomography(
        src32,
        dst32,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if H is None or mask is None:
        raise HomographyError("RANSAC_FAILED")
    mask_bool = mask.reshape(-1).astype(bool)
    if int(mask_bool.sum()) < 4:
        raise HomographyError("INSUFFICIENT_INLIERS")
    return normalize_homography(np.asarray(H, dtype=np.float64)), mask_bool


def _classify_quality(
    *,
    mean_reproj: float,
    inlier_ratio: float,
    coverage: float,
    cond: float,
    config: Mapping[str, Any],
) -> HomographyQuality:
    q = config["quality"]
    for band, quality in (
        ("valid", HomographyQuality.VALID),
        ("degraded", HomographyQuality.DEGRADED),
        ("uncertain", HomographyQuality.UNCERTAIN),
    ):
        b = q[band]
        if (
            mean_reproj <= float(b["max_mean_reproj_px"])
            and inlier_ratio >= float(b["min_inlier_ratio"])
            and coverage >= float(b["min_coverage"])
            and cond <= float(b["max_condition"])
        ):
            return quality
    return HomographyQuality.INVALID


def _physical_eligible(quality: HomographyQuality, config: Mapping[str, Any]) -> bool:
    allowed = set(config["quality"]["physical_eligible_qualities"])
    return quality.value in allowed and quality == HomographyQuality.VALID


def solve_frame_homography(
    correspondences: Sequence[Correspondence],
    *,
    config: Mapping[str, Any],
    image_width: float | None = None,
    image_height: float | None = None,
    pitch_length_m: float | None = None,
    pitch_width_m: float | None = None,
) -> FrameHomographySolution:
    """Solve image→pitch H with quality gates; never invent fake calibrations."""
    method = str(config["method"])
    scfg = config["solver"]
    L = float(pitch_length_m if pitch_length_m is not None else config["pitch"]["length_m"])
    W = float(pitch_width_m if pitch_width_m is not None else config["pitch"]["width_m"])
    n = len(correspondences)
    kp_n = sum(1 for c in correspondences if c.source_type == "keypoint")
    ix_n = sum(1 for c in correspondences if c.source_type == "line_intersection")
    diversity = len({c.canonical_pitch_feature_id for c in correspondences})

    if n < int(scfg["min_correspondences"]):
        return _empty(
            quality=HomographyQuality.NOT_AVAILABLE,
            reason="INSUFFICIENT_CORRESPONDENCES",
            method=method,
            n=n,
        )
    if diversity < int(config["correspondence"]["min_feature_diversity"]):
        return _empty(
            quality=HomographyQuality.NOT_AVAILABLE,
            reason="LOW_FEATURE_DIVERSITY",
            method=method,
            n=n,
        )

    src, dst = correspondences_to_arrays(correspondences)
    inlier_mask = np.ones(n, dtype=bool)
    try:
        if method == "ransac_opencv" and n >= 5:
            H, inlier_mask = _ransac_homography(
                src,
                dst,
                threshold=float(scfg["ransac_reproj_threshold_px"]),
                max_iters=int(scfg["ransac_max_iters"]),
                confidence=float(scfg["ransac_confidence"]),
                seed=int(scfg["ransac_seed"]),
            )
            # Optional refine with normalized DLT on inliers.
            if bool(scfg["use_normalized_dlt"]) and int(inlier_mask.sum()) >= 4:
                H = normalized_dlt(src[inlier_mask], dst[inlier_mask])
            solver_method = (
                "ransac_opencv+normalized_dlt" if scfg["use_normalized_dlt"] else "ransac_opencv"
            )
        elif bool(scfg["use_normalized_dlt"]) or method == "dlt_normalized":
            # Preflight duplicate/collinear via Stage 8A solve (raise on degenerate),
            # then use Hartley-normalized DLT as the published matrix.
            try:
                solve_homography(
                    src.tolist(),
                    dst.tolist(),
                    max_condition_number=float("inf"),
                    min_abs_determinant=0.0,
                    round_trip_tolerance_px=float("inf"),
                    round_trip_tolerance_m=float("inf"),
                    max_mean_reprojection_error_px=float("inf"),
                    reject_mirrored=False,
                    pitch_length_m=L,
                    pitch_width_m=W,
                )
            except HomographyError as pre:
                code = str(pre)
                if code in {
                    "INSUFFICIENT_CORRESPONDENCES",
                    "DUPLICATE_CORRESPONDENCES",
                    "COLLINEAR_CORRESPONDENCES",
                }:
                    raise
            H = normalized_dlt(src, dst)
            solver_method = "dlt_normalized"
        else:
            base = solve_homography(
                src.tolist(),
                dst.tolist(),
                max_condition_number=float(scfg["max_condition_number"]),
                min_abs_determinant=float(scfg["min_abs_determinant"]),
                round_trip_tolerance_px=float(scfg["round_trip_tolerance_px"]),
                round_trip_tolerance_m=float(scfg["round_trip_tolerance_m"]),
                max_mean_reprojection_error_px=float(scfg["max_mean_reprojection_error_px"]),
                reject_mirrored=bool(scfg["reject_mirrored"]),
                pitch_length_m=L,
                pitch_width_m=W,
            )
            H = base.H
            solver_method = "dlt_numpy"
    except HomographyError as exc:
        code = str(exc) or "SOLVE_FAILED"
        return _empty(
            quality=(
                HomographyQuality.INVALID
                if "INSUFFICIENT" not in code
                else HomographyQuality.NOT_AVAILABLE
            ),
            reason=code.split(":")[0][:64],
            method=method,
            n=n,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty(
            quality=HomographyQuality.INVALID,
            reason=f"SOLVE_FAILED:{type(exc).__name__}",
            method=method,
            n=n,
        )

    reasons: list[str] = []
    flags: list[str] = []
    det = float(np.linalg.det(H))
    if abs(det) < float(scfg["min_abs_determinant"]):
        return _empty(
            quality=HomographyQuality.INVALID, reason="SINGULAR_HOMOGRAPHY", method=method, n=n
        )
    cond = condition_number(H)
    if not math.isfinite(cond) or cond > float(scfg["max_condition_number"]):
        return _empty(
            quality=HomographyQuality.INVALID,
            reason="ILL_CONDITIONED_HOMOGRAPHY",
            method=method,
            n=n,
        )
    try:
        H_inv = invert_homography(H)
    except HomographyError:
        return _empty(
            quality=HomographyQuality.INVALID, reason="SINGULAR_HOMOGRAPHY", method=method, n=n
        )

    mirrored = detect_mirror(H, length_m=L, width_m=W)
    if mirrored and bool(scfg["reject_mirrored"]):
        return _empty(
            quality=HomographyQuality.INVALID, reason="MIRRORED_HOMOGRAPHY", method=method, n=n
        )

    pred = apply_homography(H, src)
    residuals = np.linalg.norm(pred - dst, axis=1)
    # Recompute inliers if DLT path (all True) or keep RANSAC mask.
    thr = float(scfg["ransac_reproj_threshold_px"])
    if method != "ransac_opencv" or n < 5:
        inlier_mask = residuals <= thr
        if int(inlier_mask.sum()) < int(scfg["min_inlier_count"]):
            inlier_mask = np.ones(n, dtype=bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / n) if n else 0.0
    if inlier_count < int(scfg["min_inlier_count"]) or inlier_ratio < float(
        scfg["min_inlier_ratio"]
    ):
        return _empty(
            quality=HomographyQuality.INVALID, reason="INSUFFICIENT_INLIERS", method=method, n=n
        )

    in_res = residuals[inlier_mask]
    mean_reproj = float(np.mean(in_res))
    median_reproj = float(np.median(in_res))
    max_reproj = float(np.max(in_res))
    mean_pitch = mean_reproj  # already in pitch metres for image_to_pitch H
    if mean_reproj > float(scfg["max_mean_reprojection_error_px"]):
        return _empty(
            quality=HomographyQuality.INVALID, reason="HIGH_REPROJECTION_ERROR", method=method, n=n
        )
    if mean_pitch > float(scfg["max_mean_pitch_error_m"]):
        return _empty(
            quality=HomographyQuality.INVALID, reason="HIGH_PITCH_ERROR", method=method, n=n
        )

    back = apply_homography(H_inv, pred)
    rt_px = float(np.mean(np.linalg.norm(back - src, axis=1)))
    rt_m = mean_pitch
    if rt_px > float(scfg["round_trip_tolerance_px"]) or rt_m > float(
        scfg["round_trip_tolerance_m"]
    ):
        # Soft: allow degraded classification rather than hard fail when close.
        reasons.append("ROUND_TRIP_SOFT_FAIL")
        flags.append("ROUND_TRIP_SOFT_FAIL")

    coverage = _coverage_fraction(src[inlier_mask], width=image_width, height=image_height)
    if coverage < float(scfg["min_coverage_hull_fraction"]):
        return _empty(quality=HomographyQuality.INVALID, reason="LOW_COVERAGE", method=method, n=n)

    # Extrapolation: fraction of inlier pitch points outside expanded pitch bounds.
    tol = float(scfg["pitch_bound_tolerance_m"])
    outside = (
        (pred[:, 0] < -tol) | (pred[:, 0] > L + tol) | (pred[:, 1] < -tol) | (pred[:, 1] > W + tol)
    )
    extrap = float(np.mean(outside.astype(np.float64)))
    if extrap > float(scfg["max_extrapolation_ratio"]):
        reasons.append("HIGH_EXTRAPOLATION")
        flags.append("HIGH_EXTRAPOLATION")

    quality = _classify_quality(
        mean_reproj=mean_reproj,
        inlier_ratio=inlier_ratio,
        coverage=coverage,
        cond=cond,
        config=config,
    )
    if quality == HomographyQuality.INVALID:
        reasons.append("QUALITY_GATES_FAILED")
    if mirrored:
        reasons.append("MIRRORED_HOMOGRAPHY")
        quality = HomographyQuality.INVALID

    eligible = _physical_eligible(quality, config)
    return FrameHomographySolution(
        status="solved" if quality != HomographyQuality.INVALID else "failed",
        quality=quality,
        H=H,
        H_inv=H_inv,
        direction=HOMOGRAPHY_DIRECTION,
        solver_method=solver_method,
        solver_version=SOLVER_VERSION,
        correspondence_count=n,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        inlier_mask=tuple(bool(x) for x in inlier_mask.tolist()),
        residuals_px=tuple(float(x) for x in residuals.tolist()),
        mean_reprojection_error_px=mean_reproj,
        median_reprojection_error_px=median_reproj,
        max_reprojection_error_px=max_reproj,
        mean_pitch_error_m=mean_pitch,
        mean_round_trip_error_px=rt_px,
        mean_round_trip_error_m=rt_m,
        condition_number=cond,
        determinant=det,
        coverage_hull_fraction=coverage,
        canonical_diversity=diversity,
        keypoint_support=kp_n,
        line_intersection_support=ix_n,
        is_mirrored=mirrored,
        extrapolation_ratio=extrap,
        physical_mapping_eligible=eligible,
        reason_codes=tuple(reasons),
        quality_flags=tuple(flags),
    )


def calibration_row_from_solution(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    calibration_id: int,
    solution: FrameHomographySolution,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, Any]:
    is_valid = solution.quality == HomographyQuality.VALID and solution.H is not None
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "calibration_id": int(calibration_id),
        "method": solution.solver_method,
        "is_valid": bool(is_valid),
        "confidence": None,  # never invent calibrated confidence
        "homography_image_to_pitch": (
            solution.matrix_row_major() if solution.H is not None else None
        ),
        "pitch_length_m": float(pitch_length_m),
        "pitch_width_m": float(pitch_width_m),
        "reprojection_error_px": (
            float(solution.mean_reprojection_error_px)
            if solution.mean_reprojection_error_px is not None
            else None
        ),
        "quality_flags": list(solution.quality_flags)
        + [f"quality:{solution.quality.value}"]
        + list(solution.reason_codes),
    }


__all__ = [
    "SOLVER_VERSION",
    "HomographyQuality",
    "FrameHomographySolution",
    "normalized_dlt",
    "solve_frame_homography",
    "calibration_row_from_solution",
]
