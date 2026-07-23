"""3x3 homography solve / validate (numpy DLT); invert; condition; round-trip."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.calibration.types import HomographyError

HOMOGRAPHY_DIRECTION = "image_to_pitch"
SOLVER_METHOD = "dlt_numpy"
SOLVER_VERSION = "1"
DEFAULT_MAX_CONDITION = 1.0e8
DEFAULT_MIN_ABS_DET = 1.0e-12
DEFAULT_ROUND_TRIP_PX = 1.0
DEFAULT_ROUND_TRIP_M = 0.05
DEFAULT_MAX_MEAN_REPROJ_PX = 5.0


@dataclass(frozen=True)
class HomographyResult:
    H: np.ndarray  # 3x3 image_to_pitch
    H_inv: np.ndarray
    direction: str
    condition_number: float
    determinant: float
    correspondence_count: int
    inlier_count: int
    mean_reprojection_error_px: float
    mean_round_trip_error_px: float
    mean_round_trip_error_m: float
    is_mirrored: bool
    solver_method: str
    solver_version: str
    status: str
    reason_codes: tuple[str, ...]

    def matrix_row_major(self) -> list[float]:
        return [float(x) for x in self.H.reshape(9)]

    def inverse_row_major(self) -> list[float]:
        return [float(x) for x in self.H_inv.reshape(9)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "homography_image_to_pitch": self.matrix_row_major(),
            "homography_pitch_to_image": self.inverse_row_major(),
            "direction": self.direction,
            "condition_number": self.condition_number,
            "determinant": self.determinant,
            "correspondence_count": self.correspondence_count,
            "inlier_count": self.inlier_count,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "mean_round_trip_error_px": self.mean_round_trip_error_px,
            "mean_round_trip_error_m": self.mean_round_trip_error_m,
            "is_mirrored": self.is_mirrored,
            "solver_method": self.solver_method,
            "solver_version": self.solver_version,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


def _as_nx2(points: Sequence[Sequence[float]] | np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise HomographyError(f"{label} must be Nx2")
    if not np.all(np.isfinite(arr)):
        raise HomographyError(f"{label} contains non-finite values")
    return arr


def _has_duplicates(pts: np.ndarray, *, tol: float = 1e-9) -> bool:
    n = pts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(pts[i] - pts[j])) <= tol:
                return True
    return False


def _is_collinear(pts: np.ndarray, *, area_tol: float = 1e-6) -> bool:
    """True if all points are (near) collinear / rank-deficient in 2D."""
    if pts.shape[0] < 3:
        return True
    centered = pts - pts.mean(axis=0, keepdims=True)
    # SVD: if second singular value ~0, points are collinear.
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.shape[0] < 2:
        return True
    span = float(s[0]) + 1e-12
    if float(s[1]) / span < area_tol:
        return True
    # Also check max triangle area relative to bbox.
    max_area = 0.0
    for i in range(pts.shape[0] - 2):
        for j in range(i + 1, pts.shape[0] - 1):
            for k in range(j + 1, pts.shape[0]):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = (
                    abs(float(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])))
                    / 2.0
                )
                max_area = max(max_area, area)
    bbox = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
    if bbox <= 0:
        return True
    return max_area < area_tol * max(bbox, 1.0)


def normalize_homography(H: np.ndarray, *, by_h22: bool = True) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(H)):
        raise HomographyError("homography contains non-finite values")
    if by_h22:
        scale = H[2, 2]
        if abs(float(scale)) < 1e-15:
            # Fall back to Frobenius normalization.
            fro = float(np.linalg.norm(H))
            if fro < 1e-15:
                raise HomographyError("zero homography")
            H = H / fro
        else:
            H = H / scale
    return H


def matrix_from_row_major(values: Sequence[float]) -> np.ndarray:
    if len(values) != 9:
        raise HomographyError("homography must have 9 values")
    return normalize_homography(np.asarray(values, dtype=np.float64).reshape(3, 3))


def invert_homography(H: np.ndarray) -> np.ndarray:
    H = normalize_homography(H)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise HomographyError("homography not invertible") from exc
    if not np.all(np.isfinite(H_inv)):
        raise HomographyError("inverse homography non-finite")
    return normalize_homography(H_inv)


def condition_number(H: np.ndarray) -> float:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    s = np.linalg.svd(H, compute_uv=False)
    if float(s[-1]) < 1e-15:
        return float("inf")
    return float(s[0] / s[-1])


def apply_homography(H: np.ndarray, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    pts = _as_nx2(points, label="points")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    mapped = (H @ homo.T).T
    w = mapped[:, 2:3]
    if np.any(np.abs(w) < 1e-15):
        raise HomographyError("homography mapped points to infinity")
    out = mapped[:, :2] / w
    if not np.all(np.isfinite(out)):
        raise HomographyError("homography mapped to non-finite")
    return out


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = src.shape[0]
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return normalize_homography(H)


def detect_mirror(H: np.ndarray, *, length_m: float = 105.0, width_m: float = 68.0) -> bool:
    """Detect orientation-reversing (mirrored / axis-flipped) mapping via corner winding."""
    corners_img = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        dtype=np.float64,
    )
    # Use synthetic image rectangle mapped to pitch — compare signed area.
    try:
        mapped = apply_homography(H, corners_img)
    except HomographyError:
        return True
    # Signed area of mapped quad (shoelace).
    x = mapped[:, 0]
    y = mapped[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # Expected image winding is positive (CCW for top-left origin y-down is CW → negative).
    # Image top-left origin with y-down: (0,0)->(100,0)->(100,100)->(0,100) has positive shoelace
    # in mathematical coords but with y-down it's clockwise → negative area.
    # Mirror flips sign relative to expected.
    expected_sign = -1.0  # CW in y-down image
    if area == 0.0:
        return True
    # Also map pitch corners and ensure length/width axes keep orientation.
    pitch_corners = np.array(
        [[0.0, 0.0], [length_m, 0.0], [length_m, width_m], [0.0, width_m]],
        dtype=np.float64,
    )
    try:
        H_inv = invert_homography(H)
        img_of_pitch = apply_homography(H_inv, pitch_corners)
    except HomographyError:
        return True
    px = img_of_pitch[:, 0]
    py = img_of_pitch[:, 1]
    pitch_area = 0.5 * float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
    if pitch_area == 0.0:
        return True
    # Mirror if image quad winding flipped unexpectedly relative to expected_sign.
    area_sign = math.copysign(1.0, area)
    pitch_sign = math.copysign(1.0, pitch_area)
    if area_sign != expected_sign and pitch_sign == expected_sign:
        return True
    # Explicit axis-flip: det of linear part.
    det2 = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
    return det2 < 0


def solve_homography(
    image_points: Sequence[Sequence[float]],
    pitch_points: Sequence[Sequence[float]],
    *,
    max_condition_number: float = DEFAULT_MAX_CONDITION,
    min_abs_determinant: float = DEFAULT_MIN_ABS_DET,
    round_trip_tolerance_px: float = DEFAULT_ROUND_TRIP_PX,
    round_trip_tolerance_m: float = DEFAULT_ROUND_TRIP_M,
    max_mean_reprojection_error_px: float = DEFAULT_MAX_MEAN_REPROJ_PX,
    reject_mirrored: bool = True,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> HomographyResult:
    """Solve image→pitch homography from correspondences; reject degenerates."""
    src = _as_nx2(image_points, label="image_points")
    dst = _as_nx2(pitch_points, label="pitch_points")
    if src.shape[0] != dst.shape[0]:
        raise HomographyError("image/pitch correspondence count mismatch")
    n = src.shape[0]
    reasons: list[str] = []
    if n < 4:
        raise HomographyError("INSUFFICIENT_CORRESPONDENCES")
    if _has_duplicates(src) or _has_duplicates(dst):
        raise HomographyError("DUPLICATE_CORRESPONDENCES")
    if _is_collinear(src) or _is_collinear(dst):
        raise HomographyError("COLLINEAR_CORRESPONDENCES")

    H = _dlt(src, dst)
    det = float(np.linalg.det(H))
    if abs(det) < min_abs_determinant:
        raise HomographyError("SINGULAR_HOMOGRAPHY")
    cond = condition_number(H)
    if not math.isfinite(cond) or cond > max_condition_number:
        raise HomographyError("ILL_CONDITIONED_HOMOGRAPHY")
    try:
        H_inv = invert_homography(H)
    except HomographyError as exc:
        raise HomographyError("SINGULAR_HOMOGRAPHY") from exc

    mirrored = detect_mirror(H, length_m=pitch_length_m, width_m=pitch_width_m)
    if mirrored and reject_mirrored:
        raise HomographyError("MIRRORED_HOMOGRAPHY")

    pred_pitch = apply_homography(H, src)
    reproj = float(np.mean(np.linalg.norm(pred_pitch - dst, axis=1)))
    if reproj > max_mean_reprojection_error_px:
        raise HomographyError("HIGH_REPROJECTION_ERROR")

    back_img = apply_homography(H_inv, pred_pitch)
    rt_px = float(np.mean(np.linalg.norm(back_img - src, axis=1)))
    forward_again = apply_homography(H, src)
    rt_m = float(np.mean(np.linalg.norm(forward_again - dst, axis=1)))
    if rt_px > round_trip_tolerance_px or rt_m > round_trip_tolerance_m:
        raise HomographyError("ROUND_TRIP_FAILURE")

    status = "valid"
    if mirrored:
        reasons.append("MIRRORED_HOMOGRAPHY")
        status = "invalid"
    return HomographyResult(
        H=H,
        H_inv=H_inv,
        direction=HOMOGRAPHY_DIRECTION,
        condition_number=cond,
        determinant=det,
        correspondence_count=n,
        inlier_count=n,
        mean_reprojection_error_px=reproj,
        mean_round_trip_error_px=rt_px,
        mean_round_trip_error_m=rt_m,
        is_mirrored=mirrored,
        solver_method=SOLVER_METHOD,
        solver_version=SOLVER_VERSION,
        status=status,
        reason_codes=tuple(reasons),
    )


def validate_homography_matrix(
    H_values: Sequence[float],
    *,
    image_points: Sequence[Sequence[float]] | None = None,
    pitch_points: Sequence[Sequence[float]] | None = None,
    max_condition_number: float = DEFAULT_MAX_CONDITION,
    min_abs_determinant: float = DEFAULT_MIN_ABS_DET,
    reject_mirrored: bool = True,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> HomographyResult:
    H = matrix_from_row_major(H_values)
    det = float(np.linalg.det(H))
    if abs(det) < min_abs_determinant:
        raise HomographyError("SINGULAR_HOMOGRAPHY")
    cond = condition_number(H)
    if not math.isfinite(cond) or cond > max_condition_number:
        raise HomographyError("ILL_CONDITIONED_HOMOGRAPHY")
    H_inv = invert_homography(H)
    mirrored = detect_mirror(H, length_m=pitch_length_m, width_m=pitch_width_m)
    if mirrored and reject_mirrored:
        raise HomographyError("MIRRORED_HOMOGRAPHY")
    n = 0
    reproj = 0.0
    rt_px = 0.0
    rt_m = 0.0
    if image_points is not None and pitch_points is not None:
        src = _as_nx2(image_points, label="image_points")
        dst = _as_nx2(pitch_points, label="pitch_points")
        n = src.shape[0]
        pred = apply_homography(H, src)
        reproj = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
        back = apply_homography(H_inv, pred)
        rt_px = float(np.mean(np.linalg.norm(back - src, axis=1)))
        rt_m = reproj
    return HomographyResult(
        H=H,
        H_inv=H_inv,
        direction=HOMOGRAPHY_DIRECTION,
        condition_number=cond,
        determinant=det,
        correspondence_count=n,
        inlier_count=n,
        mean_reprojection_error_px=reproj,
        mean_round_trip_error_px=rt_px,
        mean_round_trip_error_m=rt_m,
        is_mirrored=mirrored,
        solver_method=SOLVER_METHOD,
        solver_version=SOLVER_VERSION,
        status="valid" if not mirrored else "invalid",
        reason_codes=(),
    )


def identity_homography() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def scale_translate_homography(
    *, scale_x: float, scale_y: float, tx: float, ty: float
) -> np.ndarray:
    H = np.array([[scale_x, 0.0, tx], [0.0, scale_y, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    return normalize_homography(H)


__all__ = [
    "HOMOGRAPHY_DIRECTION",
    "SOLVER_METHOD",
    "SOLVER_VERSION",
    "DEFAULT_MAX_CONDITION",
    "DEFAULT_MIN_ABS_DET",
    "DEFAULT_ROUND_TRIP_PX",
    "DEFAULT_ROUND_TRIP_M",
    "DEFAULT_MAX_MEAN_REPROJ_PX",
    "HomographyResult",
    "normalize_homography",
    "matrix_from_row_major",
    "invert_homography",
    "condition_number",
    "apply_homography",
    "detect_mirror",
    "solve_homography",
    "validate_homography_matrix",
    "identity_homography",
    "scale_translate_homography",
]
