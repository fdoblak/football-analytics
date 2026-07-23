"""Constant-velocity bbox / center motion helpers (Stage 6B)."""

from __future__ import annotations

from collections.abc import Sequence

BBox = tuple[float, float, float, float]


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_wh(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def translate_bbox(bbox: Sequence[float], dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def velocity_from_centers(
    prev_center: Sequence[float],
    prev_time_us: int,
    curr_center: Sequence[float],
    curr_time_us: int,
) -> tuple[float, float]:
    """Return (vx, vy) in pixels per microsecond."""
    dt = int(curr_time_us) - int(prev_time_us)
    if dt <= 0:
        return (0.0, 0.0)
    px, py = float(prev_center[0]), float(prev_center[1])
    cx, cy = float(curr_center[0]), float(curr_center[1])
    return ((cx - px) / float(dt), (cy - py) / float(dt))


def predict_bbox_constant_velocity(
    bbox: Sequence[float],
    *,
    vx: float,
    vy: float,
    dt_us: int,
) -> BBox:
    """Translate bbox by constant center velocity over dt_us (size unchanged)."""
    if dt_us <= 0:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return (x1, y1, x2, y2)
    dx = float(vx) * float(dt_us)
    dy = float(vy) * float(dt_us)
    return translate_bbox(bbox, dx, dy)


def predict_center_constant_velocity(
    center: Sequence[float],
    *,
    vx: float,
    vy: float,
    dt_us: int,
) -> tuple[float, float]:
    if dt_us <= 0:
        return (float(center[0]), float(center[1]))
    return (
        float(center[0]) + float(vx) * float(dt_us),
        float(center[1]) + float(vy) * float(dt_us),
    )


__all__ = [
    "BBox",
    "bbox_center",
    "bbox_wh",
    "translate_bbox",
    "velocity_from_centers",
    "predict_bbox_constant_velocity",
    "predict_center_constant_velocity",
]
