"""Canonical source-pixel annotation coordinates (R1-F1).

Storage space is always the raw source frame:
  width=1336, height=744, origin top-left, x right, y down, bbox xyxy half-open.

Display canvases may be resized/letterboxed; saved boxes must round-trip through
source space with ≤1 px error.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SOURCE_WIDTH = 1336
SOURCE_HEIGHT = 744
COORDINATE_SPACE = "source_xyxy_px_v1"
MAX_ROUNDTRIP_ERR_PX = 1.0


class CoordinateError(ValueError):
    """Hard failure for invalid annotation coordinates."""


@dataclass(frozen=True)
class DisplayTransform:
    """Map from source pixels to a display canvas (possibly letterboxed)."""

    canvas_width: int
    canvas_height: int
    scale: float
    pad_x: float
    pad_y: float
    content_width: float
    content_height: float

    @property
    def fingerprint(self) -> str:
        raw = (
            f"{self.canvas_width}x{self.canvas_height}|s={self.scale:.10f}|"
            f"pad={self.pad_x:.10f},{self.pad_y:.10f}|"
            f"cw={self.content_width:.10f},ch={self.content_height:.10f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SourceBBox:
    frame_index: int
    video_time_us: int
    x1: float
    y1: float
    x2: float
    y2: float
    source_width: int = SOURCE_WIDTH
    source_height: int = SOURCE_HEIGHT
    coordinate_space: str = COORDINATE_SPACE
    transform_fingerprint: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transform_fingerprint(payload: Mapping[str, Any] | DisplayTransform | str) -> str:
    if isinstance(payload, DisplayTransform):
        return payload.fingerprint
    if isinstance(payload, str):
        return payload
    raw = "|".join(f"{k}={payload[k]}" for k in sorted(payload))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_letterbox_transform(
    source_w: int,
    source_h: int,
    canvas_w: int,
    canvas_h: int,
) -> DisplayTransform:
    if source_w <= 0 or source_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        raise CoordinateError("dimensions must be positive")
    scale = min(canvas_w / source_w, canvas_h / source_h)
    content_w = source_w * scale
    content_h = source_h * scale
    pad_x = (canvas_w - content_w) / 2.0
    pad_y = (canvas_h - content_h) / 2.0
    return DisplayTransform(
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        content_width=content_w,
        content_height=content_h,
    )


def make_stretch_transform(
    source_w: int,
    source_h: int,
    canvas_w: int,
    canvas_h: int,
) -> DisplayTransform:
    """Non-uniform stretch (discouraged for labeling UI; supported for tests)."""
    if source_w <= 0 or source_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        raise CoordinateError("dimensions must be positive")
    # Represent as scale_x via content mapping with pad=0 and scale on width;
    # for stretch we store scale as width scale and encode height via content_height.
    return DisplayTransform(
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        scale=canvas_w / source_w,
        pad_x=0.0,
        pad_y=0.0,
        content_width=float(canvas_w),
        content_height=float(canvas_h),
    )


def _is_stretch(t: DisplayTransform, source_w: int, source_h: int) -> bool:
    expected_h = source_h * (t.content_width / source_w) if source_w else 0
    height_mismatch = abs(t.content_height - expected_h) > 1e-6
    scale_mismatch = abs(t.scale * source_h - t.content_height) > 1e-6
    return height_mismatch or scale_mismatch


def source_point_to_canvas(
    x: float,
    y: float,
    transform: DisplayTransform,
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> tuple[float, float]:
    if _is_stretch(transform, source_w, source_h):
        sx = transform.content_width / source_w
        sy = transform.content_height / source_h
        return transform.pad_x + x * sx, transform.pad_y + y * sy
    return transform.pad_x + x * transform.scale, transform.pad_y + y * transform.scale


def canvas_point_to_source(
    x: float,
    y: float,
    transform: DisplayTransform,
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> tuple[float, float]:
    if _is_stretch(transform, source_w, source_h):
        sx = transform.content_width / source_w
        sy = transform.content_height / source_h
        if sx <= 0 or sy <= 0:
            raise CoordinateError("invalid stretch scale")
        return (x - transform.pad_x) / sx, (y - transform.pad_y) / sy
    if transform.scale <= 0:
        raise CoordinateError("invalid scale")
    return (x - transform.pad_x) / transform.scale, (y - transform.pad_y) / transform.scale


def source_bbox_to_canvas(
    bbox_xyxy: Sequence[float],
    transform: DisplayTransform,
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    cx1, cy1 = source_point_to_canvas(x1, y1, transform, source_w=source_w, source_h=source_h)
    cx2, cy2 = source_point_to_canvas(x2, y2, transform, source_w=source_w, source_h=source_h)
    return [cx1, cy1, cx2, cy2]


def canvas_bbox_to_source(
    bbox_xyxy: Sequence[float],
    transform: DisplayTransform,
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    sx1, sy1 = canvas_point_to_source(x1, y1, transform, source_w=source_w, source_h=source_h)
    sx2, sy2 = canvas_point_to_source(x2, y2, transform, source_w=source_w, source_h=source_h)
    return [sx1, sy1, sx2, sy2]


def validate_source_bbox_xyxy(
    bbox: Sequence[float],
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
    half_open: bool = True,
) -> list[float]:
    if len(bbox) != 4:
        raise CoordinateError("bbox must have 4 values")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    for name, v in ("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2):
        if not math.isfinite(v):
            raise CoordinateError(f"{name} is NaN/Inf")
    # Hard-fail common xywh confusion: if "x2/y2" look like small positive w/h
    # while also x1+x2<=w and the caller tagged format wrong — explicit API only.
    if x2 <= x1 or y2 <= y1:
        # Distinguish xywh mistaken input: if x2,y2 positive and x1+x2<=w, y1+y2<=h
        looks_xywh = (
            x2 > 0
            and y2 > 0
            and (x1 + x2) <= source_w
            and (y1 + y2) <= source_h
            and x2 < source_w
            and y2 < source_h
        )
        if looks_xywh:
            raise CoordinateError("xywh_detected_as_xyxy")
        raise CoordinateError("zero_or_negative_area")
    if half_open:
        if x1 < 0 or y1 < 0 or x2 > source_w or y2 > source_h:
            raise CoordinateError("out_of_bounds")
    else:
        if x1 < 0 or y1 < 0 or x2 >= source_w or y2 >= source_h:
            raise CoordinateError("out_of_bounds")
    if (x2 - x1) * (y2 - y1) <= 0:
        raise CoordinateError("zero_area")
    return [x1, y1, x2, y2]


def reject_xywh(bbox: Sequence[float], *, format_tag: str | None) -> None:
    if format_tag and format_tag.lower() == "xywh":
        raise CoordinateError("xywh_not_allowed")


def roundtrip_error_px(
    bbox_xyxy: Sequence[float],
    transform: DisplayTransform,
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> float:
    canvas = source_bbox_to_canvas(bbox_xyxy, transform, source_w=source_w, source_h=source_h)
    back = canvas_bbox_to_source(canvas, transform, source_w=source_w, source_h=source_h)
    return max(abs(float(a) - float(b)) for a, b in zip(bbox_xyxy, back, strict=True))


def make_source_bbox(
    *,
    frame_index: int,
    fps: float,
    bbox_xyxy: Sequence[float],
    transform: DisplayTransform | None = None,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> SourceBBox:
    if frame_index < 0:
        raise CoordinateError("frame_index must be 0-based non-negative")
    box = validate_source_bbox_xyxy(bbox_xyxy, source_w=source_w, source_h=source_h)
    video_time_us = int(round((frame_index / fps) * 1_000_000)) if fps > 0 else 0
    fp = transform.fingerprint if transform is not None else "identity"
    return SourceBBox(
        frame_index=frame_index,
        video_time_us=video_time_us,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        source_width=source_w,
        source_height=source_h,
        transform_fingerprint=fp,
    )
