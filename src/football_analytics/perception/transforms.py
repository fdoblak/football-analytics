"""BBox preprocessing transforms and inverse mapping (Stage 5A)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.perception.types import (
    ChannelOrder,
    ColorSpace,
    Orientation,
    PerceptionContractError,
    PreprocessingTransform,
    ResizeMode,
)

BBox = tuple[float, float, float, float]


class TransformError(PerceptionContractError):
    """Preprocessing / bbox transform failure."""


def _finite(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise TransformError(f"{label} must be finite (no NaN/Inf)")
    return value


def validate_bbox_xyxy(
    bbox: Sequence[float],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    allow_clip_check: bool = True,
) -> BBox:
    """Validate half-open xyxy bbox; reject NaN/Inf/zero/negative area."""
    if len(bbox) != 4:
        raise TransformError("bbox must have 4 values")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    for label, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        _finite(v, label=label)
    if not (x1 < x2 and y1 < y2):
        raise TransformError("bbox requires x1 < x2 and y1 < y2 (positive area)")
    if (
        allow_clip_check
        and frame_width is not None
        and frame_height is not None
        and (x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height)
    ):
        raise TransformError("bbox exceeds frame bounds")
    return (x1, y1, x2, y2)


def clip_bbox_xyxy(
    bbox: Sequence[float], *, frame_width: int, frame_height: int
) -> tuple[BBox, bool]:
    x1, y1, x2, y2 = validate_bbox_xyxy(bbox, allow_clip_check=False)
    cx1 = max(0.0, min(float(frame_width), x1))
    cy1 = max(0.0, min(float(frame_height), y1))
    cx2 = max(0.0, min(float(frame_width), x2))
    cy2 = max(0.0, min(float(frame_height), y2))
    clipped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    validate_bbox_xyxy((cx1, cy1, cx2, cy2), allow_clip_check=False)
    return (cx1, cy1, cx2, cy2), clipped


def letterbox_params(
    source_width: int,
    source_height: int,
    model_input_width: int,
    model_input_height: int,
) -> dict[str, float]:
    if min(source_width, source_height, model_input_width, model_input_height) < 1:
        raise TransformError("dimensions must be >= 1")
    scale = min(model_input_width / source_width, model_input_height / source_height)
    new_w = source_width * scale
    new_h = source_height * scale
    pad_x = model_input_width - new_w
    pad_y = model_input_height - new_h
    pad_left = pad_x / 2.0
    pad_right = pad_x - pad_left
    pad_top = pad_y / 2.0
    pad_bottom = pad_y - pad_top
    return {
        "scale_x": scale,
        "scale_y": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def stretch_params(
    source_width: int,
    source_height: int,
    model_input_width: int,
    model_input_height: int,
) -> dict[str, float]:
    if min(source_width, source_height, model_input_width, model_input_height) < 1:
        raise TransformError("dimensions must be >= 1")
    return {
        "scale_x": model_input_width / source_width,
        "scale_y": model_input_height / source_height,
        "pad_left": 0.0,
        "pad_top": 0.0,
        "pad_right": 0.0,
        "pad_bottom": 0.0,
    }


def build_preprocessing_transform(
    *,
    source_width: int,
    source_height: int,
    model_input_width: int,
    model_input_height: int,
    resize_mode: ResizeMode | str = ResizeMode.LETTERBOX,
    color_space: ColorSpace | str = ColorSpace.RGB,
    channel_order: ChannelOrder | str = ChannelOrder.CHANNELS_LAST,
    orientation: Orientation | str = Orientation.IDENTITY,
    normalization: Mapping[str, Any] | None = None,
    roundtrip_tolerance_px: float = 0.5,
    notes: str | None = None,
) -> PreprocessingTransform:
    mode = ResizeMode(resize_mode) if not isinstance(resize_mode, ResizeMode) else resize_mode
    if mode == ResizeMode.LETTERBOX:
        params = letterbox_params(
            source_width, source_height, model_input_width, model_input_height
        )
    elif mode == ResizeMode.STRETCH:
        params = stretch_params(source_width, source_height, model_input_width, model_input_height)
    else:
        raise TransformError(f"unsupported resize_mode: {mode}")
    norm = dict(normalization or {"kind": "none", "mean": None, "std": None})
    provisional = PreprocessingTransform(
        source_width=source_width,
        source_height=source_height,
        model_input_width=model_input_width,
        model_input_height=model_input_height,
        resize_mode=mode,
        scale_x=params["scale_x"],
        scale_y=params["scale_y"],
        pad_left=params["pad_left"],
        pad_top=params["pad_top"],
        pad_right=params["pad_right"],
        pad_bottom=params["pad_bottom"],
        color_space=(
            ColorSpace(color_space) if not isinstance(color_space, ColorSpace) else color_space
        ),
        channel_order=(
            ChannelOrder(channel_order)
            if not isinstance(channel_order, ChannelOrder)
            else channel_order
        ),
        normalization=norm,
        orientation=(
            Orientation(orientation) if not isinstance(orientation, Orientation) else orientation
        ),
        transform_fingerprint="0" * 64,
        roundtrip_tolerance_px=roundtrip_tolerance_px,
        notes=notes,
    )
    fp = compute_transform_fingerprint(provisional)
    return PreprocessingTransform(
        source_width=provisional.source_width,
        source_height=provisional.source_height,
        model_input_width=provisional.model_input_width,
        model_input_height=provisional.model_input_height,
        resize_mode=provisional.resize_mode,
        scale_x=provisional.scale_x,
        scale_y=provisional.scale_y,
        pad_left=provisional.pad_left,
        pad_top=provisional.pad_top,
        pad_right=provisional.pad_right,
        pad_bottom=provisional.pad_bottom,
        color_space=provisional.color_space,
        channel_order=provisional.channel_order,
        normalization=provisional.normalization,
        orientation=provisional.orientation,
        transform_fingerprint=fp,
        schema_version=provisional.schema_version,
        roundtrip_tolerance_px=provisional.roundtrip_tolerance_px,
        notes=provisional.notes,
    )


def compute_transform_fingerprint(transform: PreprocessingTransform) -> str:
    payload = {
        "schema_version": transform.schema_version,
        "source_width": transform.source_width,
        "source_height": transform.source_height,
        "model_input_width": transform.model_input_width,
        "model_input_height": transform.model_input_height,
        "resize_mode": transform.resize_mode.value,
        "scale_x": transform.scale_x,
        "scale_y": transform.scale_y,
        "pad_left": transform.pad_left,
        "pad_top": transform.pad_top,
        "pad_right": transform.pad_right,
        "pad_bottom": transform.pad_bottom,
        "color_space": transform.color_space.value,
        "channel_order": transform.channel_order.value,
        "normalization": dict(transform.normalization),
        "orientation": transform.orientation.value,
        "roundtrip_tolerance_px": transform.roundtrip_tolerance_px,
    }
    return hash_canonical_json(payload)


def forward_bbox(bbox_source: Sequence[float], transform: PreprocessingTransform) -> BBox:
    """Map source-frame xyxy → model-input xyxy."""
    x1, y1, x2, y2 = validate_bbox_xyxy(bbox_source, allow_clip_check=False)
    sx, sy = transform.scale_x, transform.scale_y
    return (
        x1 * sx + transform.pad_left,
        y1 * sy + transform.pad_top,
        x2 * sx + transform.pad_left,
        y2 * sy + transform.pad_top,
    )


def inverse_bbox(bbox_model: Sequence[float], transform: PreprocessingTransform) -> BBox:
    """Map model-input xyxy → canonical source-frame xyxy."""
    x1, y1, x2, y2 = validate_bbox_xyxy(bbox_model, allow_clip_check=False)
    if transform.scale_x <= 0 or transform.scale_y <= 0:
        raise TransformError("INVERSE_TRANSFORM_FAILED: non-positive scale")
    out = (
        (x1 - transform.pad_left) / transform.scale_x,
        (y1 - transform.pad_top) / transform.scale_y,
        (x2 - transform.pad_left) / transform.scale_x,
        (y2 - transform.pad_top) / transform.scale_y,
    )
    return validate_bbox_xyxy(out, allow_clip_check=False)


def roundtrip_bbox(
    bbox_source: Sequence[float],
    transform: PreprocessingTransform,
    *,
    tolerance_px: float | None = None,
) -> BBox:
    tol = transform.roundtrip_tolerance_px if tolerance_px is None else float(tolerance_px)
    fwd = forward_bbox(bbox_source, transform)
    back = inverse_bbox(fwd, transform)
    orig = validate_bbox_xyxy(bbox_source, allow_clip_check=False)
    for a, b in zip(orig, back, strict=True):
        if abs(a - b) > tol:
            raise TransformError(f"roundtrip exceeded tolerance {tol}: {orig} → {back}")
    return back


__all__ = [
    "BBox",
    "TransformError",
    "validate_bbox_xyxy",
    "clip_bbox_xyxy",
    "letterbox_params",
    "stretch_params",
    "build_preprocessing_transform",
    "compute_transform_fingerprint",
    "forward_bbox",
    "inverse_bbox",
    "roundtrip_bbox",
]
