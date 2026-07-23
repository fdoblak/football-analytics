"""Image vs pitch coordinate helpers and frame-ID enforcement (Stage 8A)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from football_analytics.calibration.types import CalibrationContractError, CoordinateFrameId

IMAGE_FRAME = CoordinateFrameId.SOURCE_IMAGE.value
PITCH_FRAME = CoordinateFrameId.CANONICAL_PITCH.value
ATTACK_FRAME = CoordinateFrameId.ATTACK_RELATIVE.value

ALLOWED_FRAMES = frozenset(f.value for f in CoordinateFrameId)


@dataclass(frozen=True)
class ImagePoint:
    x_px: float
    y_px: float
    frame_id: str = IMAGE_FRAME

    def __post_init__(self) -> None:
        if self.frame_id != IMAGE_FRAME and self.frame_id not in ALLOWED_FRAMES:
            raise CalibrationContractError(f"unknown image frame_id: {self.frame_id}")
        if not (math.isfinite(self.x_px) and math.isfinite(self.y_px)):
            raise CalibrationContractError("image coordinates must be finite")


@dataclass(frozen=True)
class PitchPointM:
    x_m: float
    y_m: float
    frame_id: str = PITCH_FRAME

    def __post_init__(self) -> None:
        if self.frame_id not in {PITCH_FRAME, ATTACK_FRAME, "unknown"}:
            raise CalibrationContractError(f"invalid pitch frame_id: {self.frame_id}")
        if not (math.isfinite(self.x_m) and math.isfinite(self.y_m)):
            raise CalibrationContractError("pitch coordinates must be finite")


def validate_coordinate_frame_id(frame_id: str) -> str:
    if frame_id not in ALLOWED_FRAMES:
        raise CalibrationContractError(f"unknown coordinate_frame_id: {frame_id}")
    return frame_id


def assert_frames_not_mixed(frame_a: str, frame_b: str) -> None:
    if frame_a != frame_b:
        raise CalibrationContractError(f"coordinate frame mixing forbidden: {frame_a} vs {frame_b}")


def validate_image_point_in_frame(
    x_px: float,
    y_px: float,
    *,
    frame_width: int,
    frame_height: int,
    half_open: bool = True,
) -> None:
    if not (math.isfinite(x_px) and math.isfinite(y_px)):
        raise CalibrationContractError("non-finite image coordinates")
    if frame_width <= 0 or frame_height <= 0:
        raise CalibrationContractError("invalid frame dimensions")
    if x_px < 0 or y_px < 0:
        raise CalibrationContractError("image coordinates outside frame (negative)")
    if half_open:
        if x_px >= frame_width or y_px >= frame_height:
            raise CalibrationContractError("image coordinates outside frame bounds")
    else:
        if x_px > frame_width or y_px > frame_height:
            raise CalibrationContractError("image coordinates outside frame bounds")


def pitch_point_in_bounds(
    x_m: float,
    y_m: float,
    *,
    length_m: float,
    width_m: float,
    tolerance_m: float = 0.0,
) -> bool:
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        return False
    return (
        -tolerance_m <= x_m <= length_m + tolerance_m
        and -tolerance_m <= y_m <= width_m + tolerance_m
    )


def human_footpoint_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> ImagePoint:
    x1, y1, x2, y2 = bbox_xyxy
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise CalibrationContractError("non-finite bbox")
    if not (x2 > x1 and y2 > y1):
        raise CalibrationContractError("invalid bbox for footpoint")
    return ImagePoint(x_px=(x1 + x2) / 2.0, y_px=y2, frame_id=IMAGE_FRAME)


def ball_centre_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> ImagePoint:
    x1, y1, x2, y2 = bbox_xyxy
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise CalibrationContractError("non-finite bbox")
    if not (x2 > x1 and y2 > y1):
        raise CalibrationContractError("invalid bbox for ball centre")
    return ImagePoint(x_px=(x1 + x2) / 2.0, y_px=(y1 + y2) / 2.0, frame_id=IMAGE_FRAME)


def default_attack_direction() -> str:
    return "unknown"


def coordinate_system_summary() -> dict[str, Any]:
    return {
        "image": {
            "frame_id": IMAGE_FRAME,
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "units": "pixels",
        },
        "pitch": {
            "frame_id": PITCH_FRAME,
            "origin": "corner_goal_a_touchline_left",
            "x_axis": "length_toward_goal_b",
            "y_axis": "width_toward_right_touchline",
            "units": "metres",
            "not_attack_direction": True,
        },
        "attack_direction_default": default_attack_direction(),
    }


__all__ = [
    "IMAGE_FRAME",
    "PITCH_FRAME",
    "ATTACK_FRAME",
    "ALLOWED_FRAMES",
    "ImagePoint",
    "PitchPointM",
    "validate_coordinate_frame_id",
    "assert_frames_not_mixed",
    "validate_image_point_in_frame",
    "pitch_point_in_bounds",
    "human_footpoint_from_bbox",
    "ball_centre_from_bbox",
    "default_attack_direction",
    "coordinate_system_summary",
]
