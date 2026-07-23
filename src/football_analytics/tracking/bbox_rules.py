"""Track bbox rules — reuse perception xyxy half-open validators."""

from __future__ import annotations

from collections.abc import Sequence

from football_analytics.perception.transforms import BBox, validate_bbox_xyxy
from football_analytics.tracking.types import TrackingContractError


def validate_track_bbox(
    bbox: Sequence[float],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> BBox:
    """Validate source-frame xyxy half-open; reject NaN/Inf/zero-area/OOB."""
    try:
        return validate_bbox_xyxy(
            bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            allow_clip_check=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise TrackingContractError(f"invalid track bbox: {exc}") from exc


__all__ = ["validate_track_bbox"]
