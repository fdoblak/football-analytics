"""SoccerTrack v2 on-disk format dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GsrPlayerObservation:
    """One player observation on one panoramic frame (reference GT only)."""

    half: int
    image_id: int
    frame_index: int
    track_id: int
    player_id: str
    role: str
    jersey_number: Optional[int]
    team_side: Optional[str]
    x_m: float
    y_m: float
    bbox_image: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class BasEvent:
    """One ball-action spotting event (reference GT only)."""

    half: int
    clock: str
    t_ms: int
    label: str
    team: Optional[str]
    player_id: Optional[str]
    visibility: Optional[str] = None


@dataclass(frozen=True)
class VideoHalfMeta:
    match_id: str
    half: int
    path: str
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_s: Optional[float]
    frame_count: Optional[int]


def maybe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
