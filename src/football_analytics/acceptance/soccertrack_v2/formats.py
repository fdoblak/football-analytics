"""SoccerTrack v2 on-disk format dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GsrPlayerObservation:
    """One player observation on one panoramic frame (reference GT only)."""

    half: int
    image_id: int
    frame_index: int
    track_id: int
    player_id: str
    role: str
    jersey_number: int | None
    team_side: str | None
    x_m: float
    y_m: float
    bbox_image: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BasEvent:
    """One ball-action spotting event (reference GT only)."""

    half: int
    clock: str
    t_ms: int
    label: str
    team: str | None
    player_id: str | None
    visibility: str | None = None


@dataclass(frozen=True)
class VideoHalfMeta:
    match_id: str
    half: int
    path: str
    width: int | None
    height: int | None
    fps: float | None
    duration_s: float | None
    frame_count: int | None


def maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
