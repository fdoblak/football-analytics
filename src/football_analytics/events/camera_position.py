"""Conservative camera_position resolution (Stage 13A).

Only supported view families map to a position; everything else stays unknown.
Never invent when uncertain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.broadcast.types import CameraPosition, ViewFamily

# Conservative defaults — only when view family is explicitly supported.
DEFAULT_SUPPORTED: Mapping[str, str] = {
    ViewFamily.MAIN_BROADCAST.value: CameraPosition.SIDELINE.value,
    ViewFamily.GOAL_VIEW.value: CameraPosition.BEHIND_GOAL.value,
    ViewFamily.AERIAL.value: CameraPosition.OVERHEAD.value,
    ViewFamily.TACTICAL.value: CameraPosition.SIDELINE.value,
}


def resolve_camera_position(
    *,
    view_family: str | None,
    supported_by_view: Mapping[str, str] | None = None,
    forced: str | None = None,
    uncertain: bool = False,
) -> str:
    """Return camera_position for supported classes only; else unknown."""
    if uncertain:
        return CameraPosition.UNKNOWN.value
    if forced is not None:
        allowed = {e.value for e in CameraPosition}
        return forced if forced in allowed else CameraPosition.UNKNOWN.value
    mapping = dict(supported_by_view or DEFAULT_SUPPORTED)
    vf = str(view_family or ViewFamily.UNKNOWN.value)
    if vf not in mapping:
        return CameraPosition.UNKNOWN.value
    pos = str(mapping[vf])
    allowed = {e.value for e in CameraPosition}
    if pos not in allowed:
        return CameraPosition.UNKNOWN.value
    return pos


def camera_position_is_supported(
    view_family: str, *, config: Mapping[str, Any] | None = None
) -> bool:
    mapping = dict((config or {}).get("supported_camera_position_by_view") or DEFAULT_SUPPORTED)
    return str(view_family) in mapping


__all__ = [
    "DEFAULT_SUPPORTED",
    "resolve_camera_position",
    "camera_position_is_supported",
]
