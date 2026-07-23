"""Canonical frame timeline time rules (no fps invent; VFR-safe µs gaps)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.tracking.types import TrackingContractError


def resolve_video_time_us(
    frame_index: int,
    frames_by_index: Mapping[int, Mapping[str, Any]],
) -> int:
    """Resolve video_time_us from canonical frames table; never invent from fps."""
    row = frames_by_index.get(int(frame_index))
    if row is None:
        raise TrackingContractError(f"missing frame for index {frame_index}")
    if "video_time_us" not in row:
        raise TrackingContractError("frames row missing video_time_us")
    t = int(row["video_time_us"])
    if t < 0:
        raise TrackingContractError("video_time_us must be >= 0")
    return t


def gap_us(earlier_us: int, later_us: int) -> int:
    """Integer microsecond gap; later must be >= earlier."""
    a = int(earlier_us)
    b = int(later_us)
    if b < a:
        raise TrackingContractError(f"timestamp reverse: {a} -> {b}")
    return b - a


def require_monotonic_times(
    pairs: Sequence[tuple[int, int]],
    *,
    label: str = "track",
) -> None:
    """Require non-decreasing (frame_index, video_time_us) sequence."""
    prev_fi: int | None = None
    prev_t: int | None = None
    for fi, t in pairs:
        fi_i = int(fi)
        t_i = int(t)
        if prev_fi is not None and fi_i < prev_fi:
            raise TrackingContractError(f"{label}: frame_index reverse")
        if prev_t is not None and t_i < prev_t:
            raise TrackingContractError(f"{label}: timestamp reverse")
        prev_fi = fi_i
        prev_t = t_i


def frames_index_map(frames: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for r in frames:
        out[int(r["frame_index"])] = r
    return out


__all__ = [
    "resolve_video_time_us",
    "gap_us",
    "require_monotonic_times",
    "frames_index_map",
]
