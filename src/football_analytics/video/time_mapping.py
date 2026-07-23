"""PTS → video_time_us mapping and quality classification (Stage 3D)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

from football_analytics.video.types import (
    FrameRateMode,
    MappingQuality,
    Rational,
    VideoContractError,
)


def pts_to_video_time_us(pts: int, time_base: Rational) -> int:
    """Convert packet PTS to video-relative microseconds using exact rationals.

    ``video_time_us = round_half_even(pts * time_base.num / time_base.den * 1_000_000)``.
    Never uses float frame-rate strings.
    """
    if not isinstance(pts, int) or isinstance(pts, bool):
        raise VideoContractError("pts must be int")
    if pts < 0:
        raise VideoContractError("pts must be >= 0 for video_time_us mapping")
    num = Decimal(pts) * Decimal(time_base.numerator) * Decimal(1_000_000)
    den = Decimal(time_base.denominator)
    value = (num / den).to_integral_value(rounding=ROUND_HALF_EVEN)
    out = int(value)
    if out < 0:
        raise VideoContractError("mapped video_time_us must be >= 0")
    return out


def duration_ts_to_us(duration_ts: int | None, time_base: Rational) -> int | None:
    if duration_ts is None:
        return None
    if not isinstance(duration_ts, int) or isinstance(duration_ts, bool):
        raise VideoContractError("duration_ts must be int")
    if duration_ts <= 0:
        return None
    return pts_to_video_time_us(duration_ts, time_base)


@dataclass
class MappingStats:
    frame_count: int = 0
    ok_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    unknown_count: int = 0
    missing_pts_count: int = 0
    duplicate_pts_count: int = 0
    non_monotonic_pts_count: int = 0
    invented_from_index_or_fps: bool = False

    def note_status(self, status: str) -> None:
        if status == "ok":
            self.ok_count += 1
        elif status == "skipped":
            self.skipped_count += 1
        elif status == "failed":
            self.failed_count += 1
        elif status == "unknown":
            self.unknown_count += 1
        else:
            raise VideoContractError(f"unknown decode_status: {status}")


def classify_mapping_quality(
    stats: MappingStats,
    *,
    frame_rate_mode: FrameRateMode,
) -> MappingQuality:
    """Classify timeline mapping quality without inventing metrics."""
    if stats.invented_from_index_or_fps:
        return MappingQuality.FAILED
    if stats.frame_count == 0:
        return MappingQuality.FAILED
    if stats.failed_count > 0 and stats.ok_count == 0:
        return MappingQuality.FAILED
    missing_ratio = stats.missing_pts_count / max(stats.frame_count, 1)
    non_mono_ratio = stats.non_monotonic_pts_count / max(stats.frame_count, 1)
    if missing_ratio > 0.25 or non_mono_ratio > 0.1:
        return MappingQuality.UNRELIABLE
    if (
        stats.missing_pts_count
        or stats.duplicate_pts_count
        or stats.non_monotonic_pts_count
        or stats.skipped_count
        or stats.unknown_count
    ):
        return MappingQuality.DEGRADED
    if frame_rate_mode == FrameRateMode.CFR and stats.ok_count == stats.frame_count:
        return MappingQuality.EXACT
    if frame_rate_mode == FrameRateMode.VFR and stats.ok_count == stats.frame_count:
        # VFR with complete PTS is good, never "exact" via invented CFR index.
        return MappingQuality.GOOD
    if stats.ok_count == stats.frame_count:
        return MappingQuality.GOOD
    return MappingQuality.DEGRADED


def assert_no_index_fps_invention(policy: dict[str, Any]) -> None:
    if policy.get("invent_time_from_index_or_fps") is not False:
        raise VideoContractError("invent_time_from_index_or_fps must be false")
