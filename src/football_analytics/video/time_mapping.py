"""PTS → video_time_us mapping and quality classification (Stage 3D / 3D-F1)."""

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

_TIME_REWRITE_TRANSFORMS = frozenset({"force_cfr"})
_SIGNIFICANT_MISSING_RATIO = 0.25
_SIGNIFICANT_NON_MONO_RATIO = 0.1
_SIGNIFICANT_DUP_RATIO = 0.1


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


@dataclass(frozen=True)
class MappingEvidence:
    """Normalization / probe provenance used to classify mapping quality."""

    has_normalization_receipt: bool
    normalization_status: str | None  # succeeded/skipped/...
    frame_rate_conversion_performed: bool | None
    frame_rate_conversion_source_mode: str | None  # cfr/vfr/unknown
    frame_rate_conversion_target_mode: str | None
    requires_stage3d_mapping: bool | None
    duration_drift_us: int | None
    applied_transforms: tuple[str, ...]
    # constant offset if proven (microseconds); None if unknown
    constant_offset_us: int | None
    identity_proven: bool  # skipped ALREADY_CANONICAL / no time rewrite


def empty_mapping_evidence() -> MappingEvidence:
    return MappingEvidence(
        has_normalization_receipt=False,
        normalization_status=None,
        frame_rate_conversion_performed=None,
        frame_rate_conversion_source_mode=None,
        frame_rate_conversion_target_mode=None,
        requires_stage3d_mapping=None,
        duration_drift_us=None,
        applied_transforms=(),
        constant_offset_us=None,
        identity_proven=False,
    )


def _has_time_transform(evidence: MappingEvidence) -> bool:
    if evidence.frame_rate_conversion_performed is True:
        return True
    if evidence.requires_stage3d_mapping is True:
        return True
    if evidence.constant_offset_us not in (None, 0):
        return True
    return any(t in _TIME_REWRITE_TRANSFORMS for t in evidence.applied_transforms)


def _significant_resampling_stats(stats: MappingStats) -> bool:
    n = max(stats.frame_count, 1)
    return (
        (stats.missing_pts_count / n) > _SIGNIFICANT_MISSING_RATIO
        or (stats.non_monotonic_pts_count / n) > _SIGNIFICANT_NON_MONO_RATIO
        or (stats.duplicate_pts_count / n) > _SIGNIFICANT_DUP_RATIO
    )


def _incomplete_pts(stats: MappingStats) -> bool:
    return bool(
        stats.missing_pts_count
        or stats.duplicate_pts_count
        or stats.non_monotonic_pts_count
        or stats.skipped_count
        or stats.unknown_count
        or stats.failed_count
    )


def _vfr_to_cfr(evidence: MappingEvidence) -> bool:
    return (
        evidence.frame_rate_conversion_source_mode == "vfr"
        and evidence.frame_rate_conversion_target_mode == "cfr"
    )


def _evidence_conflicts(evidence: MappingEvidence, frame_rate_mode: FrameRateMode) -> bool:
    if not evidence.has_normalization_receipt:
        return False
    if evidence.identity_proven and (
        evidence.frame_rate_conversion_performed is True or _has_time_transform(evidence)
    ):
        return True
    if (
        _vfr_to_cfr(evidence)
        and evidence.frame_rate_conversion_performed is False
        and evidence.requires_stage3d_mapping is not True
    ):
        return True
    tgt = evidence.frame_rate_conversion_target_mode
    if (
        evidence.frame_rate_conversion_performed is False
        and tgt in {"cfr", "vfr"}
        and frame_rate_mode not in {FrameRateMode.UNKNOWN}
        and frame_rate_mode.value != tgt
    ):
        return True
    return evidence.identity_proven and evidence.duration_drift_us not in (None, 0)


def classify_mapping_quality(
    stats: MappingStats,
    *,
    frame_rate_mode: FrameRateMode,
    evidence: MappingEvidence | None = None,
) -> MappingQuality:
    """Classify timeline mapping quality from stats + optional normalization evidence."""
    if stats.invented_from_index_or_fps:
        return MappingQuality.NOT_AVAILABLE
    if stats.frame_count == 0:
        return MappingQuality.NOT_AVAILABLE
    if stats.failed_count > 0 and stats.ok_count == 0:
        return MappingQuality.NOT_AVAILABLE

    ev = evidence if evidence is not None else empty_mapping_evidence()

    resampling = (
        ev.frame_rate_conversion_performed is True
        or _vfr_to_cfr(ev)
        or ev.requires_stage3d_mapping is True
        or _significant_resampling_stats(stats)
        or any(t in _TIME_REWRITE_TRANSFORMS for t in ev.applied_transforms)
    )
    if resampling:
        return MappingQuality.DERIVED_WITH_RESAMPLING

    if (not ev.has_normalization_receipt) or _evidence_conflicts(ev, frame_rate_mode):
        return MappingQuality.UNCERTAIN

    if ev.constant_offset_us is not None and ev.constant_offset_us != 0:
        return MappingQuality.DERIVED_WITH_CONSTANT_OFFSET

    if ev.identity_proven and not _has_time_transform(ev):
        return MappingQuality.EXACT_IDENTITY

    pts_complete = (not _incomplete_pts(stats)) and stats.ok_count == stats.frame_count
    if (
        ev.normalization_status in {"succeeded", "skipped"}
        and ev.frame_rate_conversion_performed is False
        and pts_complete
        and not _has_time_transform(ev)
    ):
        return MappingQuality.TIMESTAMP_PRESERVED

    if _incomplete_pts(stats):
        return MappingQuality.UNCERTAIN

    return MappingQuality.UNCERTAIN


def assert_no_index_fps_invention(policy: dict[str, Any]) -> None:
    if policy.get("invent_time_from_index_or_fps") is not False:
        raise VideoContractError("invent_time_from_index_or_fps must be false")
