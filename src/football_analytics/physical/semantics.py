"""Trajectory sample / segment / gap semantic helpers (Stage 9A — no real metrics)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.types import (
    GapType,
    PhysicalContractError,
    SampleSource,
    SegmentStatus,
)


def assert_finite_pitch_point(x: float, y: float, *, length_m: float, width_m: float) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        raise PhysicalContractError("NaN/Inf pitch coordinate rejected")
    if x < 0.0 or y < 0.0 or x > length_m or y > width_m:
        raise PhysicalContractError("pitch coordinate outside template bounds")


def assert_strictly_increasing_times(times_us: Sequence[int]) -> None:
    if len(times_us) < 2:
        return
    for a, b in zip(times_us, times_us[1:], strict=False):
        if int(b) <= int(a):
            raise PhysicalContractError("DUPLICATE_OR_OUT_OF_ORDER_TIME")


def assert_no_uncontrolled_duplicates(
    samples: Sequence[Mapping[str, Any]],
) -> None:
    seen: set[tuple[str, str, str, int]] = set()
    for s in samples:
        key = (
            str(s["run_id"]),
            str(s["video_id"]),
            str(s["target_player_id"]),
            int(s["video_time_us"]),
        )
        if key in seen and str(s.get("sample_source")) == SampleSource.RAW_OBSERVED.value:
            raise PhysicalContractError("duplicate raw sample at same target/time")
        seen.add(key)


def assert_raw_immutable(sample: Mapping[str, Any], *, mutated_fields: Sequence[str]) -> None:
    if str(sample.get("sample_source")) == SampleSource.RAW_OBSERVED.value and mutated_fields:
        raise PhysicalContractError("raw_observed sample is immutable")


def assert_derived_provenance(sample: Mapping[str, Any]) -> None:
    src = str(sample.get("sample_source"))
    if src in {SampleSource.FILTERED.value, SampleSource.RESAMPLED.value}:
        derived = list(sample.get("derived_from_sample_ids") or [])
        if not derived:
            raise PhysicalContractError("derived sample requires source sample IDs")


def distance_bridge_allowed(gap: Mapping[str, Any]) -> bool:
    """Gaps never allow silent distance bridging by default."""
    if gap.get("allows_distance_bridge") is True:
        raise PhysicalContractError("GAP_DISTANCE_BRIDGE_FORBIDDEN")
    return False


def segment_metric_sufficient(segment: Mapping[str, Any]) -> bool:
    if int(segment.get("eligible_sample_count", 0)) < 2:
        return False
    if str(segment.get("segment_status")) == SegmentStatus.INSUFFICIENT.value:
        return False
    return str(segment.get("metric_eligibility")) == "eligible"


def classify_gap_type(name: str) -> str:
    try:
        return GapType(name).value
    except ValueError as exc:
        raise PhysicalContractError(f"unknown gap type: {name}") from exc


def half_open_contains(start_us: int, end_us: int, t_us: int) -> bool:
    return int(start_us) <= int(t_us) < int(end_us)


def speed_delta_seconds(*, t0_us: int, t1_us: int) -> float:
    """Canonical speed time base: microseconds → seconds (never fps)."""
    dt = int(t1_us) - int(t0_us)
    if dt <= 0:
        raise PhysicalContractError("SPEED_TIME_UNIT_VIOLATION")
    return dt / 1_000_000.0


def sprint_from_single_spike(*, sample_count: int, duration_us: int, min_duration_us: int) -> bool:
    return not (sample_count < 2 or duration_us < min_duration_us)


def heatmap_weighting_is_time(*, weighting: str) -> bool:
    return weighting == "time_weighted"


def low_coverage_means_inactivity(*, policy: Mapping[str, Any]) -> bool:
    """Must always be False under Stage 9A policy."""
    return policy.get("activity_coverage", {}).get("low_coverage_is_not_low_activity") is not True


__all__ = [
    "assert_finite_pitch_point",
    "assert_strictly_increasing_times",
    "assert_no_uncontrolled_duplicates",
    "assert_raw_immutable",
    "assert_derived_provenance",
    "distance_bridge_allowed",
    "segment_metric_sufficient",
    "classify_gap_type",
    "half_open_contains",
    "speed_delta_seconds",
    "sprint_from_single_spike",
    "heatmap_weighting_is_time",
    "low_coverage_means_inactivity",
]
