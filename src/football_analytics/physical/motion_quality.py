"""Stage 9C motion metric quality report helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_motion_quality_report(
    *,
    primary_layer: str,
    segment_metric_count: int,
    sprint_bout_count: int,
    evaluable_sprint_count: int,
    measured_eligible_duration_us: int,
    observed_coverage_us: int,
    derived_coverage_us: int,
    measured_distance_m: float | None,
    robust_mean_mps: float | None,
    robust_peak_mps: float | None,
    not_evaluable_reasons: Sequence[str],
    config_fingerprint: str,
    findings: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "9C",
        "created_at_utc": _utc_now(),
        "primary_sample_layer": primary_layer,
        "segment_metric_count": int(segment_metric_count),
        "sprint_bout_count": int(sprint_bout_count),
        "evaluable_sprint_count": int(evaluable_sprint_count),
        "measured_eligible_duration_us": int(measured_eligible_duration_us),
        "observed_coverage_us": int(observed_coverage_us),
        "derived_coverage_us": int(derived_coverage_us),
        "measured_distance_m": measured_distance_m,
        "robust_mean_speed_mps": robust_mean_mps,
        "robust_peak_speed_mps": robust_peak_mps,
        "not_evaluable_reasons": list(not_evaluable_reasons),
        "config_fingerprint": config_fingerprint,
        "findings": list(findings or []),
        "notes": [
            "measured_distance ≠ full-match distance estimate",
            "no coverage extrapolation",
            "sprint not official Opta",
        ],
    }


def observed_coverage_from_points(points: Sequence[Mapping[str, Any]]) -> int:
    if len(points) < 2:
        return 0
    times = sorted(int(p["video_time_us"]) for p in points if p.get("video_time_us") is not None)
    if len(times) < 2:
        return 0
    return max(0, times[-1] - times[0])


__all__ = [
    "build_motion_quality_report",
    "observed_coverage_from_points",
]
