"""Stage 9D spatial quality report helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_spatial_quality_report(
    *,
    heatmap_status: str,
    zone_status: str,
    activity_status: str,
    coverage_ratio: float,
    observed_coverage_us: int,
    derived_coverage_us: int,
    percent_sum: float,
    mass_conservation_ok: bool,
    config_fingerprint: str,
    findings: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "9D",
        "created_at_utc": _utc_now(),
        "heatmap_status": heatmap_status,
        "zone_status": zone_status,
        "activity_status": activity_status,
        "coverage_ratio": coverage_ratio,
        "observed_coverage_us": observed_coverage_us,
        "derived_coverage_us": derived_coverage_us,
        "heatmap_percent_sum": percent_sum,
        "smoothing_mass_conservation_ok": mass_conservation_ok,
        "config_fingerprint": config_fingerprint,
        "findings": list(findings or []),
        "notes": [
            "time_weighted_heatmap_not_frame_counts",
            "missing_coverage_not_inactive",
            "attack_direction_unknown",
            "no_svg_png_committed_to_git",
        ],
    }


def observed_coverage_us(points: Sequence[Mapping[str, Any]]) -> int:
    if len(points) < 2:
        return 0
    times = sorted(int(p["video_time_us"]) for p in points if p.get("video_time_us") is not None)
    if len(times) < 2:
        return 0
    return max(0, times[-1] - times[0])


__all__ = [
    "build_spatial_quality_report",
    "observed_coverage_us",
]
