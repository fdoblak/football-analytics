"""Stage 9B trajectory quality report helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def build_trajectory_quality_report(
    *,
    raw_count: int,
    filtered_count: int,
    resampled_count: int,
    rejected_count: int,
    segment_count: int,
    gap_count: int,
    reason_counts: Mapping[str, int],
    single_point_segments: int,
    observed_coverage_us: int,
    derived_coverage_us: int,
    config_fingerprint: str,
) -> dict[str, Any]:
    status = "pass"
    findings: list[str] = []
    if rejected_count > raw_count:
        status = "fail"
        findings.append("rejected_count_exceeds_raw")
    if single_point_segments:
        findings.append(f"single_point_segments={single_point_segments}")
        status = "pass_with_findings" if status == "pass" else status
    findings.append("customer_physical_metrics_not_computed")
    findings.append("attack_direction_unknown")
    return {
        "schema_version": 1,
        "status": status,
        "config_fingerprint": config_fingerprint,
        "counts": {
            "raw": raw_count,
            "filtered": filtered_count,
            "resampled": resampled_count,
            "rejected": rejected_count,
            "segments": segment_count,
            "gaps": gap_count,
            "single_point_segments": single_point_segments,
        },
        "reason_code_distribution": dict(Counter(reason_counts)),
        "coverage": {
            "observed_coverage_us": observed_coverage_us,
            "derived_coverage_us": derived_coverage_us,
            "note": "derived_coverage_is_not_observed_coverage",
        },
        "findings": findings,
        "customer_metrics_computed": False,
    }


def observed_coverage_us(points: Sequence[Mapping[str, Any]]) -> int:
    if len(points) < 2:
        return 0
    times = sorted(int(p["video_time_us"]) for p in points)
    return max(0, times[-1] - times[0])


__all__ = ["build_trajectory_quality_report", "observed_coverage_us"]
