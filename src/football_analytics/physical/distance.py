"""Deterministic segment-internal Euclidean pitch distance (Stage 9C)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentDistanceResult:
    trajectory_segment_id: str
    sample_layer: str
    distance_m: float | None
    interval_count: int
    sample_count: int
    measured_duration_us: int
    status: str
    reason_codes: tuple[str, ...]
    uncertainty_m: float | None
    diagnostic: bool


def euclidean_m(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(float(x1) - float(x0), float(y1) - float(y0))


def _eligible_point(p: Mapping[str, Any], *, uncertainty_max_m: float) -> bool:
    if str(p.get("metric_eligibility", "not_eligible")) != "eligible":
        return False
    if str(p.get("mapping_status", "")) != "mapped":
        return False
    u = p.get("uncertainty_m")
    if u is not None and float(u) > uncertainty_max_m:
        return False
    x, y = p.get("pitch_x_m"), p.get("pitch_y_m")
    if x is None or y is None:
        return False
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return False
    t = p.get("video_time_us")
    return t is not None


def compute_segment_distance(
    points: Sequence[Mapping[str, Any]],
    *,
    trajectory_segment_id: str,
    sample_layer: str,
    config: Mapping[str, Any],
    diagnostic: bool = False,
) -> SegmentDistanceResult:
    """Sum Euclidean distances between consecutive eligible points in one segment.

    Never bridges hard gaps / other segments. Single-point → not_evaluable.
    """
    dist_cfg = config["distance"]
    unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
    min_dt = int(dist_cfg["min_delta_time_us"])
    max_dt = int(dist_cfg["max_delta_time_us"])
    ordered = sorted(
        [p for p in points if _eligible_point(p, uncertainty_max_m=unc_max)],
        key=lambda r: int(r["video_time_us"]),
    )
    if len(ordered) < int(dist_cfg["min_points_per_segment"]):
        return SegmentDistanceResult(
            trajectory_segment_id=trajectory_segment_id,
            sample_layer=sample_layer,
            distance_m=None,
            interval_count=0,
            sample_count=len(ordered),
            measured_duration_us=0,
            status="not_evaluable",
            reason_codes=("SINGLE_SAMPLE_SEGMENT_INSUFFICIENT",),
            uncertainty_m=None,
            diagnostic=diagnostic,
        )

    total = 0.0
    intervals = 0
    reasons: list[str] = []
    unc_vals: list[float] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        t0, t1 = int(a["video_time_us"]), int(b["video_time_us"])
        dt = t1 - t0
        if dt <= 0:
            reasons.append("ZERO_OR_NEGATIVE_DELTA_TIME")
            continue
        if dt < min_dt or dt > max_dt:
            reasons.append("DELTA_TIME_OUT_OF_POLICY")
            continue
        step = euclidean_m(
            float(a["pitch_x_m"]),
            float(a["pitch_y_m"]),
            float(b["pitch_x_m"]),
            float(b["pitch_y_m"]),
        )
        if not math.isfinite(step):
            reasons.append("NON_FINITE_DISTANCE")
            continue
        total += step
        intervals += 1
        for p in (a, b):
            if p.get("uncertainty_m") is not None:
                unc_vals.append(float(p["uncertainty_m"]))

    measured_us = int(ordered[-1]["video_time_us"]) - int(ordered[0]["video_time_us"])
    if intervals == 0:
        return SegmentDistanceResult(
            trajectory_segment_id=trajectory_segment_id,
            sample_layer=sample_layer,
            distance_m=None,
            interval_count=0,
            sample_count=len(ordered),
            measured_duration_us=max(0, measured_us),
            status="not_evaluable",
            reason_codes=tuple(sorted(set(reasons))) or ("NO_VALID_INTERVALS",),
            uncertainty_m=None,
            diagnostic=diagnostic,
        )

    unc = max(unc_vals) if unc_vals else None
    return SegmentDistanceResult(
        trajectory_segment_id=trajectory_segment_id,
        sample_layer=sample_layer,
        distance_m=float(total),
        interval_count=intervals,
        sample_count=len(ordered),
        measured_duration_us=max(0, measured_us),
        status="computed",
        reason_codes=tuple(sorted(set(reasons))),
        uncertainty_m=unc,
        diagnostic=diagnostic,
    )


def aggregate_measured_distance(
    segment_results: Sequence[SegmentDistanceResult],
    *,
    analysis_window_us: int,
    min_coverage_ratio: float,
) -> dict[str, Any]:
    """Aggregate segment distances without extrapolating uncovered time."""
    measured = 0.0
    measured_us = 0
    samples = 0
    segs: list[str] = []
    reasons: list[str] = []
    for r in segment_results:
        if r.diagnostic:
            continue
        if r.status == "computed" and r.distance_m is not None:
            measured += float(r.distance_m)
            measured_us += int(r.measured_duration_us)
            samples += int(r.sample_count)
            segs.append(r.trajectory_segment_id)
        else:
            reasons.extend(r.reason_codes)

    coverage = (measured_us / analysis_window_us) if analysis_window_us > 0 else 0.0
    if samples < 2 or measured_us <= 0:
        return {
            "value_m": None,
            "status": "not_evaluable",
            "measured_distance_m": None,
            "measured_eligible_duration_us": measured_us,
            "coverage_ratio": coverage,
            "included_sample_count": samples,
            "segment_ids": segs,
            "reason_codes": tuple(sorted(set(reasons))) or ("INSUFFICIENT_COVERAGE",),
        }
    if coverage < min_coverage_ratio:
        return {
            "value_m": None,
            "status": "not_evaluable",
            "measured_distance_m": float(measured),
            "measured_eligible_duration_us": measured_us,
            "coverage_ratio": coverage,
            "included_sample_count": samples,
            "segment_ids": segs,
            "reason_codes": ("INSUFFICIENT_COVERAGE", "NO_COVERAGE_EXTRAPOLATION"),
        }
    return {
        "value_m": float(measured),
        "status": "computed",
        "measured_distance_m": float(measured),
        "measured_eligible_duration_us": measured_us,
        "coverage_ratio": coverage,
        "included_sample_count": samples,
        "segment_ids": segs,
        "reason_codes": tuple(sorted(set(reasons))),
    }


__all__ = [
    "SegmentDistanceResult",
    "euclidean_m",
    "compute_segment_distance",
    "aggregate_measured_distance",
]
