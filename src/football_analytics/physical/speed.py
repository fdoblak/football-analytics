"""Speed from metre distance and video_time_us deltas (Stage 9C)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.physical.distance import euclidean_m
from football_analytics.physical.semantics import speed_delta_seconds


@dataclass(frozen=True)
class InstantSpeedSample:
    t0_us: int
    t1_us: int
    mid_time_us: int
    distance_m: float
    speed_mps: float
    uncertainty_m: float | None
    rejected: bool
    reject_reason: str | None


@dataclass(frozen=True)
class SegmentSpeedResult:
    trajectory_segment_id: str
    sample_layer: str
    instantaneous: tuple[InstantSpeedSample, ...]
    robust_mean_mps: float | None
    robust_peak_mps: float | None
    diagnostic_raw_peak_mps: float | None
    sample_count: int
    included_interval_count: int
    measured_duration_us: int
    status: str
    reason_codes: tuple[str, ...]
    uncertainty_m: float | None
    diagnostic: bool


def mps_to_kmh(mps: float) -> float:
    return float(mps) * 3.6


def _median(vals: Sequence[float]) -> float:
    ordered = sorted(vals)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _median_smooth(values: Sequence[float], *, window: int) -> list[float]:
    if window < 1 or not values:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(_median(values[lo:hi]))
    return out


def compute_segment_speeds(
    points: Sequence[Mapping[str, Any]],
    *,
    trajectory_segment_id: str,
    sample_layer: str,
    config: Mapping[str, Any],
    diagnostic: bool = False,
) -> SegmentSpeedResult:
    """Compute instantaneous + robust speeds inside one eligible segment."""
    spd = config["speed"]
    unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
    min_dt = int(spd["min_delta_time_us"])
    max_dt = int(spd["max_delta_time_us"])
    outlier = float(spd["outlier_impossible_speed_mps"])
    exclude_outliers = bool(spd.get("exclude_quality_gate_outliers_from_customer", True))
    window = int(spd["smoothing"]["window_samples"])
    peak_min_n = int(spd["peak_speed"]["min_support_samples"])
    peak_min_us = int(spd["peak_speed"]["min_support_duration_us"])

    ordered = sorted(
        [
            p
            for p in points
            if str(p.get("metric_eligibility", "not_eligible")) == "eligible"
            and str(p.get("mapping_status", "")) == "mapped"
            and p.get("video_time_us") is not None
            and p.get("pitch_x_m") is not None
            and p.get("pitch_y_m") is not None
            and math.isfinite(float(p["pitch_x_m"]))
            and math.isfinite(float(p["pitch_y_m"]))
            and (p.get("uncertainty_m") is None or float(p["uncertainty_m"]) <= unc_max)
        ],
        key=lambda r: int(r["video_time_us"]),
    )
    if len(ordered) < 2:
        return SegmentSpeedResult(
            trajectory_segment_id=trajectory_segment_id,
            sample_layer=sample_layer,
            instantaneous=(),
            robust_mean_mps=None,
            robust_peak_mps=None,
            diagnostic_raw_peak_mps=None,
            sample_count=len(ordered),
            included_interval_count=0,
            measured_duration_us=0,
            status="not_evaluable",
            reason_codes=("SINGLE_SAMPLE_SEGMENT_INSUFFICIENT",),
            uncertainty_m=None,
            diagnostic=diagnostic,
        )

    inst: list[InstantSpeedSample] = []
    reasons: list[str] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        t0, t1 = int(a["video_time_us"]), int(b["video_time_us"])
        try:
            dt_s = speed_delta_seconds(t0_us=t0, t1_us=t1)
        except Exception:
            reasons.append("ZERO_OR_NEGATIVE_DELTA_TIME")
            inst.append(
                InstantSpeedSample(
                    t0_us=t0,
                    t1_us=t1,
                    mid_time_us=(t0 + t1) // 2,
                    distance_m=0.0,
                    speed_mps=float("nan"),
                    uncertainty_m=None,
                    rejected=True,
                    reject_reason="ZERO_OR_NEGATIVE_DELTA_TIME",
                )
            )
            continue
        dt_us = t1 - t0
        if dt_us < min_dt or dt_us > max_dt:
            reasons.append("DELTA_TIME_OUT_OF_POLICY")
            inst.append(
                InstantSpeedSample(
                    t0_us=t0,
                    t1_us=t1,
                    mid_time_us=(t0 + t1) // 2,
                    distance_m=0.0,
                    speed_mps=float("nan"),
                    uncertainty_m=None,
                    rejected=True,
                    reject_reason="DELTA_TIME_OUT_OF_POLICY",
                )
            )
            continue
        dist = euclidean_m(
            float(a["pitch_x_m"]),
            float(a["pitch_y_m"]),
            float(b["pitch_x_m"]),
            float(b["pitch_y_m"]),
        )
        speed = dist / dt_s
        unc_vals = [float(p["uncertainty_m"]) for p in (a, b) if p.get("uncertainty_m") is not None]
        unc = max(unc_vals) if unc_vals else None
        rejected = False
        reject_reason = None
        if not math.isfinite(speed):
            rejected = True
            reject_reason = "NON_FINITE_SPEED"
            reasons.append(reject_reason)
        elif exclude_outliers and speed > outlier:
            rejected = True
            reject_reason = "QUALITY_GATE_OUTLIER_SPEED"
            reasons.append(reject_reason)
        inst.append(
            InstantSpeedSample(
                t0_us=t0,
                t1_us=t1,
                mid_time_us=(t0 + t1) // 2,
                distance_m=float(dist),
                speed_mps=float(speed),
                uncertainty_m=unc,
                rejected=rejected,
                reject_reason=reject_reason,
            )
        )

    accepted = [s for s in inst if not s.rejected and math.isfinite(s.speed_mps)]
    raw_peak = max((s.speed_mps for s in inst if math.isfinite(s.speed_mps)), default=None)
    measured_us = int(ordered[-1]["video_time_us"]) - int(ordered[0]["video_time_us"])
    if not accepted:
        return SegmentSpeedResult(
            trajectory_segment_id=trajectory_segment_id,
            sample_layer=sample_layer,
            instantaneous=tuple(inst),
            robust_mean_mps=None,
            robust_peak_mps=None,
            diagnostic_raw_peak_mps=raw_peak,
            sample_count=len(ordered),
            included_interval_count=0,
            measured_duration_us=max(0, measured_us),
            status="not_evaluable",
            reason_codes=tuple(sorted(set(reasons))) or ("NO_VALID_SPEED_INTERVALS",),
            uncertainty_m=None,
            diagnostic=diagnostic,
        )

    smooth = _median_smooth([s.speed_mps for s in accepted], window=window)
    robust_mean = float(sum(smooth) / len(smooth))

    # Peak requires sustained support — not a single noisy sample.
    robust_peak: float | None = None
    if len(smooth) >= peak_min_n:
        # Sliding window over accepted intervals by time span.
        for i in range(len(accepted)):
            for j in range(i + peak_min_n - 1, len(accepted)):
                span = accepted[j].t1_us - accepted[i].t0_us
                if span < peak_min_us:
                    continue
                window_vals = smooth[i : j + 1]
                cand = max(window_vals)
                if robust_peak is None or cand > robust_peak:
                    robust_peak = float(cand)
        if robust_peak is None and spd["peak_speed"].get("single_two_point_spike_not_peak"):
            reasons.append("PEAK_INSUFFICIENT_SUPPORT")
    else:
        reasons.append("PEAK_INSUFFICIENT_SUPPORT")

    unc_vals2 = [s.uncertainty_m for s in accepted if s.uncertainty_m is not None]
    unc = max(unc_vals2) if unc_vals2 else None
    return SegmentSpeedResult(
        trajectory_segment_id=trajectory_segment_id,
        sample_layer=sample_layer,
        instantaneous=tuple(inst),
        robust_mean_mps=robust_mean,
        robust_peak_mps=robust_peak,
        diagnostic_raw_peak_mps=raw_peak,
        sample_count=len(ordered),
        included_interval_count=len(accepted),
        measured_duration_us=max(0, measured_us),
        status="computed",
        reason_codes=tuple(sorted(set(reasons))),
        uncertainty_m=unc,
        diagnostic=diagnostic,
    )


def aggregate_speed_summary(
    segment_results: Sequence[SegmentSpeedResult],
    *,
    analysis_window_us: int,
    min_coverage_ratio: float,
    min_eligible_duration_us: int,
) -> dict[str, Any]:
    means: list[float] = []
    peaks: list[float] = []
    measured_us = 0
    samples = 0
    segs: list[str] = []
    reasons: list[str] = []
    for r in segment_results:
        if r.diagnostic:
            continue
        measured_us += int(r.measured_duration_us)
        samples += int(r.sample_count)
        segs.append(r.trajectory_segment_id)
        reasons.extend(r.reason_codes)
        if r.status == "computed" and r.robust_mean_mps is not None:
            means.append(float(r.robust_mean_mps))
        if r.robust_peak_mps is not None:
            peaks.append(float(r.robust_peak_mps))

    coverage = (measured_us / analysis_window_us) if analysis_window_us > 0 else 0.0
    if measured_us < min_eligible_duration_us or coverage < min_coverage_ratio or not means:
        return {
            "robust_mean_mps": None,
            "robust_peak_mps": None,
            "status": "not_evaluable",
            "measured_eligible_duration_us": measured_us,
            "coverage_ratio": coverage,
            "included_sample_count": samples,
            "segment_ids": segs,
            "reason_codes": tuple(sorted(set(reasons)))
            or ("INSUFFICIENT_COVERAGE", "SPEED_NOT_EVALUABLE"),
        }
    return {
        "robust_mean_mps": float(sum(means) / len(means)),
        "robust_peak_mps": float(max(peaks)) if peaks else None,
        "status": "computed",
        "measured_eligible_duration_us": measured_us,
        "coverage_ratio": coverage,
        "included_sample_count": samples,
        "segment_ids": segs,
        "reason_codes": tuple(sorted(set(reasons))),
    }


__all__ = [
    "InstantSpeedSample",
    "SegmentSpeedResult",
    "mps_to_kmh",
    "compute_segment_speeds",
    "aggregate_speed_summary",
]
