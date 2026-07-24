"""Trajectory-based physical movement activity profile (Stage 9D)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.distance import euclidean_m
from football_analytics.physical.semantics import speed_delta_seconds

CLASS_ORDER = (
    "stationary",
    "walking",
    "jogging",
    "running",
    "sprinting",
    "unknown",
    "not_evaluable",
)


def classify_speed_mps(speed_mps: float, *, classes: Mapping[str, Any]) -> str:
    if not math.isfinite(speed_mps) or speed_mps < 0:
        return "not_evaluable"
    for name in ("stationary", "walking", "jogging", "running", "sprinting"):
        band = classes[name]
        lo, hi = float(band["min_mps"]), float(band["max_mps"])
        if lo <= speed_mps < hi or (name == "sprinting" and speed_mps >= lo):
            return name
    return "unknown"


def compute_activity_distribution(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    analysis_window_us: int | None = None,
    gap_unobserved_us: int = 0,
) -> dict[str, Any]:
    """Speed-class durations from eligible segment intervals (no gap fill as inactive)."""
    act = config["activity"]
    classes = act["classes"]
    unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
    min_dt = int(act["min_delta_time_us"])
    max_dt = int(act["max_delta_time_us"])
    outlier = float(act["outlier_impossible_speed_mps"])

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        if str(p.get("metric_eligibility", "not_eligible")) != "eligible":
            continue
        if str(p.get("mapping_status", "")) != "mapped":
            continue
        if (
            p.get("video_time_us") is None
            or p.get("pitch_x_m") is None
            or p.get("pitch_y_m") is None
        ):
            continue
        if not math.isfinite(float(p["pitch_x_m"])) or not math.isfinite(float(p["pitch_y_m"])):
            continue
        u = p.get("uncertainty_m")
        if u is not None and float(u) > unc_max:
            continue
        sid = str(p.get("trajectory_segment_id") or "traj_seg_unknown")
        groups.setdefault(sid, []).append(dict(p))

    durations: dict[str, int] = {c: 0 for c in CLASS_ORDER}
    reasons: list[str] = []
    eligible_us = 0
    unc_vals: list[float] = []
    sample_n = 0

    for _sid, pts in groups.items():
        ordered = sorted(pts, key=lambda r: int(r["video_time_us"]))
        sample_n += len(ordered)
        if len(ordered) < 2:
            reasons.append("SINGLE_SAMPLE_NO_ACTIVITY")
            continue
        eligible_us += int(ordered[-1]["video_time_us"]) - int(ordered[0]["video_time_us"])
        for a, b in zip(ordered, ordered[1:], strict=False):
            t0, t1 = int(a["video_time_us"]), int(b["video_time_us"])
            try:
                dt_s = speed_delta_seconds(t0_us=t0, t1_us=t1)
            except Exception:
                durations["not_evaluable"] += max(0, t1 - t0)
                reasons.append("ZERO_OR_NEGATIVE_DELTA_TIME")
                continue
            dt_us = t1 - t0
            if dt_us < min_dt or dt_us > max_dt:
                durations["not_evaluable"] += dt_us
                reasons.append("DELTA_TIME_OUT_OF_POLICY")
                continue
            dist = euclidean_m(
                float(a["pitch_x_m"]),
                float(a["pitch_y_m"]),
                float(b["pitch_x_m"]),
                float(b["pitch_y_m"]),
            )
            speed = dist / dt_s
            if speed > outlier:
                durations["not_evaluable"] += dt_us
                reasons.append("QUALITY_GATE_OUTLIER_SPEED")
                continue
            label = classify_speed_mps(speed, classes=classes)
            durations[label] = durations.get(label, 0) + dt_us
            for p in (a, b):
                if p.get("uncertainty_m") is not None:
                    unc_vals.append(float(p["uncertainty_m"]))

    window = int(analysis_window_us) if analysis_window_us is not None else max(1, eligible_us)
    coverage = (eligible_us / window) if window > 0 else 0.0
    min_cov = float(act["min_coverage_ratio_for_computed"])
    min_elig = int(act["min_eligible_duration_us"])
    status = "computed"
    if eligible_us < min_elig or coverage < min_cov or sample_n < 2:
        status = "not_evaluable"
        reasons.append("INSUFFICIENT_COVERAGE")

    denom = float(eligible_us) if eligible_us > 0 else 1.0
    class_rows = []
    for name in CLASS_ORDER:
        dus = int(durations.get(name, 0))
        class_rows.append(
            {
                "class": name,
                "duration_us": dus,
                "duration_seconds": dus / 1_000_000.0,
                "eligible_percent": 100.0 * dus / denom if eligible_us > 0 else 0.0,
            }
        )

    moving = sum(int(durations.get(c, 0)) for c in act["moving_classes"])
    high_int = sum(int(durations.get(c, 0)) for c in act["high_intensity_classes"])
    sprinting = int(durations.get("sprinting", 0))

    index_cfg = act["movement_activity_index"]
    index_value: float | None = None
    index_status = status
    if status != "computed" or not index_cfg.get("enabled"):
        index_status = "not_evaluable"
        index_value = None
    else:
        weighted = 0.0
        weights = index_cfg["weights"]
        for name, w in weights.items():
            weighted += (int(durations.get(name, 0)) / 1_000_000.0) * float(w)
        index_value = weighted / (eligible_us / 1_000_000.0)

    return {
        "schema_version": 1,
        "status": status,
        "metric_origin": config["metric_origin"],
        "definition_style": config["definition_style"],
        "eligible_observed_duration_us": eligible_us,
        "gap_or_not_observed_duration_us": int(gap_unobserved_us),
        "missing_coverage_counted_as_inactive": False,
        "coverage_ratio": coverage,
        "classes": class_rows,
        "moving_duration_us": moving,
        "high_intensity_duration_us": high_int,
        "sprinting_duration_us": sprinting,
        "moving_to_eligible_ratio": (moving / denom) if eligible_us > 0 else None,
        "uncertainty_m": max(unc_vals) if unc_vals else None,
        "movement_activity_index": {
            "value": index_value,
            "status": index_status,
            "formula": "sum(class_duration_s * weight) / eligible_duration_s",
            "weights": dict(index_cfg["weights"]),
            "not_official_opta": True,
            "not_possession_or_tactical": True,
            "metric_origin": "project_generated",
        },
        "reason_codes": sorted(set(reasons)),
        "notes": [
            "Trajectory motion profile only — not tactical involvement or ball interaction.",
            "Unobserved/gap time is reported separately and is NOT inactive.",
            "Sprinting class aligns with Stage 9C sprint entry threshold.",
        ],
    }


__all__ = [
    "CLASS_ORDER",
    "classify_speed_mps",
    "compute_activity_distribution",
]
