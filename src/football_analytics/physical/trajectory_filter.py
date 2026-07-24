"""Stage 9B trajectory quality filter (deterministic; no customer metrics)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _in_pitch(x: float, y: float, *, length_m: float, width_m: float, tol: float) -> bool:
    return (-tol) <= x <= (length_m + tol) and (-tol) <= y <= (width_m + tol)


def filter_trajectory_points(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Return (accepted_filtered_rows, rejection_records, reason_counts).

    Source/raw rows are never deleted by the caller; rejections carry reason codes.
    Implied speed checks are quality_gate_only.
    """
    qf = config["quality_filter"]
    pitch = config["pitch"]
    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])
    tol = float(pitch.get("soft_bound_tolerance_m", 0.0))
    unc_max = float(qf["uncertainty_max_m"])
    max_jump = float(qf["max_jump_m"])
    max_speed = float(qf["max_implied_speed_mps"])
    spike_window = int(qf["spike_window"])
    spike_dev = float(qf["spike_max_deviation_m"])

    ordered = sorted(points, key=lambda p: (int(p["video_time_us"]), str(p["sample_id"])))
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    def reject(row: Mapping[str, Any], code: str, **extra: Any) -> None:
        reason_counts[code] = reason_counts.get(code, 0) + 1
        rec = {
            "sample_id": str(row.get("sample_id")),
            "video_time_us": int(row["video_time_us"]),
            "reason_code": code,
            "quality_gate_only_speed": True,
            **extra,
        }
        rejections.append(rec)

    # Pass 1: finite / bounds / uncertainty / duplicates
    seen_time: dict[int, dict[str, Any]] = {}
    conflict_times: set[int] = set()
    prelim: list[dict[str, Any]] = []
    for row in ordered:
        r = dict(row)
        t = int(r["video_time_us"])
        x = r.get("pitch_x_m")
        y = r.get("pitch_y_m")
        if not _finite(x) or not _finite(y):
            reject(r, "non_finite_coordinate")
            continue
        xf, yf = float(x), float(y)  # type: ignore[arg-type]
        if not _in_pitch(xf, yf, length_m=length_m, width_m=width_m, tol=tol):
            reject(r, "outside_pitch")
            continue
        unc = r.get("uncertainty_m")
        if unc is not None and _finite(unc) and float(unc) > unc_max:
            reject(r, "uncertainty_exceeded")
            continue
        if t in seen_time:
            prev = seen_time[t]
            same_xy = (
                abs(float(prev["pitch_x_m"]) - xf) < 1e-9
                and abs(float(prev["pitch_y_m"]) - yf) < 1e-9
            )
            if same_xy and qf.get("exact_duplicate_keep_first"):
                reject(r, "exact_duplicate")
                continue
            conflict_times.add(t)
            reject(r, "conflicting_duplicate")
            reject(prev, "conflicting_duplicate")
            # Mark previous for removal if already in prelim
            continue
        seen_time[t] = r
        prelim.append(r)

    prelim = [p for p in prelim if int(p["video_time_us"]) not in conflict_times]

    # Pass 2: monotonicity / jumps / implied speed / spikes
    for i, r in enumerate(prelim):
        if i > 0:
            prev = prelim[i - 1]
            dt = int(r["video_time_us"]) - int(prev["video_time_us"])
            if dt <= 0:
                reject(r, "timestamp_regression")
                continue
            dx = float(r["pitch_x_m"]) - float(prev["pitch_x_m"])
            dy = float(r["pitch_y_m"]) - float(prev["pitch_y_m"])
            dist = math.hypot(dx, dy)
            if dist > max_jump:
                reject(r, "impossible_jump", jump_m=dist)
                continue
            speed = dist / (dt / 1_000_000.0)
            if speed > max_speed:
                reject(
                    r,
                    "impossible_implied_speed",
                    implied_speed_mps=speed,
                    provenance="quality_gate_only",
                )
                continue
        # short spike: compare to local median-ish neighbors
        if spike_window >= 3 and i >= 1 and i + 1 < len(prelim):
            neigh = prelim[max(0, i - 1) : i + 2]
            if len(neigh) >= 3:
                xs = [float(n["pitch_x_m"]) for n in neigh]
                ys = [float(n["pitch_y_m"]) for n in neigh]
                mx = sorted(xs)[len(xs) // 2]
                my = sorted(ys)[len(ys) // 2]
                dev = math.hypot(float(r["pitch_x_m"]) - mx, float(r["pitch_y_m"]) - my)
                if dev > spike_dev:
                    reject(r, "short_noise_spike", deviation_m=dev)
                    continue
        out = dict(r)
        out["sample_source"] = "filtered"
        out["derived_from_sample_ids"] = [str(r["sample_id"])]
        out["eligibility_status"] = "eligible"
        out["metric_eligibility"] = "eligible"
        flags = list(out.get("quality_flags") or [])
        if "quality_gate_only" not in flags:
            flags.append("filtered_accepted")
        out["quality_flags"] = flags
        # Keep filtered sample_id distinct but linked
        out["sample_id"] = f"flt_{r['sample_id']}"
        accepted.append(out)

    return accepted, rejections, reason_counts


__all__ = ["filter_trajectory_points"]
