"""Neutral zone occupancy and entries (Stage 9D — attack direction unknown)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.physical.zones import (
    ATTACK_RELATIVE_FORBIDDEN,
    assert_zone_name_allowed,
    neutral_third_for_x,
)


@dataclass(frozen=True)
class ZoneStats:
    zone_id: str
    dwell_us: int
    eligible_percent: float
    entry_count: int
    sample_visits: int
    uncertainty_m: float | None


def corridor_for_y(y_m: float, *, width_m: float, corridors: Mapping[str, Any]) -> str:
    """Half-open left/centre/right corridors along pitch width."""
    if width_m <= 0:
        raise ValueError("invalid width")
    frac = y_m / width_m
    # Clamp exact top edge into last band
    if frac >= 1.0:
        frac = 0.999999999
    if frac < 0.0:
        raise ValueError("y out of pitch")
    left = corridors["left_frac"]
    centre = corridors["centre_frac"]
    right = corridors["right_frac"]
    if float(left[0]) <= frac < float(left[1]):
        return "left_corridor"
    if float(centre[0]) <= frac < float(centre[1]):
        return "centre_corridor"
    if float(right[0]) <= frac < float(right[1]):
        return "right_corridor"
    return "centre_corridor"


def in_penalty(
    x_m: float,
    y_m: float,
    *,
    length_m: float,
    width_m: float,
    depth_m: float,
    pen_width_m: float,
    side: str,
) -> bool:
    """Half-open penalty rectangle; physical presence only."""
    y0 = (width_m - pen_width_m) / 2.0
    y1 = y0 + pen_width_m
    if not (y0 <= y_m < y1):
        return False
    if side == "goal_a_penalty":
        return 0.0 <= x_m < depth_m
    if side == "goal_b_penalty":
        return (length_m - depth_m) <= x_m < length_m or abs(x_m - length_m) < 1e-12
    return False


def classify_point_zones(
    x_m: float,
    y_m: float,
    *,
    config: Mapping[str, Any],
) -> list[str]:
    pitch = config["pitch"]
    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])
    if x_m < 0 or y_m < 0 or x_m > length_m or y_m > width_m:
        return []
    zones_cfg = config["zones"]
    out: list[str] = [neutral_third_for_x(x_m, length_m=length_m)]
    if zones_cfg.get("corridors", {}).get("enabled"):
        out.append(corridor_for_y(y_m, width_m=width_m, corridors=zones_cfg["corridors"]))
    pen = zones_cfg.get("penalty") or {}
    if pen.get("enabled"):
        depth = float(pen["depth_m"])
        pw = float(pen["width_m"])
        if in_penalty(
            x_m,
            y_m,
            length_m=length_m,
            width_m=width_m,
            depth_m=depth,
            pen_width_m=pw,
            side="goal_a_penalty",
        ):
            out.append("goal_a_penalty")
        if in_penalty(
            x_m,
            y_m,
            length_m=length_m,
            width_m=width_m,
            depth_m=depth,
            pen_width_m=pw,
            side="goal_b_penalty",
        ):
            out.append("goal_b_penalty")
    for z in out:
        if z in ATTACK_RELATIVE_FORBIDDEN:
            raise ValueError("attack relative zone invented")
    return out


def compute_zone_occupancy(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    analysis_window_us: int | None = None,
) -> dict[str, Any]:
    """Dwell / entry counts per neutral zone; never bridges segments."""
    unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
    min_pts = int(config["zones"]["min_points_for_dwell"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        if str(p.get("metric_eligibility", "not_eligible")) != "eligible":
            continue
        if str(p.get("mapping_status", "")) != "mapped":
            continue
        if p.get("video_time_us") is None:
            continue
        if p.get("pitch_x_m") is None or p.get("pitch_y_m") is None:
            continue
        if not math.isfinite(float(p["pitch_x_m"])) or not math.isfinite(float(p["pitch_y_m"])):
            continue
        u = p.get("uncertainty_m")
        if u is not None and float(u) > unc_max:
            continue
        sid = str(p.get("trajectory_segment_id") or "traj_seg_unknown")
        groups.setdefault(sid, []).append(dict(p))

    dwell: dict[str, int] = {}
    entries: dict[str, int] = {}
    visits: dict[str, int] = {}
    unc_acc: dict[str, list[float]] = {}
    reasons: list[str] = []
    eligible_us = 0
    sample_n = 0

    for _sid, pts in groups.items():
        ordered = sorted(pts, key=lambda r: int(r["video_time_us"]))
        sample_n += len(ordered)
        if len(ordered) < min_pts:
            reasons.append("SINGLE_SAMPLE_NO_ZONE_DWELL")
            continue
        eligible_us += int(ordered[-1]["video_time_us"]) - int(ordered[0]["video_time_us"])
        prev_primary: str | None = None
        for i, (a, b) in enumerate(zip(ordered, ordered[1:], strict=False)):
            t0, t1 = int(a["video_time_us"]), int(b["video_time_us"])
            dt = t1 - t0
            if dt <= 0:
                reasons.append("ZERO_OR_NEGATIVE_DELTA_TIME")
                continue
            xa, ya = float(a["pitch_x_m"]), float(a["pitch_y_m"])
            zones_a = classify_point_zones(xa, ya, config=config)
            if not zones_a:
                reasons.append("OUT_OF_PITCH")
                continue
            primary = zones_a[0]  # third
            for z in zones_a:
                dwell[z] = dwell.get(z, 0) + dt
                visits[z] = visits.get(z, 0) + 1
                if a.get("uncertainty_m") is not None:
                    unc_acc.setdefault(z, []).append(float(a["uncertainty_m"]))
            if prev_primary is None or primary != prev_primary:
                entries[primary] = entries.get(primary, 0) + 1
            prev_primary = primary
            _ = i

    window = int(analysis_window_us) if analysis_window_us is not None else max(1, eligible_us)
    coverage = (eligible_us / window) if window > 0 else 0.0
    min_cov = float(config["heatmap"]["min_coverage_ratio_for_computed"])
    status = "computed"
    if sample_n < min_pts or eligible_us <= 0 or coverage < min_cov:
        status = "not_evaluable"
        reasons.append("INSUFFICIENT_COVERAGE" if coverage < min_cov else "NO_ZONE_DWELL")

    denom = float(eligible_us) if eligible_us > 0 else 1.0
    stats: list[dict[str, Any]] = []
    for z, dus in sorted(dwell.items()):
        if z in {"goal_a_third", "middle_third", "goal_b_third"}:
            assert_zone_name_allowed(z)
        unc_vals = unc_acc.get(z) or []
        stats.append(
            {
                "zone_id": z,
                "dwell_us": int(dus),
                "dwell_seconds": dus / 1_000_000.0,
                "eligible_percent": 100.0 * dus / denom,
                "entry_count": int(entries.get(z, 0)),
                "sample_visits": int(visits.get(z, 0)),
                "uncertainty_m": max(unc_vals) if unc_vals else None,
                "semantics": (
                    "physical_presence_only" if z.endswith("_penalty") else "neutral_geometric_zone"
                ),
                "not_touch_or_possession": z.endswith("_penalty"),
            }
        )

    return {
        "schema_version": 1,
        "status": status,
        "attack_direction": "unknown",
        "attack_relative_invented": False,
        "eligible_duration_us": eligible_us,
        "coverage_ratio": coverage,
        "observed_coverage_us": eligible_us,
        "derived_coverage_us": 0,
        "zones": stats,
        "reason_codes": sorted(set(reasons)),
        "penalty_semantics": "physical_presence_only_not_ball_touch",
        "notes": [
            "Neutral Goal A/B thirds — not attacking/defending thirds.",
            "Penalty dwell is location time only; not touch/possession/event.",
        ],
    }


__all__ = [
    "ZoneStats",
    "corridor_for_y",
    "in_penalty",
    "classify_point_zones",
    "compute_zone_occupancy",
]
