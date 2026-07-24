"""Time-weighted canonical-pitch heatmap (Stage 9D)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeatmapGridResult:
    status: str
    n_x: int
    n_y: int
    cell_size_m: float
    length_m: float
    width_m: float
    dwell_seconds: tuple[tuple[float, ...], ...]
    eligible_percent: tuple[tuple[float, ...], ...]
    total_dwell_seconds: float
    observed_dwell_seconds: float
    derived_dwell_seconds: float
    eligible_duration_us: int
    sample_count: int
    interval_count: int
    rejected_interval_count: int
    coverage_ratio: float
    reason_codes: tuple[str, ...]
    mass_before_smooth: float
    mass_after_smooth: float


def _eligible(p: Mapping[str, Any], *, unc_max: float) -> bool:
    if str(p.get("metric_eligibility", "not_eligible")) != "eligible":
        return False
    if str(p.get("mapping_status", "")) != "mapped":
        return False
    if p.get("video_time_us") is None:
        return False
    if p.get("pitch_x_m") is None or p.get("pitch_y_m") is None:
        return False
    if not math.isfinite(float(p["pitch_x_m"])) or not math.isfinite(float(p["pitch_y_m"])):
        return False
    u = p.get("uncertainty_m")
    return not (u is not None and float(u) > unc_max)


def grid_shape(*, length_m: float, width_m: float, cell_size_m: float) -> tuple[int, int]:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    n_x = max(1, int(math.ceil(length_m / cell_size_m)))
    n_y = max(1, int(math.ceil(width_m / cell_size_m)))
    return n_x, n_y


def cell_index(
    x_m: float,
    y_m: float,
    *,
    length_m: float,
    width_m: float,
    cell_size_m: float,
    n_x: int,
    n_y: int,
) -> tuple[int, int] | None:
    """Half-open cells; right/top edges map into last cell."""
    if x_m < 0.0 or y_m < 0.0 or x_m > length_m or y_m > width_m:
        return None
    ix = min(n_x - 1, int(x_m / cell_size_m)) if x_m < length_m else n_x - 1
    iy = min(n_y - 1, int(y_m / cell_size_m)) if y_m < width_m else n_y - 1
    if ix < 0 or iy < 0 or ix >= n_x or iy >= n_y:
        return None
    return ix, iy


def _gaussian_kernel(sigma: float, radius: int) -> list[tuple[int, int, float]]:
    if sigma <= 0 or radius < 0:
        return [(0, 0, 1.0)]
    weights: list[tuple[int, int, float]] = []
    s2 = 2.0 * sigma * sigma
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            w = math.exp(-(dx * dx + dy * dy) / s2)
            weights.append((dx, dy, w))
    total = sum(w for _, _, w in weights)
    return [(dx, dy, w / total) for dx, dy, w in weights]


def smooth_conserve_mass(
    grid: list[list[float]],
    *,
    sigma_cells: float,
    radius_cells: int,
) -> list[list[float]]:
    """Bounded Gaussian blur that renormalizes to preserve total mass."""
    n_y = len(grid)
    n_x = len(grid[0]) if n_y else 0
    mass0 = sum(sum(row) for row in grid)
    if mass0 <= 0:
        return [list(row) for row in grid]
    kern = _gaussian_kernel(sigma_cells, radius_cells)
    out = [[0.0 for _ in range(n_x)] for _ in range(n_y)]
    for iy in range(n_y):
        for ix in range(n_x):
            v = grid[iy][ix]
            if v == 0.0:
                continue
            for dx, dy, w in kern:
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < n_x and 0 <= jy < n_y:
                    out[jy][jx] += v * w
                # out-of-pitch mass is dropped then restored via renormalization
    mass1 = sum(sum(row) for row in out)
    if mass1 <= 0:
        return [list(row) for row in grid]
    scale = mass0 / mass1
    return [[c * scale for c in row] for row in out]


def compute_time_weighted_heatmap(
    points: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    contribution_source: str = "observed",
    analysis_window_us: int | None = None,
) -> HeatmapGridResult:
    """Accumulate dwell seconds from consecutive eligible samples within segments."""
    hm = config["heatmap"]
    pitch = config["pitch"]
    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])
    cell = float(hm["cell_size_m"])
    unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
    min_pts = int(hm["min_points_for_dwell"])
    min_dt = int(hm["min_delta_time_us"])
    max_dt = int(hm["max_support_interval_us"])
    n_x, n_y = grid_shape(length_m=length_m, width_m=width_m, cell_size_m=cell)

    # Group by trajectory segment — never bridge hard gaps
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        if not _eligible(p, unc_max=unc_max):
            continue
        sid = str(p.get("trajectory_segment_id") or "traj_seg_unknown")
        groups.setdefault(sid, []).append(dict(p))

    grid = [[0.0 for _ in range(n_x)] for _ in range(n_y)]
    reasons: list[str] = []
    intervals = 0
    rejected = 0
    sample_n = 0
    eligible_us = 0

    for _sid, pts in groups.items():
        ordered = sorted(pts, key=lambda r: int(r["video_time_us"]))
        sample_n += len(ordered)
        if len(ordered) < min_pts:
            reasons.append("SINGLE_SAMPLE_NO_DWELL")
            continue
        eligible_us += int(ordered[-1]["video_time_us"]) - int(ordered[0]["video_time_us"])
        for a, b in zip(ordered, ordered[1:], strict=False):
            t0, t1 = int(a["video_time_us"]), int(b["video_time_us"])
            dt = t1 - t0
            if dt <= 0:
                rejected += 1
                reasons.append("ZERO_OR_NEGATIVE_DELTA_TIME")
                continue
            if dt < min_dt or dt > max_dt:
                rejected += 1
                reasons.append("SUPPORT_INTERVAL_OUT_OF_POLICY")
                continue
            # Reject if either endpoint is outside pitch (no energy spill / no OOB midpoint cheat)
            for label, pt in (("a", a), ("b", b)):
                _ = label
                if (
                    cell_index(
                        float(pt["pitch_x_m"]),
                        float(pt["pitch_y_m"]),
                        length_m=length_m,
                        width_m=width_m,
                        cell_size_m=cell,
                        n_x=n_x,
                        n_y=n_y,
                    )
                    is None
                ):
                    rejected += 1
                    reasons.append("OUT_OF_PITCH")
                    break
            else:
                mx = 0.5 * (float(a["pitch_x_m"]) + float(b["pitch_x_m"]))
                my = 0.5 * (float(a["pitch_y_m"]) + float(b["pitch_y_m"]))
                idx = cell_index(
                    mx,
                    my,
                    length_m=length_m,
                    width_m=width_m,
                    cell_size_m=cell,
                    n_x=n_x,
                    n_y=n_y,
                )
                if idx is None:
                    rejected += 1
                    reasons.append("OUT_OF_PITCH")
                    continue
                ix, iy = idx
                grid[iy][ix] += dt / 1_000_000.0
                intervals += 1

    mass0 = sum(sum(row) for row in grid)
    sm = hm.get("smoothing") or {}
    if sm.get("enabled") and mass0 > 0:
        grid = smooth_conserve_mass(
            grid,
            sigma_cells=float(sm.get("sigma_cells", 1.0)),
            radius_cells=int(sm.get("radius_cells", 2)),
        )
    mass1 = sum(sum(row) for row in grid)

    window = int(analysis_window_us) if analysis_window_us is not None else max(1, eligible_us)
    coverage = (eligible_us / window) if window > 0 else 0.0
    min_cov = float(hm["min_coverage_ratio_for_computed"])

    if sample_n < min_pts or intervals == 0 or coverage < min_cov:
        status = "not_evaluable"
        if coverage < min_cov:
            reasons.append("INSUFFICIENT_COVERAGE")
        if intervals == 0:
            reasons.append("NO_VALID_DWELL_INTERVALS")
        pct = tuple(tuple(0.0 for _ in range(n_x)) for _ in range(n_y))
        dwell_t = tuple(tuple(0.0 for _ in range(n_x)) for _ in range(n_y))
        return HeatmapGridResult(
            status=status,
            n_x=n_x,
            n_y=n_y,
            cell_size_m=cell,
            length_m=length_m,
            width_m=width_m,
            dwell_seconds=dwell_t,
            eligible_percent=pct,
            total_dwell_seconds=0.0,
            observed_dwell_seconds=0.0 if contribution_source != "observed" else 0.0,
            derived_dwell_seconds=0.0,
            eligible_duration_us=eligible_us,
            sample_count=sample_n,
            interval_count=intervals,
            rejected_interval_count=rejected,
            coverage_ratio=coverage,
            reason_codes=tuple(sorted(set(reasons))),
            mass_before_smooth=mass0,
            mass_after_smooth=mass1,
        )

    denom = mass1 if mass1 > 0 else 1.0
    pct_grid = tuple(tuple(100.0 * c / denom for c in row) for row in grid)
    dwell = tuple(tuple(float(c) for c in row) for row in grid)
    obs = mass1 if contribution_source == "observed" else 0.0
    der = mass1 if contribution_source == "derived" else 0.0
    return HeatmapGridResult(
        status="computed",
        n_x=n_x,
        n_y=n_y,
        cell_size_m=cell,
        length_m=length_m,
        width_m=width_m,
        dwell_seconds=dwell,
        eligible_percent=pct_grid,
        total_dwell_seconds=float(mass1),
        observed_dwell_seconds=float(obs),
        derived_dwell_seconds=float(der),
        eligible_duration_us=eligible_us,
        sample_count=sample_n,
        interval_count=intervals,
        rejected_interval_count=rejected,
        coverage_ratio=coverage,
        reason_codes=tuple(sorted(set(reasons))),
        mass_before_smooth=float(mass0),
        mass_after_smooth=float(mass1),
    )


def heatmap_to_dict(result: HeatmapGridResult, *, config_fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": result.status,
        "n_x": result.n_x,
        "n_y": result.n_y,
        "cell_size_m": result.cell_size_m,
        "length_m": result.length_m,
        "width_m": result.width_m,
        "dwell_seconds": [list(row) for row in result.dwell_seconds],
        "eligible_percent": [list(row) for row in result.eligible_percent],
        "total_dwell_seconds": result.total_dwell_seconds,
        "observed_dwell_seconds": result.observed_dwell_seconds,
        "derived_dwell_seconds": result.derived_dwell_seconds,
        "eligible_duration_us": result.eligible_duration_us,
        "sample_count": result.sample_count,
        "interval_count": result.interval_count,
        "rejected_interval_count": result.rejected_interval_count,
        "coverage_ratio": result.coverage_ratio,
        "reason_codes": list(result.reason_codes),
        "mass_before_smooth": result.mass_before_smooth,
        "mass_after_smooth": result.mass_after_smooth,
        "percent_sum": sum(sum(row) for row in result.eligible_percent),
        "config_fingerprint": config_fingerprint,
        "weighting": "time_weighted",
        "coordinate_frame": "canonical_pitch",
    }


__all__ = [
    "HeatmapGridResult",
    "grid_shape",
    "cell_index",
    "smooth_conserve_mass",
    "compute_time_weighted_heatmap",
    "heatmap_to_dict",
]
