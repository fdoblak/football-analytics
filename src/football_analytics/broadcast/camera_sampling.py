"""Deterministic equal-time sample points within shot intervals (timeline PTS)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class CameraSamplingError(ValueError):
    """Sample-point planning failure."""


@dataclass(frozen=True)
class SamplePoint:
    frame_index: int
    time_us: int


def _nearest_timeline_index(
    timeline: Sequence[tuple[int, int]], target_us: int, *, lo: int, hi: int
) -> int:
    """Nearest timeline index in [lo, hi) by absolute time distance (deterministic)."""
    if lo >= hi:
        raise CameraSamplingError("empty timeline window")
    best_i = lo
    best_err = abs(timeline[lo][1] - target_us)
    for i in range(lo + 1, hi):
        err = abs(timeline[i][1] - target_us)
        if err < best_err or (err == best_err and timeline[i][0] < timeline[best_i][0]):
            best_err = err
            best_i = i
    return best_i


def plan_sample_points(
    timeline: Sequence[tuple[int, int]],
    *,
    start_time_us: int,
    end_time_us: int,
    config: Mapping[str, Any],
) -> list[SamplePoint]:
    """Equal-time samples in [start+margin, end-margin) using timeline PTS only.

    Does not invent fps. Returns unique SamplePoint(frame_index, time_us) sorted by time.
    """
    if not timeline:
        raise CameraSamplingError("timeline required")
    if end_time_us <= start_time_us:
        raise CameraSamplingError("end_time_us must be > start_time_us")

    sampling = config["sampling"]
    n_want = int(sampling["samples_per_shot"])
    edge = float(sampling["edge_exclude_fraction"])
    min_s = int(sampling["min_samples"])
    max_s = int(sampling["max_samples"])
    n_want = max(min_s, min(max_s, n_want))

    duration = end_time_us - start_time_us
    margin = int(round(duration * edge))
    win_start = start_time_us + margin
    win_end = end_time_us - margin
    if win_end <= win_start:
        # Degenerate short shot: use full half-open interval interior if possible
        win_start = start_time_us
        win_end = end_time_us

    # Timeline indices with time in [start, end)
    indices = [i for i, (_fi, t) in enumerate(timeline) if start_time_us <= t < end_time_us]
    if not indices:
        raise CameraSamplingError("no timeline frames inside shot interval")

    # Prefer frames inside sampling window; fall back to full shot indices
    window_indices = [i for i in indices if win_start <= timeline[i][1] < win_end]
    use = window_indices if window_indices else indices

    lo = use[0]
    hi = use[-1] + 1
    t0 = timeline[lo][1]
    t1 = timeline[use[-1]][1]
    if t1 <= t0:
        # Single-frame or identical times: repeat one sample (clamped)
        fi, tu = timeline[lo]
        n = max(1, min(n_want, min_s))
        return [SamplePoint(frame_index=fi, time_us=tu) for _ in range(n)]

    targets: list[int] = []
    if n_want == 1:
        targets.append(int(round((t0 + t1) / 2)))
    else:
        for k in range(n_want):
            frac = k / (n_want - 1)
            targets.append(int(round(t0 + frac * (t1 - t0))))

    chosen: list[SamplePoint] = []
    seen: set[int] = set()
    for target in targets:
        idx = _nearest_timeline_index(timeline, target, lo=lo, hi=hi)
        fi, tu = timeline[idx]
        if fi in seen:
            # Find next unused neighbor within window
            found = False
            for j in range(lo, hi):
                if timeline[j][0] not in seen:
                    fi, tu = timeline[j]
                    found = True
                    break
            if not found:
                continue
        seen.add(fi)
        chosen.append(SamplePoint(frame_index=fi, time_us=tu))

    if len(chosen) < min_s and len(use) >= min_s:
        # Fill from evenly spaced indices
        chosen = []
        seen = set()
        for k in range(min_s):
            pos = use[int(round(k * (len(use) - 1) / (min_s - 1)))] if min_s > 1 else use[0]
            fi, tu = timeline[pos]
            if fi in seen:
                continue
            seen.add(fi)
            chosen.append(SamplePoint(frame_index=fi, time_us=tu))

    if not chosen:
        fi, tu = timeline[indices[0]]
        chosen = [SamplePoint(frame_index=fi, time_us=tu)]

    chosen.sort(key=lambda p: (p.time_us, p.frame_index))
    if len(chosen) > max_s:
        # Keep evenly spaced subset
        keep: list[SamplePoint] = []
        for k in range(max_s):
            pos = int(round(k * (len(chosen) - 1) / (max_s - 1))) if max_s > 1 else 0
            keep.append(chosen[pos])
        # Deduplicate while preserving order
        out: list[SamplePoint] = []
        seen_f: set[int] = set()
        for p in keep:
            if p.frame_index in seen_f:
                continue
            seen_f.add(p.frame_index)
            out.append(p)
        chosen = out

    return chosen


__all__ = [
    "CameraSamplingError",
    "SamplePoint",
    "plan_sample_points",
]
