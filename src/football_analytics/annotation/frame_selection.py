"""Stratified frame selection for independent human GT (R1-F2-A)."""

from __future__ import annotations

from typing import Any

from football_analytics.annotation.independent_gt import (
    DEV_T,
    HOLDOUT_T,
    SCHEMA_SELECTION,
    TRAIN_T,
    utc_now,
)

FPS = 30.0
N_FRAMES = 1023


def _t(frame_idx: int) -> float:
    return frame_idx / FPS


def _in_range(frame_idx: int, lo: float, hi: float) -> bool:
    t = _t(frame_idx)
    return lo <= t < hi or (hi >= 34.0 and t <= hi)


def _spaced(candidates: list[int], n: int, min_gap: int) -> list[int]:
    """Greedy pick with minimum frame gap to avoid near-duplicate neighbors."""
    picked: list[int] = []
    for c in candidates:
        if all(abs(c - p) >= min_gap for p in picked):
            picked.append(c)
            if len(picked) >= n:
                break
    return picked


def _fill_band(
    *,
    start: int,
    end: int,
    n: int,
    split: str,
    anchors: dict[str, list[int]],
    prefer_gap: int = 10,
) -> list[dict[str, Any]]:
    cats_map: dict[int, list[str]] = {}
    chosen: list[int] = []

    # 1) Anchors in band
    for cat, idxs in anchors.items():
        for idx in idxs:
            if start <= idx <= end:
                cats_map.setdefault(idx, [])
                if cat not in cats_map[idx]:
                    cats_map[idx].append(cat)
    for idx in sorted(cats_map):
        if all(abs(idx - c) >= 8 for c in chosen):
            chosen.append(idx)
        if len(chosen) >= n:
            break

    # 2) Uniform lattice
    if n > 0:
        for i in range(n * 4):
            idx = start + int(round(i * (end - start) / max(n * 4 - 1, 1)))
            idx = min(end, max(start, idx))
            if idx in chosen:
                continue
            if all(abs(idx - c) >= prefer_gap for c in chosen):
                chosen.append(idx)
                cats_map.setdefault(idx, ["lattice"])
            if len(chosen) >= n:
                break

    # 3) Relax gap until full
    for gap in (prefer_gap, 8, 6, 4, 2, 1):
        if len(chosen) >= n:
            break
        idx = start
        while idx <= end and len(chosen) < n:
            if idx not in chosen and all(abs(idx - c) >= gap for c in chosen):
                chosen.append(idx)
                cats_map.setdefault(idx, ["topup"])
            idx += max(gap, 1)

    chosen = sorted(set(chosen))[:n]
    # Absolute guarantee
    if len(chosen) < n:
        for idx in range(start, end + 1):
            if idx not in chosen:
                chosen.append(idx)
                cats_map.setdefault(idx, ["forced"])
            if len(chosen) >= n:
                break
        chosen = sorted(chosen)[:n]

    rows = []
    for idx in chosen:
        rows.append(
            {
                "frame_idx": idx,
                "t_s": round(_t(idx), 4),
                "split": split,
                "categories": cats_map.get(idx, ["lattice"]),
            }
        )
    return rows


def build_independent_gt_selection(
    *,
    n_train: int = 40,
    n_dev: int = 20,
    n_holdout: int = 20,
) -> dict[str, Any]:
    """Return ~80-frame selection with time isolation and category coverage tags.

    Selection is deterministic (no RNG) using fixed index lattices + category anchors.
    """
    anchors: dict[str, list[int]] = {
        "wide_camera": [30, 90, 180, 390, 720],
        "mid_camera": [120, 240, 450, 780],
        "small_distant": [60, 210, 500, 820, 940],
        "crowded": [270, 330, 540, 860, 980],
        "occlusion": [150, 360, 580, 900],
        "dark_clothing": [200, 400, 660, 1010],
        "sideline": [45, 300, 620, 880],
        "goal_area": [480, 700, 740, 1000],
        "partial_truncated": [100, 420, 760],
        "motion_blur": [250, 520, 840],
        "off_pitch_people": [15, 350, 690],
        "easy_negative_regions": [5, 370, 710],
    }

    def band(lo: float, hi: float) -> tuple[int, int]:
        start = int(lo * FPS)
        end = min(N_FRAMES - 1, int(hi * FPS) - 1)
        if hi >= 34.0:
            end = N_FRAMES - 1
        return start, end

    t0, t1 = band(TRAIN_T[0], TRAIN_T[1])
    d0, d1 = band(DEV_T[0], DEV_T[1])
    h0, h1 = band(HOLDOUT_T[0], HOLDOUT_T[1])
    train = _fill_band(start=t0, end=t1, n=n_train, split="train", anchors=anchors)
    dev = _fill_band(start=d0, end=d1, n=n_dev, split="dev", anchors=anchors)
    holdout = _fill_band(start=h0, end=h1, n=n_holdout, split="holdout", anchors=anchors)
    frames = train + dev + holdout

    coverage: dict[str, list[str]] = {}
    for fr in frames:
        for c in fr["categories"]:
            coverage.setdefault(c, []).append(f"{fr['split']}:{fr['frame_idx']}")

    return {
        "schema": SCHEMA_SELECTION,
        "video_id": "own_video_97b298e4",
        "fps": FPS,
        "n_video_frames": N_FRAMES,
        "time_isolation_s": {
            "train": list(TRAIN_T),
            "dev": list(DEV_T),
            "holdout": list(HOLDOUT_T),
        },
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "holdout": len(holdout),
            "total": len(frames),
        },
        "min_neighbor_gap_policy": "prefer >=8–12 frames; relax only to fill quota",
        "coverage_index": {k: v[:12] for k, v in coverage.items()},
        "frames": frames,
        "written_at_utc": utc_now(),
        "note": "Manifest only — not GT. No agent/YOLO boxes included.",
    }


__all__ = ["FPS", "N_FRAMES", "build_independent_gt_selection"]
