"""Select blind holdout_v2 frames from unused regions of the canonical video."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from football_analytics.annotation.independent_gt import atomic_write_json, utc_now

# Canonical clip: 1023 frames @ 30fps ≈ 34.1s
TOTAL_FRAMES = 1023
FPS = 30.0
# Old splits are dense (~10f); require ≥4f clearance from any annotated frame.
TEMPORAL_BUFFER = 4
MIN_SPACING_AMONG_V2 = 10


def _used_indices(annotations: Mapping[str, Any]) -> set[int]:
    return {int(f["frame_idx"]) for f in annotations["frames"]}


def _near_used(idx: int, used: set[int], buffer: int = TEMPORAL_BUFFER) -> bool:
    return any(abs(idx - u) <= buffer for u in used)


class _Rng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def shuffle(self, xs: list[int]) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self._next() % (i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def select_holdout_v2_frames(
    annotations: Mapping[str, Any],
    *,
    n_target: int = 22,
    seed: int = 42,
) -> dict[str, Any]:
    """Pick 20–25 unused frames with temporal buffer; prefer 22–34s and hard scenes.

    Does not look at model predictions. Categories are planning labels only
    (no GT boxes). Caller must not expose proposals.
    """
    used = _used_indices(annotations)
    late_lo, late_hi = 660, TOTAL_FRAMES - 1

    def collect(lo: int, hi: int, buffer: int) -> list[int]:
        return [
            i for i in range(lo, hi + 1) if i not in used and not _near_used(i, used, buffer=buffer)
        ]

    candidates = collect(late_lo, late_hi, TEMPORAL_BUFFER)
    if len(candidates) < n_target * 3:
        candidates = collect(late_lo, late_hi, max(2, TEMPORAL_BUFFER - 1))
    if len(candidates) < n_target:
        candidates = sorted(set(candidates) | set(collect(0, late_lo - 1, TEMPORAL_BUFFER)))

    # Enforce mutual spacing then stratify across late window.
    spaced: list[int] = []
    last = -(10**9)
    for i in sorted(candidates):
        if i - last < MIN_SPACING_AMONG_V2:
            continue
        spaced.append(i)
        last = i

    bands = [
        (late_lo, late_lo + 70),
        (late_lo + 70, late_lo + 150),
        (late_lo + 150, late_lo + 230),
        (late_lo + 230, late_lo + 310),
        (late_lo + 310, late_hi + 1),
    ]
    rng = _Rng(seed)
    picked: list[int] = []
    per_band = max(4, (n_target + len(bands) - 1) // len(bands))
    for a, b in bands:
        pool = [i for i in spaced if a <= i < b and i not in picked]
        rng.shuffle(pool)
        for i in pool[:per_band]:
            picked.append(i)
            if len(picked) >= n_target:
                break
        if len(picked) >= n_target:
            break

    if len(picked) < 20:
        for i in spaced:
            if i in picked:
                continue
            if any(abs(i - p) < MIN_SPACING_AMONG_V2 for p in picked):
                continue
            picked.append(i)
            if len(picked) >= n_target:
                break

    picked = sorted(picked)[:25]
    if len(picked) < 20:
        raise RuntimeError(f"HOLDOUT_V2_INSUFFICIENT_CANDIDATES:{len(picked)}")

    frames_meta = []
    for i, idx in enumerate(picked):
        t_s = idx / FPS
        tags = ["small_distant", "difficult"]
        extras = ["goal", "sideline", "crowded", "occlusion", "wide_mid"]
        tags.append(extras[i % len(extras)])
        if i % 3 == 0:
            tags.append("dark_clothing")
        frames_meta.append(
            {
                "frame_idx": idx,
                "t_s": round(t_s, 4),
                "planned_categories": tags,
                "split": "holdout_v2",
            }
        )

    body = {
        "schema": "holdout_v2_selection_manifest_v1",
        "n_frames": len(picked),
        "frame_indices": picked,
        "frames": frames_meta,
        "rules": {
            "not_in_old_80": True,
            "temporal_buffer_frames": TEMPORAL_BUFFER,
            "min_spacing_among_v2": MIN_SPACING_AMONG_V2,
            "prefer_window_s": [22.0, 34.1],
            "blind": True,
            "no_proposals": True,
            "no_predictions": True,
        },
        "old_80_excluded": sorted(used),
        "seed": seed,
        "written_at_utc": utc_now(),
    }
    body["selection_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "frame_indices": picked,
                "seed": seed,
                "buffer": TEMPORAL_BUFFER,
                "old_80": sorted(used),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert not set(picked) & used
    for idx in picked:
        assert not _near_used(idx, used, buffer=TEMPORAL_BUFFER)
    return body


def write_holdout_v2_draft(
    selection: Mapping[str, Any],
    *,
    out_runtime: Path,
    source_sha256: str,
) -> Path:
    """Create empty blind draft for holdout_v2 review (no proposals)."""
    out_runtime.mkdir(parents=True, exist_ok=True)
    frames = []
    for fr in selection["frames"]:
        frames.append(
            {
                "frame_idx": int(fr["frame_idx"]),
                "t_s": float(fr["t_s"]),
                "split": "holdout",
                "categories": list(fr.get("planned_categories") or []),
                "completed": False,
                "humans": [],
                "proposals": [],
                "rejected_proposals": [],
                "no_human_confirmed": False,
                "provenance": {
                    "origin": "holdout_v2_blind",
                    "selection_fingerprint": selection["selection_fingerprint"],
                },
            }
        )
    draft = {
        "schema": "independent_gt_draft_v1",
        "dataset_id": "own_video_97b298e4_holdout_v2",
        "source_id": "own_video_97b298e4",
        "source_sha256": source_sha256,
        "holdout_v2": True,
        "blind": True,
        "frames": frames,
        "selection_fingerprint": selection["selection_fingerprint"],
    }
    path = out_runtime / "draft_annotations.json"
    atomic_write_json(path, draft, mode=0o600)
    atomic_write_json(out_runtime / "selection_manifest.json", dict(selection), mode=0o600)
    atomic_write_json(
        out_runtime / "progress.json",
        {
            "schema": "independent_gt_progress_v1",
            "n_frames": len(frames),
            "n_complete": 0,
            "by_split": {
                "train": {"n": 0, "complete": 0},
                "dev": {"n": 0, "complete": 0},
                "holdout": {"n": len(frames), "complete": 0},
            },
            "holdout_v2": True,
            "updated_at_utc": utc_now(),
        },
        mode=0o600,
    )
    return path


__all__ = [
    "select_holdout_v2_frames",
    "write_holdout_v2_draft",
]
