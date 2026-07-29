"""Blind holdout_v2 selection for R1-F2-D (30 frames, min-dist≥8 from old GT, no predictions)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.annotation.independent_gt import utc_now

TOTAL_FRAMES = 1023
FPS = 30.0
# "en az 8 frame mesafe" → |i-u| >= 8 (forbid |i-u| < 8).
MIN_DIST_FROM_OLD = 8
# Avoid near-duplicate consecutive picks inside free pockets.
MIN_BLOCK_SPACING = 3


def _used_indices(annotations: Mapping[str, Any]) -> set[int]:
    return {int(f["frame_idx"]) for f in annotations["frames"]}


def _too_close(idx: int, used: set[int], min_dist: int) -> bool:
    return any(abs(idx - u) < min_dist for u in used)


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


def candidate_pool(
    used: set[int],
    *,
    min_dist: int = MIN_DIST_FROM_OLD,
    exclude_extra: Sequence[int] | None = None,
) -> list[int]:
    blocked = set(used)
    if exclude_extra:
        blocked |= {int(x) for x in exclude_extra}
    return [
        i for i in range(TOTAL_FRAMES) if i not in blocked and not _too_close(i, blocked, min_dist)
    ]


def select_blind_holdout_v2(
    annotations: Mapping[str, Any],
    *,
    n_target: int = 30,
    seed: int = 20260729,
    exclude_extra: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Select 30 unused frames with ≥8f clearance from old 80; no model signals.

    Early clip (0–~370) is saturated by frozen GT under the ≥8 buffer; coverage
    uses earliest-available / mid / late bands of the remaining free pockets.
    Planning tags only — not GT.
    """
    used = _used_indices(annotations)
    pool = candidate_pool(used, exclude_extra=exclude_extra)
    if len(pool) < n_target:
        raise RuntimeError(f"HOLDOUT_V2_POOL_TOO_SMALL:{len(pool)}<{n_target}")

    # Relative bands over available timeline (start of free region → end of clip).
    lo, hi = pool[0], pool[-1]
    span = max(1, hi - lo)
    bands = [
        ("earliest_available", lo, lo + span // 5),
        ("early_mid", lo + span // 5, lo + 2 * span // 5),
        ("mid", lo + 2 * span // 5, lo + 3 * span // 5),
        ("late_a", lo + 3 * span // 5, lo + 4 * span // 5),
        ("late_b", lo + 4 * span // 5, hi + 1),
    ]
    rng = _Rng(seed)
    per_band = max(4, (n_target + len(bands) - 1) // len(bands))
    picked: list[int] = []

    def _try_add(cands: list[int], limit: int) -> None:
        spaced: list[int] = []
        last = -(10**9)
        for i in sorted(cands):
            if i - last < MIN_BLOCK_SPACING:
                continue
            spaced.append(i)
            last = i
        rng.shuffle(spaced)
        for i in spaced:
            if len(picked) >= n_target or limit <= 0:
                return
            if any(abs(i - p) < MIN_BLOCK_SPACING for p in picked):
                continue
            picked.append(i)
            limit -= 1

    for _name, a, b in bands:
        cands = [i for i in pool if a <= i < b]
        _try_add(cands, per_band)

    if len(picked) < n_target:
        rest = [i for i in pool if i not in picked]
        _try_add(rest, n_target - len(picked))

    # Final fill with spacing 2 if still short (still ≥8 from old GT).
    if len(picked) < n_target:
        for i in pool:
            if i in picked:
                continue
            if any(abs(i - p) < 2 for p in picked):
                continue
            picked.append(i)
            if len(picked) >= n_target:
                break

    picked = sorted(picked)[:n_target]
    if len(picked) < n_target:
        raise RuntimeError(f"HOLDOUT_V2_INSUFFICIENT:{len(picked)}<{n_target}")

    tags_cycle = [
        ["small_distant", "difficult"],
        ["crowded", "occlusion"],
        ["goal_area", "sideline"],
        ["dark_clothing", "wide_mid"],
        ["easy_negative_regions"],
    ]
    frames_meta = []
    for i, idx in enumerate(picked):
        tags = list(tags_cycle[i % len(tags_cycle)])
        frames_meta.append(
            {
                "frame_idx": idx,
                "t_s": round(idx / FPS, 4),
                "planned_categories": tags,
                "section": "holdout_v2",
                "split": "holdout",
            }
        )

    body: dict[str, Any] = {
        "schema": "holdout_v2_selection_manifest_v2",
        "n_frames": len(picked),
        "frame_indices": picked,
        "frames": frames_meta,
        "selection_method": {
            "deterministic_seed": seed,
            "min_distance_from_old_80": MIN_DIST_FROM_OLD,
            "min_block_spacing": MIN_BLOCK_SPACING,
            "bands": [{"name": n, "lo": a, "hi": b} for n, a, b in bands],
            "uses_detector": False,
            "uses_confidence": False,
            "uses_error_analysis": False,
            "note_early_clip": (
                "Frames 0–~370 have no free slots under min_distance_from_old_80; "
                "earliest_available band starts at first free pocket."
            ),
        },
        "rules": {
            "not_in_old_80": True,
            "temporal_buffer_frames": MIN_DIST_FROM_OLD,
            "disjoint_from_active_learning": True,
            "blind": True,
            "no_proposals": True,
            "no_predictions": True,
            "no_inference_during_review": True,
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
                "min_dist_old": MIN_DIST_FROM_OLD,
                "spacing": MIN_BLOCK_SPACING,
                "old_80": sorted(used),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert set(picked).isdisjoint(used)
    for idx in picked:
        assert not _too_close(idx, used, MIN_DIST_FROM_OLD)
    return body


def assert_no_holdout_v2_inference(frame_indices: Sequence[int], holdout_set: set[int]) -> None:
    bad = [i for i in frame_indices if int(i) in holdout_set]
    if bad:
        raise RuntimeError(f"HOLDOUT_V2_INFERENCE_FORBIDDEN:{bad[:10]}")


# Back-compat alias used by tests importing TEMPORAL_BUFFER.
TEMPORAL_BUFFER = MIN_DIST_FROM_OLD

__all__ = [
    "MIN_BLOCK_SPACING",
    "MIN_DIST_FROM_OLD",
    "TEMPORAL_BUFFER",
    "assert_no_holdout_v2_inference",
    "candidate_pool",
    "select_blind_holdout_v2",
]
