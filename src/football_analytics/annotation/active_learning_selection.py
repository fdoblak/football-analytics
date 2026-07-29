"""Active-learning frame selection from unused pool (development signals only)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_analytics.annotation.holdout_v2_blind import assert_no_holdout_v2_inference
from football_analytics.annotation.independent_gt import utc_now
from football_analytics.perception.full_tile_fusion import (
    FusionConfig,
    attach_frame_index,
    predict_full_tile_fused,
)

TOTAL_FRAMES = 1023
FPS = 30.0
# Disjoint from holdout; small clearance avoids near-identical neighbors only.
HOLDOUT_CLEARANCE = 2
AL_MIN_SPACING = 5
MAX_AL = 100

# Development error-analysis bands from F2-C root_cause (not GT / not accuracy).
DEV_HARD_BANDS: tuple[tuple[str, int, int, float], ...] = (
    ("early_small_heavy", 0, 350, 2.5),
    ("mid_crowded", 360, 633, 2.0),
    ("late_small_distant", 660, 1010, 2.5),
)


class _Rng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def shuffle(self, xs: list[Any]) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self._next() % (i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def _near(idx: int, blocked: set[int], buffer: int) -> bool:
    return any(abs(idx - u) < buffer for u in blocked)


def _phash_like(gray: np.ndarray, size: int = 16) -> int:
    """Tiny average-hash for near-duplicate suppression (not cryptographic)."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def candidate_pool(
    *,
    old_80: set[int],
    holdout_v2: set[int],
    holdout_clearance: int = HOLDOUT_CLEARANCE,
) -> list[int]:
    """Unused frames: not in frozen 80, not in holdout_v2 (exact + small clearance)."""
    blocked_exact = set(old_80) | set(holdout_v2)
    return [
        i
        for i in range(TOTAL_FRAMES)
        if i not in blocked_exact and not _near(i, set(holdout_v2), holdout_clearance)
    ]


def _band_boost(frame_idx: int) -> float:
    for _name, lo, hi, w in DEV_HARD_BANDS:
        if lo <= frame_idx <= hi:
            return w
    return 1.0


def score_frame_with_checkpoint(
    model: Any,
    frame: np.ndarray,
    frame_idx: int,
    *,
    device: str,
) -> dict[str, Any]:
    """Development-only heuristic scores (not GT / not accuracy)."""
    cfg_full = FusionConfig(conf=0.25, merge_iou=0.55, imgsz=960, mode="full_frame", device=device)
    cfg_hyb = FusionConfig(conf=0.25, merge_iou=0.55, imgsz=960, mode="hybrid", device=device)
    full = attach_frame_index(predict_full_tile_fused(model, frame, cfg_full), frame_idx)
    hyb = attach_frame_index(predict_full_tile_fused(model, frame, cfg_hyb), frame_idx)
    h, w = frame.shape[:2]

    def _stats(dets: Sequence[Any]) -> dict[str, Any]:
        heights = [max(0.0, d.y2 - d.y1) for d in dets]
        small = sum(1 for hh in heights if hh < 55)
        tiny = sum(1 for hh in heights if hh < 40)
        large_wide = sum(
            1
            for d in dets
            if (d.y2 - d.y1) > 1 and ((d.x2 - d.x1) / (d.y2 - d.y1)) > 0.85 and (d.x2 - d.x1) > 55
        )
        edge = sum(1 for d in dets if d.x1 < 40 or d.y1 < 40 or d.x2 > w - 40 or d.y2 > h - 40)
        return {
            "n": len(dets),
            "small": small,
            "tiny": tiny,
            "merged_diag": large_wide,
            "edge": edge,
            "mean_h": float(np.mean(heights)) if heights else 0.0,
            "scores": [float(d.score) for d in dets],
        }

    sf, sh = _stats(full), _stats(hyb)
    disagreement = abs(sf["n"] - sh["n"]) + abs(sf["small"] - sh["small"])
    reasons: list[str] = []
    score = 0.0
    if disagreement >= 2:
        reasons.append("disagreement")
        score += 3.0 * disagreement
    if sh["tiny"] >= 1 or sf["tiny"] >= 1 or sh["small"] >= 2 or sf["small"] >= 2:
        reasons.append("small_object")
        score += 4.0 * max(sh["tiny"], sf["tiny"], 1 if max(sh["small"], sf["small"]) >= 2 else 0)
    if sh["n"] >= 10 or sf["n"] >= 10:
        reasons.append("crowded")
        score += 2.0 + 0.15 * max(sh["n"], sf["n"])
    if sh["merged_diag"] or sf["merged_diag"]:
        reasons.append("likely_false_positive")
        score += 2.5 * (sh["merged_diag"] + sf["merged_diag"])
    if sh["n"] - sf["n"] >= 2:
        reasons.append("hard_negative")
        score += 3.0
    if sf["n"] > sh["n"] + 1:
        reasons.append("likely_false_negative")
        score += 2.0
    if sh["edge"] >= 2 or sf["edge"] >= 2:
        reasons.append("coverage_fill")
        score += 1.5
    if frame_idx >= 500 or frame_idx <= 200:
        reasons.append("temporal_shift")
        score += 1.0
    # F2-C root-cause temporal prior (development signal only).
    boost = _band_boost(frame_idx)
    score *= boost
    if boost > 1.0 and "temporal_shift" not in reasons:
        reasons.append("temporal_shift")
    if not reasons:
        reasons.append("coverage_fill")
        score += 0.5
    return {
        "frame_idx": frame_idx,
        "score": score,
        "reasons": sorted(set(reasons)),
        "full_n": sf["n"],
        "hybrid_n": sh["n"],
        "small_n": max(sf["small"], sh["small"]),
        "tiny_n": max(sf["tiny"], sh["tiny"]),
        "disagreement": disagreement,
        "edge_n": max(sf["edge"], sh["edge"]),
        "merged_diag": sh["merged_diag"] + sf["merged_diag"],
        "band_boost": boost,
    }


def select_active_learning_frames(
    *,
    video: Path,
    weights: Path,
    old_80: set[int],
    holdout_v2: set[int],
    n_target: int = MAX_AL,
    seed: int = 20260729,
    max_scan: int = 320,
) -> dict[str, Any]:
    """Score unused pool with YOLO11s FT ckpt; pick ≤100 stratified AL frames."""
    import torch
    from ultralytics import YOLO

    pool = candidate_pool(old_80=old_80, holdout_v2=holdout_v2)
    if len(pool) < n_target:
        raise RuntimeError(f"AL_POOL_TOO_SMALL:{len(pool)}<{n_target}")

    rng = _Rng(seed)
    # Stratified scan across time + hard bands from F2-C root cause.
    scan_set: set[int] = set()
    step = max(1, len(pool) // max_scan)
    scan_set.update(pool[::step][:max_scan])
    for _name, lo, hi, _w in DEV_HARD_BANDS:
        band = [i for i in pool if lo <= i <= hi]
        if not band:
            continue
        bstep = max(1, len(band) // 40)
        scan_set.update(band[::bstep][:40])
    scan = sorted(scan_set)[: max(max_scan, 200)]
    rng.shuffle(scan)
    assert_no_holdout_v2_inference(scan, set(holdout_v2))

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    scored: list[dict[str, Any]] = []
    hashes: list[tuple[int, int]] = []
    try:
        for idx in scan:
            if idx in holdout_v2:
                raise RuntimeError("AL_SCAN_HIT_HOLDOUT_V2")
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ph = _phash_like(gray)
            if any(_hamming(ph, h) <= 5 and abs(idx - j) < 30 for j, h in hashes):
                continue
            row = score_frame_with_checkpoint(model, frame, idx, device=device)
            row["phash"] = ph
            scored.append(row)
            hashes.append((idx, ph))
    finally:
        cap.release()

    if len(scored) < 40:
        raise RuntimeError(f"AL_SCAN_TOO_FEW:{len(scored)}")

    scored.sort(key=lambda r: (-r["score"], r["frame_idx"]))

    quotas = {
        "small_object": 30,
        "crowded": 20,
        "hard_negative": 15,
        "goal_sideline": 10,
    }
    picked: list[dict[str, Any]] = []
    picked_idx: set[int] = set()

    def _take(pred, need: int) -> None:
        for row in scored:
            if sum(1 for p in picked if pred(p)) >= need:
                return
            if row["frame_idx"] in picked_idx:
                continue
            if any(abs(row["frame_idx"] - p) < AL_MIN_SPACING for p in picked_idx):
                continue
            if not pred(row):
                continue
            picked.append(row)
            picked_idx.add(row["frame_idx"])

    _take(
        lambda r: "small_object" in r["reasons"] or r["tiny_n"] >= 1 or r["small_n"] >= 2,
        quotas["small_object"],
    )
    _take(lambda r: "crowded" in r["reasons"] or r["hybrid_n"] >= 9, quotas["crowded"])
    _take(
        lambda r: "hard_negative" in r["reasons"]
        or "likely_false_positive" in r["reasons"]
        or r["merged_diag"] > 0
        or r["disagreement"] >= 3,
        quotas["hard_negative"],
    )
    _take(
        lambda r: r["edge_n"] >= 2 or r["frame_idx"] < 220 or r["frame_idx"] > 780,
        quotas["goal_sideline"],
    )

    for row in scored:
        if len(picked) >= n_target:
            break
        if row["frame_idx"] in picked_idx:
            continue
        if any(abs(row["frame_idx"] - p) < AL_MIN_SPACING for p in picked_idx):
            continue
        if "coverage_fill" not in row["reasons"]:
            row = dict(row)
            row["reasons"] = sorted(set(row["reasons"]) | {"coverage_fill"})
        picked.append(row)
        picked_idx.add(row["frame_idx"])

    # If spacing blocked fill, relax spacing to 3 then 2.
    for spacing in (3, 2):
        if len(picked) >= min(n_target, 75):
            break
        for row in scored:
            if len(picked) >= n_target:
                break
            if row["frame_idx"] in picked_idx:
                continue
            if any(abs(row["frame_idx"] - p) < spacing for p in picked_idx):
                continue
            row = dict(row)
            row["reasons"] = sorted(set(row["reasons"]) | {"coverage_fill"})
            picked.append(row)
            picked_idx.add(row["frame_idx"])

    picked = sorted(picked, key=lambda r: r["frame_idx"])[:n_target]
    assert picked_idx.isdisjoint(holdout_v2)
    assert picked_idx.isdisjoint(old_80)

    frames = []
    for row in picked:
        reasons = list(row["reasons"])
        cats: list[str] = []
        if "small_object" in reasons:
            cats.append("small_distant")
        if "crowded" in reasons:
            cats.append("crowded")
        if "hard_negative" in reasons or "likely_false_positive" in reasons:
            cats.append("hard_negative")
        if row["edge_n"] >= 2 or row["frame_idx"] < 220 or row["frame_idx"] > 780:
            cats.extend(["sideline", "goal_area"])
        if not cats:
            cats.append("coverage_fill")
        frames.append(
            {
                "frame_idx": row["frame_idx"],
                "t_s": round(row["frame_idx"] / FPS, 4),
                "section": "active_learning",
                "split": "train",
                "selection_reasons": reasons,
                "planned_categories": sorted(set(cats)),
                "dev_score": row["score"],
                "diag": {
                    "full_n": row["full_n"],
                    "hybrid_n": row["hybrid_n"],
                    "small_n": row["small_n"],
                    "disagreement": row["disagreement"],
                    "band_boost": row.get("band_boost", 1.0),
                },
            }
        )

    counts = {
        "small_distant_weighted": sum(
            1 for f in frames if "small_distant" in f["planned_categories"]
        ),
        "crowded_occlusion": sum(1 for f in frames if "crowded" in f["planned_categories"]),
        "hard_negative_fp": sum(1 for f in frames if "hard_negative" in f["planned_categories"]),
        "goal_sideline": sum(
            1
            for f in frames
            if "sideline" in f["planned_categories"] or "goal_area" in f["planned_categories"]
        ),
        "total": len(frames),
    }
    body: dict[str, Any] = {
        "schema": "active_learning_selection_manifest_v1",
        "n_frames": len(frames),
        "frame_indices": [f["frame_idx"] for f in frames],
        "frames": frames,
        "counts": counts,
        "quotas_target": quotas,
        "checkpoint": str(weights),
        "dev_hard_bands": [
            {"name": n, "lo": a, "hi": b, "boost": w} for n, a, b, w in DEV_HARD_BANDS
        ],
        "holdout_v2_excluded": sorted(holdout_v2),
        "old_80_excluded": sorted(old_80),
        "pool_n": len(pool),
        "scan_n": len(scored),
        "note": (
            "selection_reasons are heuristics not accuracy/GT; "
            "uses F2-C root_cause bands + YOLO11s ckpt"
        ),
        "written_at_utc": utc_now(),
    }
    body["selection_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "indices": body["frame_indices"],
                "reasons": [f["selection_reasons"] for f in frames],
                "ckpt": Path(weights).name,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return body


__all__ = [
    "MAX_AL",
    "candidate_pool",
    "select_active_learning_frames",
]
