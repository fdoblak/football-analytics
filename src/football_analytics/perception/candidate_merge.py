"""Class-aware greedy merge of full-frame + tiled ball candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.detection_evaluation import bbox_iou


@dataclass(frozen=True)
class BallCandidate:
    """Pre-NMS ball candidate in source-frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str
    candidate_source: str  # full_frame | tile:<id>

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _sort_key(c: BallCandidate) -> tuple[Any, ...]:
    # Deterministic: score desc, then coords asc, then source.
    return (-float(c.score), c.x1, c.y1, c.x2, c.y2, c.candidate_source, c.class_id)


def class_aware_nms(
    candidates: Sequence[BallCandidate],
    *,
    merge_iou: float,
    class_aware: bool = True,
) -> list[BallCandidate]:
    """Greedy IoU suppression. Scores are raw detector scores — not calibrated probs."""
    if merge_iou < 0 or merge_iou > 1:
        raise ValueError("merge_iou must be in [0,1]")
    ordered = sorted(candidates, key=_sort_key)
    kept: list[BallCandidate] = []
    for cand in ordered:
        suppressed = False
        for prev in kept:
            if class_aware and (
                prev.class_id != cand.class_id or prev.class_name != cand.class_name
            ):
                continue
            if bbox_iou(cand.as_xyxy(), prev.as_xyxy()) >= float(merge_iou):
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


def merge_ball_candidates(
    full_frame: Sequence[BallCandidate],
    tiled: Sequence[BallCandidate],
    *,
    merge_iou: float,
    class_aware: bool = True,
) -> list[BallCandidate]:
    """Merge full-frame + tile candidates with class-aware NMS; preserve provenance."""
    combined = list(full_frame) + list(tiled)
    return class_aware_nms(combined, merge_iou=merge_iou, class_aware=class_aware)


__all__ = [
    "BallCandidate",
    "class_aware_nms",
    "merge_ball_candidates",
]
