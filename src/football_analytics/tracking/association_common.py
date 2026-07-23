"""Shared association primitives for Stage 6B/6C trackers.

Human and ball trackers keep entity-specific cost/gate logic; this module
holds the shared match record and deterministic greedy selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AssociationPair:
    track_id: int
    detection_id: int
    cost: float
    iou: float
    center_dist: float


def greedy_select_pairs(
    candidates: Sequence[tuple[float, int, int, float, float]],
) -> list[AssociationPair]:
    """Greedy one-to-one from (cost, track_id, detection_id, iou, center_dist).

    Sort key: cost ascending, track_id ascending, detection_id ascending.
    """
    ordered = sorted(candidates, key=lambda c: (c[0], c[1], c[2]))
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[AssociationPair] = []
    for cost, tid, did, iou, dist in ordered:
        if tid in used_tracks or did in used_dets:
            continue
        matches.append(
            AssociationPair(
                track_id=tid,
                detection_id=did,
                cost=float(cost),
                iou=float(iou),
                center_dist=float(dist),
            )
        )
        used_tracks.add(tid)
        used_dets.add(did)
    matches.sort(key=lambda m: (m.track_id, m.detection_id))
    return matches


def normalize_candidate_source(raw: str | None) -> str:
    """Map quality_flags/provenance tokens to full_frame|tile|hybrid|unknown."""
    if raw is None or not str(raw).strip():
        return "unknown"
    s = str(raw).strip().lower()
    if s.startswith("src:"):
        s = s[4:]
    if s.startswith("tile:") or s.startswith("tile"):
        return "tile"
    if "hybrid" in s:
        return "hybrid"
    if "full_frame" in s or s in {"full", "fullframe", "full-frame"}:
        return "full_frame"
    if s in {"tile", "tiled"}:
        return "tile"
    return "unknown"


__all__ = [
    "AssociationPair",
    "greedy_select_pairs",
    "normalize_candidate_source",
]
