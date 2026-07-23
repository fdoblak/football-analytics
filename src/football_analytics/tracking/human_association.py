"""Deterministic IoU + motion association (Stage 6B).

Greedy one-to-one assignment with stable tie-break:
(cost ascending, track_id ascending, detection_id ascending).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.detection_evaluation import bbox_iou, center_l2
from football_analytics.tracking.human_motion import BBox


@dataclass(frozen=True)
class AssociationPair:
    track_id: int
    detection_id: int
    cost: float
    iou: float
    center_dist: float


def association_cost(
    predicted_bbox: Sequence[float],
    detection_bbox: Sequence[float],
    *,
    iou_weight: float,
    motion_weight: float,
    motion_center_gate_px: float,
) -> tuple[float, float, float]:
    """Return (cost, iou, center_dist). Cost is finite and deterministic."""
    iou = float(bbox_iou(predicted_bbox, detection_bbox))
    dist = float(center_l2(predicted_bbox, detection_bbox))
    gate = max(float(motion_center_gate_px), 1e-6)
    motion_cost = min(1.0, dist / gate)
    iou_cost = 1.0 - iou
    cost = float(iou_weight) * iou_cost + float(motion_weight) * motion_cost
    # Quantize lightly for float stability across platforms.
    cost = round(cost, 12)
    return cost, iou, dist


def passes_association_gate(
    *,
    iou: float,
    center_dist: float,
    iou_gate: float,
    motion_center_gate_px: float,
) -> bool:
    """Accept if IoU gate OR motion center gate is satisfied."""
    if float(iou) >= float(iou_gate):
        return True
    return float(center_dist) <= float(motion_center_gate_px)


def greedy_associate(
    tracks: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    predicted_bboxes: Mapping[int, BBox],
    iou_gate: float,
    motion_center_gate_px: float,
    iou_weight: float,
    motion_weight: float,
) -> tuple[list[AssociationPair], list[int], list[int]]:
    """Greedy one-to-one association.

    ``tracks`` entries need ``track_id``.
    ``detections`` entries need ``detection_id`` and bbox fields.
    Returns (matches, unmatched_track_ids, unmatched_detection_ids).
    """
    candidates: list[tuple[float, int, int, int, int]] = []
    for ti, tr in enumerate(tracks):
        tid = int(tr["track_id"])
        pred = predicted_bboxes.get(tid)
        if pred is None:
            continue
        for di, det in enumerate(detections):
            did = int(det["detection_id"])
            det_bbox = (
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
            )
            cost, iou, dist = association_cost(
                pred,
                det_bbox,
                iou_weight=iou_weight,
                motion_weight=motion_weight,
                motion_center_gate_px=motion_center_gate_px,
            )
            if not passes_association_gate(
                iou=iou,
                center_dist=dist,
                iou_gate=iou_gate,
                motion_center_gate_px=motion_center_gate_px,
            ):
                continue
            candidates.append((cost, tid, did, ti, di))

    # Deterministic order: cost, track_id, detection_id.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[AssociationPair] = []
    for cost, tid, did, _ti, _di in candidates:
        if tid in used_tracks or did in used_dets:
            continue
        pred = predicted_bboxes[tid]
        det = next(d for d in detections if int(d["detection_id"]) == did)
        det_bbox = (
            float(det["bbox_x1"]),
            float(det["bbox_y1"]),
            float(det["bbox_x2"]),
            float(det["bbox_y2"]),
        )
        _, iou, dist = association_cost(
            pred,
            det_bbox,
            iou_weight=iou_weight,
            motion_weight=motion_weight,
            motion_center_gate_px=motion_center_gate_px,
        )
        matches.append(
            AssociationPair(
                track_id=tid,
                detection_id=did,
                cost=cost,
                iou=iou,
                center_dist=dist,
            )
        )
        used_tracks.add(tid)
        used_dets.add(did)

    unmatched_tracks = [int(t["track_id"]) for t in tracks if int(t["track_id"]) not in used_tracks]
    unmatched_dets = [
        int(d["detection_id"]) for d in detections if int(d["detection_id"]) not in used_dets
    ]
    unmatched_tracks.sort()
    unmatched_dets.sort()
    matches.sort(key=lambda m: (m.track_id, m.detection_id))
    return matches, unmatched_tracks, unmatched_dets


__all__ = [
    "AssociationPair",
    "association_cost",
    "passes_association_gate",
    "greedy_associate",
]
