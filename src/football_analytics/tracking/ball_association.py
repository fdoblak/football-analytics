"""Motion-first ball association (Stage 6C).

IoU alone is insufficient for small/fast balls. Cost combines center
displacement (vs constant-velocity predict), size consistency, confidence,
and IoU as support. Greedy one-to-one with deterministic tie-break.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.perception.detection_evaluation import bbox_iou, center_l2
from football_analytics.tracking.association_common import AssociationPair, greedy_select_pairs
from football_analytics.tracking.human_motion import BBox, bbox_wh


def _size_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    aw, ah = bbox_wh(a)
    bw, bh = bbox_wh(b)
    area_a = max(aw * ah, 1e-6)
    area_b = max(bw * bh, 1e-6)
    ratio = max(area_a / area_b, area_b / area_a)
    return float(ratio)


def effective_motion_gate_px(
    *,
    base_gate_px: float,
    dt_us: int,
    scale_per_us: float,
) -> float:
    """Time-scaled image-space motion gate (not a physical impossibility claim)."""
    extra = max(0, int(dt_us)) * float(scale_per_us)
    return float(base_gate_px) + float(extra)


def ball_association_cost(
    predicted_bbox: Sequence[float],
    detection_bbox: Sequence[float],
    *,
    detection_confidence: float | None,
    motion_gate_px: float,
    size_ratio_gate: float,
    motion_weight: float,
    size_weight: float,
    confidence_weight: float,
    iou_weight: float,
) -> tuple[float, float, float, float]:
    """Return (cost, iou, center_dist, size_ratio)."""
    iou = float(bbox_iou(predicted_bbox, detection_bbox))
    dist = float(center_l2(predicted_bbox, detection_bbox))
    gate = max(float(motion_gate_px), 1e-6)
    motion_cost = min(1.0, dist / gate)
    ratio = _size_ratio(predicted_bbox, detection_bbox)
    size_norm = max(float(size_ratio_gate), 1e-6)
    size_cost = min(1.0, max(0.0, (ratio - 1.0) / max(size_norm - 1.0, 1e-6)))
    conf = 0.5 if detection_confidence is None else float(detection_confidence)
    conf = min(1.0, max(0.0, conf))
    conf_cost = 1.0 - conf
    iou_cost = 1.0 - iou
    cost = (
        float(motion_weight) * motion_cost
        + float(size_weight) * size_cost
        + float(confidence_weight) * conf_cost
        + float(iou_weight) * iou_cost
    )
    cost = round(cost, 12)
    return cost, iou, dist, ratio


def passes_ball_association_gate(
    *,
    center_dist: float,
    motion_gate_px: float,
    size_ratio: float,
    size_ratio_gate: float,
    iou: float,
    iou_support_min: float,
    require_motion_gate: bool,
) -> bool:
    """Motion/size gates required; IoU is support only (never sufficient alone)."""
    if bool(require_motion_gate) and float(center_dist) > float(motion_gate_px):
        return False
    if float(size_ratio) > float(size_ratio_gate):
        return False
    # Optional IoU support floor (0.0 = IoU not required).
    return not (float(iou_support_min) > 0.0 and float(iou) < float(iou_support_min))


def greedy_ball_associate(
    tracks: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    predicted_bboxes: Mapping[int, BBox],
    dt_us_by_track: Mapping[int, int],
    motion_center_gate_px: float,
    motion_gate_scale_per_us: float,
    size_ratio_gate: float,
    iou_support_min: float,
    require_motion_gate: bool,
    motion_weight: float,
    size_weight: float,
    confidence_weight: float,
    iou_weight: float,
) -> tuple[list[AssociationPair], list[int], list[int]]:
    """Greedy motion-first ball association."""
    candidates: list[tuple[float, int, int, float, float]] = []
    for tr in tracks:
        tid = int(tr["track_id"])
        pred = predicted_bboxes.get(tid)
        if pred is None:
            continue
        dt = int(dt_us_by_track.get(tid, 0))
        gate = effective_motion_gate_px(
            base_gate_px=motion_center_gate_px,
            dt_us=dt,
            scale_per_us=motion_gate_scale_per_us,
        )
        for det in detections:
            did = int(det["detection_id"])
            det_bbox = (
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
            )
            conf = det.get("confidence")
            conf_f = None if conf is None else float(conf)
            cost, iou, dist, ratio = ball_association_cost(
                pred,
                det_bbox,
                detection_confidence=conf_f,
                motion_gate_px=gate,
                size_ratio_gate=size_ratio_gate,
                motion_weight=motion_weight,
                size_weight=size_weight,
                confidence_weight=confidence_weight,
                iou_weight=iou_weight,
            )
            if not passes_ball_association_gate(
                center_dist=dist,
                motion_gate_px=gate,
                size_ratio=ratio,
                size_ratio_gate=size_ratio_gate,
                iou=iou,
                iou_support_min=iou_support_min,
                require_motion_gate=require_motion_gate,
            ):
                continue
            candidates.append((cost, tid, did, iou, dist))

    matches = greedy_select_pairs(candidates)
    used_tracks = {m.track_id for m in matches}
    used_dets = {m.detection_id for m in matches}
    unmatched_tracks = [int(t["track_id"]) for t in tracks if int(t["track_id"]) not in used_tracks]
    unmatched_dets = [
        int(d["detection_id"]) for d in detections if int(d["detection_id"]) not in used_dets
    ]
    unmatched_tracks.sort()
    unmatched_dets.sort()
    return matches, unmatched_tracks, unmatched_dets


def primary_ball_score(
    *,
    association_cost_value: float | None,
    confidence: float | None,
    lifecycle_confirmed: bool,
    size_ratio: float,
    prefer_confirmed: bool,
) -> float:
    """Higher is better. Confidence alone is never decisive."""
    cost = 1.0 if association_cost_value is None else float(association_cost_value)
    conf = 0.5 if confidence is None else min(1.0, max(0.0, float(confidence)))
    size_pen = min(1.0, max(0.0, (float(size_ratio) - 1.0) / 2.0))
    score = (1.0 - min(1.0, cost)) * 0.55 + conf * 0.25 + (1.0 - size_pen) * 0.20
    if prefer_confirmed and lifecycle_confirmed:
        score += 0.05
    return round(score, 12)


__all__ = [
    "AssociationPair",
    "ball_association_cost",
    "effective_motion_gate_px",
    "passes_ball_association_gate",
    "greedy_ball_associate",
    "primary_ball_score",
]
