"""Lightweight MOT evaluation helpers for TeamTrack pilot (no GT leakage into predict)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from football_analytics.acceptance.teamtrack.loader import MotBox


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def scale_box(
    box: tuple[float, float, float, float], *, sx: float, sy: float
) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x * sx, y * sy, w * sx, h * sy)


def evaluate_detection_frames(
    *,
    gt_by_frame: dict[int, list[tuple[float, float, float, float]]],
    pred_by_frame: dict[int, list[tuple[float, float, float, float]]],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for frame, gts in gt_by_frame.items():
        preds = list(pred_by_frame.get(frame, []))
        matched_p: set[int] = set()
        for g in gts:
            best_i, best = -1, 0.0
            for i, p in enumerate(preds):
                if i in matched_p:
                    continue
                v = iou_xywh(g, p)
                if v > best:
                    best, best_i = v, i
            if best >= iou_thresh and best_i >= 0:
                tp += 1
                matched_p.add(best_i)
            else:
                fn += 1
        fp += max(0, len(preds) - len(matched_p))
    # also count preds on frames without GT as FP
    for frame, preds in pred_by_frame.items():
        if frame not in gt_by_frame:
            fp += len(preds)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_thresh": iou_thresh,
        "note": "framewise greedy IoU matching; not COCO mAP",
    }


def evaluate_target_track(
    *,
    gt_boxes: list[MotBox],
    pred_boxes_by_frame: dict[int, tuple[float, float, float, float] | None],
    iou_thresh: float = 0.3,
) -> dict[str, Any]:
    """Target coverage + position agreement for one GT track vs predicted box stream."""
    hits = 0
    ious: list[float] = []
    center_err: list[float] = []
    for g in gt_boxes:
        p = pred_boxes_by_frame.get(g.frame)
        if p is None:
            continue
        v = iou_xywh((g.x, g.y, g.w, g.h), p)
        ious.append(v)
        if v >= iou_thresh:
            hits += 1
        gcx, gcy = g.x + g.w / 2.0, g.y + g.h / 2.0
        pcx, pcy = p[0] + p[2] / 2.0, p[1] + p[3] / 2.0
        center_err.append(((gcx - pcx) ** 2 + (gcy - pcy) ** 2) ** 0.5)
    n = len(gt_boxes)
    return {
        "gt_frames": n,
        "matched_frames_iou_ge_thresh": hits,
        "target_coverage_ratio": hits / n if n else 0.0,
        "mean_iou": (sum(ious) / len(ious)) if ious else 0.0,
        "mean_center_error_px": (sum(center_err) / len(center_err)) if center_err else None,
        "iou_thresh": iou_thresh,
        "id_switches": "not_evaluable_without_full_mot_tracker_ids",
        "hota_mota_idf1": "not_evaluable_lightweight_pilot_evaluator",
    }


def group_gt_boxes(
    boxes: tuple[MotBox, ...], track_id: int | None = None
) -> dict[int, list[MotBox]]:
    out: dict[int, list[MotBox]] = defaultdict(list)
    for b in boxes:
        if track_id is not None and b.track_id != track_id:
            continue
        out[b.frame].append(b)
    return dict(out)


__all__ = [
    "evaluate_detection_frames",
    "evaluate_target_track",
    "group_gt_boxes",
    "iou_xywh",
    "scale_box",
]
