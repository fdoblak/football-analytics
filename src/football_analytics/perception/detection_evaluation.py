"""Deterministic human-detection evaluation (greedy one-to-one IoU matching)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class DetectionEvaluationError(ValueError):
    """Detection evaluation failure."""


NOT_EVALUATED = "NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH"


@dataclass(frozen=True)
class BBoxDetection:
    frame_index: int
    entity_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None
    detection_id: int | None = None
    reviewed: bool = True


@dataclass(frozen=True)
class MatchPair:
    pred_index: int
    gt_index: int
    iou: float
    frame_index: int


@dataclass(frozen=True)
class DetectionEvalMetrics:
    status: str
    true_positives: int | None
    false_positives: int | None
    false_negatives: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    ap50: float | None
    ap50_95: float | None
    mean_matched_iou: float | None
    mean_localization_error: float | None
    fp_per_frame: float | None
    missed_humans_per_frame: float | None
    empty_frame_accuracy: float | None
    mean_coverage_pred: float | None
    mean_coverage_gt: float | None
    predicted_count: int
    ground_truth_count: int
    matched_count: int
    frame_count: int
    iou_threshold: float | None
    matches: tuple[MatchPair, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ap50": self.ap50,
            "ap50_95": self.ap50_95,
            "mean_matched_iou": self.mean_matched_iou,
            "mean_localization_error": self.mean_localization_error,
            "fp_per_frame": self.fp_per_frame,
            "missed_humans_per_frame": self.missed_humans_per_frame,
            "empty_frame_accuracy": self.empty_frame_accuracy,
            "mean_coverage_pred": self.mean_coverage_pred,
            "mean_coverage_gt": self.mean_coverage_gt,
            "predicted_count": self.predicted_count,
            "ground_truth_count": self.ground_truth_count,
            "matched_count": self.matched_count,
            "frame_count": self.frame_count,
            "iou_threshold": self.iou_threshold,
            "matches": [
                {
                    "pred_index": m.pred_index,
                    "gt_index": m.gt_index,
                    "iou": m.iou,
                    "frame_index": m.frame_index,
                }
                for m in self.matches
            ],
            "notes": list(self.notes),
            "role_accuracy": None,
        }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_l2(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def greedy_match_iou(
    predictions: Sequence[BBoxDetection],
    ground_truth: Sequence[BBoxDetection],
    *,
    iou_threshold: float,
    require_entity_type: str = "human",
) -> list[MatchPair]:
    """Greedy deterministic one-to-one match by descending IoU, then indices."""
    if iou_threshold < 0 or iou_threshold > 1:
        raise DetectionEvaluationError("iou_threshold must be in [0,1]")
    preds = [(i, p) for i, p in enumerate(predictions) if p.entity_type == require_entity_type]
    gts = [(i, g) for i, g in enumerate(ground_truth) if g.entity_type == require_entity_type]
    candidates: list[tuple[float, int, int, int]] = []
    for pi, p in preds:
        for gi, g in gts:
            if int(p.frame_index) != int(g.frame_index):
                continue
            iou = bbox_iou((p.x1, p.y1, p.x2, p.y2), (g.x1, g.y1, g.x2, g.y2))
            if iou >= iou_threshold:
                # Sort key: higher IoU first → negate; then pred idx, gt idx.
                candidates.append((-iou, pi, gi, int(p.frame_index)))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[MatchPair] = []
    for neg_iou, pi, gi, frame_index in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append(
            MatchPair(pred_index=pi, gt_index=gi, iou=float(-neg_iou), frame_index=frame_index)
        )
    matches.sort(key=lambda m: (m.frame_index, m.gt_index, m.pred_index))
    return matches


def _frame_indices(*groups: Sequence[BBoxDetection]) -> list[int]:
    frames: set[int] = set()
    for group in groups:
        for d in group:
            frames.add(int(d.frame_index))
    return sorted(frames)


def _coverage(dets: Sequence[BBoxDetection], frame_index: int) -> float:
    boxes = [
        (d.x1, d.y1, d.x2, d.y2)
        for d in dets
        if int(d.frame_index) == frame_index and d.entity_type == "human"
    ]
    if not boxes:
        return 0.0
    # Approximate coverage as summed area normalized by first-box-derived frame proxy:
    # without frame size, report mean relative area sum clipped later by caller.
    areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in boxes]
    return float(sum(areas))


def average_precision_at_iou(
    predictions: Sequence[BBoxDetection],
    ground_truth: Sequence[BBoxDetection],
    *,
    iou_threshold: float,
) -> float | None:
    """Simple AP via score-sorted greedy matching at a fixed IoU threshold."""
    preds = sorted(
        [p for p in predictions if p.entity_type == "human"],
        key=lambda p: (-(p.score if p.score is not None else 0.0), p.frame_index),
    )
    gts = [g for g in ground_truth if g.entity_type == "human"]
    if not gts and not preds:
        return None
    if not gts:
        return 0.0 if preds else None

    matched_gt: set[int] = set()
    tp_flags: list[int] = []
    for p in preds:
        best_iou = -1.0
        best_gi: int | None = None
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            if int(g.frame_index) != int(p.frame_index):
                continue
            iou = bbox_iou((p.x1, p.y1, p.x2, p.y2), (g.x1, g.y1, g.x2, g.y2))
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None:
            matched_gt.add(best_gi)
            tp_flags.append(1)
        else:
            tp_flags.append(0)

    if not tp_flags:
        return 0.0

    tps = 0
    fps = 0
    precisions: list[float] = []
    recalls: list[float] = []
    n_gt = len(gts)
    for flag in tp_flags:
        if flag:
            tps += 1
        else:
            fps += 1
        precisions.append(tps / float(tps + fps))
        recalls.append(tps / float(n_gt))

    # 11-point interpolation style AP.
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        candidates = [p for p, r in zip(precisions, recalls, strict=True) if r >= t]
        ap += max(candidates) if candidates else 0.0
    return ap / 11.0


def not_evaluated_metrics(*, reason: str = NOT_EVALUATED) -> DetectionEvalMetrics:
    return DetectionEvalMetrics(
        status=reason,
        true_positives=None,
        false_positives=None,
        false_negatives=None,
        precision=None,
        recall=None,
        f1=None,
        ap50=None,
        ap50_95=None,
        mean_matched_iou=None,
        mean_localization_error=None,
        fp_per_frame=None,
        missed_humans_per_frame=None,
        empty_frame_accuracy=None,
        mean_coverage_pred=None,
        mean_coverage_gt=None,
        predicted_count=0,
        ground_truth_count=0,
        matched_count=0,
        frame_count=0,
        iou_threshold=None,
        matches=(),
        notes=(reason, "No invented mAP without reviewed ground truth."),
    )


def ground_truth_is_reviewed(rows: Sequence[Mapping[str, Any]] | None) -> bool:
    if not rows:
        return False
    for r in rows:
        status = str(r.get("review_status", r.get("status", ""))).lower()
        if status in {"reviewed", "accepted"}:
            return True
        if r.get("reviewed") is True:
            return True
    # Explicit fixture flag
    return any(bool(r.get("is_reviewed_ground_truth")) for r in rows)


def parse_detection_boxes(
    rows: Sequence[Mapping[str, Any]],
    *,
    default_entity: str = "human",
    require_reviewed: bool = False,
) -> list[BBoxDetection]:
    out: list[BBoxDetection] = []
    for r in rows:
        entity = str(r.get("entity_type", default_entity)).lower()
        reviewed = True
        status = str(r.get("review_status", "")).lower()
        if "review_status" in r:
            reviewed = status in {"reviewed", "accepted"}
        if r.get("reviewed") is False:
            reviewed = False
        if require_reviewed and not reviewed and not r.get("is_reviewed_ground_truth"):
            continue
        if "bbox_x1" in r:
            x1, y1, x2, y2 = (
                float(r["bbox_x1"]),
                float(r["bbox_y1"]),
                float(r["bbox_x2"]),
                float(r["bbox_y2"]),
            )
        elif "bbox" in r and isinstance(r["bbox"], (list, tuple)) and len(r["bbox"]) == 4:
            x1, y1, x2, y2 = (float(v) for v in r["bbox"])
        else:
            raise DetectionEvaluationError("detection row missing bbox fields")
        out.append(
            BBoxDetection(
                frame_index=int(r["frame_index"]),
                entity_type=entity,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=None if r.get("confidence") is None else float(r["confidence"]),
                detection_id=None if r.get("detection_id") is None else int(r["detection_id"]),
                reviewed=reviewed or bool(r.get("is_reviewed_ground_truth")),
            )
        )
    return out


def evaluate_human_detections(
    predictions: Sequence[BBoxDetection],
    ground_truth: Sequence[BBoxDetection],
    *,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> DetectionEvalMetrics:
    preds = [p for p in predictions if p.entity_type == "human"]
    gts = [g for g in ground_truth if g.entity_type == "human"]
    frames = _frame_indices(preds, gts)
    matches = greedy_match_iou(preds, gts, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _f1(precision, recall)

    mean_iou = (sum(m.iou for m in matches) / len(matches)) if matches else None
    loc_errors = [
        center_l2(
            (
                preds[m.pred_index].x1,
                preds[m.pred_index].y1,
                preds[m.pred_index].x2,
                preds[m.pred_index].y2,
            ),
            (gts[m.gt_index].x1, gts[m.gt_index].y1, gts[m.gt_index].x2, gts[m.gt_index].y2),
        )
        for m in matches
    ]
    mean_loc = (sum(loc_errors) / len(loc_errors)) if loc_errors else None

    n_frames = len(frames) if frames else 0
    fp_per_frame = _safe_div(float(fp), float(n_frames)) if n_frames else None
    miss_per_frame = _safe_div(float(fn), float(n_frames)) if n_frames else None

    empty_correct = 0
    empty_total = 0
    cov_pred: list[float] = []
    cov_gt: list[float] = []
    frame_area = None
    if frame_width and frame_height and frame_width > 0 and frame_height > 0:
        frame_area = float(frame_width) * float(frame_height)
    for fi in frames:
        p_count = sum(1 for p in preds if int(p.frame_index) == fi)
        g_count = sum(1 for g in gts if int(g.frame_index) == fi)
        if g_count == 0:
            empty_total += 1
            if p_count == 0:
                empty_correct += 1
        if frame_area:
            cov_pred.append(min(1.0, _coverage(preds, fi) / frame_area))
            cov_gt.append(min(1.0, _coverage(gts, fi) / frame_area))

    empty_acc = _safe_div(float(empty_correct), float(empty_total))
    mean_cov_p = (sum(cov_pred) / len(cov_pred)) if cov_pred else None
    mean_cov_g = (sum(cov_gt) / len(cov_gt)) if cov_gt else None

    ap50 = average_precision_at_iou(preds, gts, iou_threshold=0.5)
    thr = list(iou_thresholds) if iou_thresholds else [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = [average_precision_at_iou(preds, gts, iou_threshold=float(t)) for t in thr]
    aps_valid = [a for a in aps if a is not None]
    ap50_95 = (sum(aps_valid) / len(aps_valid)) if aps_valid else None

    return DetectionEvalMetrics(
        status="EVALUATED",
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        ap50_95=ap50_95,
        mean_matched_iou=mean_iou,
        mean_localization_error=mean_loc,
        fp_per_frame=fp_per_frame,
        missed_humans_per_frame=miss_per_frame,
        empty_frame_accuracy=empty_acc,
        mean_coverage_pred=mean_cov_p,
        mean_coverage_gt=mean_cov_g,
        predicted_count=len(preds),
        ground_truth_count=len(gts),
        matched_count=tp,
        frame_count=n_frames,
        iou_threshold=float(iou_threshold),
        matches=tuple(matches),
        notes=(),
    )


def evaluate_from_rows(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]] | None,
    *,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] | None = None,
    require_reviewed_gt: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> DetectionEvalMetrics:
    if ground_truth is None or not ground_truth:
        return not_evaluated_metrics()
    if require_reviewed_gt and not ground_truth_is_reviewed(ground_truth):
        return not_evaluated_metrics()
    preds = parse_detection_boxes(predictions, default_entity="human")
    gts = parse_detection_boxes(
        ground_truth, default_entity="human", require_reviewed=require_reviewed_gt
    )
    if require_reviewed_gt and not gts:
        return not_evaluated_metrics()
    return evaluate_human_detections(
        preds,
        gts,
        iou_threshold=iou_threshold,
        iou_thresholds=iou_thresholds,
        frame_width=frame_width,
        frame_height=frame_height,
    )


__all__ = [
    "NOT_EVALUATED",
    "DetectionEvaluationError",
    "BBoxDetection",
    "MatchPair",
    "DetectionEvalMetrics",
    "bbox_iou",
    "center_l2",
    "greedy_match_iou",
    "average_precision_at_iou",
    "not_evaluated_metrics",
    "ground_truth_is_reviewed",
    "parse_detection_boxes",
    "evaluate_human_detections",
    "evaluate_from_rows",
]
