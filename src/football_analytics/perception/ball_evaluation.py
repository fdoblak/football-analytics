"""Deterministic ball-detection evaluation (greedy one-to-one IoU matching)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.detection_evaluation import (
    BBoxDetection,
    DetectionEvalMetrics,
    DetectionEvaluationError,
    MatchPair,
    bbox_iou,
    center_l2,
    greedy_match_iou,
    ground_truth_is_reviewed,
    parse_detection_boxes,
)

NOT_EVALUATED_BALL = "NOT_EVALUATED_NO_REVIEWED_BALL_GROUND_TRUTH"
SMALL_OBJECT_AREA_FRACTION = 0.001  # relative to frame when frame size known; else absolute px^2


@dataclass(frozen=True)
class BallEvalMetrics:
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
    mean_center_distance_error: float | None
    mean_bbox_size_error: float | None
    small_object_recall: float | None
    fp_per_frame: float | None
    no_ball_negative_accuracy: float | None
    mean_coverage_pred: float | None
    mean_coverage_gt: float | None
    predicted_count: int
    ground_truth_count: int
    matched_count: int
    frame_count: int
    iou_threshold: float | None
    matches: tuple[MatchPair, ...]
    mode_comparison: Mapping[str, Any] | None
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
            "mean_center_distance_error": self.mean_center_distance_error,
            "mean_bbox_size_error": self.mean_bbox_size_error,
            "small_object_recall": self.small_object_recall,
            "fp_per_frame": self.fp_per_frame,
            "no_ball_negative_accuracy": self.no_ball_negative_accuracy,
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
            "mode_comparison": dict(self.mode_comparison) if self.mode_comparison else None,
            "notes": list(self.notes),
            "role_accuracy": None,
            "player_ownership": None,
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


def _frame_indices(*groups: Sequence[BBoxDetection]) -> list[int]:
    frames: set[int] = set()
    for group in groups:
        for d in group:
            frames.add(int(d.frame_index))
    return sorted(frames)


def _area(d: BBoxDetection) -> float:
    return max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)


def _size_l1(a: BBoxDetection, b: BBoxDetection) -> float:
    aw = max(0.0, a.x2 - a.x1)
    ah = max(0.0, a.y2 - a.y1)
    bw = max(0.0, b.x2 - b.x1)
    bh = max(0.0, b.y2 - b.y1)
    return abs(aw - bw) + abs(ah - bh)


def average_precision_ball(
    predictions: Sequence[BBoxDetection],
    ground_truth: Sequence[BBoxDetection],
    *,
    iou_threshold: float,
) -> float | None:
    preds = sorted(
        [p for p in predictions if p.entity_type == "ball"],
        key=lambda p: (-(p.score if p.score is not None else 0.0), p.frame_index),
    )
    gts = [g for g in ground_truth if g.entity_type == "ball"]
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

    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        candidates = [p for p, r in zip(precisions, recalls, strict=True) if r >= t]
        ap += max(candidates) if candidates else 0.0
    return ap / 11.0


def not_evaluated_ball_metrics(*, reason: str = NOT_EVALUATED_BALL) -> BallEvalMetrics:
    return BallEvalMetrics(
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
        mean_center_distance_error=None,
        mean_bbox_size_error=None,
        small_object_recall=None,
        fp_per_frame=None,
        no_ball_negative_accuracy=None,
        mean_coverage_pred=None,
        mean_coverage_gt=None,
        predicted_count=0,
        ground_truth_count=0,
        matched_count=0,
        frame_count=0,
        iou_threshold=None,
        matches=(),
        mode_comparison=None,
        notes=(reason, "No invented mAP without reviewed ball ground truth."),
    )


def evaluate_ball_detections(
    predictions: Sequence[BBoxDetection],
    ground_truth: Sequence[BBoxDetection],
    *,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
    mode_comparison: Mapping[str, Any] | None = None,
) -> BallEvalMetrics:
    if iou_threshold < 0 or iou_threshold > 1:
        raise DetectionEvaluationError("iou_threshold must be in [0,1]")
    preds = [p for p in predictions if p.entity_type == "ball"]
    gts = [g for g in ground_truth if g.entity_type == "ball"]
    frames = _frame_indices(preds, gts)
    matches = greedy_match_iou(preds, gts, iou_threshold=iou_threshold, require_entity_type="ball")
    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _f1(precision, recall)

    mean_iou = (sum(m.iou for m in matches) / len(matches)) if matches else None
    center_errs = [
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
    size_errs = [_size_l1(preds[m.pred_index], gts[m.gt_index]) for m in matches]
    mean_center = (sum(center_errs) / len(center_errs)) if center_errs else None
    mean_size = (sum(size_errs) / len(size_errs)) if size_errs else None

    n_frames = len(frames) if frames else 0
    fp_per_frame = _safe_div(float(fp), float(n_frames)) if n_frames else None

    # Small-object recall among GT boxes below area threshold.
    frame_area = None
    if frame_width and frame_height and frame_width > 0 and frame_height > 0:
        frame_area = float(frame_width) * float(frame_height)
    small_gts = []
    for g in gts:
        area = _area(g)
        if frame_area:
            if area / frame_area <= SMALL_OBJECT_AREA_FRACTION:
                small_gts.append(g)
        elif area <= 32.0 * 32.0:
            small_gts.append(g)
    if small_gts:
        small_matches = greedy_match_iou(
            preds, small_gts, iou_threshold=iou_threshold, require_entity_type="ball"
        )
        small_object_recall = _safe_div(float(len(small_matches)), float(len(small_gts)))
    else:
        small_object_recall = None

    # No-ball negative accuracy: frames with zero GT ball and zero preds.
    neg_correct = 0
    neg_total = 0
    cov_pred: list[float] = []
    cov_gt: list[float] = []
    for fi in frames:
        p_count = sum(1 for p in preds if int(p.frame_index) == fi)
        g_count = sum(1 for g in gts if int(g.frame_index) == fi)
        if g_count == 0:
            neg_total += 1
            if p_count == 0:
                neg_correct += 1
        if frame_area:
            p_area = sum(_area(p) for p in preds if int(p.frame_index) == fi)
            g_area = sum(_area(g) for g in gts if int(g.frame_index) == fi)
            cov_pred.append(min(1.0, p_area / frame_area))
            cov_gt.append(min(1.0, g_area / frame_area))

    no_ball_acc = _safe_div(float(neg_correct), float(neg_total))
    mean_cov_p = (sum(cov_pred) / len(cov_pred)) if cov_pred else None
    mean_cov_g = (sum(cov_gt) / len(cov_gt)) if cov_gt else None

    ap50 = average_precision_ball(preds, gts, iou_threshold=0.5)
    thr = list(iou_thresholds) if iou_thresholds else [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = [average_precision_ball(preds, gts, iou_threshold=float(t)) for t in thr]
    aps_valid = [a for a in aps if a is not None]
    ap50_95 = (sum(aps_valid) / len(aps_valid)) if aps_valid else None

    return BallEvalMetrics(
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
        mean_center_distance_error=mean_center,
        mean_bbox_size_error=mean_size,
        small_object_recall=small_object_recall,
        fp_per_frame=fp_per_frame,
        no_ball_negative_accuracy=no_ball_acc,
        mean_coverage_pred=mean_cov_p,
        mean_coverage_gt=mean_cov_g,
        predicted_count=len(preds),
        ground_truth_count=len(gts),
        matched_count=tp,
        frame_count=n_frames,
        iou_threshold=float(iou_threshold),
        matches=tuple(matches),
        mode_comparison=dict(mode_comparison) if mode_comparison else None,
        notes=(),
    )


def evaluate_ball_from_rows(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]] | None,
    *,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] | None = None,
    require_reviewed_gt: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
    mode_comparison: Mapping[str, Any] | None = None,
) -> BallEvalMetrics:
    if ground_truth is None or not ground_truth:
        return not_evaluated_ball_metrics()
    if require_reviewed_gt and not ground_truth_is_reviewed(ground_truth):
        return not_evaluated_ball_metrics()
    preds = parse_detection_boxes(predictions, default_entity="ball")
    gts = parse_detection_boxes(
        ground_truth, default_entity="ball", require_reviewed=require_reviewed_gt
    )
    if require_reviewed_gt and not gts:
        return not_evaluated_ball_metrics()
    return evaluate_ball_detections(
        preds,
        gts,
        iou_threshold=iou_threshold,
        iou_thresholds=iou_thresholds,
        frame_width=frame_width,
        frame_height=frame_height,
        mode_comparison=mode_comparison,
    )


# Frozen synthetic fixtures for unit tests (never use model preds as GT).
FROZEN_BALL_FIXTURES: dict[str, dict[str, Any]] = {
    "single_ball": {
        "predictions": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [10, 10, 20, 20],
                "confidence": 0.9,
            }
        ],
        "ground_truth": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [10, 10, 20, 20],
                "is_reviewed_ground_truth": True,
            }
        ],
        "expect_tp": 1,
        "expect_fp": 0,
        "expect_fn": 0,
    },
    "two_cand_one_gt": {
        "predictions": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [10, 10, 20, 20],
                "confidence": 0.9,
            },
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [50, 50, 60, 60],
                "confidence": 0.8,
            },
        ],
        "ground_truth": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [11, 11, 21, 21],
                "is_reviewed_ground_truth": True,
            }
        ],
        "expect_tp": 1,
        "expect_fp": 1,
        "expect_fn": 0,
    },
    "no_ball": {
        "predictions": [],
        "ground_truth": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [0, 0, 0, 0],
                "is_reviewed_ground_truth": True,
                "_skip": True,
            }
        ],
        "expect_tp": 0,
        "expect_fp": 0,
        "expect_fn": 0,
        "empty_gt": True,
    },
    "missed": {
        "predictions": [],
        "ground_truth": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [5, 5, 15, 15],
                "is_reviewed_ground_truth": True,
            }
        ],
        "expect_tp": 0,
        "expect_fp": 0,
        "expect_fn": 1,
    },
    "false_positive": {
        "predictions": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [40, 40, 50, 50],
                "confidence": 0.7,
            }
        ],
        "ground_truth": [],
        "expect_tp": 0,
        "expect_fp": 1,
        "expect_fn": 0,
        "empty_gt_list": True,
    },
    "small_bbox": {
        "predictions": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [100, 100, 104, 104],
                "confidence": 0.6,
            }
        ],
        "ground_truth": [
            {
                "frame_index": 0,
                "entity_type": "ball",
                "bbox": [100, 100, 104, 104],
                "is_reviewed_ground_truth": True,
            }
        ],
        "expect_tp": 1,
        "expect_fp": 0,
        "expect_fn": 0,
    },
}


__all__ = [
    "NOT_EVALUATED_BALL",
    "BallEvalMetrics",
    "average_precision_ball",
    "not_evaluated_ball_metrics",
    "evaluate_ball_detections",
    "evaluate_ball_from_rows",
    "FROZEN_BALL_FIXTURES",
    "DetectionEvalMetrics",
    "BBoxDetection",
    "bbox_iou",
]
