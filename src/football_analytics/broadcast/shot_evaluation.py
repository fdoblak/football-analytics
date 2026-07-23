"""Deterministic shot-boundary evaluation (greedy one-to-one matching)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ShotEvaluationError(ValueError):
    """Evaluation failure."""


@dataclass(frozen=True)
class BoundaryEvent:
    time_us: int
    transition_type: str | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class MatchPair:
    pred_index: int
    gt_index: int
    abs_error_us: int


@dataclass(frozen=True)
class EvaluationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_abs_timing_error_us: float | None
    median_abs_timing_error_us: float | None
    max_abs_timing_error_us: int | None
    fp_per_minute: float | None
    over_segmentation: int
    under_segmentation: int
    shot_count_error: int
    predicted_count: int
    ground_truth_count: int
    matches: tuple[MatchPair, ...]
    per_transition: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_abs_timing_error_us": self.mean_abs_timing_error_us,
            "median_abs_timing_error_us": self.median_abs_timing_error_us,
            "max_abs_timing_error_us": self.max_abs_timing_error_us,
            "fp_per_minute": self.fp_per_minute,
            "over_segmentation": self.over_segmentation,
            "under_segmentation": self.under_segmentation,
            "shot_count_error": self.shot_count_error,
            "predicted_count": self.predicted_count,
            "ground_truth_count": self.ground_truth_count,
            "matches": [
                {
                    "pred_index": m.pred_index,
                    "gt_index": m.gt_index,
                    "abs_error_us": m.abs_error_us,
                }
                for m in self.matches
            ],
            "per_transition": {k: dict(v) for k, v in self.per_transition.items()},
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


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def parse_boundary_events(rows: Sequence[Mapping[str, Any]]) -> list[BoundaryEvent]:
    events: list[BoundaryEvent] = []
    for r in rows:
        if "boundary_time_us" not in r:
            raise ShotEvaluationError("boundary row missing boundary_time_us")
        events.append(
            BoundaryEvent(
                time_us=int(r["boundary_time_us"]),
                transition_type=(
                    str(r["transition_type"]) if r.get("transition_type") is not None else None
                ),
                event_id=str(r["boundary_id"]) if r.get("boundary_id") is not None else None,
            )
        )
    events.sort(key=lambda e: (e.time_us, e.event_id or ""))
    return events


def greedy_match(
    predictions: Sequence[BoundaryEvent],
    ground_truth: Sequence[BoundaryEvent],
    *,
    tolerance_us: int,
) -> list[MatchPair]:
    """Greedy deterministic one-to-one match by min |pred-gt| within tolerance.

    Both sides sorted; candidates considered in ascending absolute error, then
    pred index, then gt index. No double-counting.
    """
    if tolerance_us < 0:
        raise ShotEvaluationError("tolerance_us must be >= 0")
    preds = list(enumerate(predictions))
    gts = list(enumerate(ground_truth))
    candidates: list[tuple[int, int, int]] = []
    for pi, p in preds:
        for gi, g in gts:
            err = abs(p.time_us - g.time_us)
            if err <= tolerance_us:
                candidates.append((err, pi, gi))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[MatchPair] = []
    for err, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append(MatchPair(pred_index=pi, gt_index=gi, abs_error_us=err))
    matches.sort(key=lambda m: (m.gt_index, m.pred_index))
    return matches


def evaluate_boundaries(
    predictions: Sequence[BoundaryEvent],
    ground_truth: Sequence[BoundaryEvent],
    *,
    tolerance_us: int,
    duration_us: int | None = None,
) -> EvaluationMetrics:
    preds = sorted(predictions, key=lambda e: (e.time_us, e.event_id or ""))
    gts = sorted(ground_truth, key=lambda e: (e.time_us, e.event_id or ""))
    matches = greedy_match(preds, gts, tolerance_us=tolerance_us)
    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _f1(precision, recall)
    errors = [m.abs_error_us for m in matches]
    mean_err = (sum(errors) / len(errors)) if errors else None
    med_err = _median(errors)
    max_err = max(errors) if errors else None

    fp_per_minute = None
    if duration_us is not None and duration_us > 0:
        minutes = duration_us / 60_000_000.0
        if minutes > 0:
            fp_per_minute = fp / minutes

    # Shot count = boundaries + 1 (for coverage [0, duration))
    pred_shots = len(preds) + 1
    gt_shots = len(gts) + 1
    shot_count_error = pred_shots - gt_shots
    over_segmentation = max(0, shot_count_error)
    under_segmentation = max(0, -shot_count_error)

    # Per-transition metrics (by GT type; unmatched GT → FN; unmatched pred typed → FP)
    types: set[str] = set()
    for e in gts:
        if e.transition_type:
            types.add(e.transition_type)
    for e in preds:
        if e.transition_type:
            types.add(e.transition_type)
    matched_gt = {m.gt_index for m in matches}
    matched_pred = {m.pred_index for m in matches}
    per: dict[str, dict[str, Any]] = {}
    for t in sorted(types):
        gt_idx = [i for i, e in enumerate(gts) if e.transition_type == t]
        pred_idx = [i for i, e in enumerate(preds) if e.transition_type == t]
        tp_t = sum(1 for i in gt_idx if i in matched_gt)
        fn_t = sum(1 for i in gt_idx if i not in matched_gt)
        # FP for type: unmatched predictions labeled as this type
        fp_t = sum(1 for i in pred_idx if i not in matched_pred)
        # Also count matched preds whose GT type differs? Keep simple: type metrics by GT.
        p_t = _safe_div(float(tp_t), float(tp_t + fp_t))
        r_t = _safe_div(float(tp_t), float(tp_t + fn_t))
        per[t] = {
            "true_positives": tp_t,
            "false_positives": fp_t,
            "false_negatives": fn_t,
            "precision": p_t,
            "recall": r_t,
            "f1": _f1(p_t, r_t),
        }

    return EvaluationMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_abs_timing_error_us=mean_err,
        median_abs_timing_error_us=med_err,
        max_abs_timing_error_us=max_err,
        fp_per_minute=fp_per_minute,
        over_segmentation=over_segmentation,
        under_segmentation=under_segmentation,
        shot_count_error=shot_count_error,
        predicted_count=len(preds),
        ground_truth_count=len(gts),
        matches=tuple(matches),
        per_transition=per,
    )


def evaluate_from_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    ground_truth_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_us: int,
    duration_us: int | None = None,
) -> EvaluationMetrics:
    return evaluate_boundaries(
        parse_boundary_events(prediction_rows),
        parse_boundary_events(ground_truth_rows),
        tolerance_us=tolerance_us,
        duration_us=duration_us,
    )


def assert_finite_metrics(metrics: EvaluationMetrics) -> None:
    for name in ("precision", "recall", "f1", "mean_abs_timing_error_us", "fp_per_minute"):
        val = getattr(metrics, name)
        if val is None:
            continue
        if isinstance(val, float) and not math.isfinite(val):
            raise ShotEvaluationError(f"{name} is non-finite")


__all__ = [
    "ShotEvaluationError",
    "BoundaryEvent",
    "MatchPair",
    "EvaluationMetrics",
    "parse_boundary_events",
    "greedy_match",
    "evaluate_boundaries",
    "evaluate_from_rows",
    "assert_finite_metrics",
]
