"""Stage 4D broadcast routing evaluation (safety metrics, deterministic)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.broadcast.playability import TASK_AXES
from football_analytics.broadcast.types import Eligibility, Playability, ReplayStatus

ELIGIBILITY_LABELS = tuple(e.value for e in Eligibility)


class BroadcastEvaluationError(ValueError):
    """Broadcast evaluation failure."""


@dataclass(frozen=True)
class BroadcastEvaluationReport:
    window_temporal_coverage: float | None
    gap_rate: float | None
    overlap_rate: float | None
    unexplained_gap_rate: float | None
    eligibility_axes: Mapping[str, Mapping[str, Any]]
    macro_f1: Mapping[str, float | None]
    manual_review_recall: float | None
    unsafe_tracking_false_positive_rate: float | None
    unsafe_calibration_false_positive_rate: float | None
    unsafe_live_event_false_positive_rate: float | None
    unsafe_physical_metric_false_positive_rate: float | None
    non_playable_eligible_false_positive_rate: float | None
    decision_code_accuracy: float | None
    deterministic_repeat: bool | None
    n_pred: int
    n_gt: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_temporal_coverage": self.window_temporal_coverage,
            "gap_rate": self.gap_rate,
            "overlap_rate": self.overlap_rate,
            "unexplained_gap_rate": self.unexplained_gap_rate,
            "eligibility_axes": {k: dict(v) for k, v in self.eligibility_axes.items()},
            "macro_f1": dict(self.macro_f1),
            "manual_review_recall": self.manual_review_recall,
            "unsafe_tracking_false_positive_rate": self.unsafe_tracking_false_positive_rate,
            "unsafe_calibration_false_positive_rate": self.unsafe_calibration_false_positive_rate,
            "unsafe_live_event_false_positive_rate": self.unsafe_live_event_false_positive_rate,
            "unsafe_physical_metric_false_positive_rate": (
                self.unsafe_physical_metric_false_positive_rate
            ),
            "non_playable_eligible_false_positive_rate": (
                self.non_playable_eligible_false_positive_rate
            ),
            "decision_code_accuracy": self.decision_code_accuracy,
            "deterministic_repeat": self.deterministic_repeat,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "status": self.status,
        }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _overlap_rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if len(rows) < 2:
        return 0.0 if rows else None
    ordered = sorted(rows, key=lambda r: (int(r["start_time_us"]), str(r["analysis_window_id"])))
    overlaps = 0
    for a, b in zip(ordered, ordered[1:], strict=False):
        if _intervals_overlap(
            int(a["start_time_us"]),
            int(a["end_time_us"]),
            int(b["start_time_us"]),
            int(b["end_time_us"]),
        ):
            overlaps += 1
    return overlaps / max(1, len(ordered) - 1)


def _coverage_vs_span(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    start = min(int(r["start_time_us"]) for r in rows)
    end = max(int(r["end_time_us"]) for r in rows)
    span = end - start
    if span <= 0:
        return None
    covered = sum(int(r["end_time_us"]) - int(r["start_time_us"]) for r in rows)
    return min(1.0, covered / span)


def _axis_confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = list(ELIGIBILITY_LABELS)
    conf = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        tt = t if t in conf else Eligibility.UNKNOWN.value
        pp = p if p in conf[tt] else Eligibility.UNKNOWN.value
        conf.setdefault(tt, {x: 0 for x in labels})
        conf[tt].setdefault(pp, 0)
        for row in conf.values():
            row.setdefault(pp, 0)
        conf[tt][pp] += 1

    f1s: list[float] = []
    per_f1: dict[str, float | None] = {}
    for lab in labels:
        tp = conf.get(lab, {}).get(lab, 0)
        fp = sum(conf.get(t, {}).get(lab, 0) for t in labels if t != lab)
        fn = sum(conf.get(lab, {}).get(p, 0) for p in labels if p != lab)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        if prec is None or rec is None or (prec + rec) == 0:
            per_f1[lab] = None
        else:
            per_f1[lab] = 2.0 * prec * rec / (prec + rec)
            f1s.append(per_f1[lab])  # type: ignore[arg-type]
    macro = sum(f1s) / len(f1s) if f1s else None
    return {
        "confusion": conf,
        "per_class_f1": per_f1,
        "macro_f1": macro,
        "status": "ok" if y_true else "not_evaluable",
    }


def _pair_by_id(
    preds: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    gt_map = {str(g["analysis_window_id"]): g for g in gts}
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for p in preds:
        gid = str(p["analysis_window_id"])
        if gid in gt_map:
            pairs.append((p, gt_map[gid]))
    return pairs


def _unsafe_fp_rate(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    axis: str,
    unsafe_true: set[str],
) -> float | None:
    """Rate of predicting eligible when GT says axis is unsafe (ineligible/unknown)."""
    denom = 0
    bad = 0
    for pred, gt in pairs:
        gt_val = str(gt[axis])
        if gt_val not in unsafe_true:
            continue
        denom += 1
        if str(pred[axis]) == Eligibility.ELIGIBLE.value:
            bad += 1
    return _safe_div(bad, denom)


def _non_playable_eligible_fp(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> float | None:
    axes = (
        "tracking_eligibility",
        "calibration_eligibility",
        "ball_analysis_eligibility",
        "live_event_eligibility",
        "physical_metric_eligibility",
    )
    denom = 0
    bad = 0
    for pred, gt in pairs:
        if str(gt.get("playability", pred.get("playability"))) != Playability.NON_PLAYABLE.value:
            # Also treat GT decision codes / playability on pred when GT omits.
            play = str(gt.get("playability", pred.get("playability", "")))
            if play != Playability.NON_PLAYABLE.value:
                continue
        denom += 1
        if any(str(pred[a]) == Eligibility.ELIGIBLE.value for a in axes):
            bad += 1
        if str(pred.get("playability")) == Playability.PLAYABLE.value:
            bad += 1
    return _safe_div(bad, denom)


def _manual_review_recall(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> float | None:
    """Recall of manual_review_required among GT unknown/conflict cases."""
    denom = 0
    hits = 0
    for pred, gt in pairs:
        require = bool(gt.get("manual_review_required"))
        codes = set(gt.get("decision_codes") or [])
        if not require and not codes.intersection(
            {
                "UNKNOWN_VIEW_REVIEW_REQUIRED",
                "CONFLICTING_CAMERA_LABELS",
                "CAMERA_GAP",
                "REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING",
            }
        ):
            continue
        denom += 1
        if bool(pred.get("manual_review_required")):
            hits += 1
    return _safe_div(hits, denom)


def _decision_code_accuracy(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> float | None:
    if not pairs:
        return None
    hits = 0
    for pred, gt in pairs:
        if set(pred.get("decision_codes") or []) == set(gt.get("decision_codes") or []):
            hits += 1
    return hits / len(pairs)


def evaluate_broadcast_windows(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    expected_span_us: int | None = None,
    repeat_predictions: Sequence[Mapping[str, Any]] | None = None,
    unexplained_gap_windows: int = 0,
    total_gap_windows: int | None = None,
) -> BroadcastEvaluationReport:
    """Deterministic evaluator for routed analysis windows vs reviewed GT."""
    pairs = _pair_by_id(predictions, ground_truth)
    eligibility_axes: dict[str, Mapping[str, Any]] = {}
    macro_f1: dict[str, float | None] = {}
    for axis_short in TASK_AXES:
        field = f"{axis_short}_eligibility"
        y_true = [str(gt[field]) for _, gt in pairs]
        y_pred = [str(pred[field]) for pred, _ in pairs]
        metrics = _axis_confusion(y_true, y_pred)
        eligibility_axes[axis_short] = metrics
        macro_f1[axis_short] = metrics["macro_f1"]

    unsafe = {Eligibility.INELIGIBLE.value, Eligibility.UNKNOWN.value}
    unsafe_track = _unsafe_fp_rate(pairs, axis="tracking_eligibility", unsafe_true=unsafe)
    unsafe_cal = _unsafe_fp_rate(pairs, axis="calibration_eligibility", unsafe_true=unsafe)
    unsafe_live = _unsafe_fp_rate(pairs, axis="live_event_eligibility", unsafe_true=unsafe)
    unsafe_phys = _unsafe_fp_rate(pairs, axis="physical_metric_eligibility", unsafe_true=unsafe)

    # Extra live-event safety: replay unknown GT must never predict eligible.
    live_extra_denom = 0
    live_extra_bad = 0
    for pred, gt in pairs:
        if str(gt.get("replay_status", pred.get("replay_status"))) == ReplayStatus.UNKNOWN.value:
            live_extra_denom += 1
            if str(pred["live_event_eligibility"]) == Eligibility.ELIGIBLE.value:
                live_extra_bad += 1
        if str(gt.get("replay_status")) in {
            ReplayStatus.REPLAY.value,
            ReplayStatus.REPLAY_TRANSITION.value,
        }:
            live_extra_denom += 1
            if str(pred["live_event_eligibility"]) == Eligibility.ELIGIBLE.value:
                live_extra_bad += 1
    if live_extra_denom:
        extra = live_extra_bad / live_extra_denom
        unsafe_live = extra if unsafe_live is None else max(unsafe_live, extra)

    gap_rate = None
    if total_gap_windows is not None:
        gap_rate = _safe_div(float(unexplained_gap_windows), float(max(1, total_gap_windows)))
    unexplained = _safe_div(float(unexplained_gap_windows), float(max(1, len(predictions))))

    coverage = _coverage_vs_span(predictions)
    if expected_span_us is not None and expected_span_us > 0 and predictions:
        covered = sum(int(r["end_time_us"]) - int(r["start_time_us"]) for r in predictions)
        coverage = min(1.0, covered / expected_span_us)

    det = None
    if repeat_predictions is not None:
        a = [
            (
                r.get("analysis_window_id"),
                r.get("start_time_us"),
                r.get("end_time_us"),
                tuple(r.get("decision_codes") or []),
                r.get("tracking_eligibility"),
                r.get("live_event_eligibility"),
                r.get("physical_metric_eligibility"),
                r.get("manual_review_required"),
            )
            for r in predictions
        ]
        b = [
            (
                r.get("analysis_window_id"),
                r.get("start_time_us"),
                r.get("end_time_us"),
                tuple(r.get("decision_codes") or []),
                r.get("tracking_eligibility"),
                r.get("live_event_eligibility"),
                r.get("physical_metric_eligibility"),
                r.get("manual_review_required"),
            )
            for r in repeat_predictions
        ]
        det = a == b

    status = "ok" if pairs else "not_evaluable"
    return BroadcastEvaluationReport(
        window_temporal_coverage=coverage,
        gap_rate=gap_rate,
        overlap_rate=_overlap_rate(predictions),
        unexplained_gap_rate=unexplained,
        eligibility_axes=eligibility_axes,
        macro_f1=macro_f1,
        manual_review_recall=_manual_review_recall(pairs),
        unsafe_tracking_false_positive_rate=unsafe_track,
        unsafe_calibration_false_positive_rate=unsafe_cal,
        unsafe_live_event_false_positive_rate=unsafe_live,
        unsafe_physical_metric_false_positive_rate=unsafe_phys,
        non_playable_eligible_false_positive_rate=_non_playable_eligible_fp(pairs),
        decision_code_accuracy=_decision_code_accuracy(pairs),
        deterministic_repeat=det,
        n_pred=len(predictions),
        n_gt=len(ground_truth),
        status=status,
    )


def passes_safety_gates(
    report: BroadcastEvaluationReport,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Return (ok, failures) against frozen safety thresholds."""
    th = {
        "unsafe_live_event_fp_max": 0.0,
        "unsafe_physical_metric_fp_max": 0.0,
        "unsafe_calibration_fp_max": 0.0,
        "unsafe_tracking_fp_max": 0.0,
        "non_playable_eligible_fp_max": 0.0,
        "manual_review_recall_min": 1.0,
        "overlap_rate_max": 0.0,
        "unexplained_gap_rate_max": 0.0,
    }
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items()})
    failures: list[str] = []

    def _check(
        name: str,
        value: float | None,
        *,
        maximum: float | None = None,
        minimum: float | None = None,
    ) -> None:
        if value is None:
            return
        if maximum is not None and value > maximum + 1e-12:
            failures.append(f"{name}={value} > {maximum}")
        if minimum is not None and value + 1e-12 < minimum:
            failures.append(f"{name}={value} < {minimum}")

    _check(
        "unsafe_live_event_false_positive_rate",
        report.unsafe_live_event_false_positive_rate,
        maximum=th["unsafe_live_event_fp_max"],
    )
    _check(
        "unsafe_physical_metric_false_positive_rate",
        report.unsafe_physical_metric_false_positive_rate,
        maximum=th["unsafe_physical_metric_fp_max"],
    )
    _check(
        "unsafe_calibration_false_positive_rate",
        report.unsafe_calibration_false_positive_rate,
        maximum=th["unsafe_calibration_fp_max"],
    )
    _check(
        "unsafe_tracking_false_positive_rate",
        report.unsafe_tracking_false_positive_rate,
        maximum=th["unsafe_tracking_fp_max"],
    )
    _check(
        "non_playable_eligible_false_positive_rate",
        report.non_playable_eligible_false_positive_rate,
        maximum=th["non_playable_eligible_fp_max"],
    )
    _check(
        "manual_review_recall",
        report.manual_review_recall,
        minimum=th["manual_review_recall_min"],
    )
    _check("overlap_rate", report.overlap_rate, maximum=th["overlap_rate_max"])
    _check(
        "unexplained_gap_rate",
        report.unexplained_gap_rate,
        maximum=th["unexplained_gap_rate_max"],
    )
    if report.deterministic_repeat is False:
        failures.append("deterministic_repeat=false")
    return (len(failures) == 0, failures)


__all__ = [
    "BroadcastEvaluationError",
    "BroadcastEvaluationReport",
    "evaluate_broadcast_windows",
    "passes_safety_gates",
]
