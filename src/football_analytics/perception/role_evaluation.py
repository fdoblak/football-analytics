"""Human role evaluation (Stage 5D). No invented accuracy without reviewed GT."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NOT_EVALUATED_ROLE = "NOT_EVALUATED_NO_REVIEWED_HUMAN_ROLE_GROUND_TRUTH"
ROLE_LABELS = (
    "player",
    "goalkeeper",
    "referee",
    "assistant_referee",
    "staff",
    "unknown",
)


class RoleEvaluationError(ValueError):
    """Role evaluation failure."""


@dataclass(frozen=True)
class RoleEvalMetrics:
    status: str
    confusion_matrix: Mapping[str, Mapping[str, int]] | None
    per_role_precision: Mapping[str, float | None] | None
    per_role_recall: Mapping[str, float | None] | None
    per_role_f1: Mapping[str, float | None] | None
    macro_f1: float | None
    coverage: float | None
    abstention_rate: float | None
    selective_accuracy: float | None
    unsafe_known_role_false_positive: float | None
    unknown_review_recall: float | None
    predicted_count: int
    ground_truth_count: int
    matched_count: int
    notes: tuple[str, ...]
    synthetic_fixture_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confusion_matrix": (
                None
                if self.confusion_matrix is None
                else {k: dict(v) for k, v in self.confusion_matrix.items()}
            ),
            "per_role_precision": (
                None if self.per_role_precision is None else dict(self.per_role_precision)
            ),
            "per_role_recall": None if self.per_role_recall is None else dict(self.per_role_recall),
            "per_role_f1": None if self.per_role_f1 is None else dict(self.per_role_f1),
            "macro_f1": self.macro_f1,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "selective_accuracy": self.selective_accuracy,
            "unsafe_known_role_false_positive": self.unsafe_known_role_false_positive,
            "unknown_review_recall": self.unknown_review_recall,
            "predicted_count": self.predicted_count,
            "ground_truth_count": self.ground_truth_count,
            "matched_count": self.matched_count,
            "notes": list(self.notes),
            "synthetic_fixture_only": self.synthetic_fixture_only,
            "other_maps_to": "staff",
            "team_id": None,
            "real_football_accuracy_claimed": False,
        }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def not_evaluated_role_metrics(*, reason: str = NOT_EVALUATED_ROLE) -> RoleEvalMetrics:
    return RoleEvalMetrics(
        status=reason,
        confusion_matrix=None,
        per_role_precision=None,
        per_role_recall=None,
        per_role_f1=None,
        macro_f1=None,
        coverage=None,
        abstention_rate=None,
        selective_accuracy=None,
        unsafe_known_role_false_positive=None,
        unknown_review_recall=None,
        predicted_count=0,
        ground_truth_count=0,
        matched_count=0,
        notes=(reason, "Synthetic fixtures are not real football performance."),
        synthetic_fixture_only=False,
    )


def _normalize_role(label: Any) -> str:
    text = str(label).strip().lower()
    if text == "other":
        return "staff"
    if text not in ROLE_LABELS:
        raise RoleEvaluationError(f"unsupported role_label: {label}")
    return text


def ground_truth_roles_reviewed(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if row.get("is_reviewed_ground_truth") is True:
            return True
        rs = str(row.get("review_status", "")).lower()
        if rs in {"reviewed", "accepted"}:
            return True
    return False


def evaluate_role_assignments(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    synthetic_fixture_only: bool = False,
) -> RoleEvalMetrics:
    """Evaluate role labels joined on detection_id (and frame_index when present)."""
    if not ground_truth:
        return not_evaluated_role_metrics()

    gt_by_key: dict[tuple[int, int], str] = {}
    for g in ground_truth:
        fi = int(g.get("frame_index", 0))
        did = int(g["detection_id"])
        gt_by_key[(fi, did)] = _normalize_role(g.get("role_label", "unknown"))

    abstained = 0
    for p in predictions:
        status = str(p.get("assignment_status", "classified")).lower()
        if status == "abstained" or _normalize_role(p.get("role_label", "unknown")) == "unknown":
            abstained += 1

    cm: dict[str, dict[str, int]] = {r: {c: 0 for c in ROLE_LABELS} for r in ROLE_LABELS}
    matched = 0
    correct_selective = 0
    selective_total = 0
    unsafe_fp = 0
    known_pred = 0
    unknown_gt_hit = 0
    unknown_gt = 0

    for p in predictions:
        fi = int(p.get("frame_index", 0))
        did = int(p["detection_id"])
        key = (fi, did)
        if key not in gt_by_key:
            continue
        pred = _normalize_role(p.get("role_label", "unknown"))
        gt = gt_by_key[key]
        cm[gt][pred] += 1
        matched += 1
        if gt == "unknown":
            unknown_gt += 1
            if pred == "unknown":
                unknown_gt_hit += 1
        if pred != "unknown":
            known_pred += 1
            selective_total += 1
            if pred == gt:
                correct_selective += 1
            if (
                gt in {"player", "goalkeeper", "referee", "assistant_referee", "staff"}
                and pred != gt
                and pred in {"player", "goalkeeper", "referee", "assistant_referee"}
            ):
                unsafe_fp += 1

    per_p: dict[str, float | None] = {}
    per_r: dict[str, float | None] = {}
    per_f: dict[str, float | None] = {}
    f1s: list[float] = []
    for role in ROLE_LABELS:
        tp = cm[role][role]
        fp = sum(cm[g][role] for g in ROLE_LABELS if g != role)
        fn = sum(cm[role][c] for c in ROLE_LABELS if c != role)
        prec = _safe_div(float(tp), float(tp + fp))
        rec = _safe_div(float(tp), float(tp + fn))
        if prec is None or rec is None or (prec + rec) == 0:
            f1 = None
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        per_p[role] = prec
        per_r[role] = rec
        per_f[role] = f1
        if f1 is not None:
            f1s.append(f1)

    macro = (sum(f1s) / len(f1s)) if f1s else None
    coverage = _safe_div(float(matched), float(len(gt_by_key))) if gt_by_key else None
    abst_rate = _safe_div(float(abstained), float(len(predictions))) if predictions else None
    sel_acc = _safe_div(float(correct_selective), float(selective_total))
    unsafe = _safe_div(float(unsafe_fp), float(known_pred)) if known_pred else None
    unk_rec = _safe_div(float(unknown_gt_hit), float(unknown_gt))

    notes: list[str] = []
    if synthetic_fixture_only:
        notes.append("SYNTHETIC_FIXTURE_ONLY — not real football accuracy")
    return RoleEvalMetrics(
        status="EVALUATED",
        confusion_matrix=cm,
        per_role_precision=per_p,
        per_role_recall=per_r,
        per_role_f1=per_f,
        macro_f1=macro,
        coverage=coverage,
        abstention_rate=abst_rate,
        selective_accuracy=sel_acc,
        unsafe_known_role_false_positive=unsafe,
        unknown_review_recall=unk_rec,
        predicted_count=len(predictions),
        ground_truth_count=len(gt_by_key),
        matched_count=matched,
        notes=tuple(notes),
        synthetic_fixture_only=synthetic_fixture_only,
    )


def evaluate_roles_from_rows(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]] | None,
    *,
    require_reviewed_gt: bool = True,
    synthetic_fixture_only: bool = False,
) -> RoleEvalMetrics:
    if ground_truth is None or not ground_truth:
        return not_evaluated_role_metrics()
    if require_reviewed_gt and not ground_truth_roles_reviewed(ground_truth):
        return not_evaluated_role_metrics()
    if require_reviewed_gt and not synthetic_fixture_only:
        reviewed = [g for g in ground_truth if g.get("is_reviewed_ground_truth") is True]
        if not reviewed:
            return not_evaluated_role_metrics()
        return evaluate_role_assignments(predictions, reviewed, synthetic_fixture_only=False)
    return evaluate_role_assignments(
        predictions, ground_truth, synthetic_fixture_only=synthetic_fixture_only
    )


def count_roles(assignments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for a in assignments:
        c[_normalize_role(a.get("role_label", "unknown"))] += 1
    return {k: int(c.get(k, 0)) for k in ROLE_LABELS}


__all__ = [
    "NOT_EVALUATED_ROLE",
    "ROLE_LABELS",
    "RoleEvaluationError",
    "RoleEvalMetrics",
    "not_evaluated_role_metrics",
    "ground_truth_roles_reviewed",
    "evaluate_role_assignments",
    "evaluate_roles_from_rows",
    "count_roles",
]
