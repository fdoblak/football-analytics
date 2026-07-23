"""Camera-view classification evaluation metrics (macro P/R/F1, selective, OOD)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class CameraEvaluationError(ValueError):
    """Evaluation failure."""


SUPPORTED_EVAL_AXES = (
    "view_family",
    "framing_scale",
    "camera_motion",
    "graphics_status",
    "playability",
)

SUITABILITY_AXES = (
    "calibration_suitability",
    "tracking_suitability",
    "target_identity_suitability",
)


@dataclass(frozen=True)
class AxisMetrics:
    axis: str
    labels: tuple[str, ...]
    confusion: Mapping[str, Mapping[str, int]]
    precision: Mapping[str, float | None]
    recall: Mapping[str, float | None]
    f1: Mapping[str, float | None]
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    coverage: float | None
    abstention_rate: float | None
    selective_accuracy: float | None
    support: int
    predicted: int
    status: str  # ok | not_evaluable

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "labels": list(self.labels),
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
            "precision": dict(self.precision),
            "recall": dict(self.recall),
            "f1": dict(self.f1),
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "selective_accuracy": self.selective_accuracy,
            "support": self.support,
            "predicted": self.predicted,
            "status": self.status,
        }


@dataclass(frozen=True)
class CameraEvaluationReport:
    axes: Mapping[str, AxisMetrics]
    suitability: Mapping[str, AxisMetrics]
    unsafe_playable_false_positive_rate: float | None
    ood_abstention_rate: float | None
    n_pairs: int
    n_ood: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "suitability": {k: v.to_dict() for k, v in self.suitability.items()},
            "unsafe_playable_false_positive_rate": self.unsafe_playable_false_positive_rate,
            "ood_abstention_rate": self.ood_abstention_rate,
            "n_pairs": self.n_pairs,
            "n_ood": self.n_ood,
            "status": self.status,
        }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if p + r == 0:
        return None
    return 2.0 * p * r / (p + r)


def _is_abstain(label: str, *, axis: str) -> bool:
    if axis == "playability":
        return label in {"uncertain", "unknown"}
    return label == "unknown"


def _confusion(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    conf: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        tt = t if t in conf else "unknown"
        pp = p if p in conf[tt] else "unknown"
        if tt not in conf:
            conf[tt] = {x: 0 for x in labels}
        if pp not in conf[tt]:
            for row in conf.values():
                row.setdefault(pp, 0)
            conf[tt][pp] = 0
        conf[tt][pp] = conf[tt].get(pp, 0) + 1
    return conf


def evaluate_axis(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    axis: str,
    labels: Sequence[str] | None = None,
    unknown_labels: Sequence[str] | None = None,
) -> AxisMetrics:
    if len(y_true) != len(y_pred):
        raise CameraEvaluationError(f"{axis}: length mismatch")
    if not y_true:
        return AxisMetrics(
            axis=axis,
            labels=tuple(labels or ()),
            confusion={},
            precision={},
            recall={},
            f1={},
            macro_precision=None,
            macro_recall=None,
            macro_f1=None,
            coverage=None,
            abstention_rate=None,
            selective_accuracy=None,
            support=0,
            predicted=0,
            status="not_evaluable",
        )

    unk = set(unknown_labels or ("unknown", "uncertain"))
    label_set: list[str] = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    # Ensure unknowns present for confusion completeness when used
    for u in unk:
        if u not in label_set and any(x == u for x in list(y_true) + list(y_pred)):
            label_set.append(u)

    conf = _confusion(y_true, y_pred, label_set)
    precision: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}
    f1s: dict[str, float | None] = {}

    # Macro over supported non-unknown labels that appear in GT
    macro_labels = [lb for lb in label_set if lb not in unk]
    p_list: list[float] = []
    r_list: list[float] = []
    f_list: list[float] = []
    for lb in label_set:
        tp = conf.get(lb, {}).get(lb, 0)
        fp = sum(conf.get(t, {}).get(lb, 0) for t in label_set if t != lb)
        fn = sum(v for p, v in conf.get(lb, {}).items() if p != lb)
        pr = _safe_div(float(tp), float(tp + fp))
        rc = _safe_div(float(tp), float(tp + fn))
        ff = _f1(pr, rc)
        precision[lb] = pr
        recall[lb] = rc
        f1s[lb] = ff
        if lb in macro_labels and (tp + fn) > 0:
            if pr is not None:
                p_list.append(pr)
            if rc is not None:
                r_list.append(rc)
            if ff is not None:
                f_list.append(ff)

    macro_p = (sum(p_list) / len(p_list)) if p_list else None
    macro_r = (sum(r_list) / len(r_list)) if r_list else None
    macro_f = (sum(f_list) / len(f_list)) if f_list else None
    if not macro_labels or (sum(1 for t in y_true if t not in unk) == 0):
        status = "not_evaluable"
        macro_p = macro_r = macro_f = None
    else:
        status = "ok"

    abstain_pred = sum(1 for p in y_pred if _is_abstain(p, axis=axis))
    abstention_rate = abstain_pred / len(y_pred)
    # Coverage: fraction of GT non-unknown that received non-abstain prediction
    gt_known_idx = [i for i, t in enumerate(y_true) if t not in unk]
    if gt_known_idx:
        covered = sum(1 for i in gt_known_idx if not _is_abstain(y_pred[i], axis=axis))
        coverage = covered / len(gt_known_idx)
    else:
        coverage = None

    # Selective accuracy: among non-abstain predictions, accuracy vs GT
    sel_idx = [i for i, p in enumerate(y_pred) if not _is_abstain(p, axis=axis)]
    if sel_idx:
        selective = sum(1 for i in sel_idx if y_pred[i] == y_true[i]) / len(sel_idx)
    else:
        selective = None

    return AxisMetrics(
        axis=axis,
        labels=tuple(label_set),
        confusion=conf,
        precision=precision,
        recall=recall,
        f1=f1s,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        coverage=coverage,
        abstention_rate=abstention_rate,
        selective_accuracy=selective,
        support=len(y_true),
        predicted=len(y_pred),
        status=status,
    )


def unsafe_playable_fp_rate(
    y_true_playability: Sequence[str], y_pred_playability: Sequence[str]
) -> float | None:
    """Rate of GT non_playable predicted as playable (must be 0)."""
    if len(y_true_playability) != len(y_pred_playability):
        raise CameraEvaluationError("playability length mismatch")
    denom = sum(1 for t in y_true_playability if t == "non_playable")
    if denom == 0:
        return None
    bad = sum(
        1
        for t, p in zip(y_true_playability, y_pred_playability, strict=True)
        if t == "non_playable" and p == "playable"
    )
    return bad / denom


def ood_abstention_rate(
    predictions: Sequence[Mapping[str, Any]],
    *,
    ood_ids: Sequence[str] | None = None,
    id_key: str = "fixture_id",
) -> float | None:
    """Fraction of OOD fixtures where primary axes abstain (view unknown)."""
    if not predictions:
        return None
    if ood_ids is not None:
        ood_set = set(ood_ids)
        rows = [r for r in predictions if str(r.get(id_key, "")) in ood_set]
    else:
        rows = [r for r in predictions if bool(r.get("is_ood"))]
    if not rows:
        return None
    abstain = sum(1 for r in rows if str(r.get("view_family", "")) == "unknown")
    return abstain / len(rows)


def evaluate_camera_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    supported_labels: Mapping[str, Sequence[str]] | None = None,
    ood_fixture_ids: Sequence[str] | None = None,
) -> CameraEvaluationReport:
    """Evaluate multi-axis camera classifications.

    Rows are paired by (video_id, shot_id) when present, else by fixture_id/name.
    """
    # Build flexible pairing
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    if predictions and ground_truth:
        # Index GT by multiple keys
        by_shot: dict[tuple[Any, Any], Mapping[str, Any]] = {}
        by_fixture: dict[str, Mapping[str, Any]] = {}
        for g in ground_truth:
            if g.get("video_id") is not None and g.get("shot_id") is not None:
                by_shot[(g["video_id"], g["shot_id"])] = g
            fid = g.get("fixture_id") or g.get("name") or g.get("shot_id")
            if fid is not None:
                by_fixture[str(fid)] = g
        used_gt: set[int] = set()
        for p in predictions:
            matched: Mapping[str, Any] | None = None
            if p.get("video_id") is not None and p.get("shot_id") is not None:
                matched = by_shot.get((p["video_id"], p["shot_id"]))
            if matched is None:
                fid = p.get("fixture_id") or p.get("name") or p.get("shot_id")
                if fid is not None:
                    matched = by_fixture.get(str(fid))
            if matched is None:
                continue
            gid = id(matched)
            if gid in used_gt:
                continue
            used_gt.add(gid)
            pairs.append((p, matched))

    axes_out: dict[str, AxisMetrics] = {}
    for axis in SUPPORTED_EVAL_AXES:
        y_t = [str(g.get(axis, "unknown")) for _, g in pairs]
        y_p = [str(p.get(axis, "unknown")) for p, _ in pairs]
        labels = None
        if supported_labels and axis in supported_labels:
            labels = list(supported_labels[axis])
        axes_out[axis] = evaluate_axis(y_t, y_p, axis=axis, labels=labels)

    suit_out: dict[str, AxisMetrics] = {}
    for axis in SUITABILITY_AXES:
        y_t = [str(g.get(axis, "unknown")) for _, g in pairs]
        y_p = [str(p.get(axis, "unknown")) for p, _ in pairs]
        suit_out[axis] = evaluate_axis(
            y_t,
            y_p,
            axis=axis,
            labels=["suitable", "conditionally_suitable", "unsuitable", "unknown"],
        )

    play_t = [str(g.get("playability", "uncertain")) for _, g in pairs]
    play_p = [str(p.get("playability", "uncertain")) for p, _ in pairs]
    unsafe = unsafe_playable_fp_rate(play_t, play_p)

    # OOD abstention from GT-marked or id list
    ood_pairs = []
    for p, g in pairs:
        fid = str(g.get("fixture_id") or g.get("name") or p.get("fixture_id") or "")
        is_ood = bool(g.get("is_ood")) or (
            ood_fixture_ids is not None and fid in set(ood_fixture_ids)
        )
        if is_ood:
            ood_pairs.append(p)
    if ood_pairs:
        ood_rate = sum(1 for r in ood_pairs if str(r.get("view_family", "")) == "unknown") / len(
            ood_pairs
        )
    else:
        ood_rate = None

    status = "ok"
    if not pairs:
        status = "not_evaluable"

    return CameraEvaluationReport(
        axes=axes_out,
        suitability=suit_out,
        unsafe_playable_false_positive_rate=unsafe,
        ood_abstention_rate=ood_rate,
        n_pairs=len(pairs),
        n_ood=len(ood_pairs),
        status=status,
    )


def combined_view_framing_macro_f1(report: CameraEvaluationReport) -> float | None:
    """Mean of view_family and framing_scale macro F1 (supported classes)."""
    vals: list[float] = []
    for axis in ("view_family", "framing_scale"):
        m = report.axes.get(axis)
        if m is None or m.macro_f1 is None or m.status != "ok":
            return None
        vals.append(m.macro_f1)
    if not vals:
        return None
    return sum(vals) / len(vals)


__all__ = [
    "CameraEvaluationError",
    "AxisMetrics",
    "CameraEvaluationReport",
    "SUPPORTED_EVAL_AXES",
    "evaluate_axis",
    "unsafe_playable_fp_rate",
    "ood_abstention_rate",
    "evaluate_camera_predictions",
    "combined_view_framing_macro_f1",
]
