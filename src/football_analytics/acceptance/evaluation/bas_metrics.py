"""BAS event evaluation against external reference GT."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional


def _key(label: str, half: int, t_ms: int) -> tuple[str, int, int]:
    return (label, half, t_ms)


def match_events(
    *,
    predicted: Iterable[dict[str, Any]],
    reference: Iterable[dict[str, Any]],
    tolerance_ms: int = 1000,
    label_whitelist: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Greedy temporal matching per label within tolerance_ms."""
    refs = [dict(r) for r in reference]
    preds = [dict(p) for p in predicted]
    if label_whitelist:
        refs = [r for r in refs if r.get("label") in label_whitelist]
        preds = [p for p in preds if p.get("label") in label_whitelist]

    by_label_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in refs:
        by_label_ref[str(r.get("label"))].append(r)
    by_label_pred: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        by_label_pred[str(p.get("label"))].append(p)

    per_label: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    timing_errors: list[int] = []
    attribution_ok = 0
    attribution_n = 0

    labels = sorted(set(by_label_ref) | set(by_label_pred))
    for label in labels:
        rlist = sorted(by_label_ref.get(label, []), key=lambda x: (int(x.get("half") or 0), int(x.get("t_ms") or 0)))
        plist = sorted(by_label_pred.get(label, []), key=lambda x: (int(x.get("half") or 0), int(x.get("t_ms") or 0)))
        used = [False] * len(rlist)
        tp = fp = 0
        for p in plist:
            ph = int(p.get("half") or 0)
            pt = int(p.get("t_ms") or 0)
            best_i = None
            best_dt = None
            for i, r in enumerate(rlist):
                if used[i]:
                    continue
                if int(r.get("half") or 0) != ph:
                    continue
                dt = abs(int(r.get("t_ms") or 0) - pt)
                if dt <= tolerance_ms and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_i = i
            if best_i is None:
                fp += 1
                continue
            used[best_i] = True
            tp += 1
            timing_errors.append(int(best_dt or 0))
            ref_pid = rlist[best_i].get("player_id")
            pred_pid = p.get("player_id")
            if ref_pid is not None and pred_pid is not None:
                attribution_n += 1
                if str(ref_pid) == str(pred_pid):
                    attribution_ok += 1
        fn = sum(1 for u in used if not u)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    mean_timing = sum(timing_errors) / len(timing_errors) if timing_errors else None
    attr = attribution_ok / attribution_n if attribution_n else None
    return {
        "tolerance_ms": tolerance_ms,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None,
        "mean_abs_timing_error_ms": mean_timing,
        "target_attribution_accuracy": attr,
        "per_label": per_label,
        "not_evaluable_outcomes": [
            "pass_accuracy",
            "duel_win_rate",
            "clearance_outcome",
            "recovery_turnover_outcome",
        ],
    }
