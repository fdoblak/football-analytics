"""Stage 17-R2 own-video recovery helpers: role/team gates and metrics stubs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ALLOWED_ROLES = {"player", "goalkeeper", "referee", "staff", "unknown"}
PLAYER_TEAMS = {"yellow", "white", "unknown"}
ROLE_EVAL_LABELS = ("player", "goalkeeper", "referee", "staff")


def normalize_team_for_role(role: str, team: str | None) -> str | None:
    """Referee/staff/unknown must never receive yellow/white team labels."""
    if role in {"referee", "staff"}:
        return None
    if role == "unknown":
        if team in {"yellow", "white"}:
            return "unknown"
        return team
    if role in {"player", "goalkeeper"}:
        if team in PLAYER_TEAMS:
            return team
        return "unknown"
    return None


def count_non_player_team_assignments(humans: list[dict[str, Any]]) -> int:
    n = 0
    for h in humans:
        role = str(h.get("role", "unknown"))
        team = h.get("team")
        if role in {"referee", "staff"} and team in {"yellow", "white"}:
            n += 1
    return n


def role_confusion_referee_as_player(humans_gt: list[dict], humans_pred: list[dict]) -> int:
    """Count GT referees matched to pred player (IoU>0.3)."""
    from football_analytics.acceptance.final_perception_repair.pipeline import _iou

    bad = 0
    for g in humans_gt:
        if g.get("role") != "referee":
            continue
        gbox = (float(g["bbox"][0]), float(g["bbox"][1]), float(g["bbox"][2]), float(g["bbox"][3]))
        for p in humans_pred:
            pbox = (
                float(p["bbox"][0]),
                float(p["bbox"][1]),
                float(p["bbox"][2]),
                float(p["bbox"][3]),
            )
            if _iou(gbox, pbox) >= 0.3 and p.get("role") == "player":
                bad += 1
                break
    return bad


def role_macro_f1(
    pairs: list[tuple[str, str]], labels: tuple[str, ...] = ROLE_EVAL_LABELS
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Macro-F1 over role labels present in GT or predictions; skip unknown."""
    confusion: dict[str, int] = defaultdict(int)
    tp = {lab: 0 for lab in labels}
    fp = {lab: 0 for lab in labels}
    fn = {lab: 0 for lab in labels}
    for gt, pred in pairs:
        g = gt if gt in labels else "unknown"
        p = pred if pred in labels else "unknown"
        confusion[f"{g}->{p}"] += 1
        if g == "unknown" and p == "unknown":
            continue
        if g == p and g in labels:
            tp[g] += 1
        else:
            if p in labels:
                fp[p] += 1
            if g in labels:
                fn[g] += 1
    per: dict[str, float] = {}
    scores: list[float] = []
    for lab in labels:
        prec = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rec = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        # Only average labels with GT support
        if tp[lab] + fn[lab] > 0:
            per[lab] = f1
            scores.append(f1)
        else:
            per[lab] = float("nan")
    macro = float(sum(scores) / len(scores)) if scores else 0.0
    return macro, per, dict(confusion)
