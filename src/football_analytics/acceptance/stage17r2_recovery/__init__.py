"""Stage 17-R2 own-video recovery helpers: role/team gates and metrics stubs."""

from __future__ import annotations

from typing import Any

ALLOWED_ROLES = {"player", "goalkeeper", "referee", "staff", "unknown"}
PLAYER_TEAMS = {"yellow", "white", "unknown"}


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
