"""Explainable period/half-scoped attack direction resolver (Stage 13B).

Extends Stage 11 stub with period scope, anonymous team, manual override,
conflict → unknown, and half-boundary direction change. Never invents team names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.passing.types import AttackDirection

ALLOWED = {
    AttackDirection.TOWARD_GOAL_A.value,
    AttackDirection.TOWARD_GOAL_B.value,
    AttackDirection.UNKNOWN.value,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize_direction(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value)
    return v if v in ALLOWED else AttackDirection.UNKNOWN.value


def _flip(direction: str) -> str:
    if direction == AttackDirection.TOWARD_GOAL_A.value:
        return AttackDirection.TOWARD_GOAL_B.value
    if direction == AttackDirection.TOWARD_GOAL_B.value:
        return AttackDirection.TOWARD_GOAL_A.value
    return AttackDirection.UNKNOWN.value


def resolve_period_attack_direction(
    *,
    run_id: str,
    video_id: str,
    period_id: str = "period_1",
    half_id: str = "first_half",
    anonymous_team_id: str | None = None,
    config_direction: str | None = None,
    manual_direction: str | None = None,
    prior_half_direction: str | None = None,
    apply_half_boundary_flip: bool = False,
    evidence_refs: Sequence[str] | None = None,
    team_display_name: str | None = None,
) -> dict[str, Any]:
    """Resolve attack direction for one period/half scope.

    Real team display names are rejected (never invented / never stored as truth).
    """
    if team_display_name:
        # Explicit guard: callers must not pass real team names into the resolver.
        return {
            "schema_version": 1,
            "run_id": run_id,
            "video_id": video_id,
            "period_id": period_id,
            "half_id": half_id,
            "anonymous_team_id": anonymous_team_id,
            "attack_direction": AttackDirection.UNKNOWN.value,
            "evidence_source": "conflict",
            "evidence_refs": list(evidence_refs or []),
            "conflict": True,
            "invented": False,
            "half_boundary_flipped": False,
            "manual_override": False,
            "notes": "team_display_name_forbidden",
            "created_at_utc": _utc_now(),
        }

    cfg = _normalize_direction(config_direction)
    man = _normalize_direction(manual_direction)
    conflict = False
    source = "none"
    direction = AttackDirection.UNKNOWN.value
    manual_override = False

    if cfg and man and cfg != man and AttackDirection.UNKNOWN.value not in {cfg, man}:
        conflict = True
        source = "conflict"
        direction = AttackDirection.UNKNOWN.value
    elif man and man != AttackDirection.UNKNOWN.value:
        source = "manual"
        direction = man
        manual_override = True
    elif cfg and cfg != AttackDirection.UNKNOWN.value:
        source = "config"
        direction = cfg
    elif man == AttackDirection.UNKNOWN.value or cfg == AttackDirection.UNKNOWN.value:
        source = "manual" if man is not None else "config"
        direction = AttackDirection.UNKNOWN.value

    half_flipped = False
    if (
        apply_half_boundary_flip
        and not conflict
        and not manual_override
        and prior_half_direction
        and prior_half_direction
        in {
            AttackDirection.TOWARD_GOAL_A.value,
            AttackDirection.TOWARD_GOAL_B.value,
        }
        and direction == AttackDirection.UNKNOWN.value
    ):
        direction = _flip(str(prior_half_direction))
        source = "half_boundary"
        half_flipped = True

    # If config/manual resolved AND half boundary flip requested with prior, flip
    # only when config explicitly asks for boundary change without conflicting manual.
    if (
        apply_half_boundary_flip
        and not conflict
        and not manual_override
        and prior_half_direction
        and direction in {AttackDirection.TOWARD_GOAL_A.value, AttackDirection.TOWARD_GOAL_B.value}
        and source == "config"
        and direction == prior_half_direction
    ):
        direction = _flip(direction)
        source = "half_boundary"
        half_flipped = True

    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "period_id": period_id,
        "half_id": half_id,
        "anonymous_team_id": anonymous_team_id,
        "attack_direction": direction,
        "evidence_source": source if not conflict else "conflict",
        "evidence_refs": list(evidence_refs or []),
        "conflict": conflict,
        "invented": False,
        "half_boundary_flipped": half_flipped,
        "manual_override": manual_override,
        "notes": "period_scoped_resolver_never_invents",
        "created_at_utc": _utc_now(),
    }


def resolve_match_attack_directions(
    *,
    run_id: str,
    video_id: str,
    periods: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve a sequence of period/half scopes with optional boundary flips."""
    out: list[dict[str, Any]] = []
    prior: str | None = None
    for i, p in enumerate(periods):
        apply_flip = bool(p.get("apply_half_boundary_flip", i > 0))
        ev = resolve_period_attack_direction(
            run_id=run_id,
            video_id=video_id,
            period_id=str(p.get("period_id", f"period_{i+1}")),
            half_id=str(p.get("half_id", "first_half" if i == 0 else "second_half")),
            anonymous_team_id=p.get("anonymous_team_id"),
            config_direction=p.get("config_direction"),
            manual_direction=p.get("manual_direction"),
            prior_half_direction=prior,
            apply_half_boundary_flip=apply_flip,
            evidence_refs=list(p.get("evidence_refs") or []),
            team_display_name=p.get("team_display_name"),
        )
        out.append(ev)
        d = str(ev["attack_direction"])
        if d in {AttackDirection.TOWARD_GOAL_A.value, AttackDirection.TOWARD_GOAL_B.value}:
            prior = d
        else:
            prior = None
    return out


def attack_relative_evaluable(evidence: Mapping[str, Any]) -> bool:
    return (
        str(evidence.get("attack_direction"))
        in {AttackDirection.TOWARD_GOAL_A.value, AttackDirection.TOWARD_GOAL_B.value}
        and evidence.get("invented") is False
        and evidence.get("conflict") is not True
    )


__all__ = [
    "resolve_period_attack_direction",
    "resolve_match_attack_directions",
    "attack_relative_evaluable",
]
