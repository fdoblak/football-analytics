"""Attack direction resolver (Stage 11C stub + Stage 13B-compatible).

Manual/config evidence only. Conflict → unknown. Never invent.
Period/half-scoped API: football_analytics.events.attack_direction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.events.attack_direction import (
    attack_relative_evaluable as _period_evaluable,
)
from football_analytics.events.attack_direction import (
    resolve_period_attack_direction,
)
from football_analytics.passing.receipt import build_attack_direction_evidence
from football_analytics.passing.types import AttackDirection


def resolve_attack_direction(
    *,
    run_id: str,
    video_id: str,
    config_direction: str | None = None,
    manual_direction: str | None = None,
    evidence_refs: Sequence[str] | None = None,
    period_id: str = "period_1",
    half_id: str = "first_half",
    anonymous_team_id: str | None = None,
    prior_half_direction: str | None = None,
    apply_half_boundary_flip: bool = False,
) -> dict[str, Any]:
    """Resolve attack direction from config and/or manual evidence only."""
    period_ev = resolve_period_attack_direction(
        run_id=run_id,
        video_id=video_id,
        period_id=period_id,
        half_id=half_id,
        anonymous_team_id=anonymous_team_id,
        config_direction=config_direction,
        manual_direction=manual_direction,
        prior_half_direction=prior_half_direction,
        apply_half_boundary_flip=apply_half_boundary_flip,
        evidence_refs=evidence_refs,
    )
    # Preserve Stage 11 JSON schema shape (additionalProperties=false).
    return build_attack_direction_evidence(
        run_id=run_id,
        video_id=video_id,
        attack_direction=str(period_ev["attack_direction"]),
        evidence_source=str(period_ev["evidence_source"]),
        evidence_refs=list(period_ev.get("evidence_refs") or []),
        conflict=bool(period_ev.get("conflict")),
        notes=str(period_ev.get("notes") or "resolver_never_invents"),
    )


def attack_relative_evaluable(evidence: Mapping[str, Any]) -> bool:
    if "period_id" in evidence:
        return _period_evaluable(evidence)
    return (
        str(evidence.get("attack_direction"))
        in {AttackDirection.TOWARD_GOAL_A.value, AttackDirection.TOWARD_GOAL_B.value}
        and evidence.get("invented") is False
        and evidence.get("conflict") is not True
    )


__all__ = [
    "resolve_attack_direction",
    "attack_relative_evaluable",
]
