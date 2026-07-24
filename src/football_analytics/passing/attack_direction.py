"""Attack direction resolver stub (Stage 11C).

Manual/config evidence only. Conflict → unknown. Never invent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.passing.receipt import build_attack_direction_evidence
from football_analytics.passing.types import AttackDirection


def resolve_attack_direction(
    *,
    run_id: str,
    video_id: str,
    config_direction: str | None = None,
    manual_direction: str | None = None,
    evidence_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve attack direction from config and/or manual evidence only."""
    allowed = {
        AttackDirection.TOWARD_GOAL_A.value,
        AttackDirection.TOWARD_GOAL_B.value,
        AttackDirection.UNKNOWN.value,
    }
    cfg = str(config_direction) if config_direction else None
    man = str(manual_direction) if manual_direction else None
    if cfg is not None and cfg not in allowed:
        cfg = AttackDirection.UNKNOWN.value
    if man is not None and man not in allowed:
        man = AttackDirection.UNKNOWN.value

    conflict = False
    source = "none"
    direction = AttackDirection.UNKNOWN.value

    if cfg and man and cfg != man and AttackDirection.UNKNOWN.value not in {cfg, man}:
        conflict = True
        source = "conflict"
        direction = AttackDirection.UNKNOWN.value
    elif man and man != AttackDirection.UNKNOWN.value:
        source = "manual"
        direction = man
    elif cfg and cfg != AttackDirection.UNKNOWN.value:
        source = "config"
        direction = cfg
    elif man == AttackDirection.UNKNOWN.value or cfg == AttackDirection.UNKNOWN.value:
        source = "manual" if man is not None else "config"
        direction = AttackDirection.UNKNOWN.value

    return build_attack_direction_evidence(
        run_id=run_id,
        video_id=video_id,
        attack_direction=direction,
        evidence_source=source if not conflict else "conflict",
        evidence_refs=list(evidence_refs or []),
        conflict=conflict,
        notes="resolver_stub_never_invents",
    )


def attack_relative_evaluable(evidence: Mapping[str, Any]) -> bool:
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
