"""Neutral geometric zone helpers (Stage 9A — no attack-relative progression)."""

from __future__ import annotations

from football_analytics.physical.types import NeutralZone, PhysicalContractError

ATTACK_RELATIVE_FORBIDDEN = frozenset(
    {
        "first_third",
        "middle_third_attack",
        "final_third",
        "attacking_third",
        "defensive_third",
    }
)


def assert_zone_name_allowed(name: str) -> None:
    if name in ATTACK_RELATIVE_FORBIDDEN or name.startswith("attack_"):
        raise PhysicalContractError("ATTACK_RELATIVE_FORBIDDEN")
    allowed = {z.value for z in NeutralZone}
    if name not in allowed:
        raise PhysicalContractError(f"unknown or forbidden zone: {name}")


def neutral_third_for_x(x_m: float, *, length_m: float) -> str:
    """Map absolute pitch x to Goal A / middle / Goal B thirds (not attack-relative)."""
    if length_m <= 0:
        raise PhysicalContractError("invalid pitch length")
    third = length_m / 3.0
    if x_m < third:
        return NeutralZone.GOAL_A_THIRD.value
    if x_m < 2.0 * third:
        return NeutralZone.MIDDLE_THIRD.value
    return NeutralZone.GOAL_B_THIRD.value


def progression_enabled(*, attack_direction: str, policy_enabled: bool) -> bool:
    if attack_direction == "unknown":
        return False
    return bool(policy_enabled)


__all__ = [
    "ATTACK_RELATIVE_FORBIDDEN",
    "assert_zone_name_allowed",
    "neutral_third_for_x",
    "progression_enabled",
]
