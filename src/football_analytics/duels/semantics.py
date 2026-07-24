"""Duels / competitive-events semantic helpers (Stage 12A — no real inference)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.duels.types import RESULT_LEVELS, DuelsContractError


def assert_finite_optional(value: float | None, *, label: str) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise DuelsContractError(f"NAN_INF_REJECTED:{label}")


def assert_half_open_interval(start_us: int, end_us: int) -> None:
    if int(end_us) < int(start_us):
        raise DuelsContractError("NEGATIVE_DURATION")
    if int(start_us) < 0 or int(end_us) < 0:
        raise DuelsContractError("NEGATIVE_DURATION")


def assert_result_level(level: str) -> str:
    if level not in RESULT_LEVELS:
        raise DuelsContractError(f"unknown result level: {level}")
    return level


def nearby_opponent_alone_is_take_on(*, nearby_opponent_alone: bool) -> bool:
    """Nearby opponent alone never implies a take-on."""
    _ = nearby_opponent_alone
    return False


def nearest_switch_alone_is_duel_outcome(*, nearest_switch_alone: bool) -> bool:
    """Nearest / track switch alone never implies a duel outcome."""
    _ = nearest_switch_alone
    return False


def monocular_aerial_allows_exact_height(*, monocular_only: bool) -> bool:
    """Monocular aerial never supports an exact 3D height claim."""
    _ = monocular_only
    return False


def monocular_aerial_evaluability(*, monocular_only: bool) -> str:
    if monocular_only:
        return "not_evaluable"
    return "provisional"


def long_ball_alone_is_clearance(*, long_ball_alone: bool) -> bool:
    """Long ball alone never implies a clearance."""
    _ = long_ball_alone
    return False


def cut_replay_gap_allows_event(*, cut_or_replay: bool, hard_gap: bool) -> bool:
    return not (cut_or_replay or hard_gap)


def automatic_confirmed_allowed() -> bool:
    return False


def assert_no_duplicate_pk(rows: Sequence[Mapping[str, Any]], pk_fields: Sequence[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[f] for f in pk_fields)
        if key in seen:
            raise DuelsContractError("DUPLICATE_PK")
        seen.add(key)


def assert_scope_match(rows: Sequence[Mapping[str, Any]], *, run_id: str, video_id: str) -> None:
    for row in rows:
        if str(row.get("run_id")) != run_id or str(row.get("video_id")) != video_id:
            raise DuelsContractError("CROSS_SCOPE_FK")


def opta_accuracy_claim_forbidden(metrics: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if metrics.get("opta_accuracy_validated") is True:
        errors.append("OPTA_CLAIM_FORBIDDEN")
    if metrics.get("real_football_accuracy_validated") is True:
        errors.append("OPTA_CLAIM_FORBIDDEN:real_football")
    return errors


def neutral_zone_from_x(*, x_m: float | None, pitch_length_m: float = 105.0) -> str:
    if x_m is None or not math.isfinite(float(x_m)):
        return "unknown"
    third = float(pitch_length_m) / 3.0
    x = float(x_m)
    if x < third:
        return "goal_a"
    if x < 2.0 * third:
        return "middle"
    return "goal_b"


__all__ = [
    "assert_finite_optional",
    "assert_half_open_interval",
    "assert_result_level",
    "nearby_opponent_alone_is_take_on",
    "nearest_switch_alone_is_duel_outcome",
    "monocular_aerial_allows_exact_height",
    "monocular_aerial_evaluability",
    "long_ball_alone_is_clearance",
    "cut_replay_gap_allows_event",
    "automatic_confirmed_allowed",
    "assert_no_duplicate_pk",
    "assert_scope_match",
    "opta_accuracy_claim_forbidden",
    "neutral_zone_from_x",
]
