"""Passing / reception / progression semantic helpers (Stage 11A — no real inference)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.passing.types import RESULT_LEVELS, PassingContractError


def assert_finite_optional(value: float | None, *, label: str) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise PassingContractError(f"NAN_INF_REJECTED:{label}")


def assert_half_open_interval(start_us: int, end_us: int) -> None:
    if int(end_us) < int(start_us):
        raise PassingContractError("NEGATIVE_DURATION")
    if int(start_us) < 0 or int(end_us) < 0:
        raise PassingContractError("NEGATIVE_DURATION")


def assert_result_level(level: str) -> str:
    if level not in RESULT_LEVELS:
        raise PassingContractError(f"unknown result level: {level}")
    return level


def owner_change_alone_is_completed_pass(*, owner_changed: bool) -> bool:
    """Owner change alone never implies a completed pass."""
    _ = owner_changed
    return False


def cut_replay_gap_allows_pass(*, cut_or_replay: bool, hard_gap: bool) -> bool:
    return not (cut_or_replay or hard_gap)


def attack_direction_unknown_blocks_directional(*, attack_direction: str) -> bool:
    return str(attack_direction) in {"", "unknown", "None"}


def directional_metric_status(*, attack_direction: str) -> str:
    if attack_direction_unknown_blocks_directional(attack_direction=attack_direction):
        return "not_evaluable"
    return "evaluable"


def penalty_presence_is_box_touch(*, in_penalty: bool) -> bool:
    _ = in_penalty
    return False


def box_touch_eligible(
    *,
    in_penalty: bool,
    has_possession_or_contact: bool,
    has_pitch_mapping: bool,
    playability_status: str,
) -> bool:
    return (
        bool(in_penalty)
        and bool(has_possession_or_contact)
        and bool(has_pitch_mapping)
        and str(playability_status) == "playable"
    )


def proximity_alone_is_reception(*, proximity_only: bool) -> bool:
    return not proximity_only


def automatic_confirmed_allowed() -> bool:
    return False


def assert_no_duplicate_pk(rows: Sequence[Mapping[str, Any]], pk_fields: Sequence[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[f] for f in pk_fields)
        if key in seen:
            raise PassingContractError("DUPLICATE_PK")
        seen.add(key)


def assert_scope_match(rows: Sequence[Mapping[str, Any]], *, run_id: str, video_id: str) -> None:
    for row in rows:
        if str(row.get("run_id")) != run_id or str(row.get("video_id")) != video_id:
            raise PassingContractError("CROSS_SCOPE_FK")


def append_only_decision(
    previous_log: Sequence[Mapping[str, Any]],
    new_entry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    out = [dict(e) for e in previous_log]
    out.append(dict(new_entry))
    return out


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


def neutral_transition(start_zone: str, end_zone: str) -> str:
    if start_zone in {"unknown", "not_evaluable"} or end_zone in {"unknown", "not_evaluable"}:
        return "not_evaluable"
    if start_zone == end_zone:
        return "same_zone"
    key = f"{start_zone}_to_{end_zone}"
    allowed = {
        "goal_a_to_middle",
        "middle_to_goal_b",
        "goal_b_to_middle",
        "middle_to_goal_a",
        "goal_a_to_goal_b",
        "goal_b_to_goal_a",
    }
    if key in allowed:
        if key in {"goal_a_to_goal_b"}:
            return "goal_a_to_middle"  # coarse; mid implied
        if key in {"goal_b_to_goal_a"}:
            return "goal_b_to_middle"
        return key
    return "unknown"


__all__ = [
    "assert_finite_optional",
    "assert_half_open_interval",
    "assert_result_level",
    "owner_change_alone_is_completed_pass",
    "cut_replay_gap_allows_pass",
    "attack_direction_unknown_blocks_directional",
    "directional_metric_status",
    "penalty_presence_is_box_touch",
    "box_touch_eligible",
    "proximity_alone_is_reception",
    "automatic_confirmed_allowed",
    "assert_no_duplicate_pk",
    "assert_scope_match",
    "append_only_decision",
    "opta_accuracy_claim_forbidden",
    "neutral_zone_from_x",
    "neutral_transition",
]
