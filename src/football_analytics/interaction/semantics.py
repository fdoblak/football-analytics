"""Human-ball interaction semantic helpers (Stage 10A — no real inference)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.interaction.types import RESULT_LEVELS, InteractionContractError


def assert_finite_optional(value: float | None, *, label: str) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise InteractionContractError(f"NAN_INF_REJECTED:{label}")


def assert_half_open_interval(start_us: int, end_us: int) -> None:
    if int(end_us) < int(start_us):
        raise InteractionContractError("NEGATIVE_DURATION")
    if int(start_us) < 0 or int(end_us) < 0:
        raise InteractionContractError("NEGATIVE_DURATION")


def assert_result_level(level: str) -> str:
    if level not in RESULT_LEVELS:
        raise InteractionContractError(f"unknown result level: {level}")
    return level


def nearest_player_is_possession(*, is_nearest: bool) -> bool:
    """Nearest-human flag never implies possession."""
    _ = is_nearest
    return False


def proximity_is_contact(*, proximity_only: bool) -> bool:
    return not proximity_only


def contact_is_controlled_possession(*, contact_candidate: bool) -> bool:
    _ = contact_candidate
    return False


def possession_is_completed_pass(*, possession_transition: bool) -> bool:
    _ = possession_transition
    return False


def ball_leaving_is_ball_loss(*, ball_leaving: bool) -> bool:
    _ = ball_leaving
    return False


def approaching_opponent_is_duel(*, approaching: bool) -> bool:
    _ = approaching
    return False


def penalty_presence_is_box_touch(*, in_penalty: bool) -> bool:
    _ = in_penalty
    return False


def ball_near_head_is_aerial(*, near_head: bool) -> bool:
    _ = near_head
    return False


def direction_change_is_dribble(*, direction_changed: bool) -> bool:
    _ = direction_changed
    return False


def single_frame_proximity_is_contact(*, frame_count: int) -> bool:
    return frame_count >= 2


def hard_gap_allows_possession_carry(*, hard_gap: bool) -> bool:
    return not hard_gap


def terminate_hypothesis_on(event: str) -> bool:
    return event in {
        "shot_cut",
        "replay",
        "non_playable",
        "ball_loss",
        "hard_gap",
        "track_termination",
    }


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return int(a_start) < int(b_end) and int(b_start) < int(a_end)


def assert_no_duplicate_pk(rows: Sequence[Mapping[str, Any]], pk_fields: Sequence[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[f] for f in pk_fields)
        if key in seen:
            raise InteractionContractError("DUPLICATE_PK")
        seen.add(key)


def assert_scope_match(rows: Sequence[Mapping[str, Any]], *, run_id: str, video_id: str) -> None:
    for row in rows:
        if str(row.get("run_id")) != run_id or str(row.get("video_id")) != video_id:
            raise InteractionContractError("CROSS_SCOPE_FK")


def owner_transition_requires_evidence(
    *, previous_owner: int | None, new_owner: int | None, evidence_refs: Sequence[str]
) -> bool:
    if previous_owner == new_owner:
        return True
    return len(list(evidence_refs)) > 0


def append_only_decision(
    previous_log: Sequence[Mapping[str, Any]],
    new_entry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return extended decision log; never mutates prior entries."""
    out = [dict(e) for e in previous_log]
    out.append(dict(new_entry))
    return out


def event_metrics_forbidden(metrics: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("pass", "dribble", "duel", "aerial", "turnover", "box_touch", "recovery"):
        val = metrics.get(key)
        if val not in (False, None, 0):
            errors.append(f"EVENT_METRIC_FORBIDDEN:{key}")
    return errors


__all__ = [
    "assert_finite_optional",
    "assert_half_open_interval",
    "assert_result_level",
    "nearest_player_is_possession",
    "proximity_is_contact",
    "contact_is_controlled_possession",
    "possession_is_completed_pass",
    "ball_leaving_is_ball_loss",
    "approaching_opponent_is_duel",
    "penalty_presence_is_box_touch",
    "ball_near_head_is_aerial",
    "direction_change_is_dribble",
    "single_frame_proximity_is_contact",
    "hard_gap_allows_possession_carry",
    "terminate_hypothesis_on",
    "intervals_overlap",
    "assert_no_duplicate_pk",
    "assert_scope_match",
    "owner_transition_requires_evidence",
    "append_only_decision",
    "event_metrics_forbidden",
]
