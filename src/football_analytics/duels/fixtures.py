"""Synthetic duels contract fixtures (Stage 12A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.duels.types import CONTRACT_VERSION, DEFINITION_STYLE, METRIC_ORIGIN


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {"run_id": generate_run_id(), "video_id": "video_synth_01"}


def _common(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    event_state: str = "provisional",
    target_human_track_id: int = 1,
    opponent_human_track_id: int | None = 7,
    start_time_us: int = 1_000_000,
    end_time_us: int = 2_000_000,
    x_m: float | None = 40.0,
    y_m: float | None = 34.0,
    zone_neutral: str = "middle",
    cut_or_replay: bool = False,
    hard_gap: bool = False,
    reason_codes: Sequence[str] | None = None,
    evidence_refs: Sequence[str] | None = None,
    contact_candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_human_track_id": target_human_track_id,
        "opponent_human_track_id": opponent_human_track_id,
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "x_m": x_m,
        "y_m": y_m,
        "zone_neutral": zone_neutral,
        "event_state": event_state,
        "target_relationship": "confirmed_target",
        "possession_hypothesis_id": "poss_01",
        "contact_candidate_ids": list(contact_candidate_ids or ["contact_01"]),
        "evidence_refs": list(evidence_refs or ["ev_duel_01"]),
        "cut_or_replay": cut_or_replay,
        "hard_gap": hard_gap,
        "playability_status": "playable",
        "calibration_status": "valid",
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "uncertainty": 0.4,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def take_on_row(
    run_id: str,
    video_id: str,
    take_on_attempt_id: str,
    *,
    policy_fingerprint: str,
    nearby_opponent_alone: bool = False,
    direction_change_alone: bool = False,
    implies_take_on: bool = True,
    implies_take_on_success: bool = False,
    outcome: str = "attempted",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "take_on_attempt_id": take_on_attempt_id,
            "nearby_opponent_alone": nearby_opponent_alone,
            "direction_change_alone": direction_change_alone,
            "implies_take_on": implies_take_on,
            "implies_take_on_success": implies_take_on_success,
            "outcome": outcome,
        }
    )
    return row


def ground_duel_row(
    run_id: str,
    video_id: str,
    ground_duel_candidate_id: str,
    *,
    policy_fingerprint: str,
    nearest_switch_alone: bool = False,
    contested_possession: bool = True,
    implies_duel_outcome: bool = False,
    outcome: str = "uncertain",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "ground_duel_candidate_id": ground_duel_candidate_id,
            "nearest_switch_alone": nearest_switch_alone,
            "contested_possession": contested_possession,
            "implies_duel_outcome": implies_duel_outcome,
            "outcome": outcome,
        }
    )
    return row


def aerial_duel_row(
    run_id: str,
    video_id: str,
    aerial_duel_candidate_id: str,
    *,
    policy_fingerprint: str,
    monocular_only: bool = True,
    exact_3d_height_claimed: bool = False,
    exact_3d_height_m: float | None = None,
    aerial_evaluability: str = "not_evaluable",
    implies_aerial_outcome: bool = False,
    outcome: str = "not_evaluable",
    event_state: str = "not_evaluable",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "aerial_duel_candidate_id": aerial_duel_candidate_id,
            "monocular_only": monocular_only,
            "exact_3d_height_claimed": exact_3d_height_claimed,
            "exact_3d_height_m": exact_3d_height_m,
            "aerial_evaluability": aerial_evaluability,
            "implies_aerial_outcome": implies_aerial_outcome,
            "outcome": outcome,
        }
    )
    return row


def tackle_row(
    run_id: str,
    video_id: str,
    tackle_event_id: str,
    *,
    policy_fingerprint: str,
    related_ground_duel_candidate_id: str | None = "gduel_01",
    implies_tackle: bool = True,
    implies_tackle_success: bool = False,
    outcome: str = "uncertain",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "tackle_event_id": tackle_event_id,
            "related_ground_duel_candidate_id": related_ground_duel_candidate_id,
            "implies_tackle": implies_tackle,
            "implies_tackle_success": implies_tackle_success,
            "outcome": outcome,
        }
    )
    return row


def recovery_row(
    run_id: str,
    video_id: str,
    recovery_event_id: str,
    *,
    policy_fingerprint: str,
    related_turnover_event_id: str | None = "turn_01",
    implies_recovery: bool = True,
    outcome: str = "recovered",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "recovery_event_id": recovery_event_id,
            "related_turnover_event_id": related_turnover_event_id,
            "implies_recovery": implies_recovery,
            "outcome": outcome,
        }
    )
    return row


def turnover_row(
    run_id: str,
    video_id: str,
    turnover_event_id: str,
    *,
    policy_fingerprint: str,
    related_recovery_event_id: str | None = "rec_01",
    implies_turnover: bool = True,
    outcome: str = "lost",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "turnover_event_id": turnover_event_id,
            "related_recovery_event_id": related_recovery_event_id,
            "implies_turnover": implies_turnover,
            "outcome": outcome,
        }
    )
    return row


def clearance_row(
    run_id: str,
    video_id: str,
    clearance_event_id: str,
    *,
    policy_fingerprint: str,
    long_ball_alone: bool = False,
    ball_distance_m: float | None = 25.0,
    implies_clearance: bool = True,
    outcome: str = "cleared",
    event_state: str = "provisional",
    **kwargs: Any,
) -> dict[str, Any]:
    row = _common(
        run_id=run_id,
        video_id=video_id,
        policy_fingerprint=policy_fingerprint,
        event_state=event_state,
        **kwargs,
    )
    row.update(
        {
            "clearance_event_id": clearance_event_id,
            "long_ball_alone": long_ball_alone,
            "ball_distance_m": ball_distance_m,
            "implies_clearance": implies_clearance,
            "outcome": outcome,
        }
    )
    return row


def single_target_duels_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    take_ons = [
        take_on_row(
            rid,
            vid,
            "take_01",
            policy_fingerprint=policy_fingerprint,
            implies_take_on=True,
            nearby_opponent_alone=False,
            evidence_refs=["ev_take_01"],
            contact_candidate_ids=["contact_take"],
        )
    ]
    ground = [
        ground_duel_row(
            rid,
            vid,
            "gduel_01",
            policy_fingerprint=policy_fingerprint,
            contested_possession=True,
            nearest_switch_alone=False,
            implies_duel_outcome=False,
        )
    ]
    aerial = [
        aerial_duel_row(
            rid,
            vid,
            "aduels_01",
            policy_fingerprint=policy_fingerprint,
            monocular_only=True,
            aerial_evaluability="not_evaluable",
            event_state="not_evaluable",
            outcome="not_evaluable",
            reason_codes=["MONOCULAR_AERIAL_NO_EXACT_HEIGHT"],
        )
    ]
    tackles = [
        tackle_row(
            rid,
            vid,
            "tack_01",
            policy_fingerprint=policy_fingerprint,
            related_ground_duel_candidate_id="gduel_01",
        )
    ]
    turnovers = [
        turnover_row(
            rid,
            vid,
            "turn_01",
            policy_fingerprint=policy_fingerprint,
            related_recovery_event_id="rec_01",
            start_time_us=3_000_000,
            end_time_us=3_500_000,
        )
    ]
    recoveries = [
        recovery_row(
            rid,
            vid,
            "rec_01",
            policy_fingerprint=policy_fingerprint,
            related_turnover_event_id="turn_01",
            start_time_us=3_500_000,
            end_time_us=4_000_000,
        )
    ]
    clearances = [
        clearance_row(
            rid,
            vid,
            "clr_01",
            policy_fingerprint=policy_fingerprint,
            long_ball_alone=False,
            implies_clearance=True,
            evidence_refs=["ev_clr_01"],
            contact_candidate_ids=["contact_clr"],
        )
    ]
    return {
        "run_id": rid,
        "video_id": vid,
        "take_on_attempts": _cast("take_on_attempts", take_ons),
        "ground_duel_candidates": _cast("ground_duel_candidates", ground),
        "aerial_duel_candidates": _cast("aerial_duel_candidates", aerial),
        "tackle_events": _cast("tackle_events", tackles),
        "recovery_events": _cast("recovery_events", recoveries),
        "turnover_events": _cast("turnover_events", turnovers),
        "clearance_events": _cast("clearance_events", clearances),
        "take_on_rows": take_ons,
        "ground_rows": ground,
        "aerial_rows": aerial,
        "tackle_rows": tackles,
        "recovery_rows": recoveries,
        "turnover_rows": turnovers,
        "clearance_rows": clearances,
    }


def nearby_opponent_alone_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = take_on_row(
        rid,
        vid,
        "take_nearby",
        policy_fingerprint=policy_fingerprint,
        nearby_opponent_alone=True,
        implies_take_on=False,
        implies_take_on_success=False,
        outcome="not_evaluable",
        event_state="not_evaluable",
        evidence_refs=[],
        contact_candidate_ids=[],
        reason_codes=["NEARBY_OPPONENT_ALONE_NOT_TAKE_ON"],
    )
    return {"run_id": rid, "video_id": vid, "take_on_rows": [row]}


def nearest_switch_alone_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = ground_duel_row(
        rid,
        vid,
        "gduel_switch",
        policy_fingerprint=policy_fingerprint,
        nearest_switch_alone=True,
        contested_possession=False,
        implies_duel_outcome=False,
        outcome="not_evaluable",
        event_state="not_evaluable",
        reason_codes=["NEAREST_SWITCH_ALONE_NOT_DUEL_OUTCOME"],
    )
    return {"run_id": rid, "video_id": vid, "ground_rows": [row]}


def monocular_aerial_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = aerial_duel_row(
        rid,
        vid,
        "aduels_mono",
        policy_fingerprint=policy_fingerprint,
        monocular_only=True,
        exact_3d_height_claimed=False,
        exact_3d_height_m=None,
        aerial_evaluability="candidate",
        event_state="candidate",
        outcome="uncertain",
        reason_codes=["MONOCULAR_AERIAL_NO_EXACT_HEIGHT"],
    )
    return {"run_id": rid, "video_id": vid, "aerial_rows": [row]}


def long_ball_alone_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = clearance_row(
        rid,
        vid,
        "clr_long",
        policy_fingerprint=policy_fingerprint,
        long_ball_alone=True,
        implies_clearance=False,
        outcome="not_evaluable",
        event_state="not_evaluable",
        evidence_refs=[],
        contact_candidate_ids=[],
        reason_codes=["LONG_BALL_ALONE_NOT_CLEARANCE"],
    )
    return {"run_id": rid, "video_id": vid, "clearance_rows": [row]}


def coverage_example() -> dict[str, Any]:
    return {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "opponent_context_us": 7_000_000,
        "not_observed_us": 1_000_000,
        "joint_coverage_ratio": 0.75,
        "nearby_opponent_alone_is_not_take_on": True,
        "nearest_switch_alone_is_not_duel_outcome": True,
        "monocular_aerial_no_exact_height": True,
        "long_ball_alone_is_not_clearance": True,
    }


__all__ = [
    "base_ids",
    "take_on_row",
    "ground_duel_row",
    "aerial_duel_row",
    "tackle_row",
    "recovery_row",
    "turnover_row",
    "clearance_row",
    "single_target_duels_bundle",
    "nearby_opponent_alone_rows",
    "nearest_switch_alone_rows",
    "monocular_aerial_rows",
    "long_ball_alone_rows",
    "coverage_example",
]
