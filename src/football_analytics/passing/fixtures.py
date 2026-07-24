"""Synthetic passing contract fixtures (Stage 11A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.passing.types import (
    CONTRACT_VERSION,
    DEFINITION_STYLE,
    METRIC_ORIGIN,
)


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {"run_id": generate_run_id(), "video_id": "video_synth_01"}


def pass_row(
    run_id: str,
    video_id: str,
    pass_candidate_id: str,
    *,
    policy_fingerprint: str,
    passer_human_track_id: int = 1,
    intended_receiver_track_id: int | None = 2,
    start_time_us: int = 1_000_000,
    end_time_us: int = 2_000_000,
    start_x_m: float | None = 30.0,
    start_y_m: float | None = 34.0,
    end_x_m: float | None = 55.0,
    end_y_m: float | None = 40.0,
    pass_distance_m: float | None = 25.0,
    start_zone_neutral: str = "goal_a",
    end_zone_neutral: str = "middle",
    candidate_state: str = "provisional",
    target_relationship: str = "confirmed_target",
    possession_hypothesis_id: str | None = "poss_01",
    contact_candidate_ids: Sequence[str] | None = None,
    evidence_refs: Sequence[str] | None = None,
    cut_or_replay: bool = False,
    hard_gap: bool = False,
    owner_change_alone: bool = False,
    implies_completed_pass: bool = False,
    playability_status: str = "playable",
    calibration_status: str = "valid",
    review_status: str = "unreviewed",
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "pass_candidate_id": pass_candidate_id,
        "passer_human_track_id": passer_human_track_id,
        "intended_receiver_track_id": intended_receiver_track_id,
        "passer_team_id": "team_a",
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "start_x_m": start_x_m,
        "start_y_m": start_y_m,
        "end_x_m": end_x_m,
        "end_y_m": end_y_m,
        "pass_distance_m": pass_distance_m,
        "start_zone_neutral": start_zone_neutral,
        "end_zone_neutral": end_zone_neutral,
        "candidate_state": candidate_state,
        "target_relationship": target_relationship,
        "possession_hypothesis_id": possession_hypothesis_id,
        "contact_candidate_ids": list(contact_candidate_ids or ["contact_01"]),
        "evidence_refs": list(evidence_refs or ["ev_pass_01"]),
        "cut_or_replay": cut_or_replay,
        "hard_gap": hard_gap,
        "owner_change_alone": owner_change_alone,
        "implies_completed_pass": implies_completed_pass,
        "playability_status": playability_status,
        "calibration_status": calibration_status,
        "automatic_ceiling": "provisional",
        "review_status": review_status,
        "manual_review_required": False,
        "uncertainty": 0.4,
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def reception_row(
    run_id: str,
    video_id: str,
    reception_candidate_id: str,
    *,
    policy_fingerprint: str,
    receiver_human_track_id: int = 2,
    source_pass_candidate_id: str | None = "pass_01",
    start_time_us: int = 2_000_000,
    end_time_us: int = 2_500_000,
    reception_x_m: float | None = 55.0,
    reception_y_m: float | None = 40.0,
    zone_neutral: str = "middle",
    candidate_state: str = "provisional",
    target_relationship: str = "non_target",
    cut_or_replay: bool = False,
    hard_gap: bool = False,
    implies_completed_pass: bool = False,
    contact_candidate_ids: Sequence[str] | None = None,
    evidence_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "reception_candidate_id": reception_candidate_id,
        "receiver_human_track_id": receiver_human_track_id,
        "source_pass_candidate_id": source_pass_candidate_id,
        "receiver_team_id": "team_a",
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "reception_x_m": reception_x_m,
        "reception_y_m": reception_y_m,
        "zone_neutral": zone_neutral,
        "candidate_state": candidate_state,
        "target_relationship": target_relationship,
        "possession_hypothesis_id": "poss_02",
        "contact_candidate_ids": list(contact_candidate_ids or ["contact_02"]),
        "evidence_refs": list(evidence_refs or ["ev_recv_01"]),
        "cut_or_replay": cut_or_replay,
        "hard_gap": hard_gap,
        "implies_completed_pass": implies_completed_pass,
        "playability_status": "playable",
        "calibration_status": "valid",
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "uncertainty": 0.35,
        "reason_codes": [],
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def outcome_row(
    run_id: str,
    video_id: str,
    outcome_id: str,
    *,
    policy_fingerprint: str,
    pass_candidate_id: str = "pass_01",
    reception_candidate_id: str | None = "recv_01",
    outcome: str = "completed",
    outcome_state: str = "provisional",
    passer_is_target: bool = True,
    receiver_is_teammate: bool | None = True,
    is_long_pass: bool = False,
    pass_distance_m: float | None = 25.0,
    long_pass_threshold_m: float | None = 30.0,
    start_zone_neutral: str = "goal_a",
    end_zone_neutral: str = "middle",
    attack_relative_evaluable: bool = False,
    progression_1_to_2: str = "not_evaluable",
    progression_2_to_3: str = "not_evaluable",
    cut_or_replay: bool = False,
    hard_gap: bool = False,
    owner_change_alone: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "outcome_id": outcome_id,
        "pass_candidate_id": pass_candidate_id,
        "reception_candidate_id": reception_candidate_id,
        "outcome": outcome,
        "outcome_state": outcome_state,
        "passer_is_target": passer_is_target,
        "receiver_is_teammate": receiver_is_teammate,
        "is_long_pass": is_long_pass,
        "pass_distance_m": pass_distance_m,
        "long_pass_threshold_m": long_pass_threshold_m,
        "start_zone_neutral": start_zone_neutral,
        "end_zone_neutral": end_zone_neutral,
        "attack_relative_evaluable": attack_relative_evaluable,
        "progression_1_to_2": progression_1_to_2,
        "progression_2_to_3": progression_2_to_3,
        "cut_or_replay": cut_or_replay,
        "hard_gap": hard_gap,
        "owner_change_alone": owner_change_alone,
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "uncertainty": 0.4,
        "evidence_refs": ["ev_out_01"],
        "reason_codes": [],
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def progression_row(
    run_id: str,
    video_id: str,
    segment_id: str,
    *,
    policy_fingerprint: str,
    pass_candidate_id: str | None = "pass_01",
    outcome_id: str | None = "out_01",
    start_time_us: int = 1_000_000,
    end_time_us: int = 2_000_000,
    start_zone_neutral: str = "goal_a",
    end_zone_neutral: str = "middle",
    neutral_transition: str = "goal_a_to_middle",
    attack_direction: str = "unknown",
    progression_1_to_2: str = "not_evaluable",
    progression_2_to_3: str = "not_evaluable",
    segment_state: str = "provisional",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "segment_id": segment_id,
        "pass_candidate_id": pass_candidate_id,
        "outcome_id": outcome_id,
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "start_x_m": 30.0,
        "start_y_m": 34.0,
        "end_x_m": 55.0,
        "end_y_m": 40.0,
        "start_zone_neutral": start_zone_neutral,
        "end_zone_neutral": end_zone_neutral,
        "neutral_transition": neutral_transition,
        "attack_direction": attack_direction,
        "progression_1_to_2": progression_1_to_2,
        "progression_2_to_3": progression_2_to_3,
        "segment_state": segment_state,
        "target_relationship": "confirmed_target",
        "cut_or_replay": False,
        "hard_gap": False,
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "evidence_refs": ["ev_prog_01"],
        "reason_codes": [],
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def touch_row(
    run_id: str,
    video_id: str,
    touch_id: str,
    *,
    policy_fingerprint: str,
    human_track_id: int = 1,
    touch_time_us: int = 3_000_000,
    in_penalty_area: bool = True,
    is_box_touch_candidate: bool = True,
    penalty_presence_alone: bool = False,
    has_possession_or_contact: bool = True,
    has_pitch_mapping: bool = True,
    playability_status: str = "playable",
    touch_state: str = "provisional",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "touch_id": touch_id,
        "human_track_id": human_track_id,
        "touch_time_us": touch_time_us,
        "touch_x_m": 100.0,
        "touch_y_m": 34.0,
        "in_penalty_area": in_penalty_area,
        "penalty_side_neutral": "goal_b",
        "is_box_touch_candidate": is_box_touch_candidate,
        "penalty_presence_alone": penalty_presence_alone,
        "has_possession_or_contact": has_possession_or_contact,
        "has_pitch_mapping": has_pitch_mapping,
        "playability_status": playability_status,
        "calibration_status": "valid",
        "touch_state": touch_state,
        "target_relationship": "confirmed_target",
        "possession_hypothesis_id": "poss_03",
        "contact_candidate_ids": ["contact_03"],
        "evidence_refs": ["ev_touch_01"],
        "automatic_ceiling": "provisional",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "uncertainty": 0.3,
        "reason_codes": [],
        "quality_flags": [],
        "metric_origin": METRIC_ORIGIN,
        "definition_style": DEFINITION_STYLE,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def single_target_pass_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    passes = [pass_row(rid, vid, "pass_01", policy_fingerprint=policy_fingerprint)]
    receptions = [reception_row(rid, vid, "recv_01", policy_fingerprint=policy_fingerprint)]
    outcomes = [outcome_row(rid, vid, "out_01", policy_fingerprint=policy_fingerprint)]
    progression = [progression_row(rid, vid, "seg_01", policy_fingerprint=policy_fingerprint)]
    touches = [touch_row(rid, vid, "touch_01", policy_fingerprint=policy_fingerprint)]
    return {
        "run_id": rid,
        "video_id": vid,
        "pass_candidates": _cast("pass_candidates", passes),
        "reception_candidates": _cast("reception_candidates", receptions),
        "pass_outcomes": _cast("pass_outcomes", outcomes),
        "ball_progression_segments": _cast("ball_progression_segments", progression),
        "target_ball_touches": _cast("target_ball_touches", touches),
        "pass_rows": passes,
        "reception_rows": receptions,
        "outcome_rows": outcomes,
        "progression_rows": progression,
        "touch_rows": touches,
    }


def owner_change_alone_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = pass_row(
        rid,
        vid,
        "pass_owner_only",
        policy_fingerprint=policy_fingerprint,
        owner_change_alone=True,
        implies_completed_pass=False,
        evidence_refs=[],
        contact_candidate_ids=[],
        candidate_state="not_evaluable",
        reason_codes=["OWNER_CHANGE_ALONE_NOT_PASS"],
    )
    return {"run_id": rid, "video_id": vid, "pass_rows": [row]}


def cut_replay_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = pass_row(
        rid,
        vid,
        "pass_cut",
        policy_fingerprint=policy_fingerprint,
        cut_or_replay=True,
        candidate_state="rejected",
        playability_status="replay",
        reason_codes=["CUT_REPLAY_GAP_NO_PASS"],
    )
    return {"run_id": rid, "video_id": vid, "pass_rows": [row]}


def penalty_presence_only_rows(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    row = touch_row(
        rid,
        vid,
        "touch_presence",
        policy_fingerprint=policy_fingerprint,
        in_penalty_area=True,
        is_box_touch_candidate=False,
        penalty_presence_alone=True,
        has_possession_or_contact=False,
        touch_state="rejected",
    )
    row["reason_codes"] = ["PENALTY_PRESENCE_NOT_BOX_TOUCH"]
    return {"run_id": rid, "video_id": vid, "touch_rows": [row]}


def coverage_example() -> dict[str, Any]:
    return {
        "target_confirmed_us": 10_000_000,
        "possession_or_contact_us": 8_000_000,
        "calibration_valid_us": 9_000_000,
        "playable_us": 9_500_000,
        "attack_direction_resolved_us": 0,
        "not_observed_us": 1_000_000,
        "joint_coverage_ratio": 0.8,
        "owner_change_alone_is_not_completed_pass": True,
        "penalty_presence_is_not_box_touch": True,
        "attack_direction_unknown_blocks_directional": True,
    }


__all__ = [
    "base_ids",
    "pass_row",
    "reception_row",
    "outcome_row",
    "progression_row",
    "touch_row",
    "single_target_pass_bundle",
    "owner_change_alone_rows",
    "cut_replay_rows",
    "penalty_presence_only_rows",
    "coverage_example",
]
