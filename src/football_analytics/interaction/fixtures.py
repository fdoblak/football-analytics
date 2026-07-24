"""Synthetic human-ball interaction contract fixtures only (Stage 10A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.interaction.types import CONTRACT_VERSION


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {
        "run_id": generate_run_id(),
        "video_id": "video_synth_01",
    }


def proximity_row(
    run_id: str,
    video_id: str,
    proximity_id: str,
    *,
    human_track_id: int,
    ball_track_id: int | None,
    frame_index: int,
    video_time_us: int,
    policy_fingerprint: str,
    image_distance_px: float | None = 12.0,
    pitch_distance_m: float | None = 0.8,
    foot_reference_type: str = "bbox_bottom_centre",
    ball_reference_type: str = "detection_centre",
    human_observation_state: str = "observed",
    ball_observation_state: str = "observed",
    evidence_space: str = "both",
    ball_air_state: str = "grounded",
    ball_candidate_status: str = "primary",
    calibration_status: str = "valid",
    playability_status: str = "playable",
    target_relationship: str = "confirmed_target",
    evidence_level: str = "candidate",
    is_nearest_human: bool = False,
    pitch_distance_usable: bool = True,
    uncertainty: float | None = 0.3,
    eligibility_status: str = "eligible",
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "proximity_id": proximity_id,
        "human_track_id": human_track_id,
        "ball_track_id": ball_track_id,
        "frame_index": frame_index,
        "video_time_us": video_time_us,
        "image_distance_px": image_distance_px,
        "pitch_distance_m": pitch_distance_m,
        "foot_reference_type": foot_reference_type,
        "ball_reference_type": ball_reference_type,
        "human_observation_state": human_observation_state,
        "ball_observation_state": ball_observation_state,
        "evidence_space": evidence_space,
        "ball_air_state": ball_air_state,
        "ball_candidate_status": ball_candidate_status,
        "calibration_status": calibration_status,
        "playability_status": playability_status,
        "target_relationship": target_relationship,
        "evidence_level": evidence_level,
        "is_nearest_human": is_nearest_human,
        "nearest_implies_possession": False,
        "pitch_distance_usable": pitch_distance_usable,
        "uncertainty": uncertainty,
        "eligibility_status": eligibility_status,
        "human_observation_id": f"hobs_{frame_index}",
        "ball_observation_id": f"bobs_{frame_index}" if ball_track_id is not None else None,
        "human_projection_id": "hproj_01",
        "ball_projection_id": "bproj_01" if ball_track_id is not None else None,
        "identity_assignment_id": "asn_confirmed_01",
        "analysis_window_id": "aw_01",
        "manual_review_required": False,
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def contact_row(
    run_id: str,
    video_id: str,
    candidate_id: str,
    *,
    human_track_id: int,
    ball_track_id: int | None,
    start_time_us: int,
    peak_time_us: int,
    end_time_us: int,
    policy_fingerprint: str,
    contact_state: str = "candidate",
    proximity_support: bool = True,
    trajectory_change_support: bool = False,
    multi_frame_support: bool = True,
    confidence: float | None = None,
    proximity_ids: Sequence[str] | None = None,
    evidence_types: Sequence[str] | None = None,
    reason_codes: Sequence[str] | None = None,
    ball_air_state: str = "grounded",
    ball_candidate_status: str = "primary",
    target_relationship: str = "confirmed_target",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "candidate_id": candidate_id,
        "human_track_id": human_track_id,
        "ball_track_id": ball_track_id,
        "start_time_us": start_time_us,
        "peak_time_us": peak_time_us,
        "end_time_us": end_time_us,
        "contact_state": contact_state,
        "evidence_types": list(evidence_types or ["proximity_support"]),
        "proximity_support": proximity_support,
        "trajectory_change_support": trajectory_change_support,
        "multi_frame_support": multi_frame_support,
        "confidence": confidence,
        "review_status": "unreviewed",
        "rejection_reason_codes": [],
        "implies_controlled_possession": False,
        "implies_pass_or_event": False,
        "implies_box_touch": False,
        "ball_air_state": ball_air_state,
        "ball_candidate_status": ball_candidate_status,
        "target_relationship": target_relationship,
        "proximity_ids": list(proximity_ids or []),
        "evidence_refs": list(proximity_ids or []),
        "manual_review_required": False,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def possession_row(
    run_id: str,
    video_id: str,
    hypothesis_id: str,
    *,
    start_time_us: int,
    end_time_us: int,
    policy_fingerprint: str,
    owner_human_track_id: int | None = 1,
    owner_team_id: str | None = None,
    ownership_kind: str = "human_track",
    possession_state: str = "candidate",
    contested_participant_track_ids: Sequence[int] | None = None,
    evidence_refs: Sequence[str] | None = None,
    contact_candidate_ids: Sequence[str] | None = None,
    proximity_ids: Sequence[str] | None = None,
    termination_reason: str = "none",
    target_relationship: str = "confirmed_target",
    observed_coverage_ratio: float | None = 0.7,
    penalty_area_presence_only: bool = False,
    transition_from_hypothesis_id: str | None = None,
    decision_log_json: str | None = None,
    reason_codes: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "hypothesis_id": hypothesis_id,
        "owner_human_track_id": owner_human_track_id,
        "owner_team_id": owner_team_id,
        "ownership_kind": ownership_kind,
        "target_relationship": target_relationship,
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "possession_state": possession_state,
        "contested_participant_track_ids": list(contested_participant_track_ids or []),
        "evidence_refs": list(evidence_refs or []),
        "contact_candidate_ids": list(contact_candidate_ids or []),
        "proximity_ids": list(proximity_ids or []),
        "observed_coverage_ratio": observed_coverage_ratio,
        "derived_coverage_ratio": None,
        "uncertainty": 0.4,
        "termination_reason": termination_reason,
        "transition_from_hypothesis_id": transition_from_hypothesis_id,
        "automatic_ceiling": "provisional",
        "implies_completed_pass": False,
        "implies_dribble_or_take_on": False,
        "implies_duel_or_aerial": False,
        "implies_box_touch": False,
        "implies_turnover": False,
        "penalty_area_presence_only": penalty_area_presence_only,
        "manual_review_required": False,
        "review_status": "unreviewed",
        "decision_log_json": decision_log_json,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def single_player_proximity_bundle(policy_fp: str) -> dict[str, Any]:
    ids = base_ids()
    prox = [
        proximity_row(
            ids["run_id"],
            ids["video_id"],
            "prox_01",
            human_track_id=1,
            ball_track_id=100,
            frame_index=0,
            video_time_us=0,
            policy_fingerprint=policy_fp,
            is_nearest_human=True,
            evidence_level="candidate",
        ),
        proximity_row(
            ids["run_id"],
            ids["video_id"],
            "prox_02",
            human_track_id=1,
            ball_track_id=100,
            frame_index=1,
            video_time_us=40_000,
            policy_fingerprint=policy_fp,
            is_nearest_human=True,
            evidence_level="candidate",
        ),
    ]
    contacts = [
        contact_row(
            ids["run_id"],
            ids["video_id"],
            "contact_01",
            human_track_id=1,
            ball_track_id=100,
            start_time_us=0,
            peak_time_us=40_000,
            end_time_us=80_000,
            policy_fingerprint=policy_fp,
            contact_state="candidate",
            multi_frame_support=True,
            proximity_ids=["prox_01", "prox_02"],
            evidence_types=["proximity_support", "multi_frame_support"],
        )
    ]
    possessions = [
        possession_row(
            ids["run_id"],
            ids["video_id"],
            "poss_01",
            owner_human_track_id=1,
            start_time_us=0,
            end_time_us=80_000,
            policy_fingerprint=policy_fp,
            possession_state="candidate",
            proximity_ids=["prox_01", "prox_02"],
            contact_candidate_ids=["contact_01"],
            evidence_refs=["prox_01", "prox_02", "contact_01"],
        )
    ]
    return {
        **ids,
        "human_ball_proximity": _cast("human_ball_proximity", prox),
        "ball_contact_candidates": _cast("ball_contact_candidates", contacts),
        "possession_hypotheses": _cast("possession_hypotheses", possessions),
        "proximity_rows": prox,
        "contact_rows": contacts,
        "possession_rows": possessions,
    }


def nearest_not_possession_rows(policy_fp: str) -> dict[str, Any]:
    ids = base_ids()
    prox = proximity_row(
        ids["run_id"],
        ids["video_id"],
        "prox_nearest",
        human_track_id=2,
        ball_track_id=100,
        frame_index=0,
        video_time_us=0,
        policy_fingerprint=policy_fp,
        is_nearest_human=True,
        evidence_level="candidate",
    )
    return {**ids, "proximity_rows": [prox]}


def contested_two_player_bundle(policy_fp: str) -> dict[str, Any]:
    ids = base_ids()
    prox = [
        proximity_row(
            ids["run_id"],
            ids["video_id"],
            "prox_a",
            human_track_id=1,
            ball_track_id=100,
            frame_index=0,
            video_time_us=0,
            policy_fingerprint=policy_fp,
            is_nearest_human=False,
        ),
        proximity_row(
            ids["run_id"],
            ids["video_id"],
            "prox_b",
            human_track_id=2,
            ball_track_id=100,
            frame_index=0,
            video_time_us=0,
            policy_fingerprint=policy_fp,
            is_nearest_human=True,
        ),
    ]
    possessions = [
        possession_row(
            ids["run_id"],
            ids["video_id"],
            "poss_contested",
            owner_human_track_id=None,
            ownership_kind="unknown",
            start_time_us=0,
            end_time_us=100_000,
            policy_fingerprint=policy_fp,
            possession_state="contested",
            contested_participant_track_ids=[1, 2],
            proximity_ids=["prox_a", "prox_b"],
            evidence_refs=["prox_a", "prox_b"],
            termination_reason="contested",
        )
    ]
    return {
        **ids,
        "human_ball_proximity": _cast("human_ball_proximity", prox),
        "ball_contact_candidates": _cast("ball_contact_candidates", []),
        "possession_hypotheses": _cast("possession_hypotheses", possessions),
        "proximity_rows": prox,
        "contact_rows": [],
        "possession_rows": possessions,
    }


def coverage_example() -> dict[str, Any]:
    return {
        "human_observed_us": 1_000_000,
        "ball_observed_us": 800_000,
        "joint_observed_us": 500_000,
        "calibration_valid_us": 900_000,
        "playable_us": 950_000,
        "target_confirmed_us": 700_000,
        "ambiguous_ball_us": 100_000,
        "contested_us": 50_000,
        "not_observed_us": 200_000,
        "joint_coverage_ratio": 0.5,
        "low_joint_coverage_is_not_evaluable": True,
        "missing_ball_is_not_no_possession": True,
    }


__all__ = [
    "base_ids",
    "proximity_row",
    "contact_row",
    "possession_row",
    "single_player_proximity_bundle",
    "nearest_not_possession_rows",
    "contested_two_player_bundle",
    "coverage_example",
]
