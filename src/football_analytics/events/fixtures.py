"""Synthetic fixtures for Stage 13A–13E."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id


def replay_contexts_fixture(name: str = "mixed") -> list[dict[str, Any]]:
    if name == "uncertain_blocks_live":
        return [
            {
                "replay_candidate_id": "rep_unc_01",
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "replay_status_hint": "live",
                "confidence": 0.4,
                "view_family": "main_broadcast",
                "dissolve_score": 0.7,
                "transition_hint": "dissolve",
            }
        ]
    if name == "supported_camera":
        return [
            {
                "replay_candidate_id": "rep_cam_01",
                "start_time_us": 1_000_000,
                "end_time_us": 1_500_000,
                "explicit_live": True,
                "live_confidence": 0.9,
                "view_family": "main_broadcast",
            },
            {
                "replay_candidate_id": "rep_cam_02",
                "start_time_us": 2_000_000,
                "end_time_us": 2_500_000,
                "explicit_live": True,
                "live_confidence": 0.9,
                "view_family": "crowd",
            },
        ]
    # mixed default
    return [
        {
            "replay_candidate_id": "rep_live_01",
            "start_time_us": 1_000_000,
            "end_time_us": 2_000_000,
            "explicit_live": True,
            "live_confidence": 0.95,
            "view_family": "main_broadcast",
            "transition_hint": "none",
        },
        {
            "replay_candidate_id": "rep_unk_01",
            "start_time_us": 3_000_000,
            "end_time_us": 4_000_000,
            "overlay_fraction": 0.7,
            "scoreboard_interruption": True,
            "view_family": "graphics",
            "transition_hint": "scoreboard",
        },
        {
            "replay_candidate_id": "rep_replay_01",
            "start_time_us": 5_000_000,
            "end_time_us": 6_000_000,
            "replay_status_hint": "replay",
            "confidence": 0.8,
            "dissolve_score": 0.8,
            "view_family": "goal_view",
            "transition_hint": "dissolve",
        },
    ]


def source_events_fixture(name: str = "full_package") -> dict[str, list[dict[str, Any]]]:
    run_id = generate_run_id()
    video_id = "synthetic_video_13"
    base = {
        "run_id": run_id,
        "video_id": video_id,
        "target_relationship": "confirmed_target",
        "target_human_track_id": 7,
        "cut_or_replay": False,
        "hard_gap": False,
        "playability_status": "playable",
        "review_status": "unreviewed",
        "manual_review_required": False,
        "evidence_refs": ["fx"],
        "reason_codes": [],
        "quality_flags": [],
        "confidence": 0.8,
    }

    if name == "duplicate_overlap":
        return {
            "take_on_attempts": [
                {
                    **base,
                    "take_on_attempt_id": "to_01",
                    "start_time_us": 1_000_000,
                    "end_time_us": 1_400_000,
                    "outcome": "beaten",
                    "event_state": "provisional",
                    "implies_take_on": True,
                },
                {
                    **base,
                    "take_on_attempt_id": "to_02",
                    "start_time_us": 1_100_000,
                    "end_time_us": 1_500_000,
                    "outcome": "beaten",
                    "event_state": "provisional",
                    "implies_take_on": True,
                    "confidence": 0.5,
                },
            ]
        }

    if name == "replay_blocked":
        return {
            "pass_outcomes": [
                {
                    **base,
                    "outcome_id": "out_replay",
                    "pass_candidate_id": "pass_replay",
                    "start_time_us": 2_000_000,
                    "end_time_us": 2_400_000,
                    "outcome": "completed",
                    "event_state": "provisional",
                    "cut_or_replay": True,
                    "passer_is_target": True,
                }
            ]
        }

    sources: dict[str, list[dict[str, Any]]] = {
        "pass_outcomes": [
            {
                **base,
                "outcome_id": "out_01",
                "pass_candidate_id": "pass_01",
                "start_time_us": 1_000_000,
                "end_time_us": 1_400_000,
                "outcome": "completed",
                "event_state": "provisional",
                "passer_is_target": True,
                "is_long_pass": True,
                "pass_distance_m": 35.0,
                "attributes_json": (
                    '{"is_long_pass":true,"pass_distance_m":35.0,'
                    '"start_zone_neutral":"goal_a","end_zone_neutral":"middle"}'
                ),
            },
            {
                **base,
                "outcome_id": "out_02",
                "pass_candidate_id": "pass_02",
                "start_time_us": 2_000_000,
                "end_time_us": 2_400_000,
                "outcome": "incomplete",
                "event_state": "provisional",
                "passer_is_target": True,
                "is_long_pass": False,
                "attributes_json": (
                    '{"is_long_pass":false,"start_zone_neutral":"middle",'
                    '"end_zone_neutral":"goal_b"}'
                ),
            },
        ],
        "reception_candidates": [
            {
                **base,
                "reception_candidate_id": "rec_01",
                "start_time_us": 1_400_000,
                "end_time_us": 1_600_000,
                "candidate_state": "provisional",
                "event_state": "provisional",
                "outcome": "received",
            }
        ],
        "take_on_attempts": [
            {
                **base,
                "take_on_attempt_id": "to_01",
                "start_time_us": 3_000_000,
                "end_time_us": 3_500_000,
                "outcome": "beaten",
                "event_state": "provisional",
                "implies_take_on": True,
            },
            {
                **base,
                "take_on_attempt_id": "to_02",
                "start_time_us": 4_000_000,
                "end_time_us": 4_500_000,
                "outcome": "lost",
                "event_state": "provisional",
                "implies_take_on": True,
            },
        ],
        "ground_duel_candidates": [
            {
                **base,
                "ground_duel_candidate_id": "gd_01",
                "start_time_us": 5_000_000,
                "end_time_us": 5_400_000,
                "outcome": "won",
                "event_state": "provisional",
            },
            {
                **base,
                "ground_duel_candidate_id": "gd_02",
                "start_time_us": 6_000_000,
                "end_time_us": 6_400_000,
                "outcome": "lost",
                "event_state": "provisional",
            },
        ],
        "aerial_duel_candidates": [
            {
                **base,
                "aerial_duel_candidate_id": "ad_01",
                "start_time_us": 7_000_000,
                "end_time_us": 7_400_000,
                "outcome": "won",
                "event_state": "provisional",
                "monocular_only": True,
                "exact_3d_height_claimed": False,
            }
        ],
        "tackle_events": [
            {
                **base,
                "tackle_event_id": "tk_01",
                "start_time_us": 8_000_000,
                "end_time_us": 8_300_000,
                "outcome": "won",
                "event_state": "provisional",
                "implies_tackle": True,
            }
        ],
        "recovery_events": [
            {
                **base,
                "recovery_event_id": "rv_01",
                "start_time_us": 9_000_000,
                "end_time_us": 9_300_000,
                "outcome": "won",
                "event_state": "provisional",
                "implies_recovery": True,
            }
        ],
        "turnover_events": [
            {
                **base,
                "turnover_event_id": "tn_01",
                "start_time_us": 10_000_000,
                "end_time_us": 10_300_000,
                "outcome": "lost",
                "event_state": "provisional",
                "implies_turnover": True,
            }
        ],
        "clearance_events": [
            {
                **base,
                "clearance_event_id": "cl_01",
                "start_time_us": 11_000_000,
                "end_time_us": 11_400_000,
                "outcome": "cleared",
                "event_state": "provisional",
                "implies_clearance": True,
                "long_ball_alone": False,
                "attributes_json": '{"implies_clearance":true}',
            }
        ],
        "target_ball_touches": [
            {
                **base,
                "touch_id": "touch_01",
                "touch_time_us": 12_000_000,
                "human_track_id": 7,
                "in_penalty_area": True,
                "is_box_touch_candidate": True,
                "penalty_presence_alone": False,
                "has_possession_or_contact": True,
                "event_state": "provisional",
                "outcome": "touch",
                "attributes_json": (
                    '{"in_penalty_area":true,"is_box_touch_candidate":true,'
                    '"penalty_presence_alone":false,"has_possession_or_contact":true}'
                ),
            }
        ],
    }
    if name == "passes_only":
        return {
            "pass_outcomes": sources["pass_outcomes"],
            "reception_candidates": sources["reception_candidates"],
        }
    return sources


def pipeline_fixture(name: str = "full_package") -> dict[str, Any]:
    sources = source_events_fixture(name if name != "nearby_dup" else "duplicate_overlap")
    # pull run/video from first row
    first_rows = next(iter(sources.values()))
    run_id = str(first_rows[0]["run_id"])
    video_id = str(first_rows[0]["video_id"])
    return {
        "name": name,
        "run_id": run_id,
        "video_id": video_id,
        "replay_contexts": replay_contexts_fixture("mixed"),
        "sources": sources,
        "attack_periods": [
            {
                "period_id": "period_1",
                "half_id": "first_half",
                "anonymous_team_id": "anon_team_a",
                "config_direction": "toward_goal_b",
                "apply_half_boundary_flip": False,
            },
            {
                "period_id": "period_2",
                "half_id": "second_half",
                "anonymous_team_id": "anon_team_a",
                "config_direction": "toward_goal_b",
                "apply_half_boundary_flip": True,
            },
        ],
        "interaction_coverage": 0.85,
    }


def load_fixture(name: str) -> Mapping[str, Any]:
    if name.startswith("replay_"):
        return {"contexts": replay_contexts_fixture(name.replace("replay_", "", 1))}
    if name.startswith("sources_"):
        return {"sources": source_events_fixture(name.replace("sources_", "", 1))}
    return pipeline_fixture(name)


__all__ = [
    "replay_contexts_fixture",
    "source_events_fixture",
    "pipeline_fixture",
    "load_fixture",
]
