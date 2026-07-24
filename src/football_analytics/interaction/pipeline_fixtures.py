"""Stage 10D synthetic fixtures for interaction fusion E2E scenarios."""

from __future__ import annotations

from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.interaction.possession_fixtures import (
    contested_fixture,
    hard_gap_fixture,
    loose_ball_fixture,
    missing_ball_possession_fixture,
    nearest_not_owner_fixture,
    predicted_ball_fixture,
    provisional_control_fixture,
    replay_terminate_fixture,
)
from football_analytics.interaction.proximity_fixtures import (
    controlled_carry_fixture,
    nearest_false_owner_fixture,
)


def _base(run_id: str | None = None) -> dict[str, str]:
    return {
        "run_id": run_id or generate_run_id(),
        "video_id": "video_synth_10d",
    }


def _pt(
    ids: dict[str, str],
    *,
    human_track_id: int,
    ball_track_id: int | None,
    frame_index: int,
    video_time_us: int,
    human_image_x: float,
    human_image_y: float,
    ball_image_x: float,
    ball_image_y: float,
    **kwargs: Any,
) -> dict[str, Any]:
    row = {
        **ids,
        "human_track_id": human_track_id,
        "ball_track_id": ball_track_id,
        "frame_index": frame_index,
        "video_time_us": video_time_us,
        "human_image_x": human_image_x,
        "human_image_y": human_image_y,
        "ball_image_x": ball_image_x,
        "ball_image_y": ball_image_y,
        "human_pitch_x_m": kwargs.pop("human_pitch_x_m", 40.0),
        "human_pitch_y_m": kwargs.pop("human_pitch_y_m", 34.0),
        "ball_pitch_x_m": kwargs.pop("ball_pitch_x_m", 40.3),
        "ball_pitch_y_m": kwargs.pop("ball_pitch_y_m", 34.1),
        "human_observation_state": "observed",
        "ball_observation_state": "observed",
        "ball_air_state": "grounded",
        "ball_candidate_status": "primary",
        "calibration_status": "valid",
        "playability_status": "playable",
        "target_relationship": "confirmed_target",
        "uncertainty": 0.3,
        "human_observation_id": f"hobs_{frame_index}",
        "ball_observation_id": f"bobs_{frame_index}",
        "human_projection_id": "hproj_01",
        "ball_projection_id": "bproj_01",
        "identity_assignment_id": "asn_confirmed_01",
        "analysis_window_id": "aw_01",
    }
    row.update(kwargs)
    return row


def rapid_owner_change_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points: list[dict[str, Any]] = []
    # Player 1 controls then player 2 takes over (with contact support each)
    for i in range(4):
        points.append(
            _pt(
                ids,
                human_track_id=1,
                ball_track_id=100,
                frame_index=i,
                video_time_us=i * 40_000,
                human_image_x=100.0 + i,
                human_image_y=200.0,
                ball_image_x=105.0 + i,
                ball_image_y=202.0,
                human_pitch_x_m=40.0 + 0.2 * i,
                human_pitch_y_m=34.0,
                ball_pitch_x_m=40.3 + 0.2 * i,
                ball_pitch_y_m=34.1,
            )
        )
    for i in range(4):
        points.append(
            _pt(
                ids,
                human_track_id=2,
                ball_track_id=100,
                frame_index=10 + i,
                video_time_us=200_000 + i * 40_000,
                human_image_x=150.0 + i,
                human_image_y=200.0,
                ball_image_x=155.0 + i,
                ball_image_y=202.0,
                human_pitch_x_m=50.0 + 0.2 * i,
                human_pitch_y_m=34.0,
                ball_pitch_x_m=50.3 + 0.2 * i,
                ball_pitch_y_m=34.1,
            )
        )
    return {**ids, "points": points, "label": "rapid_owner_change"}


def target_revoked_fixture(run_id: str | None = None) -> dict[str, Any]:
    fx = controlled_carry_fixture(run_id)
    fx["video_id"] = "video_synth_10d"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10d"
        p["target_relationship"] = "non_target"
        p["identity_assignment_id"] = "asn_revoked_01"
    fx["label"] = "target_revoked"
    fx["identity_status"] = "revoked"
    return fx


def target_confirmed_fixture(run_id: str | None = None) -> dict[str, Any]:
    fx = provisional_control_fixture(run_id)
    fx["video_id"] = "video_synth_10d"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10d"
        p["target_relationship"] = "confirmed_target"
    fx["label"] = "target_confirmed"
    fx["identity_status"] = "confirmed"
    return fx


def two_player_ambiguity_fixture(run_id: str | None = None) -> dict[str, Any]:
    return contested_fixture(run_id)


def false_nearest_fixture(run_id: str | None = None) -> dict[str, Any]:
    fx = nearest_false_owner_fixture(run_id)
    fx["video_id"] = "video_synth_10d"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10d"
    fx["label"] = "false_nearest"
    return fx


def cut_replay_fixture(run_id: str | None = None) -> dict[str, Any]:
    return replay_terminate_fixture(run_id)


FIXTURES = {
    "controlled_carry": lambda run_id=None: _retag(
        controlled_carry_fixture(run_id), "controlled_carry"
    ),
    "loose_ball": loose_ball_fixture,
    "contested_ball": contested_fixture,
    "two_player_ambiguity": two_player_ambiguity_fixture,
    "missing_ball": missing_ball_possession_fixture,
    "predicted_ball": predicted_ball_fixture,
    "cut_replay": cut_replay_fixture,
    "rapid_owner_change": rapid_owner_change_fixture,
    "false_nearest": false_nearest_fixture,
    "target_confirmed": target_confirmed_fixture,
    "target_revoked": target_revoked_fixture,
    "hard_gap": hard_gap_fixture,
    "nearest_not_owner": nearest_not_owner_fixture,
    "provisional_control": provisional_control_fixture,
}


def _retag(fx: dict[str, Any], label: str) -> dict[str, Any]:
    fx = dict(fx)
    fx["video_id"] = "video_synth_10d"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10d"
    fx["label"] = label
    return fx


def load_pipeline_fixture(name: str, *, run_id: str | None = None) -> dict[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown pipeline fixture: {name}")
    return FIXTURES[name](run_id)


__all__ = [
    "FIXTURES",
    "load_pipeline_fixture",
    "rapid_owner_change_fixture",
    "target_revoked_fixture",
    "target_confirmed_fixture",
    "two_player_ambiguity_fixture",
    "false_nearest_fixture",
    "cut_replay_fixture",
]
