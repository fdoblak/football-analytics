"""Synthetic fixtures for Stage 10C possession / control baseline."""

from __future__ import annotations

from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.interaction.proximity_fixtures import (
    controlled_carry_fixture,
    missing_ball_fixture,
    replay_fixture,
)


def _base(run_id: str | None = None) -> dict[str, str]:
    return {
        "run_id": run_id or generate_run_id(),
        "video_id": "video_synth_10c",
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
    human_pitch_x_m: float | None = None,
    human_pitch_y_m: float | None = None,
    ball_pitch_x_m: float | None = None,
    ball_pitch_y_m: float | None = None,
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
        "human_pitch_x_m": human_pitch_x_m,
        "human_pitch_y_m": human_pitch_y_m,
        "ball_pitch_x_m": ball_pitch_x_m,
        "ball_pitch_y_m": ball_pitch_y_m,
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


def provisional_control_fixture(run_id: str | None = None) -> dict[str, Any]:
    """Long controlled carry → provisional possession (not confirmed)."""
    fx = controlled_carry_fixture(run_id)
    fx["video_id"] = "video_synth_10c"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10c"
    # Extend for provisional duration + co-motion
    ids = {"run_id": fx["run_id"], "video_id": fx["video_id"]}
    extra = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=100,
            frame_index=5 + i,
            video_time_us=(5 + i) * 40_000,
            human_image_x=105.0 + i,
            human_image_y=200.0,
            ball_image_x=110.0 + i,
            ball_image_y=202.0,
            human_pitch_x_m=41.0 + 0.25 * i,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=41.4 + 0.25 * i,
            ball_pitch_y_m=34.2,
        )
        for i in range(4)
    ]
    fx["points"] = list(fx["points"]) + extra
    fx["label"] = "provisional_control"
    return fx


def contested_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points: list[dict[str, Any]] = []
    for i in range(4):
        t = i * 40_000
        points.append(
            _pt(
                ids,
                human_track_id=1,
                ball_track_id=100,
                frame_index=i,
                video_time_us=t,
                human_image_x=100.0 + i,
                human_image_y=200.0,
                ball_image_x=110.0,
                ball_image_y=200.0,
                human_pitch_x_m=40.0 + 0.1 * i,
                human_pitch_y_m=34.0,
                ball_pitch_x_m=40.8,
                ball_pitch_y_m=34.0,
            )
        )
        points.append(
            _pt(
                ids,
                human_track_id=2,
                ball_track_id=100,
                frame_index=i,
                video_time_us=t,
                human_image_x=115.0 - i,
                human_image_y=200.0,
                ball_image_x=110.0,
                ball_image_y=200.0,
                human_pitch_x_m=41.0 - 0.1 * i,
                human_pitch_y_m=34.0,
                ball_pitch_x_m=40.8,
                ball_pitch_y_m=34.0,
            )
        )
    return {**ids, "points": points, "label": "contested"}


def loose_ball_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=100,
            frame_index=i,
            video_time_us=i * 40_000,
            human_image_x=50.0,
            human_image_y=50.0,
            ball_image_x=400.0,
            ball_image_y=300.0,
            human_pitch_x_m=10.0,
            human_pitch_y_m=10.0,
            ball_pitch_x_m=50.0,
            ball_pitch_y_m=40.0,
        )
        for i in range(3)
    ]
    return {**ids, "points": points, "label": "loose_ball"}


def hard_gap_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points = []
    for i in range(3):
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
    # Hard gap > 500ms then resume
    for i in range(3):
        points.append(
            _pt(
                ids,
                human_track_id=1,
                ball_track_id=100,
                frame_index=10 + i,
                video_time_us=1_000_000 + i * 40_000,
                human_image_x=120.0 + i,
                human_image_y=200.0,
                ball_image_x=125.0 + i,
                ball_image_y=202.0,
                human_pitch_x_m=45.0 + 0.2 * i,
                human_pitch_y_m=34.0,
                ball_pitch_x_m=45.3 + 0.2 * i,
                ball_pitch_y_m=34.1,
            )
        )
    return {**ids, "points": points, "label": "hard_gap"}


def predicted_ball_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=100,
            frame_index=i,
            video_time_us=i * 40_000,
            human_image_x=100.0,
            human_image_y=200.0,
            ball_image_x=105.0,
            ball_image_y=202.0,
            human_pitch_x_m=40.0,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=40.3,
            ball_pitch_y_m=34.1,
            ball_observation_state="predicted",
        )
        for i in range(3)
    ]
    return {**ids, "points": points, "label": "predicted_ball"}


def missing_ball_possession_fixture(run_id: str | None = None) -> dict[str, Any]:
    fx = missing_ball_fixture(run_id)
    fx["video_id"] = "video_synth_10c"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10c"
    fx["label"] = "missing_ball"
    return fx


def nearest_not_owner_fixture(run_id: str | None = None) -> dict[str, Any]:
    """Single-frame nearest human — must not become possession owner."""
    ids = _base(run_id)
    points = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=100,
            frame_index=0,
            video_time_us=0,
            human_image_x=100.0,
            human_image_y=200.0,
            ball_image_x=105.0,
            ball_image_y=202.0,
            human_pitch_x_m=40.0,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=40.3,
            ball_pitch_y_m=34.1,
        )
    ]
    return {**ids, "points": points, "label": "nearest_not_owner"}


def replay_terminate_fixture(run_id: str | None = None) -> dict[str, Any]:
    fx = replay_fixture(run_id)
    fx["video_id"] = "video_synth_10c"
    for p in fx["points"]:
        p["video_id"] = "video_synth_10c"
    fx["label"] = "replay"
    return fx


FIXTURES = {
    "provisional_control": provisional_control_fixture,
    "contested": contested_fixture,
    "loose_ball": loose_ball_fixture,
    "hard_gap": hard_gap_fixture,
    "predicted_ball": predicted_ball_fixture,
    "missing_ball": missing_ball_possession_fixture,
    "nearest_not_owner": nearest_not_owner_fixture,
    "replay": replay_terminate_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> dict[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown fixture: {name}")
    return FIXTURES[name](run_id)


__all__ = [
    "FIXTURES",
    "load_fixture",
    "provisional_control_fixture",
    "contested_fixture",
    "loose_ball_fixture",
    "hard_gap_fixture",
    "predicted_ball_fixture",
    "missing_ball_possession_fixture",
    "nearest_not_owner_fixture",
    "replay_terminate_fixture",
]
