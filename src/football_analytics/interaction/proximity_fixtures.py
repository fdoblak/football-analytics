"""Synthetic fixtures for Stage 10B proximity / contact baseline."""

from __future__ import annotations

from typing import Any

from football_analytics.core.run_id import generate_run_id


def _base(run_id: str | None = None) -> dict[str, str]:
    return {
        "run_id": run_id or generate_run_id(),
        "video_id": "video_synth_10b",
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


def controlled_carry_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points = [
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
        for i in range(5)
    ]
    return {**ids, "points": points, "label": "controlled_carry"}


def nearest_false_owner_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    t = 0
    points = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=100,
            frame_index=0,
            video_time_us=t,
            human_image_x=100.0,
            human_image_y=200.0,
            ball_image_x=130.0,
            ball_image_y=200.0,
            human_pitch_x_m=40.0,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=42.0,
            ball_pitch_y_m=34.0,
        ),
        _pt(
            ids,
            human_track_id=2,
            ball_track_id=100,
            frame_index=0,
            video_time_us=t,
            human_image_x=120.0,
            human_image_y=200.0,
            ball_image_x=130.0,
            ball_image_y=200.0,
            human_pitch_x_m=41.5,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=42.0,
            ball_pitch_y_m=34.0,
        ),
    ]
    return {**ids, "points": points, "label": "false_nearest"}


def single_frame_fixture(run_id: str | None = None) -> dict[str, Any]:
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
    return {**ids, "points": points, "label": "single_frame"}


def missing_ball_fixture(run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    points = [
        _pt(
            ids,
            human_track_id=1,
            ball_track_id=None,
            frame_index=0,
            video_time_us=0,
            human_image_x=100.0,
            human_image_y=200.0,
            ball_image_x=100.0,
            ball_image_y=200.0,
            ball_observation_state="missing",
            ball_candidate_status="missing",
            ball_air_state="unknown",
            calibration_status="valid",
        )
    ]
    return {**ids, "points": points, "label": "missing_ball"}


def replay_fixture(run_id: str | None = None) -> dict[str, Any]:
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
            playability_status="replay",
        )
        for i in range(3)
    ]
    return {**ids, "points": points, "label": "replay"}


def airborne_unknown_fixture(run_id: str | None = None) -> dict[str, Any]:
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
            ball_image_y=180.0,
            human_pitch_x_m=40.0,
            human_pitch_y_m=34.0,
            ball_pitch_x_m=40.3,
            ball_pitch_y_m=34.1,
            ball_air_state="unknown",
        )
        for i in range(3)
    ]
    return {**ids, "points": points, "label": "airborne_unknown"}


FIXTURES = {
    "controlled_carry": controlled_carry_fixture,
    "false_nearest": nearest_false_owner_fixture,
    "single_frame": single_frame_fixture,
    "missing_ball": missing_ball_fixture,
    "replay": replay_fixture,
    "airborne_unknown": airborne_unknown_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> dict[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown fixture: {name}")
    return FIXTURES[name](run_id)


__all__ = [
    "FIXTURES",
    "load_fixture",
    "controlled_carry_fixture",
    "nearest_false_owner_fixture",
    "single_frame_fixture",
    "missing_ball_fixture",
    "replay_fixture",
    "airborne_unknown_fixture",
]
