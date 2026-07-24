"""Stage 11B synthetic possession-transition fixtures for pass/reception."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id


def _base(run_id: str | None = None) -> dict[str, str]:
    return {"run_id": run_id or generate_run_id(), "video_id": "video_synth_01"}


def completed_pass_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": 2,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "start_x_m": 30.0,
                "start_y_m": 34.0,
                "end_x_m": 55.0,
                "end_y_m": 40.0,
                "same_team": True,
                "contact_candidate_ids": ["contact_01"],
                "evidence_refs": ["ev_01"],
                "cut_or_replay": False,
                "hard_gap": False,
                "owner_change_alone": False,
                "playability_status": "playable",
                "calibration_status": "valid",
                "possession_hypothesis_id": "poss_01",
                "to_possession_hypothesis_id": "poss_02",
            }
        ],
    }


def incomplete_pass_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": None,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "start_x_m": 40.0,
                "start_y_m": 20.0,
                "end_x_m": 70.0,
                "end_y_m": 50.0,
                "same_team": False,
                "contact_candidate_ids": [],
                "evidence_refs": ["ev_incomplete"],
                "cut_or_replay": False,
                "hard_gap": False,
                "owner_change_alone": False,
                "playability_status": "playable",
                "calibration_status": "valid",
            }
        ],
    }


def owner_change_alone_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": 3,
                "start_time_us": 1_000_000,
                "end_time_us": 1_500_000,
                "start_x_m": 20.0,
                "start_y_m": 30.0,
                "end_x_m": 22.0,
                "end_y_m": 31.0,
                "same_team": True,
                "contact_candidate_ids": [],
                "evidence_refs": [],
                "owner_change_alone": True,
                "cut_or_replay": False,
                "hard_gap": False,
                "playability_status": "playable",
                "calibration_status": "valid",
            }
        ],
    }


def cut_replay_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": 2,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "start_x_m": 30.0,
                "start_y_m": 34.0,
                "end_x_m": 60.0,
                "end_y_m": 40.0,
                "same_team": True,
                "contact_candidate_ids": ["contact_cut"],
                "evidence_refs": ["ev_cut"],
                "cut_or_replay": True,
                "hard_gap": False,
                "owner_change_alone": False,
                "playability_status": "replay",
                "calibration_status": "valid",
            }
        ],
    }


def long_pass_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": 2,
                "start_time_us": 1_000_000,
                "end_time_us": 2_500_000,
                "start_x_m": 10.0,
                "start_y_m": 34.0,
                "end_x_m": 80.0,
                "end_y_m": 40.0,
                "same_team": True,
                "contact_candidate_ids": ["contact_long"],
                "evidence_refs": ["ev_long"],
                "cut_or_replay": False,
                "hard_gap": False,
                "owner_change_alone": False,
                "playability_status": "playable",
                "calibration_status": "valid",
            }
        ],
    }


def hard_gap_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "transitions": [
            {
                "from_owner_track_id": 1,
                "to_owner_track_id": 2,
                "start_time_us": 1_000_000,
                "end_time_us": 5_000_000,
                "start_x_m": 30.0,
                "start_y_m": 34.0,
                "end_x_m": 50.0,
                "end_y_m": 34.0,
                "same_team": True,
                "contact_candidate_ids": ["contact_gap"],
                "evidence_refs": ["ev_gap"],
                "cut_or_replay": False,
                "hard_gap": True,
                "owner_change_alone": False,
                "playability_status": "playable",
                "calibration_status": "valid",
            }
        ],
    }


FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "completed_pass": completed_pass_fixture,
    "incomplete_pass": incomplete_pass_fixture,
    "owner_change_alone": owner_change_alone_fixture,
    "cut_replay": cut_replay_fixture,
    "long_pass": long_pass_fixture,
    "hard_gap": hard_gap_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown pass fixture: {name}")
    return FIXTURES[name](run_id=run_id)


__all__ = [
    "FIXTURES",
    "completed_pass_fixture",
    "incomplete_pass_fixture",
    "owner_change_alone_fixture",
    "cut_replay_fixture",
    "long_pass_fixture",
    "hard_gap_fixture",
    "load_fixture",
]
