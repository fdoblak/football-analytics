"""Stage 12B synthetic take-on fixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id


def _base(run_id: str | None = None) -> dict[str, str]:
    return {"run_id": run_id or generate_run_id(), "video_id": "video_synth_01"}


def successful_take_on_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "target_track_id": 1,
                "opponent_track_id": 7,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "x_m": 40.0,
                "y_m": 34.0,
                "carry_distance_m": 4.0,
                "nearby_opponent_alone": False,
                "direction_change_alone": False,
                "has_possession_or_contact": True,
                "beaten_opponent": True,
                "contact_candidate_ids": ["contact_take"],
                "evidence_refs": ["ev_take"],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


def nearby_opponent_alone_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "target_track_id": 1,
                "opponent_track_id": 7,
                "start_time_us": 1_000_000,
                "end_time_us": 1_500_000,
                "x_m": 40.0,
                "y_m": 34.0,
                "carry_distance_m": 0.2,
                "nearby_opponent_alone": True,
                "direction_change_alone": False,
                "has_possession_or_contact": False,
                "beaten_opponent": False,
                "contact_candidate_ids": [],
                "evidence_refs": [],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


def cut_replay_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "target_track_id": 1,
                "opponent_track_id": 7,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "x_m": 40.0,
                "y_m": 34.0,
                "carry_distance_m": 3.0,
                "nearby_opponent_alone": False,
                "direction_change_alone": False,
                "has_possession_or_contact": True,
                "beaten_opponent": False,
                "contact_candidate_ids": ["contact_cut"],
                "evidence_refs": ["ev_cut"],
                "cut_or_replay": True,
                "hard_gap": False,
                "playability_status": "replay",
            }
        ],
    }


FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "successful_take_on": successful_take_on_fixture,
    "nearby_opponent_alone": nearby_opponent_alone_fixture,
    "cut_replay": cut_replay_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown take-on fixture: {name}")
    return FIXTURES[name](run_id=run_id)


__all__ = [
    "FIXTURES",
    "successful_take_on_fixture",
    "nearby_opponent_alone_fixture",
    "cut_replay_fixture",
    "load_fixture",
]
