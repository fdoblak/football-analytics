"""Stage 12D synthetic aerial / clearance fixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id


def _base(run_id: str | None = None) -> dict[str, str]:
    return {"run_id": run_id or generate_run_id(), "video_id": "video_synth_01"}


def monocular_aerial_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "kind": "aerial",
                "target_track_id": 1,
                "opponent_track_id": 10,
                "start_time_us": 1_000_000,
                "end_time_us": 1_500_000,
                "x_m": 60.0,
                "y_m": 34.0,
                "monocular_only": True,
                "exact_3d_height_claimed": False,
                "exact_3d_height_m": None,
                "contact_candidate_ids": ["contact_air"],
                "evidence_refs": ["ev_air"],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


def clearance_with_evidence_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "kind": "clearance",
                "target_track_id": 1,
                "opponent_track_id": None,
                "start_time_us": 2_000_000,
                "end_time_us": 2_500_000,
                "x_m": 20.0,
                "y_m": 34.0,
                "long_ball_alone": False,
                "ball_distance_m": 35.0,
                "defensive_context": True,
                "contact_candidate_ids": ["contact_clr"],
                "evidence_refs": ["ev_clr"],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


def long_ball_alone_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "kind": "clearance",
                "target_track_id": 1,
                "opponent_track_id": None,
                "start_time_us": 2_000_000,
                "end_time_us": 2_500_000,
                "x_m": 40.0,
                "y_m": 34.0,
                "long_ball_alone": True,
                "ball_distance_m": 40.0,
                "defensive_context": False,
                "contact_candidate_ids": [],
                "evidence_refs": [],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "monocular_aerial": monocular_aerial_fixture,
    "clearance_with_evidence": clearance_with_evidence_fixture,
    "long_ball_alone": long_ball_alone_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown aerial fixture: {name}")
    return FIXTURES[name](run_id=run_id)


__all__ = [
    "FIXTURES",
    "monocular_aerial_fixture",
    "clearance_with_evidence_fixture",
    "long_ball_alone_fixture",
    "load_fixture",
]
