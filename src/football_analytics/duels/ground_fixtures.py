"""Stage 12C synthetic ground/tackle/recovery/turnover fixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id


def _base(run_id: str | None = None) -> dict[str, str]:
    return {"run_id": run_id or generate_run_id(), "video_id": "video_synth_01"}


def contested_ground_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "kind": "ground_duel",
                "target_track_id": 1,
                "opponent_track_id": 8,
                "start_time_us": 1_000_000,
                "end_time_us": 2_000_000,
                "x_m": 50.0,
                "y_m": 30.0,
                "nearest_switch_alone": False,
                "contested_possession": True,
                "implies_duel_outcome": False,
                "outcome": "uncertain",
                "contact_candidate_ids": ["contact_g"],
                "evidence_refs": ["ev_g"],
                "cut_or_replay": False,
                "hard_gap": False,
            },
            {
                "kind": "tackle",
                "target_track_id": 1,
                "opponent_track_id": 8,
                "start_time_us": 1_200_000,
                "end_time_us": 1_800_000,
                "x_m": 50.0,
                "y_m": 30.0,
                "related_ground_duel_candidate_id": "gduel_01",
                "implies_tackle": True,
                "implies_tackle_success": False,
                "outcome": "uncertain",
                "contact_candidate_ids": ["contact_t"],
                "evidence_refs": ["ev_t"],
                "cut_or_replay": False,
                "hard_gap": False,
            },
            {
                "kind": "turnover",
                "target_track_id": 1,
                "opponent_track_id": 8,
                "start_time_us": 3_000_000,
                "end_time_us": 3_500_000,
                "x_m": 55.0,
                "y_m": 32.0,
                "related_recovery_event_id": "rec_01",
                "implies_turnover": True,
                "outcome": "lost",
                "contact_candidate_ids": ["contact_turn"],
                "evidence_refs": ["ev_turn"],
                "cut_or_replay": False,
                "hard_gap": False,
            },
            {
                "kind": "recovery",
                "target_track_id": 1,
                "opponent_track_id": 8,
                "start_time_us": 3_500_000,
                "end_time_us": 4_000_000,
                "x_m": 56.0,
                "y_m": 33.0,
                "related_turnover_event_id": "turn_01",
                "implies_recovery": True,
                "outcome": "recovered",
                "contact_candidate_ids": ["contact_rec"],
                "evidence_refs": ["ev_rec"],
                "cut_or_replay": False,
                "hard_gap": False,
            },
        ],
    }


def nearest_switch_alone_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    ids = _base(run_id)
    return {
        **ids,
        "contexts": [
            {
                "kind": "ground_duel",
                "target_track_id": 1,
                "opponent_track_id": 9,
                "start_time_us": 1_000_000,
                "end_time_us": 1_500_000,
                "x_m": 40.0,
                "y_m": 20.0,
                "nearest_switch_alone": True,
                "contested_possession": False,
                "implies_duel_outcome": False,
                "outcome": "not_evaluable",
                "contact_candidate_ids": [],
                "evidence_refs": [],
                "cut_or_replay": False,
                "hard_gap": False,
            }
        ],
    }


FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "contested_ground": contested_ground_fixture,
    "nearest_switch_alone": nearest_switch_alone_fixture,
}


def load_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown ground fixture: {name}")
    return FIXTURES[name](run_id=run_id)


__all__ = [
    "FIXTURES",
    "contested_ground_fixture",
    "nearest_switch_alone_fixture",
    "load_fixture",
]
