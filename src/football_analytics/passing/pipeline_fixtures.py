"""Stage 11D synthetic E2E fixtures for passing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.passing.pass_fixtures import (
    completed_pass_fixture,
    cut_replay_fixture,
    hard_gap_fixture,
    incomplete_pass_fixture,
    long_pass_fixture,
    owner_change_alone_fixture,
)


def _with_touches(fx: Mapping[str, Any], *, eligible: bool = True) -> dict[str, Any]:
    out = dict(fx)
    if eligible:
        out["touch_inputs"] = [
            {
                "human_track_id": 1,
                "touch_time_us": 3_000_000,
                "touch_x_m": 100.0,
                "touch_y_m": 34.0,
                "in_penalty_area": True,
                "has_possession_or_contact": True,
                "has_pitch_mapping": True,
                "playability_status": "playable",
                "contact_candidate_ids": ["contact_box"],
                "evidence_refs": ["ev_box"],
                "calibration_status": "valid",
            }
        ]
    else:
        out["touch_inputs"] = [
            {
                "human_track_id": 1,
                "touch_time_us": 3_000_000,
                "touch_x_m": 100.0,
                "touch_y_m": 34.0,
                "in_penalty_area": True,
                "has_possession_or_contact": False,
                "has_pitch_mapping": True,
                "playability_status": "playable",
                "contact_candidate_ids": [],
                "evidence_refs": [],
                "calibration_status": "valid",
            }
        ]
    return out


def completed_with_box(*, run_id: str | None = None) -> dict[str, Any]:
    return _with_touches(completed_pass_fixture(run_id=run_id), eligible=True)


def presence_only_box(*, run_id: str | None = None) -> dict[str, Any]:
    return _with_touches(completed_pass_fixture(run_id=run_id), eligible=False)


def multi_transition(*, run_id: str | None = None) -> dict[str, Any]:
    a = completed_pass_fixture(run_id=run_id)
    b = long_pass_fixture(run_id=a["run_id"])
    transitions = list(a["transitions"]) + list(b["transitions"])
    # stagger second transition times
    transitions[1] = dict(transitions[1])
    transitions[1]["start_time_us"] = 4_000_000
    transitions[1]["end_time_us"] = 5_500_000
    return _with_touches(
        {"run_id": a["run_id"], "video_id": a["video_id"], "transitions": transitions},
        eligible=True,
    )


PIPELINE_FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "completed_with_box": completed_with_box,
    "incomplete_pass": lambda **kw: _with_touches(incomplete_pass_fixture(**kw), eligible=False),
    "owner_change_alone": lambda **kw: _with_touches(
        owner_change_alone_fixture(**kw), eligible=False
    ),
    "cut_replay": lambda **kw: _with_touches(cut_replay_fixture(**kw), eligible=False),
    "hard_gap": lambda **kw: _with_touches(hard_gap_fixture(**kw), eligible=False),
    "long_pass": lambda **kw: _with_touches(long_pass_fixture(**kw), eligible=True),
    "presence_only_box": presence_only_box,
    "multi_transition": multi_transition,
}


def load_pipeline_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in PIPELINE_FIXTURES:
        raise KeyError(f"unknown pipeline fixture: {name}")
    return PIPELINE_FIXTURES[name](run_id=run_id)


__all__ = [
    "PIPELINE_FIXTURES",
    "load_pipeline_fixture",
    "completed_with_box",
    "presence_only_box",
    "multi_transition",
]
