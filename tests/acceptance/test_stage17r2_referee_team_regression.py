"""Regression: referee/staff must never receive yellow/white team assignment."""

from __future__ import annotations

from football_analytics.acceptance.stage17r2_recovery import (
    count_non_player_team_assignments,
    normalize_team_for_role,
)


def test_referee_team_must_be_null() -> None:
    assert normalize_team_for_role("referee", "yellow") is None
    assert normalize_team_for_role("referee", "white") is None
    assert normalize_team_for_role("staff", "yellow") is None


def test_player_keeps_yellow_white() -> None:
    assert normalize_team_for_role("player", "yellow") == "yellow"
    assert normalize_team_for_role("player", "white") == "white"


def test_count_non_player_team_assignments_detects_bug() -> None:
    humans = [
        {"role": "referee", "team": "white", "bbox": [1200, 400, 30, 80]},
        {"role": "player", "team": "yellow", "bbox": [100, 400, 40, 90]},
    ]
    assert count_non_player_team_assignments(humans) == 1
    fixed = [
        {**humans[0], "team": normalize_team_for_role("referee", humans[0]["team"])},
        humans[1],
    ]
    assert count_non_player_team_assignments(fixed) == 0


def test_own_video_holdout_seed_has_no_referee_team() -> None:
    """Seed reviewed holdout decisions must not assign team to referee/staff."""
    import json
    from pathlib import Path

    path = Path(
        "/home/fdoblak/workspace/own_video_analysis/stage17r2/review_decisions/decisions_human.jsonl"
    )
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        assert count_non_player_team_assignments(d.get("humans", [])) == 0
