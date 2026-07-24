"""BAS/GSR evaluation smoke tests."""

from __future__ import annotations

from football_analytics.acceptance.evaluation.bas_metrics import match_events
from football_analytics.acceptance.evaluation.gsr_metrics import compare_trajectories
from football_analytics.acceptance.evaluation.metric_taxonomy import (
    classify_metric,
    taxonomy_table,
)


def test_bas_match_smoke() -> None:
    ref = [
        {"label": "Pass", "half": 1, "t_ms": 1000, "player_id": "1"},
        {"label": "Pass", "half": 1, "t_ms": 5000, "player_id": "1"},
    ]
    pred = [
        {"label": "Pass", "half": 1, "t_ms": 1100, "player_id": "1"},
        {"label": "Pass", "half": 1, "t_ms": 9000, "player_id": "2"},
    ]
    out = match_events(predicted=pred, reference=ref, tolerance_ms=1000)
    assert out["tp"] == 1
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert "pass_accuracy" in out["not_evaluable_outcomes"]


def test_gsr_trajectory_smoke() -> None:
    ref = [{"half": 1, "t_ms": 0, "x_m": 0.0, "y_m": 0.0}]
    pred = [{"half": 1, "t_ms": 40, "x_m": 0.5, "y_m": 0.0}]
    out = compare_trajectories(predicted=pred, reference=ref)
    assert out["matched"] == 1
    assert out["detection_map"] == "not_evaluable"


def test_taxonomy_pass_accuracy_not_evaluable() -> None:
    assert classify_metric("pass_accuracy") == "not_evaluable"
    assert any(r["metric"] == "pass_events" for r in taxonomy_table())
