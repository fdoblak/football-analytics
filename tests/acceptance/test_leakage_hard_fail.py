"""Leakage hard-fail tests for Stage 16."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analytics.acceptance.contracts import (
    EXTERNAL_CC_BY_REFERENCE_GT,
    LEAKAGE_SEPARATION_VIOLATION,
)
from football_analytics.acceptance.leakage import (
    LeakageError,
    validate_event_ledger_not_copied_from_gt,
    validate_no_gt_under_predictions,
    validate_prediction_bundle_not_gt,
    validate_run_dir,
)


def test_gt_marker_under_predictions_hard_fails(tmp_path: Path) -> None:
    run = tmp_path / "run"
    pred = run / "predictions"
    pred.mkdir(parents=True)
    (pred / "bad.json").write_text(
        json.dumps({"annotation_provenance": EXTERNAL_CC_BY_REFERENCE_GT}),
        encoding="utf-8",
    )
    with pytest.raises(LeakageError) as exc:
        validate_no_gt_under_predictions(run)
    assert LEAKAGE_SEPARATION_VIOLATION in str(exc.value)


def test_prediction_bundle_rejects_gt_provenance() -> None:
    with pytest.raises(LeakageError):
        validate_prediction_bundle_not_gt({"provenance": EXTERNAL_CC_BY_REFERENCE_GT})


def test_event_ledger_copy_from_gt_hard_fails() -> None:
    ref = [
        {
            "label": "Pass",
            "t_ms": 1000,
            "player_id": "1",
            "source": EXTERNAL_CC_BY_REFERENCE_GT,
        }
    ]
    pred = [
        {
            "label": "Pass",
            "t_ms": 1000,
            "player_id": "1",
            "source": EXTERNAL_CC_BY_REFERENCE_GT,
        }
    ]
    with pytest.raises(LeakageError):
        validate_event_ledger_not_copied_from_gt(pred, ref)


def test_validate_run_dir_pass(tmp_path: Path) -> None:
    run = tmp_path / "run"
    receipt = validate_run_dir(run)
    assert receipt["status"] == "PASS"
    assert (run / "evaluation" / "leakage_validation.json").is_file()
