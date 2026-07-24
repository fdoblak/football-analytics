"""Stage 13 evaluation helpers — no reviewed GT ⇒ NOT_EVALUATED."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.events.types import NOT_EVALUATED_EVENTS

NOT_EVALUATED = NOT_EVALUATED_EVENTS


def evaluate_events(
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None = None,
    reviewed_ground_truth: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _ = ledger_rows
    if reviewed_ground_truth:
        return {
            "schema_version": 1,
            "evaluation_status": "REVIEWED_GT_PRESENT_BUT_NOT_SCORED",
            "notes": "Stage 13 does not claim real football accuracy even with GT present",
            "real_football_accuracy_validated": False,
            "opta_accuracy_validated": False,
            "metric_origin": "project_generated",
        }
    return {
        "schema_version": 1,
        "evaluation_status": NOT_EVALUATED,
        "notes": "No reviewed target-events ground truth",
        "real_football_accuracy_validated": False,
        "opta_accuracy_validated": False,
        "metric_origin": "project_generated",
    }


__all__ = ["NOT_EVALUATED", "NOT_EVALUATED_EVENTS", "evaluate_events"]
