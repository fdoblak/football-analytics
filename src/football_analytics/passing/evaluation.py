"""Passing evaluation stubs (Stage 11A — no reviewed GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.passing.types import NOT_EVALUATED_PASSING

NULL_METRICS: dict[str, Any] = {
    "pass_accuracy": None,
    "pass_completion_rate": None,
    "reception_precision_recall": None,
    "long_pass_ratio": None,
    "box_touch_precision_recall": None,
    "progression_agreement": None,
    "coverage_calibration": None,
    "not_evaluable_correctness": None,
}

METRIC_REASON = NOT_EVALUATED_PASSING


@dataclass(frozen=True)
class PassingEvaluationReport:
    status: str
    ground_truth_evaluation_status: str
    metrics: dict[str, Any]
    metric_reasons: dict[str, str]
    findings: tuple[str, ...]
    created_at_utc: str
    adapter_notes: str

    def to_dict(
        self, *, run_id: str, video_id: str, config_fingerprint: str | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "video_id": video_id,
            "config_fingerprint": config_fingerprint,
            "status": self.status,
            "ground_truth_evaluation_status": self.ground_truth_evaluation_status,
            "metrics": dict(self.metrics),
            "metric_reasons": dict(self.metric_reasons),
            "findings": list(self.findings),
            "adapter_notes": self.adapter_notes,
            "created_at_utc": self.created_at_utc,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def evaluate_passing(
    *,
    passes: Sequence[Mapping[str, Any]] | None = None,
    receptions: Sequence[Mapping[str, Any]] | None = None,
    outcomes: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> PassingEvaluationReport:
    _ = passes, receptions, outcomes
    if not has_reviewed_ground_truth:
        reasons = {k: METRIC_REASON for k in NULL_METRICS}
        eval_status = NOT_EVALUATED_PASSING
        return PassingEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons=reasons,
            findings=(
                "No reviewed passing ground truth available.",
                "Stage 11 synthetic fixtures are not real accuracy.",
                "No Opta or official event accuracy claim is made.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="contracts_and_synthetic_baseline_stage_11",
        )
    return PassingEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="reviewed_gt_adapter_not_implemented",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "adapter_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT path is not implemented in Stage 11.",),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed_gt_not_implemented",
    )


__all__ = [
    "NOT_EVALUATED_PASSING",
    "NULL_METRICS",
    "PassingEvaluationReport",
    "evaluate_passing",
]
