"""Physical metric evaluation stubs (Stage 9A — no real GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.physical.types import NOT_EVALUATED_PHYSICAL

NULL_METRICS: dict[str, Any] = {
    "position_error_m": None,
    "distance_absolute_relative_error": None,
    "speed_mae_rmse": None,
    "peak_speed_error": None,
    "sprint_event_precision_recall_f1": None,
    "sprint_timing_error": None,
    "heatmap_similarity": None,
    "coverage_calibration": None,
    "false_sprint_rate": None,
    "not_evaluable_correctness": None,
}

METRIC_REASON = NOT_EVALUATED_PHYSICAL


@dataclass(frozen=True)
class PhysicalEvaluationReport:
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


def evaluate_physical_metrics(
    *,
    samples: Sequence[Mapping[str, Any]] | None = None,
    metric_results: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> PhysicalEvaluationReport:
    _ = samples, metric_results
    if not has_reviewed_ground_truth:
        reasons = {k: METRIC_REASON for k in NULL_METRICS}
        eval_status = NOT_EVALUATED_PHYSICAL
        return PhysicalEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons=reasons,
            findings=(
                "No reviewed trajectory/physical-metric ground truth available.",
                "Stage 9A is contracts-only; synthetic fixtures are not real accuracy.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="contracts_only_stage_9a",
        )
    return PhysicalEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="evaluator_not_implemented_with_reviewed_gt",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "evaluator_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT present but Stage 9A does not compute accuracy metrics.",),
        created_at_utc=_utc_now(),
        adapter_notes="contracts_only_stage_9a",
    )


__all__ = [
    "NOT_EVALUATED_PHYSICAL",
    "NULL_METRICS",
    "PhysicalEvaluationReport",
    "evaluate_physical_metrics",
]
