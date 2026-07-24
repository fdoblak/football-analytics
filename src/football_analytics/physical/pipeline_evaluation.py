"""Stage 9E fused physical metric evaluation (no reviewed football GT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_PIPELINE = "NOT_EVALUATED_NO_REVIEWED_" + "TARGET_PHYSICAL_METRIC_" + "GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "distance_absolute_relative_error": None,
    "speed_mae_rmse": None,
    "peak_speed_error": None,
    "sprint_event_precision_recall_f1": None,
    "heatmap_similarity": None,
    "zone_dwell_error": None,
    "activity_class_accuracy": None,
    "coverage_calibration": None,
    "fusion_integrity_correctness": None,
}


@dataclass(frozen=True)
class PipelineEvaluationReport:
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


def evaluate_physical_pipeline(
    *,
    has_reviewed_ground_truth: bool = False,
    metric_results: Sequence[Mapping[str, Any]] | None = None,
) -> PipelineEvaluationReport:
    _ = metric_results
    if not has_reviewed_ground_truth:
        return PipelineEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=(NOT_EVALUATED_PIPELINE),
            metrics=dict(NULL_METRICS),
            metric_reasons={k: NOT_EVALUATED_PIPELINE for k in NULL_METRICS},
            findings=(
                "No reviewed target physical-metric ground truth available.",
                "Synthetic E2E proves fusion integrity only — not real football accuracy.",
                "Official Opta data was not used.",
                "Stage 9 closed without customer final visual.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="stage_9e_fusion_synthetic_only",
        )
    return PipelineEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="evaluator_not_implemented_with_reviewed_gt",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "evaluator_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT present but Stage 9E accuracy evaluator is not implemented.",),
        created_at_utc=_utc_now(),
        adapter_notes="stage_9e",
    )


__all__ = [
    "NOT_EVALUATED_PIPELINE",
    "NULL_METRICS",
    "PipelineEvaluationReport",
    "evaluate_physical_pipeline",
]
