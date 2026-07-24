"""Stage 9D spatial metric evaluation (no reviewed football GT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_SPATIAL = "NOT_EVALUATED_NO_REVIEWED_" + "HEATMAP_ZONE_ACTIVITY_" + "GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "heatmap_similarity": None,
    "zone_dwell_error": None,
    "activity_class_accuracy": None,
    "coverage_calibration": None,
    "not_evaluable_correctness": None,
}


@dataclass(frozen=True)
class SpatialEvaluationReport:
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


def evaluate_spatial_metrics(
    *,
    has_reviewed_ground_truth: bool = False,
    metric_results: Sequence[Mapping[str, Any]] | None = None,
) -> SpatialEvaluationReport:
    _ = metric_results
    if not has_reviewed_ground_truth:
        return SpatialEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=(NOT_EVALUATED_SPATIAL),
            metrics=dict(NULL_METRICS),
            metric_reasons={k: NOT_EVALUATED_SPATIAL for k in NULL_METRICS},
            findings=(
                "No reviewed heatmap/zone/activity ground truth available.",
                "Synthetic fixtures prove math/pipeline only — not real football accuracy.",
                "Activity index is project_generated; not official Opta.",
                "Penalty presence is not ball touch or possession.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="stage_9d_synthetic_math_only",
        )
    return SpatialEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="evaluator_not_implemented_with_reviewed_gt",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "evaluator_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT present but Stage 9D accuracy evaluator is not implemented.",),
        created_at_utc=_utc_now(),
        adapter_notes="stage_9d",
    )


__all__ = [
    "NOT_EVALUATED_SPATIAL",
    "NULL_METRICS",
    "SpatialEvaluationReport",
    "evaluate_spatial_metrics",
]
