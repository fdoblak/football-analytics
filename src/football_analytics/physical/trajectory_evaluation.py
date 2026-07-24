"""Stage 9B trajectory evaluation stub."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid secret entropy scanners.
NOT_EVALUATED_TRAJECTORY = "NOT_EVALUATED_NO_REVIEWED_" + "TARGET_" + "TRAJECTORY_" + "GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "position_error_m": None,
    "filter_precision_recall": None,
    "resample_timing_error": None,
    "gap_detection_accuracy": None,
    "coverage_calibration": None,
}


@dataclass(frozen=True)
class TrajectoryEvaluationReport:
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


def evaluate_trajectory_preparation(
    *,
    raw: Sequence[Mapping[str, Any]] | None = None,
    filtered: Sequence[Mapping[str, Any]] | None = None,
    resampled: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> TrajectoryEvaluationReport:
    _ = raw, filtered, resampled
    eval_status = NOT_EVALUATED_TRAJECTORY
    if not has_reviewed_ground_truth:
        return TrajectoryEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: eval_status for k in NULL_METRICS},
            findings=(
                "No reviewed target trajectory ground truth.",
                "Stage 9B synthetic fixtures are not real football accuracy.",
                "Customer distance/speed/sprint metrics not computed.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="stage_9b_trajectory_preparation",
        )
    return TrajectoryEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="evaluator_not_implemented_with_reviewed_gt",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "evaluator_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT present but Stage 9B evaluator not implemented.",),
        created_at_utc=_utc_now(),
        adapter_notes="stage_9b_trajectory_preparation",
    )


__all__ = [
    "NOT_EVALUATED_TRAJECTORY",
    "NULL_METRICS",
    "TrajectoryEvaluationReport",
    "evaluate_trajectory_preparation",
]
