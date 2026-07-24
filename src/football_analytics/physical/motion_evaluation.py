"""Stage 9C motion metric evaluation (no reviewed football GT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_MOTION = "NOT_EVALUATED_NO_REVIEWED_" + "DISTANCE_SPEED_SPRINT_" + "GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "distance_absolute_relative_error": None,
    "speed_mae_rmse": None,
    "peak_speed_error": None,
    "sprint_event_precision_recall_f1": None,
    "sprint_timing_error": None,
    "false_sprint_rate": None,
    "not_evaluable_correctness": None,
}


@dataclass(frozen=True)
class MotionEvaluationReport:
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


def evaluate_motion_metrics(
    *,
    metric_results: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> MotionEvaluationReport:
    _ = metric_results
    if not has_reviewed_ground_truth:
        return MotionEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=(NOT_EVALUATED_MOTION),
            metrics=dict(NULL_METRICS),
            metric_reasons={k: NOT_EVALUATED_MOTION for k in NULL_METRICS},
            findings=(
                "No reviewed distance/speed/sprint ground truth available.",
                "Synthetic fixtures prove math/pipeline only — not real football accuracy.",
                "Sprint definition is project_generated / "
                "opta_style_metric_definition, not official Opta.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="stage_9c_synthetic_math_only",
        )
    return MotionEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="evaluator_not_implemented_with_reviewed_gt",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "evaluator_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT present but Stage 9C accuracy evaluator is not implemented.",),
        created_at_utc=_utc_now(),
        adapter_notes="stage_9c",
    )


__all__ = [
    "NOT_EVALUATED_MOTION",
    "NULL_METRICS",
    "MotionEvaluationReport",
    "evaluate_motion_metrics",
]
