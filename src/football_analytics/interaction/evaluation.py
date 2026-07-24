"""Human-ball interaction evaluation stubs (Stage 10A — no real GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.interaction.types import NOT_EVALUATED_INTERACTION

NULL_METRICS: dict[str, Any] = {
    "possession_precision_recall": None,
    "contact_candidate_precision_recall": None,
    "proximity_timing_error": None,
    "contested_agreement": None,
    "coverage_calibration": None,
    "false_possession_rate": None,
    "not_evaluable_correctness": None,
}

METRIC_REASON = NOT_EVALUATED_INTERACTION


@dataclass(frozen=True)
class InteractionEvaluationReport:
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


def evaluate_human_ball_interaction(
    *,
    proximity: Sequence[Mapping[str, Any]] | None = None,
    contacts: Sequence[Mapping[str, Any]] | None = None,
    possessions: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> InteractionEvaluationReport:
    _ = proximity, contacts, possessions
    if not has_reviewed_ground_truth:
        reasons = {k: METRIC_REASON for k in NULL_METRICS}
        eval_status = NOT_EVALUATED_INTERACTION
        return InteractionEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons=reasons,
            findings=(
                "No reviewed human-ball interaction ground truth available.",
                "Stage 10A is contracts-only; synthetic fixtures are not real accuracy.",
                "No Opta or official event accuracy claim is made.",
            ),
            created_at_utc=_utc_now(),
            adapter_notes="contracts_only_stage_10a",
        )
    return InteractionEvaluationReport(
        status="failed",
        ground_truth_evaluation_status="reviewed_gt_adapter_not_implemented",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "adapter_not_implemented" for k in NULL_METRICS},
        findings=("Reviewed GT path is not implemented in Stage 10A.",),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed_gt_not_implemented",
    )


__all__ = [
    "NOT_EVALUATED_INTERACTION",
    "NULL_METRICS",
    "InteractionEvaluationReport",
    "evaluate_human_ball_interaction",
]
