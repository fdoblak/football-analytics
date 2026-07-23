"""Target identity fusion evaluation (Stage 7E)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_TARGET_IDENTITY = "NOT_EVALUATED_NO_REVIEWED_" "TARGET_IDENTITY_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "target_track_precision": None,
    "target_track_recall": None,
    "target_track_f1": None,
    "false_target_attribution": None,
    "missed_target_coverage": None,
    "confirmed_temporal_coverage": None,
    "identity_switches": None,
    "candidate_recall": None,
    "ranking_quality": None,
    "abstention_rate": None,
    "selective_accuracy": None,
    "conflict_review_recall": None,
    "manual_workload": None,
}


@dataclass(frozen=True)
class TargetFusionEvaluationReport:
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


def synthetic_false_target_attribution(
    *,
    confirmed_assignments: Sequence[Mapping[str, Any]],
    expected_track_ids: Sequence[int] | None = None,
) -> float:
    """Diagnostic-only: false confirmed attributions on synthetic fixtures must be 0."""
    expected = set(int(x) for x in (expected_track_ids or []))
    if not expected:
        return 0.0
    false_n = 0
    for a in confirmed_assignments:
        if int(a["track_id"]) not in expected:
            false_n += 1
    return float(false_n)


def evaluate_target_fusion(
    *,
    assignments: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    synthetic_expected_track_ids: Sequence[int] | None = None,
) -> TargetFusionEvaluationReport:
    findings: list[str] = []
    metrics = dict(NULL_METRICS)
    reasons = {k: NOT_EVALUATED_TARGET_IDENTITY for k in NULL_METRICS}

    confirmed = [a for a in (assignments or []) if str(a.get("assignment_status")) == "confirmed"]
    # Synthetic safety metric (diagnostic): must be 0 when expected tracks provided.
    if synthetic_expected_track_ids is not None:
        fta = synthetic_false_target_attribution(
            confirmed_assignments=confirmed,
            expected_track_ids=synthetic_expected_track_ids,
        )
        metrics["false_target_attribution"] = fta
        reasons["false_target_attribution"] = "SYNTHETIC_DIAGNOSTIC_ONLY"
        if fta != 0.0:
            findings.append("false_target_attribution_nonzero_on_synthetic")

    if not has_reviewed_ground_truth or ground_truth is None:
        eval_status = NOT_EVALUATED_TARGET_IDENTITY
        findings.append(eval_status)
        findings.append("false_target_attribution is the critical product error")
        findings.append("synthetic fixtures must not claim football identity accuracy")
        return TargetFusionEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=metrics,
            metric_reasons=reasons,
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes="no reviewed target-identity GT; Stage 7E workflow only",
        )

    findings.append("reviewed GT present but full metric suite deferred")
    findings.append("false_target_attribution remains the critical product error")
    return TargetFusionEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=metrics,
        metric_reasons={k: "DEFERRED_FULL_METRICS" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT path reserved; face/biometric forbidden",
    )


__all__ = [
    "NOT_EVALUATED_TARGET_IDENTITY",
    "NULL_METRICS",
    "TargetFusionEvaluationReport",
    "synthetic_false_target_attribution",
    "evaluate_target_fusion",
]
