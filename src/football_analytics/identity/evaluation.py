"""Identity evaluation stubs (Stage 7A — no real GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.identity.types import NOT_EVALUATED_IDENTITY

NULL_METRICS: dict[str, Any] = {
    "pairwise_precision": None,
    "pairwise_recall": None,
    "pairwise_f1": None,
    "cmc": None,
    "map": None,
    "target_track_precision": None,
    "target_track_recall": None,
    "target_temporal_coverage": None,
    "false_target_attribution": None,
    "missed_target_coverage": None,
    "identity_switches": None,
    "conflict_review_recall": None,
    "selective_accuracy": None,
    "abstention_rate": None,
}

METRIC_REASON = NOT_EVALUATED_IDENTITY


@dataclass(frozen=True)
class IdentityEvaluationReport:
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


def evaluate_identity(
    *,
    assignments: Sequence[Mapping[str, Any]] | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> IdentityEvaluationReport:
    """Return null metrics + NOT_EVALUATED unless reviewed identity GT is present.

    False target attribution is the critical product error — documented only here;
    no fabricated accuracy. sn-reid reserved as future adapter candidate only.
    """
    _ = (assignments, evidence)
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_IDENTITY)
        findings.append("false_target_attribution is the critical product error")
        findings.append("sn-reid reserved as future adapter only; not executed in Stage 7A")
        findings.append("synthetic fixtures must not claim football identity accuracy")
        eval_status = NOT_EVALUATED_IDENTITY
        return IdentityEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: METRIC_REASON for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes="sn-reid future adapter candidate only; face/biometric forbidden",
        )
    findings.append("reviewed GT present but metric computation deferred to Stage 7B+")
    findings.append("false_target_attribution remains the critical product error")
    return IdentityEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED_STAGE_7B" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="sn-reid future adapter candidate only; face/biometric forbidden",
    )


__all__ = [
    "NOT_EVALUATED_IDENTITY",
    "NULL_METRICS",
    "IdentityEvaluationReport",
    "evaluate_identity",
]
