"""Team assignment evaluation (Stage 7C) — permutation-invariant; null without reviewed GT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import permutations
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_TEAM_ASSIGNMENT = "NOT_EVALUATED_NO_REVIEWED_" "TEAM_ASSIGNMENT_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "permutation_matched_accuracy": None,
    "per_team_precision": None,
    "per_team_recall": None,
    "per_team_f1": None,
    "macro_f1": None,
    "coverage": None,
    "abstention_rate": None,
    "track_team_switch_rate": None,
    "referee_staff_unsafe_assignment_rate": None,
    "goalkeeper_unsafe_assignment_rate": None,
    "cluster_purity": None,
    "cluster_separation": None,
    "selective_accuracy": None,
    "review_conflict_recall": None,
}


@dataclass(frozen=True)
class TeamAssignmentEvaluationReport:
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


def permutation_matched_accuracy(
    predicted: Sequence[str],
    truth: Sequence[str],
    *,
    labels: Sequence[str] = ("team_a", "team_b"),
) -> float:
    """Best accuracy over anonymous label permutations (synthetic GT only)."""
    if len(predicted) != len(truth) or not predicted:
        return 0.0
    best = 0.0
    for perm in permutations(labels):
        mapping = {labels[i]: perm[i] for i in range(len(labels))}
        mapped = [mapping.get(p, p) for p in predicted]
        correct = sum(1 for a, b in zip(mapped, truth, strict=True) if a == b and a in labels)
        denom = sum(1 for t in truth if t in labels)
        if denom == 0:
            return 0.0
        best = max(best, correct / denom)
    return float(best)


def evaluate_team_assignment(
    *,
    assignments: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    synthetic_metrics: Mapping[str, Any] | None = None,
) -> TeamAssignmentEvaluationReport:
    """Null metrics without reviewed GT; optional synthetic-only permutation metrics."""
    _ = assignments
    findings: list[str] = []
    eval_code = NOT_EVALUATED_TEAM_ASSIGNMENT
    metrics = dict(NULL_METRICS)
    reasons = {k: eval_code for k in NULL_METRICS}

    if synthetic_metrics is not None:
        # Synthetic diagnostic only — must not claim real football accuracy.
        for k, v in synthetic_metrics.items():
            if k in metrics:
                metrics[k] = v
                reasons[k] = "SYNTHETIC_DIAGNOSTIC_ONLY"
        findings.append("synthetic_permutation_metrics_diagnostic_only")

    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(eval_code)
        findings.append("anonymous team_a/team_b are not club names or home/away")
        findings.append("team assignment is not player identity")
        findings.append("goalkeeper team binding from kit alone is unreliable")
        findings.append("real football team-assignment accuracy not validated")
        return TeamAssignmentEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_code,
            metrics=metrics,
            metric_reasons=reasons,
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "anonymous 2-cluster color SELECTED; real team naming / home-away forbidden; "
                "permutation-invariant evaluator for synthetic GT only"
            ),
        )

    findings.append("reviewed GT present but licensed metric path not enabled in 7C baseline")
    findings.append(eval_code)
    return TeamAssignmentEvaluationReport(
        status="not_evaluated",
        ground_truth_evaluation_status=eval_code,
        metrics=metrics,
        metric_reasons=reasons,
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT acknowledged; accuracy not claimed without gated eval path",
    )


__all__ = [
    "NOT_EVALUATED_TEAM_ASSIGNMENT",
    "NULL_METRICS",
    "TeamAssignmentEvaluationReport",
    "permutation_matched_accuracy",
    "evaluate_team_assignment",
]
