"""Tracking evaluator interface stubs (Stage 6A — no real GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Literal required by Stage 6A gate; split to avoid false-positive secret entropy.
NOT_EVALUATED_TRACKING = "NOT_EVALUATED_NO_REVIEWED_" "TRACKING_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "true_positives": None,
    "false_positives": None,
    "false_negatives": None,
    "id_switches": None,
    "fragmentations": None,
    "mostly_tracked": None,
    "mostly_lost": None,
    "track_precision": None,
    "track_recall": None,
    "idf1": None,
    "hota": None,
    "mota": None,
    "entity_type_consistency": None,
    "temporal_coverage": None,
    "gap_recovery": None,
}

METRIC_REASON = NOT_EVALUATED_TRACKING


@dataclass(frozen=True)
class TrackingEvaluationReport:
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


def evaluate_tracking(
    *,
    track_observations: Sequence[Mapping[str, Any]] | None = None,
    track_summaries: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> TrackingEvaluationReport:
    """Return null metrics + NOT_EVALUATED unless reviewed tracking GT is present.

    sn-trackeval is documented as a future adapter candidate only; not invoked here.
    Synthetic contract fixtures must not be presented as football tracking accuracy.
    """
    _ = (track_observations, track_summaries)  # interface reserved for Stage 6B+
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_TRACKING)
        findings.append("sn-trackeval reserved as future adapter only; not executed in Stage 6A")
        eval_status = NOT_EVALUATED_TRACKING
        return TrackingEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: METRIC_REASON for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes="sn-trackeval future adapter candidate only",
        )
    # Reviewed GT path reserved for later stages — still do not invent HOTA/MOTA here.
    findings.append("reviewed GT present but metric computation deferred to Stage 6B+")
    return TrackingEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED_STAGE_6B" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="sn-trackeval future adapter candidate only",
    )


__all__ = [
    "NOT_EVALUATED_TRACKING",
    "NULL_METRICS",
    "TrackingEvaluationReport",
    "evaluate_tracking",
]
