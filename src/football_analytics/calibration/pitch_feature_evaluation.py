"""Pitch feature detection evaluation (Stage 8B — no reviewed GT → NOT_EVALUATED)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_PITCH_FEATURES = "NOT_EVALUATED_NO_REVIEWED_" "PITCH_FEATURE_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "keypoint_precision": None,
    "keypoint_recall": None,
    "keypoint_pixel_error": None,
    "line_precision": None,
    "line_recall": None,
    "line_endpoint_distance": None,
    "feature_coverage": None,
    "no_feature_rate": None,
    "catastrophic_class_mismatch_rate": None,
    "calibration_readiness_rate": None,
}


@dataclass(frozen=True)
class PitchFeatureEvaluationReport:
    status: str
    ground_truth_evaluation_status: str
    metrics: dict[str, Any]
    metric_reasons: dict[str, str]
    findings: tuple[str, ...]
    created_at_utc: str
    adapter_notes: str

    def to_dict(
        self,
        *,
        run_id: str,
        video_id: str,
        config_fingerprint: str | None = None,
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


def evaluate_pitch_features(
    *,
    predictions: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> PitchFeatureEvaluationReport:
    """Null metrics unless reviewed pitch-feature GT is present (not implemented)."""
    _ = predictions
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_PITCH_FEATURES)
        findings.append("synthetic/model smoke is not football pitch-feature accuracy")
        findings.append("homography solve deferred to Stage 8C")
        findings.append("SV weights evaluation_only; production_approved=false")
        eval_status = NOT_EVALUATED_PITCH_FEATURES
        return PitchFeatureEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: eval_status for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "ai-dev lazy importlib HRNet from locked NBJW paths; "
                "GPL-2.0 linking risk; evaluation_only"
            ),
        )
    findings.append("reviewed GT present but metric computation not enabled in Stage 8B")
    return PitchFeatureEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT path reserved; no accuracy claim",
    )


__all__ = [
    "NOT_EVALUATED_PITCH_FEATURES",
    "NULL_METRICS",
    "PitchFeatureEvaluationReport",
    "evaluate_pitch_features",
]
