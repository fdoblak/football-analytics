"""Calibration evaluation stubs (Stage 8A — no real GT metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from football_analytics.calibration.types import NOT_EVALUATED_CALIBRATION

NULL_METRICS: dict[str, Any] = {
    "keypoint_reprojection_error_px": None,
    "pitch_coordinate_error_m": None,
    "line_alignment_error": None,
    "calibration_success_rate": None,
    "coverage": None,
    "catastrophic_mirrored_failure_rate": None,
    "mapping_in_bounds_accuracy": None,
    "temporal_stability": None,
}

METRIC_REASON = NOT_EVALUATED_CALIBRATION


@dataclass(frozen=True)
class CalibrationEvaluationReport:
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


def evaluate_calibration(
    *,
    segments: Sequence[Mapping[str, Any]] | None = None,
    projected_positions: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> CalibrationEvaluationReport:
    """Return null metrics + NOT_EVALUATED unless reviewed calibration GT is present."""
    _ = (segments, projected_positions)
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_CALIBRATION)
        findings.append("SV_kp/SV_lines not executed in Stage 8A")
        findings.append("synthetic fixtures must not claim football calibration accuracy")
        findings.append("projected positions are not physical-metric guarantees")
        eval_status = NOT_EVALUATED_CALIBRATION
        return CalibrationEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: METRIC_REASON for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes="sn-calibration / SV_kp / SV_lines future adapters only",
        )
    findings.append("reviewed GT present but metric computation deferred to Stage 8B+")
    return CalibrationEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED_STAGE_8B" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="sn-calibration / SV_kp / SV_lines future adapters only",
    )


__all__ = [
    "NOT_EVALUATED_CALIBRATION",
    "NULL_METRICS",
    "CalibrationEvaluationReport",
    "evaluate_calibration",
]
