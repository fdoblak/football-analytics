"""Homography / calibration-segment evaluation (Stage 8C — no reviewed GT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_HOMOGRAPHY = "NOT_EVALUATED_NO_REVIEWED_" "HOMOGRAPHY_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "keypoint_reprojection_error_px": None,
    "pitch_coordinate_error_m": None,
    "line_alignment_distance": None,
    "homography_success_rate": None,
    "inlier_precision": None,
    "inlier_recall": None,
    "segment_temporal_coverage": None,
    "catastrophic_mirrored_failure_rate": None,
    "camera_change_boundary_accuracy": None,
    "calibration_readiness_rate": None,
}


@dataclass(frozen=True)
class HomographyEvaluationReport:
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


def evaluate_homography(
    *,
    calibrations: Sequence[Mapping[str, Any]] | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> HomographyEvaluationReport:
    """Null metrics unless reviewed homography GT is present (not claimed here)."""
    _ = (calibrations, segments)
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_HOMOGRAPHY)
        findings.append("synthetic known-H is not football match accuracy")
        findings.append("feature detection does not guarantee correct homography")
        findings.append("projected player/ball positions deferred to Stage 8D")
        findings.append("attack_direction remains unknown")
        findings.append("SV/NBJW adapter remains evaluation_only / GPL linking risk")
        eval_status = NOT_EVALUATED_HOMOGRAPHY
        return HomographyEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: eval_status for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "Stage 8C baseline: normalized DLT + OpenCV RANSAC; " "no reviewed homography GT"
            ),
        )
    findings.append("reviewed GT present but metric computation not enabled in Stage 8C")
    return HomographyEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT path reserved; no accuracy claim",
    )


__all__ = [
    "NOT_EVALUATED_HOMOGRAPHY",
    "NULL_METRICS",
    "HomographyEvaluationReport",
    "evaluate_homography",
]
