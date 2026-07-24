"""Projected-position evaluation (Stage 8D — no reviewed GT claimed)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_PROJECTED_POS = "NOT_EVALUATED_NO_REVIEWED_" "PROJECTED_POSITION_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "pitch_coordinate_error_m": None,
    "human_footpoint_projection_error_m": None,
    "ball_image_plane_projection_error_m": None,
    "in_bounds_classification_accuracy": None,
    "extrapolation_detection_rate": None,
    "mapping_coverage": None,
    "calibration_gap_handling": None,
    "catastrophic_mirrored_projection_rate": None,
    "target_eligible_coverage": None,
}


@dataclass(frozen=True)
class PitchProjectionEvaluationReport:
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


def evaluate_pitch_projection(
    *,
    projections: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> PitchProjectionEvaluationReport:
    """Null metrics unless reviewed projected-position GT is present (not claimed)."""
    _ = projections
    findings: list[str] = []
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(NOT_EVALUATED_PROJECTED_POS)
        findings.append("synthetic known-H projection is not football match accuracy")
        findings.append("human footpoint is bbox_bottom_centre approximation (no pose model)")
        findings.append("ball projection is image-plane centre; airborne/grounded unknown")
        findings.append("attack_direction remains unknown")
        findings.append("no distance/speed/sprint/heatmap/events computed")
        findings.append("SV/NBJW adapter remains evaluation_only / GPL linking risk")
        eval_status = NOT_EVALUATED_PROJECTED_POS
        return PitchProjectionEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_status,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: eval_status for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "Stage 8D baseline: image_to_pitch projection from calibration_segments; "
                "no reviewed projected-position GT"
            ),
        )
    findings.append("reviewed GT present but metric computation not enabled in Stage 8D")
    return PitchProjectionEvaluationReport(
        status="partial",
        ground_truth_evaluation_status="partial",
        metrics=dict(NULL_METRICS),
        metric_reasons={k: "DEFERRED" for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT path reserved; no accuracy claim",
    )


__all__ = [
    "NOT_EVALUATED_PROJECTED_POS",
    "NULL_METRICS",
    "PitchProjectionEvaluationReport",
    "evaluate_pitch_projection",
]
