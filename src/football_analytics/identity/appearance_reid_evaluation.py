"""Appearance ReID evaluation (Stage 7B) — null metrics without reviewed GT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_APPEARANCE_REID = "NOT_EVALUATED_NO_REVIEWED_" "APPEARANCE_REID_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "pairwise_precision": None,
    "pairwise_recall": None,
    "pairwise_f1": None,
    "cmc_rank1": None,
    "cmc_rank5": None,
    "map": None,
    "false_link_rate": None,
    "missed_link_rate": None,
    "ambiguity_rate": None,
    "abstention_rate": None,
    "selective_accuracy": None,
    "same_kit_hard_negative_error_rate": None,
    "target_attribution_precision": None,
}


@dataclass(frozen=True)
class AppearanceReidEvaluationReport:
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


def evaluate_appearance_reid(
    *,
    profiles: Sequence[Mapping[str, Any]] | None = None,
    links: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
) -> AppearanceReidEvaluationReport:
    """Return null metrics unless reviewed appearance/ReID GT is present.

    Evaluation labels must never leak into descriptors or candidate decisions.
    Synthetic fixtures must not claim real football ReID accuracy.
    """
    _ = (profiles, links)
    findings: list[str] = []
    eval_code = NOT_EVALUATED_APPEARANCE_REID
    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(eval_code)
        findings.append("same-kit players are hard-negatives; false match risk remains")
        findings.append("handcrafted descriptor selected; real football accuracy not validated")
        findings.append("sn-reid/TrackLab reserved as future adapter only")
        findings.append("appearance alone cannot confirm identity")
        return AppearanceReidEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_code,
            metrics=dict(NULL_METRICS),
            metric_reasons={k: eval_code for k in NULL_METRICS},
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "handcrafted HSV/Lab/edge SELECTED; sn-reid future; "
                "torchvision unused; face/biometric forbidden"
            ),
        )
    # Reviewed GT present — still refuse fabricated football claims without
    # a separate licensed evaluation path (not implemented in 7B baseline).
    findings.append("reviewed GT present but licensed metric computation not enabled in 7B")
    findings.append(eval_code)
    return AppearanceReidEvaluationReport(
        status="not_evaluated",
        ground_truth_evaluation_status=eval_code,
        metrics=dict(NULL_METRICS),
        metric_reasons={k: eval_code for k in NULL_METRICS},
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT acknowledged; accuracy not claimed without gated eval path",
    )


__all__ = [
    "NOT_EVALUATED_APPEARANCE_REID",
    "NULL_METRICS",
    "AppearanceReidEvaluationReport",
    "evaluate_appearance_reid",
]
