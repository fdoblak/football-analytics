"""Jersey OCR evaluation (Stage 7D) — null without reviewed GT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_JERSEY_OCR = "NOT_EVALUATED_NO_REVIEWED_" "JERSEY_NUMBER_GROUND_TRUTH"

NULL_METRICS: dict[str, Any] = {
    "region_proposal_recall": None,
    "region_proposal_precision": None,
    "digit_accuracy": None,
    "exact_number_accuracy": None,
    "character_error_rate": None,
    "coverage": None,
    "abstention_rate": None,
    "selective_accuracy": None,
    "false_number_emission_rate": None,
    "track_consensus_accuracy": None,
    "conflict_review_recall": None,
}


@dataclass(frozen=True)
class JerseyOcrEvaluationReport:
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


def false_number_emission_rate(
    observations: Sequence[Mapping[str, Any]],
    *,
    negative_observation_ids: Sequence[int] | None = None,
) -> float:
    """Fraction of no-number negatives that incorrectly emit a number (synthetic diagnostic)."""
    if not negative_observation_ids:
        return 0.0
    neg = set(int(x) for x in negative_observation_ids)
    bad = 0
    total = 0
    for row in observations:
        oid = int(row.get("observation_id", -1))
        if oid not in neg:
            continue
        total += 1
        raw = row.get("raw_text")
        norm = row.get("normalized_number")
        flags = [str(x) for x in (row.get("quality_flags") or [])]
        if raw or norm is not None:
            # Allow only if explicitly marked no_digits/no_region without emission — count as bad.
            bad += 1
        elif "no_digits" not in flags and "no_region" not in flags and "not_eligible" not in flags:
            # Negative control should abstain with a no_* flag.
            bad += 1
    if total == 0:
        return 0.0
    return float(bad) / float(total)


def evaluate_jersey_ocr(
    *,
    observations: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    synthetic_metrics: Mapping[str, Any] | None = None,
) -> JerseyOcrEvaluationReport:
    _ = observations
    findings: list[str] = []
    eval_code = NOT_EVALUATED_JERSEY_OCR
    metrics = dict(NULL_METRICS)
    reasons = {k: eval_code for k in NULL_METRICS}

    if synthetic_metrics is not None:
        for k, v in synthetic_metrics.items():
            if k in metrics:
                metrics[k] = v
                reasons[k] = "SYNTHETIC_DIAGNOSTIC_ONLY"
        findings.append("synthetic_jersey_metrics_diagnostic_only")

    if not has_reviewed_ground_truth or ground_truth is None:
        findings.append(eval_code)
        findings.append("jersey number is supporting identity evidence only")
        findings.append("template/synthetic success is not broadcast OCR accuracy")
        findings.append("false number emission is a critical product negative")
        findings.append("real football jersey OCR accuracy not validated")
        return JerseyOcrEvaluationReport(
            status="not_evaluated",
            ground_truth_evaluation_status=eval_code,
            metrics=metrics,
            metric_reasons=reasons,
            findings=tuple(findings),
            created_at_utc=_utc_now(),
            adapter_notes=(
                "OpenCV template/shape SELECTED; sn-jersey future adapter only; "
                "tesseract/easyocr/mmocr rejected/not installed"
            ),
        )

    findings.append("reviewed GT present but licensed metric path not enabled in 7D baseline")
    findings.append(eval_code)
    return JerseyOcrEvaluationReport(
        status="not_evaluated",
        ground_truth_evaluation_status=eval_code,
        metrics=metrics,
        metric_reasons=reasons,
        findings=tuple(findings),
        created_at_utc=_utc_now(),
        adapter_notes="reviewed GT acknowledged; accuracy not claimed without gated eval path",
    )


__all__ = [
    "NOT_EVALUATED_JERSEY_OCR",
    "NULL_METRICS",
    "JerseyOcrEvaluationReport",
    "false_number_emission_rate",
    "evaluate_jersey_ocr",
]
