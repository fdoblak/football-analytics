"""Per-metric quality gate helpers for Stage 9E."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

METRIC_STATUSES = frozenset(
    {
        "evaluable",
        "not_evaluable",
        "not_observed",
        "insufficient_coverage",
        "identity_unconfirmed",
        "calibration_unavailable",
        "source_inconsistent",
        "failed",
    }
)


def normalize_status(raw: str | None, *, fallback: str = "not_evaluable") -> str:
    if raw is None:
        return fallback
    s = str(raw)
    # Map Stage 9C/9D statuses onto 9E vocabulary
    mapping = {
        "computed": "evaluable",
        "partial": "evaluable",
        "not_evaluable": "not_evaluable",
        "insufficient_coverage": "insufficient_coverage",
        "failed": "failed",
        "contract_stub": "not_evaluable",
        "evaluable": "evaluable",
        "not_observed": "not_observed",
        "identity_unconfirmed": "identity_unconfirmed",
        "calibration_unavailable": "calibration_unavailable",
        "source_inconsistent": "source_inconsistent",
    }
    out = mapping.get(s, s)
    return out if out in METRIC_STATUSES else fallback


def metric_entry(
    *,
    name: str,
    value: float | int | None,
    unit: str,
    status: str,
    coverage_ratio: float | None = None,
    confidence: float | None = None,
    reason_codes: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    st = normalize_status(status)
    # 0 is a real measurement when status is evaluable
    return {
        "metric_name": name,
        "value": value,
        "unit": unit,
        "status": st,
        "coverage_ratio": coverage_ratio,
        "confidence": confidence,
        "reason_codes": list(reason_codes or []),
        "provenance": dict(provenance or {}),
        "zero_means_measured_zero": bool(st == "evaluable" and value == 0),
    }


def derive_overall_status(
    metrics: Sequence[Mapping[str, Any]],
    *,
    critical: Sequence[str],
    identity_ok: bool,
) -> str:
    by_name = {str(m["metric_name"]): m for m in metrics}
    if not identity_ok:
        return "not_evaluable"
    crit = [by_name[n] for n in critical if n in by_name]
    if not crit:
        return "failed"
    statuses = [str(m["status"]) for m in crit]
    if any(s in {"failed", "source_inconsistent"} for s in statuses):
        return "failed"
    if any(s == "identity_unconfirmed" for s in statuses):
        return "not_evaluable"
    if all(s == "evaluable" for s in statuses):
        return "succeeded"
    if any(s == "evaluable" for s in statuses):
        return "partial"
    return "not_evaluable"


__all__ = [
    "METRIC_STATUSES",
    "normalize_status",
    "metric_entry",
    "derive_overall_status",
]
