"""Trajectory input eligibility rules (Stage 9A — contracts only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.physical.types import (
    EligibilityStatus,
    MetricEligibility,
    PhysicalContractError,
)


def input_is_trajectory_eligible(candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a projected/identity candidate may enter target trajectory."""
    reasons: list[str] = []
    if str(candidate.get("identity_status", "")) != "confirmed":
        reasons.append("PROVISIONAL_TARGET_EXCLUDED")
    if str(candidate.get("entity_type", "human")) != "human":
        reasons.append("BALL_NOT_TRAJECTORY_INPUT")
    if str(candidate.get("observation_source", "")) != "detection_associated":
        if str(candidate.get("observation_source", "")) in {"predicted", "interpolated"}:
            reasons.append("PREDICTED_INTERPOLATED_EXCLUDED")
        else:
            reasons.append("INELIGIBLE_INPUT")
    if str(candidate.get("mapping_status", "")) != "mapped":
        reasons.append("INELIGIBLE_INPUT")
    if str(candidate.get("physical_metric_eligibility", "")) != "eligible":
        reasons.append("INELIGIBLE_INPUT")
    if candidate.get("is_extrapolated") is True:
        reasons.append("INELIGIBLE_INPUT")
    if candidate.get("assignment_revoked_or_conflicted") is True:
        reasons.append("INELIGIBLE_INPUT")
    if candidate.get("playable_non_replay") is False:
        reasons.append("INELIGIBLE_INPUT")
    if candidate.get("fingerprints_match") is False:
        reasons.append("FINGERPRINT_MISMATCH")
    return (len(reasons) == 0, reasons)


def sample_metric_eligibility(
    *,
    eligible_input: bool,
    sample_source: str,
    identity_quality: str,
) -> str:
    if not eligible_input:
        return MetricEligibility.NOT_ELIGIBLE.value
    if sample_source != "raw_observed":
        # Derived samples are ineligible-by-default for customer physical metrics.
        return MetricEligibility.NOT_ELIGIBLE.value
    if identity_quality != "confirmed":
        return MetricEligibility.PROVISIONAL_ONLY.value
    return MetricEligibility.ELIGIBLE.value


def eligibility_status_for_reasons(reasons: Sequence[str]) -> str:
    if not reasons:
        return EligibilityStatus.ELIGIBLE.value
    if any(r.endswith("_GAP") or "GAP" in r for r in reasons):
        return EligibilityStatus.GAP.value
    if any("BOUNDARY" in r for r in reasons):
        return EligibilityStatus.BOUNDARY.value
    if "NOT_EVALUABLE" in reasons:
        return EligibilityStatus.NOT_EVALUABLE.value
    return EligibilityStatus.INELIGIBLE.value


def assert_no_attack_relative(frame_id: str) -> None:
    if frame_id == "attack_relative":
        raise PhysicalContractError("ATTACK_RELATIVE_FORBIDDEN")


def distinguish_zero_null_not_evaluable(*, value: float | None, status: str, observed: bool) -> str:
    """Return semantic label for value/status (does not invent metrics)."""
    if not observed:
        return "not_observed"
    if status == "not_evaluable":
        return "not_evaluable"
    if value is None:
        return "null"
    if value == 0.0:
        return "zero"
    return "value"


__all__ = [
    "input_is_trajectory_eligible",
    "sample_metric_eligibility",
    "eligibility_status_for_reasons",
    "assert_no_attack_relative",
    "distinguish_zero_null_not_evaluable",
]
