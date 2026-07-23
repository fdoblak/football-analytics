"""Metric eligibility rules for identity assignments (Stage 7A)."""

from __future__ import annotations

from football_analytics.identity.types import AssignmentStatus, MetricEligibility, TargetScope


def resolve_metric_eligibility(
    *,
    assignment_status: str,
    target_scope: str,
    has_observed_tracking: bool,
    sufficient_coverage: bool,
    unresolved_hard_conflict: bool,
    observation_state: str | None = None,
) -> str:
    """Map assignment + tracking context → metric eligibility.

    Predicted/interpolated observations are never customer-metric eligible.
    Revoked / rejected / unknown are not eligible.
    """
    if observation_state in {"predicted", "interpolated"}:
        return MetricEligibility.NOT_ELIGIBLE.value
    if assignment_status == AssignmentStatus.REVOKED.value:
        return MetricEligibility.NOT_ELIGIBLE.value
    if assignment_status == AssignmentStatus.REJECTED.value:
        return MetricEligibility.NOT_ELIGIBLE.value
    if unresolved_hard_conflict:
        return MetricEligibility.NOT_ELIGIBLE.value
    if not sufficient_coverage:
        return MetricEligibility.NOT_EVALUABLE.value
    if not has_observed_tracking:
        return MetricEligibility.NOT_EVALUABLE.value
    if assignment_status == AssignmentStatus.PROVISIONAL.value:
        return MetricEligibility.PROVISIONAL_ONLY.value
    if assignment_status == AssignmentStatus.UNKNOWN.value:
        return MetricEligibility.NOT_ELIGIBLE.value
    if assignment_status == AssignmentStatus.CANDIDATE.value:
        return MetricEligibility.NOT_ELIGIBLE.value
    if (
        assignment_status == AssignmentStatus.CONFIRMED.value
        and target_scope == TargetScope.TARGET.value
        and has_observed_tracking
        and sufficient_coverage
        and not unresolved_hard_conflict
    ):
        return MetricEligibility.ELIGIBLE.value
    return MetricEligibility.NOT_ELIGIBLE.value


def customer_metric_allowed(eligibility: str) -> bool:
    return eligibility == MetricEligibility.ELIGIBLE.value


__all__ = [
    "resolve_metric_eligibility",
    "customer_metric_allowed",
]
