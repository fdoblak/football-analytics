"""Metric eligibility timeline builder (Stage 7E — no real metrics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.metric_eligibility import resolve_metric_eligibility
from football_analytics.identity.types import IdentityContractError, MetricEligibility


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _frame_count(start: int, end: int) -> int:
    return max(0, int(end) - int(start) + 1)


def build_eligibility_timeline(
    assignments: Sequence[Mapping[str, Any]],
    *,
    timeline_id: str,
    run_id: str,
    video_id: str,
    target_player_id: str,
    observation_state_by_assignment: Mapping[str, str] | None = None,
    sufficient_coverage_by_assignment: Mapping[str, bool] | None = None,
    conflict_by_assignment: Mapping[str, bool] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    obs_map = dict(observation_state_by_assignment or {})
    cov_map = dict(sufficient_coverage_by_assignment or {})
    conf_map = dict(conflict_by_assignment or {})
    intervals: list[dict[str, Any]] = []
    summary = {
        "eligible_frame_count": 0,
        "provisional_only_frame_count": 0,
        "not_eligible_frame_count": 0,
        "not_evaluable_frame_count": 0,
    }
    for a in assignments:
        aid = str(a["assignment_id"])
        obs = obs_map.get(aid, "observed")
        sufficient = bool(cov_map.get(aid, True))
        conflict = bool(conf_map.get(aid, False))
        eligibility = resolve_metric_eligibility(
            assignment_status=str(a["assignment_status"]),
            target_scope=str(a.get("target_scope", "target")),
            has_observed_tracking=obs == "observed",
            sufficient_coverage=sufficient,
            unresolved_hard_conflict=conflict,
            observation_state=obs,
        )
        reasons: list[str] = []
        if eligibility == MetricEligibility.ELIGIBLE.value:
            reasons.append("CONFIRMED_OBSERVED_ELIGIBLE")
        elif eligibility == MetricEligibility.PROVISIONAL_ONLY.value:
            reasons.append("PROVISIONAL_ONLY")
        elif eligibility == MetricEligibility.NOT_EVALUABLE.value:
            reasons.append("INSUFFICIENT_COVERAGE" if not sufficient else "MISSING_TRACKING")
        else:
            if str(a["assignment_status"]) == "revoked":
                reasons.append("REVOKED_NOT_METRIC_ELIGIBLE")
            elif obs in {"predicted", "interpolated"}:
                reasons.append("PREDICTED_INTERPOLATED_NOT_ELIGIBLE")
            else:
                reasons.append("NOT_ELIGIBLE")
        start = int(a["start_frame_index"])
        end = int(a["end_frame_index"])
        if end < start:
            raise IdentityContractError("assignment interval invalid")
        n = _frame_count(start, end)
        if eligibility == MetricEligibility.ELIGIBLE.value:
            summary["eligible_frame_count"] += n
        elif eligibility == MetricEligibility.PROVISIONAL_ONLY.value:
            summary["provisional_only_frame_count"] += n
        elif eligibility == MetricEligibility.NOT_EVALUABLE.value:
            summary["not_evaluable_frame_count"] += n
        else:
            summary["not_eligible_frame_count"] += n
        intervals.append(
            {
                "start_frame_index": start,
                "end_frame_index": end,
                "track_id": int(a["track_id"]),
                "assignment_id": aid,
                "assignment_status": str(a["assignment_status"]),
                "eligibility": eligibility,
                "observation_state": obs,
                "reason_codes": reasons,
                "sufficient_coverage": sufficient,
                "unresolved_hard_conflict": conflict,
            }
        )
    payload = {
        "schema_version": 1,
        "timeline_id": timeline_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "intervals": intervals,
        "summary": summary,
        "created_at_utc": created_at_utc or _utc_now(),
        "provenance": {
            "stage": "7E",
            "no_real_metrics": True,
            "notes": "eligibility contract only; no customer metric computation",
        },
    }
    return validate_eligibility_timeline(payload)


def validate_eligibility_timeline(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    schema = load_identity_json_schema("metric_eligibility_timeline")
    validate_against_json_schema(data, schema)
    return data


__all__ = [
    "build_eligibility_timeline",
    "validate_eligibility_timeline",
]
