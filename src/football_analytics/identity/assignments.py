"""Track identity assignment + revoke/supersede (append-only, Stage 7A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.identity.metric_eligibility import resolve_metric_eligibility
from football_analytics.identity.types import (
    CONTRACT_VERSION,
    AssignmentStatus,
    IdentityContractError,
    TargetScope,
)


def validate_assignment_record(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id",
        "video_id",
        "assignment_id",
        "track_id",
        "target_player_id",
        "assignment_status",
        "target_scope",
        "evidence_ids",
        "supporting_evidence_count",
        "conflicting_evidence_count",
        "start_frame_index",
        "end_frame_index",
        "metric_eligibility",
        "manual_review_required",
        "assignment_version",
        "reason_codes",
        "quality_flags",
        "producer",
        "producer_version",
        "policy_fingerprint",
        "leakage_class",
        "contract_version",
    }
    missing = required - set(row)
    if missing:
        raise IdentityContractError(f"assignment missing keys: {sorted(missing)}")
    status = str(row["assignment_status"])
    if status not in {s.value for s in AssignmentStatus}:
        raise IdentityContractError(f"invalid assignment_status: {status}")
    if str(row["target_scope"]) not in {s.value for s in TargetScope}:
        raise IdentityContractError("invalid target_scope")
    if int(row["end_frame_index"]) < int(row["start_frame_index"]):
        raise IdentityContractError("assignment interval invalid")
    if int(row["contract_version"]) != CONTRACT_VERSION:
        raise IdentityContractError("contract_version mismatch")
    if int(row["assignment_version"]) < 1:
        raise IdentityContractError("assignment_version must be >= 1")
    if status == AssignmentStatus.CONFIRMED.value:
        # Confirmed must not be presented from a single alone-insufficient cue.
        reasons = [str(x) for x in (row.get("reason_codes") or [])]
        alone = {
            "JERSEY_ALONE_INSUFFICIENT",
            "TEAM_ALONE_INSUFFICIENT",
            "ROLE_ALONE_INSUFFICIENT",
            "APPEARANCE_ALONE_INSUFFICIENT",
            "SINGLE_WEAK_CANNOT_CONFIRM",
        }
        if alone & set(reasons):
            raise IdentityContractError("SINGLE_WEAK_CANNOT_CONFIRM")
        if int(row["supporting_evidence_count"]) < 1:
            raise IdentityContractError("confirmed requires supporting evidence")
    return dict(row)


def build_revocation(
    previous: Mapping[str, Any],
    *,
    new_assignment_id: str,
    reason: str,
    actor_provenance: str | None = None,
) -> dict[str, Any]:
    """Append-only revoke: new row with status=revoked referencing previous."""
    prev = validate_assignment_record(previous)
    if prev["assignment_status"] == AssignmentStatus.REVOKED.value:
        raise IdentityContractError("already revoked")
    revoked = dict(prev)
    revoked["assignment_id"] = new_assignment_id
    revoked["assignment_status"] = AssignmentStatus.REVOKED.value
    revoked["assignment_version"] = int(prev["assignment_version"]) + 1
    revoked["supersedes_assignment_id"] = prev["assignment_id"]
    revoked["revoked_by_assignment_id"] = new_assignment_id
    revoked["reason_codes"] = list(prev.get("reason_codes") or []) + ["REVOKED", reason]
    revoked["metric_eligibility"] = resolve_metric_eligibility(
        assignment_status=AssignmentStatus.REVOKED.value,
        target_scope=str(prev["target_scope"]),
        has_observed_tracking=True,
        sufficient_coverage=True,
        unresolved_hard_conflict=False,
    )
    if actor_provenance:
        revoked["provenance_json"] = actor_provenance
    return validate_assignment_record(revoked)


def build_supersede(
    previous: Mapping[str, Any],
    *,
    new_assignment_id: str,
    new_status: str,
    evidence_ids: Sequence[str],
    supporting_count: int,
    conflicting_count: int,
    reason: str,
) -> dict[str, Any]:
    prev = validate_assignment_record(previous)
    row = dict(prev)
    row["assignment_id"] = new_assignment_id
    row["assignment_status"] = new_status
    row["assignment_version"] = int(prev["assignment_version"]) + 1
    row["supersedes_assignment_id"] = prev["assignment_id"]
    row["evidence_ids"] = list(evidence_ids)
    row["supporting_evidence_count"] = int(supporting_count)
    row["conflicting_evidence_count"] = int(conflicting_count)
    row["reason_codes"] = list(prev.get("reason_codes") or []) + ["SUPERSEDE", reason]
    row["metric_eligibility"] = resolve_metric_eligibility(
        assignment_status=new_status,
        target_scope=str(prev["target_scope"]),
        has_observed_tracking=True,
        sufficient_coverage=True,
        unresolved_hard_conflict=int(conflicting_count) > 0,
    )
    return validate_assignment_record(row)


def validate_assignment_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [validate_assignment_record(r) for r in rows]
    ids = [r["assignment_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise IdentityContractError("duplicate assignment_id")
    return out


__all__ = [
    "validate_assignment_record",
    "validate_assignment_rows",
    "build_revocation",
    "build_supersede",
]
