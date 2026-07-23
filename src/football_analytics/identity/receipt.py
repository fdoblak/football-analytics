"""Identity request/receipt builders for contract fixtures only (Stage 7A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.evaluation import NOT_EVALUATED_IDENTITY
from football_analytics.identity.types import (
    AssignmentStatus,
    IdentityContractError,
    LinkDecisionStatus,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_target_request(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    target_player_id: str = "target_player_01",
    request_id: str = "id_req_01",
    manual_anchors: Sequence[Mapping[str, Any]] | None = None,
    expected_jersey_number: int | None = None,
    expected_team_id: str | None = None,
) -> dict[str, Any]:
    anchors = list(manual_anchors or [])
    return {
        "schema_version": 1,
        "request_id": request_id,
        "target_player_id": target_player_id,
        "display_label": "Synthetic Target",
        "run_id": run_id,
        "video_id": video_id,
        "match_scope": {"match_id": "match_synth_01", "video_ids": [video_id]},
        "expected_team_id": expected_team_id,
        "expected_jersey_number": expected_jersey_number,
        "expected_role": "player",
        "reference_image_hashes": [],
        "manual_anchors": anchors,
        "policy_fingerprint": policy_fingerprint,
        "created_at_utc": _utc_now(),
        "reviewed_at_utc": None,
        "provenance": {
            "stage": "7A",
            "label": "synthetic_contract_fixture",
            "notes": "no_reid_inference",
            "face_biometric_forbidden": True,
        },
    }


def recount_assignment_counts(assignments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {s.value: 0 for s in AssignmentStatus}
    for a in assignments:
        st = str(a["assignment_status"])
        counts[st] = counts.get(st, 0) + 1
    counts["total"] = len(assignments)
    return counts


def recount_link_counts(links: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {s.value: 0 for s in LinkDecisionStatus}
    for link in links:
        st = str(link["decision_status"])
        counts[st] = counts.get(st, 0) + 1
    counts["total"] = len(links)
    return counts


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    assignments: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]] | None = None,
    request_id: str = "id_req_01",
    receipt_id: str = "id_receipt_01",
    status: str = "succeeded",
    conflict_count: int = 0,
    review_required_count: int = 0,
    metric_eligible_track_count: int = 0,
    metric_eligible_frame_count: int = 0,
) -> dict[str, Any]:
    links = list(links or [])
    ev_counts: dict[str, int] = {}
    for e in evidence:
        et = str(e["evidence_type"])
        ev_counts[et] = ev_counts.get(et, 0) + 1
    eval_gt_status = NOT_EVALUATED_IDENTITY
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "run_id": run_id,
        "video_id": video_id,
        "request_id": request_id,
        "target_player_request_ref": "target_player_request.json",
        "tracking_bundle_ref": "tracking_bundle/",
        "team_assignments_ref": "team_assignments.parquet",
        "jersey_observations_ref": "jersey_observations.parquet",
        "appearance_evidence_ref": None,
        "config_fingerprint": policy_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "assignment_counts": recount_assignment_counts(assignments),
        "evidence_counts": ev_counts,
        "link_counts": recount_link_counts(links),
        "target_coverage": {
            "confirmed_frame_count": sum(
                1 + int(a["end_frame_index"]) - int(a["start_frame_index"])
                for a in assignments
                if a["assignment_status"] == "confirmed"
            ),
            "provisional_frame_count": sum(
                1 + int(a["end_frame_index"]) - int(a["start_frame_index"])
                for a in assignments
                if a["assignment_status"] == "provisional"
            ),
            "unknown_frame_count": 0,
        },
        "conflict_count": conflict_count,
        "review_required_count": review_required_count,
        "metric_eligible_track_count": metric_eligible_track_count,
        "metric_eligible_frame_count": metric_eligible_frame_count,
        "ground_truth_evaluation_status": eval_gt_status,
        "output_artifacts": {
            "identity_evidence": {
                "path": "identity_evidence.parquet",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now(),
        "status": status,
        "warnings": [],
        "errors": [],
        "provenance": {
            "stage": "7A",
            "label": "synthetic_contract_fixture",
            "face_biometric_forbidden": True,
            "no_auto_target_selection": True,
            "no_track_merge": True,
            "notes": "contracts_only",
        },
    }


def validate_receipt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    schema = load_identity_json_schema("identity_run_receipt")
    validate_against_json_schema(data, schema)
    if data["provenance"].get("face_biometric_forbidden") is not True:
        raise IdentityContractError("face_biometric_forbidden must be true")
    if data["provenance"].get("no_auto_target_selection") is not True:
        raise IdentityContractError("no_auto_target_selection must be true")
    return data


def validate_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    from football_analytics.identity.target_profile import validate_target_player_request

    return validate_target_player_request(payload)


__all__ = [
    "build_synthetic_target_request",
    "build_synthetic_receipt",
    "recount_assignment_counts",
    "recount_link_counts",
    "validate_receipt_payload",
    "validate_request_payload",
]
