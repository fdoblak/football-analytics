"""Passing request/receipt builders (Stage 11A fixtures)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.passing.contracts import (
    load_passing_json_schema,
    validate_against_json_schema,
)
from football_analytics.passing.evaluation import NOT_EVALUATED_PASSING
from football_analytics.passing.types import PassingContractError
from football_analytics.passing.validation import recount_passing_counts


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    passing_policy_fingerprint: str,
    request_id: str = "pass_req_01",
    output_root: str = "/home/fdoblak/workspace/passing_contract_checks",
    target_player_id: str | None = "target_player_01",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "possession_hypotheses_ref": "synthetic://possession_hypotheses",
        "contact_candidates_ref": "synthetic://ball_contact_candidates",
        "projected_positions_ref": "synthetic://projected_positions",
        "calibration_artifact_ref": "synthetic://calibration",
        "identity_artifact_ref": "synthetic://identity",
        "passing_policy_fingerprint": passing_policy_fingerprint,
        "output_root": output_root,
        "no_overwrite": True,
        "cache_enabled": False,
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "11A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_passing_inference",
            "no_real_passing_inference": True,
        },
    }


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    passing_policy_fingerprint: str,
    passes: Sequence[Mapping[str, Any]],
    receptions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    progression: Sequence[Mapping[str, Any]],
    touches: Sequence[Mapping[str, Any]],
    coverage_summary: Mapping[str, Any] | None = None,
    request_id: str = "pass_req_01",
    receipt_id: str = "pass_receipt_01",
    status: str = "contract_stub",
    review_count: int = 0,
    target_player_id: str | None = "target_player_01",
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    for row in list(passes) + list(receptions) + list(outcomes) + list(progression) + list(touches):
        st = str(
            row.get("candidate_state")
            or row.get("outcome_state")
            or row.get("segment_state")
            or row.get("touch_state")
            or "unknown"
        )
        state_counts[st] = state_counts.get(st, 0) + 1
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "status": status,
        "passing_policy_fingerprint": passing_policy_fingerprint,
        "input_fingerprints": {
            "possession_hypotheses": None,
            "contact_candidates": None,
            "projected_positions": None,
            "calibration": None,
            "identity": None,
        },
        "output_fingerprints": {
            "pass_candidates": None,
            "reception_candidates": None,
            "pass_outcomes": None,
            "ball_progression_segments": None,
            "target_ball_touches": None,
        },
        "pass_candidate_count": len(passes),
        "reception_candidate_count": len(receptions),
        "pass_outcome_count": len(outcomes),
        "progression_segment_count": len(progression),
        "target_ball_touch_count": len(touches),
        "state_counts": state_counts,
        "coverage_summary": dict(coverage_summary or {}),
        "review_count": review_count,
        "evaluation_status": NOT_EVALUATED_PASSING,
        "accuracy_claims": {
            "opta_accuracy_validated": False,
            "real_football_accuracy_validated": False,
        },
        "artifact_hashes": {},
        "warning_codes": [],
        "error_codes": [],
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "11A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_passing_inference",
            "no_real_passing_inference": True,
        },
    }


def build_synthetic_quality(
    *,
    run_id: str,
    video_id: str,
    coverage: Mapping[str, Any],
    passing_policy_fingerprint: str | None = None,
    quality_flags: Sequence[str] | None = None,
    not_evaluable_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "passing_policy_fingerprint": passing_policy_fingerprint,
        "coverage": dict(coverage),
        "quality_flags": list(quality_flags or []),
        "not_evaluable_reasons": list(not_evaluable_reasons or []),
        "created_at_utc": _utc_now(),
    }


def build_synthetic_review_queue(
    *,
    run_id: str,
    video_id: str,
    entries: Sequence[Mapping[str, Any]] | None = None,
    queue_id: str = "pass_queue_01",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue_id": queue_id,
        "run_id": run_id,
        "video_id": video_id,
        "entries": [dict(e) for e in (entries or [])],
        "created_at_utc": _utc_now(),
    }


def build_attack_direction_evidence(
    *,
    run_id: str,
    video_id: str,
    attack_direction: str = "unknown",
    evidence_source: str = "none",
    evidence_refs: Sequence[str] | None = None,
    conflict: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "attack_direction": attack_direction,
        "evidence_source": evidence_source,
        "evidence_refs": list(evidence_refs or []),
        "conflict": conflict,
        "invented": False,
        "notes": notes,
        "created_at_utc": _utc_now(),
    }


def validate_request_payload(payload: Mapping[str, Any]) -> None:
    schema = load_passing_json_schema("passing_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise PassingContractError(f"request schema invalid: {exc}") from exc
    if payload.get("provenance", {}).get("no_real_passing_inference") is not True:
        raise PassingContractError("request must declare no_real_passing_inference")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_passing_json_schema("passing_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise PassingContractError(f"receipt schema invalid: {exc}") from exc
    claims = payload.get("accuracy_claims") or {}
    if claims.get("opta_accuracy_validated") is not False:
        raise PassingContractError("accuracy_claims.opta_accuracy_validated must be false")
    if claims.get("real_football_accuracy_validated") is not False:
        raise PassingContractError("accuracy_claims.real_football_accuracy_validated must be false")


def validate_quality_payload(payload: Mapping[str, Any]) -> None:
    schema = load_passing_json_schema("passing_quality")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise PassingContractError(f"quality schema invalid: {exc}") from exc
    cov = payload.get("coverage") or {}
    if cov.get("owner_change_alone_is_not_completed_pass") is not True:
        raise PassingContractError("coverage must declare owner_change_alone_is_not_completed_pass")


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "build_synthetic_quality",
    "build_synthetic_review_queue",
    "build_attack_direction_evidence",
    "validate_request_payload",
    "validate_receipt_payload",
    "validate_quality_payload",
    "recount_passing_counts",
]
