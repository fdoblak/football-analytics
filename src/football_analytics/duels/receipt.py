"""Duels request/receipt builders (Stage 12A fixtures)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.duels.contracts import (
    load_duels_json_schema,
    validate_against_json_schema,
)
from football_analytics.duels.evaluation import NOT_EVALUATED_DUELS
from football_analytics.duels.types import DuelsContractError
from football_analytics.duels.validation import count_duels_rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    duels_policy_fingerprint: str,
    request_id: str = "duels_req_01",
    output_root: str = "/home/fdoblak/workspace/duels_contract_checks",
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
        "duels_policy_fingerprint": duels_policy_fingerprint,
        "output_root": output_root,
        "no_overwrite": True,
        "cache_enabled": False,
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "12A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_duels_inference",
            "no_real_duels_inference": True,
        },
    }


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    duels_policy_fingerprint: str,
    take_ons: Sequence[Mapping[str, Any]],
    ground_duels: Sequence[Mapping[str, Any]],
    aerial_duels: Sequence[Mapping[str, Any]],
    tackles: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
    turnovers: Sequence[Mapping[str, Any]],
    clearances: Sequence[Mapping[str, Any]],
    coverage_summary: Mapping[str, Any] | None = None,
    request_id: str = "duels_req_01",
    receipt_id: str = "duels_receipt_01",
    status: str = "contract_stub",
    review_count: int = 0,
    target_player_id: str | None = "target_player_01",
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    for row in (
        list(take_ons)
        + list(ground_duels)
        + list(aerial_duels)
        + list(tackles)
        + list(recoveries)
        + list(turnovers)
        + list(clearances)
    ):
        st = str(row.get("event_state") or "unknown")
        state_counts[st] = state_counts.get(st, 0) + 1
    counts = count_duels_rows(
        take_ons=take_ons,
        ground_duels=ground_duels,
        aerial_duels=aerial_duels,
        tackles=tackles,
        recoveries=recoveries,
        turnovers=turnovers,
        clearances=clearances,
    )
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "status": status,
        "duels_policy_fingerprint": duels_policy_fingerprint,
        "input_fingerprints": {
            "possession_hypotheses": None,
            "contact_candidates": None,
            "projected_positions": None,
            "calibration": None,
            "identity": None,
        },
        "output_fingerprints": {
            "take_on_attempts": None,
            "ground_duel_candidates": None,
            "aerial_duel_candidates": None,
            "tackle_events": None,
            "recovery_events": None,
            "turnover_events": None,
            "clearance_events": None,
        },
        **counts,
        "state_counts": state_counts,
        "coverage_summary": dict(coverage_summary or {}),
        "review_count": review_count,
        "evaluation_status": NOT_EVALUATED_DUELS,
        "accuracy_claims": {
            "opta_accuracy_validated": False,
            "real_football_accuracy_validated": False,
        },
        "artifact_hashes": {},
        "warning_codes": [],
        "error_codes": [],
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "12A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_duels_inference",
            "no_real_duels_inference": True,
        },
    }


def build_synthetic_quality(
    *,
    run_id: str,
    video_id: str,
    coverage: Mapping[str, Any],
    duels_policy_fingerprint: str | None = None,
    quality_flags: Sequence[str] | None = None,
    not_evaluable_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "duels_policy_fingerprint": duels_policy_fingerprint,
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
    queue_id: str = "duels_queue_01",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue_id": queue_id,
        "run_id": run_id,
        "video_id": video_id,
        "entries": [dict(e) for e in (entries or [])],
        "created_at_utc": _utc_now(),
    }


def validate_request_payload(payload: Mapping[str, Any]) -> None:
    schema = load_duels_json_schema("duels_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise DuelsContractError(f"request schema invalid: {exc}") from exc
    if payload.get("provenance", {}).get("no_real_duels_inference") is not True:
        raise DuelsContractError("request must declare no_real_duels_inference")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_duels_json_schema("duels_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise DuelsContractError(f"receipt schema invalid: {exc}") from exc
    claims = payload.get("accuracy_claims") or {}
    if claims.get("opta_accuracy_validated") is not False:
        raise DuelsContractError("accuracy_claims.opta_accuracy_validated must be false")
    if claims.get("real_football_accuracy_validated") is not False:
        raise DuelsContractError("accuracy_claims.real_football_accuracy_validated must be false")


def validate_quality_payload(payload: Mapping[str, Any]) -> None:
    schema = load_duels_json_schema("duels_quality")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise DuelsContractError(f"quality schema invalid: {exc}") from exc
    cov = payload.get("coverage") or {}
    if cov.get("nearby_opponent_alone_is_not_take_on") is not True:
        raise DuelsContractError("coverage must declare nearby_opponent_alone_is_not_take_on")


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "build_synthetic_quality",
    "build_synthetic_review_queue",
    "validate_request_payload",
    "validate_receipt_payload",
    "validate_quality_payload",
    "count_duels_rows",
]
