"""Human-ball interaction request/receipt builders (Stage 10A fixtures only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.interaction.contracts import (
    load_interaction_json_schema,
    validate_against_json_schema,
)
from football_analytics.interaction.evaluation import NOT_EVALUATED_INTERACTION
from football_analytics.interaction.types import InteractionContractError
from football_analytics.interaction.validation import recount_interaction_counts


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    interaction_policy_fingerprint: str,
    request_id: str = "hbi_req_01",
    output_root: str = "/home/fdoblak/workspace/human_ball_contract_checks",
    target_player_id: str | None = "target_player_01",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "detections_ref": "synthetic://detections",
        "human_tracks_ref": "synthetic://tracks/human",
        "ball_tracks_ref": "synthetic://tracks/ball",
        "identity_artifact_ref": "synthetic://identity",
        "projected_positions_ref": "synthetic://projected_positions",
        "calibration_artifact_ref": "synthetic://calibration",
        "analysis_windows_ref": "synthetic://analysis_windows",
        "interaction_policy_fingerprint": interaction_policy_fingerprint,
        "output_root": output_root,
        "no_overwrite": True,
        "cache_enabled": False,
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "10A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_interaction_inference",
            "no_real_interaction_inference": True,
        },
    }


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    interaction_policy_fingerprint: str,
    proximity: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
    possessions: Sequence[Mapping[str, Any]],
    coverage_summary: Mapping[str, Any] | None = None,
    request_id: str = "hbi_req_01",
    receipt_id: str = "hbi_receipt_01",
    status: str = "contract_stub",
    review_count: int = 0,
    target_player_id: str | None = "target_player_01",
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    for row in list(proximity) + list(contacts) + list(possessions):
        st = str(
            row.get("evidence_level")
            or row.get("contact_state")
            or row.get("possession_state")
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
        "interaction_policy_fingerprint": interaction_policy_fingerprint,
        "input_fingerprints": {
            "detections": None,
            "tracks": None,
            "identity": None,
            "projected_positions": None,
            "calibration": None,
        },
        "output_fingerprints": {
            "human_ball_proximity": None,
            "ball_contact_candidates": None,
            "possession_hypotheses": None,
        },
        "proximity_row_count": len(proximity),
        "contact_candidate_count": len(contacts),
        "possession_hypothesis_count": len(possessions),
        "state_counts": state_counts,
        "coverage_summary": dict(coverage_summary or {}),
        "review_count": review_count,
        "evaluation_status": NOT_EVALUATED_INTERACTION,
        "event_metrics_produced": {
            "pass": False,
            "dribble": False,
            "duel": False,
            "aerial": False,
            "turnover": False,
            "box_touch": False,
        },
        "artifact_hashes": {},
        "warning_codes": [],
        "error_codes": [],
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "10A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_interaction_inference",
            "no_real_interaction_inference": True,
        },
    }


def build_synthetic_quality(
    *,
    run_id: str,
    video_id: str,
    coverage: Mapping[str, Any],
    interaction_policy_fingerprint: str | None = None,
    quality_flags: Sequence[str] | None = None,
    not_evaluable_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "interaction_policy_fingerprint": interaction_policy_fingerprint,
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
    queue_id: str = "hbi_queue_01",
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
    schema = load_interaction_json_schema("human_ball_interaction_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise InteractionContractError(f"request schema invalid: {exc}") from exc
    if payload.get("provenance", {}).get("no_real_interaction_inference") is not True:
        raise InteractionContractError("request must declare no_real_interaction_inference")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_interaction_json_schema("human_ball_interaction_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise InteractionContractError(f"receipt schema invalid: {exc}") from exc
    metrics = payload.get("event_metrics_produced") or {}
    for key in ("pass", "dribble", "duel", "aerial", "turnover", "box_touch"):
        if metrics.get(key) is not False:
            raise InteractionContractError(f"event_metrics_produced.{key} must be false")


def validate_quality_payload(payload: Mapping[str, Any]) -> None:
    schema = load_interaction_json_schema("human_ball_interaction_quality")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise InteractionContractError(f"quality schema invalid: {exc}") from exc
    cov = payload.get("coverage") or {}
    if cov.get("missing_ball_is_not_no_possession") is not True:
        raise InteractionContractError("coverage must declare missing_ball_is_not_no_possession")


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "build_synthetic_quality",
    "build_synthetic_review_queue",
    "validate_request_payload",
    "validate_receipt_payload",
    "validate_quality_payload",
    "recount_interaction_counts",
]
