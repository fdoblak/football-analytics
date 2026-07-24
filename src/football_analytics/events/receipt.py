"""Synthetic receipt/request/quality builders for Stage 13."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.events.contracts import (
    load_events_json_schema,
    validate_against_json_schema,
)
from football_analytics.events.types import EventsContractError


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "provenance": {"no_real_events_inference": True},
        "created_at_utc": _utc_now(),
    }
    if extras:
        payload.update(dict(extras))
    return payload


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    ledger_count: int = 0,
    revision_count: int = 0,
    replay_count: int = 0,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "ledger_count": ledger_count,
        "revision_count": revision_count,
        "replay_count": replay_count,
        "accuracy_claims": {
            "opta_accuracy_validated": False,
            "real_football_accuracy_validated": False,
        },
        "created_at_utc": _utc_now(),
    }
    if extras:
        payload.update(dict(extras))
    return payload


def build_synthetic_quality(
    *,
    run_id: str,
    video_id: str,
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "coverage": dict(
            coverage
            or {
                "never_invent_live_when_replay_uncertain": True,
                "append_only_ledger": True,
                "interaction_coverage": 0.8,
            }
        ),
        "created_at_utc": _utc_now(),
    }


def build_synthetic_review_queue(
    *,
    queue_id: str,
    run_id: str,
    video_id: str,
    entries: Sequence[Mapping[str, Any]] | None = None,
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
    schema = load_events_json_schema("events_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise EventsContractError(f"request schema invalid: {exc}") from exc
    if payload.get("provenance", {}).get("no_real_events_inference") is not True:
        raise EventsContractError("request must declare no_real_events_inference")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_events_json_schema("events_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise EventsContractError(f"receipt schema invalid: {exc}") from exc
    claims = payload.get("accuracy_claims") or {}
    if claims.get("opta_accuracy_validated") is not False:
        raise EventsContractError("accuracy_claims.opta_accuracy_validated must be false")
    if claims.get("real_football_accuracy_validated") is not False:
        raise EventsContractError("accuracy_claims.real_football_accuracy_validated must be false")


def validate_quality_payload(payload: Mapping[str, Any]) -> None:
    schema = load_events_json_schema("events_quality")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise EventsContractError(f"quality schema invalid: {exc}") from exc
    cov = payload.get("coverage") or {}
    if cov.get("never_invent_live_when_replay_uncertain") is not True:
        raise EventsContractError("coverage must declare never_invent_live_when_replay_uncertain")
    if cov.get("append_only_ledger") is not True:
        raise EventsContractError("coverage must declare append_only_ledger")


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "build_synthetic_quality",
    "build_synthetic_review_queue",
    "validate_request_payload",
    "validate_receipt_payload",
    "validate_quality_payload",
]
