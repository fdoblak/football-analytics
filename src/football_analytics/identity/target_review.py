"""Review manifest build + validation (Stage 7E)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.review_audit import sample_review_items
from football_analytics.identity.types import IdentityContractError


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_manifest_hash(manifest_without_hash: Mapping[str, Any]) -> str:
    payload = dict(manifest_without_hash)
    payload.pop("manifest_hash", None)
    return hash_canonical_json(payload)


def build_review_manifest(
    *,
    manifest_id: str,
    run_id: str,
    video_id: str,
    request_id: str,
    target_player_id: str,
    config_fingerprint: str,
    policy_fingerprint: str,
    expected_assignment_version: int,
    candidates: Sequence[Mapping[str, Any]],
    allowed_decisions: Sequence[str],
    artifact_refs: Mapping[str, Mapping[str, Any]],
    conflict_flags: Sequence[str] | None = None,
    max_review_items: int = 32,
    created_at_utc: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    sampled = sample_review_items(
        [
            {
                "priority": "a" if c.get("manual_review_required") else "b",
                "run_id": run_id,
                "video_id": video_id,
                "item_id": c["candidate_id"],
                **dict(c),
            }
            for c in candidates
        ],
        max_items=max_review_items,
    )
    # Preserve ranked order among sampled ids.
    keep_ids = {str(s["candidate_id"]) for s in sampled}
    ranked = [dict(c) for c in candidates if str(c["candidate_id"]) in keep_ids]
    body: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "run_id": run_id,
        "video_id": video_id,
        "request_id": request_id,
        "target_player_id": target_player_id,
        "config_fingerprint": config_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "expected_assignment_version": int(expected_assignment_version),
        "candidates": ranked,
        "allowed_decisions": list(allowed_decisions),
        "artifact_refs": {k: dict(v) for k, v in artifact_refs.items()},
        "conflict_flags": list(conflict_flags or []),
        "created_at_utc": created_at_utc or _utc_now(),
        "provenance": {
            "stage": "7E",
            "face_biometric_forbidden": True,
            "auto_confirm_forbidden": True,
            "ranking_is_review_aid_only": True,
            "notes": notes,
        },
    }
    body["manifest_hash"] = compute_manifest_hash(body)
    return validate_review_manifest(body)


def validate_review_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    schema = load_identity_json_schema("target_review_manifest")
    validate_against_json_schema(data, schema)
    expected = compute_manifest_hash(data)
    if data["manifest_hash"] != expected:
        raise IdentityContractError("HASH_VERSION_MISMATCH")
    for c in data["candidates"]:
        if int(c["end_frame_index"]) < int(c["start_frame_index"]):
            raise IdentityContractError("candidate interval invalid")
        if str(c["proposed_status"]) == "confirmed":
            raise IdentityContractError("AUTO_TARGET_SELECTION_FORBIDDEN")
    return data


__all__ = [
    "compute_manifest_hash",
    "build_review_manifest",
    "validate_review_manifest",
]
