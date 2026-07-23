"""Target player request/profile validation (Stage 7A)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.types import IdentityContractError


def validate_target_player_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate target player request against JSON schema + semantic safety rules."""
    data = dict(payload)
    schema = load_identity_json_schema("target_player_request")
    validate_against_json_schema(data, schema)

    prov = data.get("provenance") or {}
    if prov.get("face_biometric_forbidden") is not True:
        raise IdentityContractError("face_biometric_forbidden must be true")

    # Display name / jersey / team are never algorithmic confirmation alone.
    anchors = data.get("manual_anchors") or []
    for a in anchors:
        if int(a["end_frame_index"]) < int(a["start_frame_index"]):
            raise IdentityContractError("manual anchor interval invalid")

    match_scope = data["match_scope"]
    if data["video_id"] not in match_scope["video_ids"]:
        raise IdentityContractError("video_id must be listed in match_scope.video_ids")

    return data


def assert_target_hints_not_identity(payload: Mapping[str, Any]) -> None:
    """Documented invariant: name/jersey/team hints ≠ confirmed identity."""
    _ = payload  # interface reserved; hints never auto-confirm
    return None


__all__ = [
    "validate_target_player_request",
    "assert_target_hints_not_identity",
]
