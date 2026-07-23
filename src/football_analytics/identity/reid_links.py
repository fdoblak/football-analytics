"""ReID candidate link validation (Stage 7A — no physical merge)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.identity.types import (
    CONTRACT_VERSION,
    IdentityContractError,
    LinkDecisionStatus,
)

REQUIRED_LINK_KEYS = frozenset(
    {
        "run_id",
        "video_id",
        "link_id",
        "source_track_id",
        "target_track_id",
        "source_video_id",
        "target_video_id",
        "time_gap_us",
        "evidence_ids",
        "decision_status",
        "conflict_flag",
        "manual_review_required",
        "producer",
        "config_fingerprint",
        "reason_codes",
        "quality_flags",
        "contract_version",
    }
)


def validate_reid_link(
    row: Mapping[str, Any],
    *,
    allow_cross_video_manual: bool = False,
) -> dict[str, Any]:
    missing = REQUIRED_LINK_KEYS - set(row)
    if missing:
        raise IdentityContractError(f"reid link missing keys: {sorted(missing)}")
    status = str(row["decision_status"])
    if status not in {s.value for s in LinkDecisionStatus}:
        raise IdentityContractError(f"invalid decision_status: {status}")
    if int(row["contract_version"]) != CONTRACT_VERSION:
        raise IdentityContractError("contract_version mismatch")
    if int(row["source_track_id"]) == int(row["target_track_id"]) and str(
        row["source_video_id"]
    ) == str(row["target_video_id"]):
        raise IdentityContractError("self-link forbidden")

    cross = str(row["source_video_id"]) != str(row["target_video_id"])
    if cross:
        reasons = [str(x) for x in (row.get("reason_codes") or [])]
        manual = "MANUAL_CROSS_VIDEO" in reasons or bool(row.get("manual_review_required"))
        if not allow_cross_video_manual or not manual:
            raise IdentityContractError("CROSS_VIDEO_AUTO_LINK_FORBIDDEN")
        if (
            status
            not in {
                LinkDecisionStatus.REJECTED.value,
                LinkDecisionStatus.REVIEW_REQUIRED.value,
                LinkDecisionStatus.UNKNOWN.value,
            }
            and not allow_cross_video_manual
        ):
            raise IdentityContractError("CROSS_VIDEO_AUTO_LINK_FORBIDDEN")

    # Candidate links never imply physical merge.
    flags = [str(x) for x in (row.get("quality_flags") or [])]
    if "physical_merge" in flags or "merged_track" in flags:
        raise IdentityContractError("PHYSICAL_MERGE_FORBIDDEN")
    return dict(row)


def validate_reid_links(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_cross_video_manual: bool = False,
) -> list[dict[str, Any]]:
    out = [validate_reid_link(r, allow_cross_video_manual=allow_cross_video_manual) for r in rows]
    ids = [r["link_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise IdentityContractError("duplicate link_id")
    return out


__all__ = [
    "REQUIRED_LINK_KEYS",
    "validate_reid_link",
    "validate_reid_links",
]
