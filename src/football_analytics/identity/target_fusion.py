"""Fuse multi-cue identity evidence into per-track candidates (Stage 7E)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.identity.evidence import (
    assert_no_face_biometric_evidence,
    validate_evidence_rows,
)
from football_analytics.identity.policy import decide_assignment_status
from football_analytics.identity.types import (
    ALONE_INSUFFICIENT_TYPES,
    AssignmentStatus,
    EvidencePolarity,
    EvidenceType,
    IdentityContractError,
    LeakageClass,
    ReliabilityTier,
)


class TargetFusionError(IdentityContractError):
    """Target evidence fusion failure."""


AUTO_SUPPORTING_TYPES = frozenset(
    {
        EvidenceType.APPEARANCE_SIMILARITY.value,
        EvidenceType.JERSEY_NUMBER.value,
        EvidenceType.TEAM_ASSIGNMENT.value,
        EvidenceType.ROLE_CONSISTENCY.value,
        EvidenceType.TEMPORAL_CONTINUITY.value,
        EvidenceType.SPATIAL_MOTION_CONTINUITY.value,
        EvidenceType.CAMERA_VIEW_SUITABILITY.value,
    }
)


def _intervals_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int, *, tolerance: int = 0
) -> bool:
    return not (a_end + tolerance < b_start or b_end + tolerance < a_start)


def assert_no_evaluation_leakage(evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    """Hard-fail if evaluation labels enter decision-facing evidence."""
    for row in evidence_rows:
        leakage = str(row.get("leakage_class", ""))
        if leakage == LeakageClass.EVALUATION.value:
            raise TargetFusionError("LEAKAGE_SEPARATION_VIOLATION")
        reasons = [str(x).lower() for x in (row.get("reason_codes") or [])]
        flags = [str(x).lower() for x in (row.get("quality_flags") or [])]
        blob = " ".join(reasons + flags + [str(row.get("evidence_type", "")).lower()])
        if "evaluation_label" in blob or "eval_gt_feature" in blob:
            raise TargetFusionError("LEAKAGE_SEPARATION_VIOLATION")


def assert_no_cross_video_auto_link(evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    for row in evidence_rows:
        reasons = [str(x) for x in (row.get("reason_codes") or [])]
        if "CROSS_VIDEO_AUTO_LINK" in reasons or "CROSS_VIDEO_AUTO_LINK_FORBIDDEN" in reasons:
            raise TargetFusionError("CROSS_VIDEO_AUTO_LINK_FORBIDDEN")
        flags = [str(x) for x in (row.get("quality_flags") or [])]
        if any("cross_video_auto" in f.lower() for f in flags):
            raise TargetFusionError("CROSS_VIDEO_AUTO_LINK_FORBIDDEN")


def group_evidence_by_track(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        tid = row.get("track_id")
        if tid is None:
            continue
        grouped.setdefault(int(tid), []).append(dict(row))
    return grouped


def fuse_track_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    within_manual_anchor_scope: bool = False,
    long_gap_after_cut: bool = False,
) -> tuple[str, list[str], list[str], list[str]]:
    """Return (proposed_status, reason_codes, supporting_ids, conflicting_ids).

    Never returns confirmed from automatic cues alone.
    """
    validated = validate_evidence_rows(evidence_rows)
    assert_no_face_biometric_evidence(validated)
    assert_no_evaluation_leakage(validated)
    assert_no_cross_video_auto_link(validated)

    status, reasons = decide_assignment_status(
        validated,
        policy=policy,
        within_manual_anchor_scope=within_manual_anchor_scope,
    )
    # Hard safety: auto path never confirms.
    if status == AssignmentStatus.CONFIRMED.value and not within_manual_anchor_scope:
        raise TargetFusionError("AUTO_TARGET_SELECTION_FORBIDDEN")

    supporting = [
        e
        for e in validated
        if str(e.get("polarity")) == EvidencePolarity.SUPPORTS.value
        and str(e.get("reliability_tier"))
        not in {ReliabilityTier.CONFLICTING.value, ReliabilityTier.UNAVAILABLE.value}
    ]
    conflicting = [
        e
        for e in validated
        if str(e.get("polarity")) == EvidencePolarity.CONFLICTS.value
        or str(e.get("reliability_tier")) == ReliabilityTier.CONFLICTING.value
    ]

    # Alone-insufficient single auto cue stays candidate.
    if (
        status == AssignmentStatus.CANDIDATE.value
        and len(supporting) == 1
        and str(supporting[0].get("evidence_type")) in ALONE_INSUFFICIENT_TYPES
    ):
        et = str(supporting[0].get("evidence_type"))
        alone_map = {
            EvidenceType.APPEARANCE_SIMILARITY.value: "APPEARANCE_ALONE_INSUFFICIENT",
            EvidenceType.JERSEY_NUMBER.value: "JERSEY_ALONE_INSUFFICIENT",
            EvidenceType.TEAM_ASSIGNMENT.value: "TEAM_ALONE_INSUFFICIENT",
        }
        code = alone_map.get(et, "SINGLE_WEAK_CANNOT_CONFIRM")
        if code not in reasons:
            reasons = list(reasons) + [code]

    # Two auto cues → at most provisional (policy already enforces).
    auto_types = {
        str(e.get("evidence_type"))
        for e in supporting
        if str(e.get("evidence_type")) in AUTO_SUPPORTING_TYPES
    }
    if (
        status == AssignmentStatus.PROVISIONAL.value
        and len(auto_types) >= 2
        and "MULTI_SUPPORTING_PROVISIONAL" not in reasons
    ):
        reasons = list(reasons) + ["MULTI_SUPPORTING_PROVISIONAL"]

    if long_gap_after_cut and status in {
        AssignmentStatus.CONFIRMED.value,
        AssignmentStatus.PROVISIONAL.value,
    }:
        # After cut/gap without ReID continuity, demote to candidate.
        status = AssignmentStatus.CANDIDATE.value
        reasons = list(reasons) + ["LONG_GAP_NEW_TRACK_CANDIDATE"]

    # Fusion prepare-review never emits confirmed; manual decide does.
    if status == AssignmentStatus.CONFIRMED.value:
        status = AssignmentStatus.PROVISIONAL.value
        reasons = list(reasons) + ["MANUAL_CONFIRM_REQUIRED"]

    return (
        status,
        list(reasons),
        [str(e["evidence_id"]) for e in supporting],
        [str(e["evidence_id"]) for e in conflicting],
    )


def detect_confirmed_overlaps(
    assignments: Sequence[Mapping[str, Any]],
    *,
    overlap_tolerance_frames: int = 0,
) -> list[dict[str, Any]]:
    """Hard-fail findings for simultaneous confirmed targets overlapping in time."""
    confirmed = [
        a
        for a in assignments
        if str(a.get("assignment_status")) == AssignmentStatus.CONFIRMED.value
    ]
    findings: list[dict[str, Any]] = []
    for i, a in enumerate(confirmed):
        for b in confirmed[i + 1 :]:
            if int(a["track_id"]) == int(b["track_id"]) and str(a["target_player_id"]) == str(
                b["target_player_id"]
            ):
                continue
            if _intervals_overlap(
                int(a["start_frame_index"]),
                int(a["end_frame_index"]),
                int(b["start_frame_index"]),
                int(b["end_frame_index"]),
                tolerance=overlap_tolerance_frames,
            ):
                findings.append(
                    {
                        "code": "DUPLICATE_CONFIRMED_IDENTITY",
                        "assignment_a": a["assignment_id"],
                        "assignment_b": b["assignment_id"],
                        "track_a": int(a["track_id"]),
                        "track_b": int(b["track_id"]),
                    }
                )
    return findings


def detect_track_multi_identity(
    assignments: Sequence[Mapping[str, Any]],
    *,
    overlap_tolerance_frames: int = 0,
) -> list[dict[str, Any]]:
    active = [
        a
        for a in assignments
        if str(a.get("assignment_status"))
        in {
            AssignmentStatus.CONFIRMED.value,
            AssignmentStatus.PROVISIONAL.value,
            AssignmentStatus.CANDIDATE.value,
        }
    ]
    findings: list[dict[str, Any]] = []
    by_track: dict[int, list[Mapping[str, Any]]] = {}
    for a in active:
        by_track.setdefault(int(a["track_id"]), []).append(a)
    for track_id, rows in by_track.items():
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if str(a["target_player_id"]) == str(b["target_player_id"]):
                    continue
                if _intervals_overlap(
                    int(a["start_frame_index"]),
                    int(a["end_frame_index"]),
                    int(b["start_frame_index"]),
                    int(b["end_frame_index"]),
                    tolerance=overlap_tolerance_frames,
                ):
                    findings.append(
                        {
                            "code": "TRACK_MULTI_IDENTITY",
                            "track_id": track_id,
                            "assignment_a": a["assignment_id"],
                            "assignment_b": b["assignment_id"],
                        }
                    )
    return findings


__all__ = [
    "TargetFusionError",
    "AUTO_SUPPORTING_TYPES",
    "assert_no_evaluation_leakage",
    "assert_no_cross_video_auto_link",
    "group_evidence_by_track",
    "fuse_track_evidence",
    "detect_confirmed_overlaps",
    "detect_track_multi_identity",
]
