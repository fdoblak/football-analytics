"""Synthetic identity contract fixtures only (Stage 7A — no real accuracy claims)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.identity.types import CONTRACT_VERSION


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {
        "run_id": generate_run_id(),
        "video_id": "video_synth_01",
        "target_player_id": "target_player_01",
    }


def evidence_row(
    run_id: str,
    video_id: str,
    evidence_id: str,
    *,
    evidence_type: str,
    reliability_tier: str,
    polarity: str = "supports",
    track_id: int | None = 0,
    frame_index: int | None = 0,
    start_frame_index: int | None = 0,
    end_frame_index: int | None = 10,
    score: float | None = None,
    leakage_class: str = "synthetic",
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
    source_artifact_ref: str | None = None,
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "evidence_id": evidence_id,
        "track_id": track_id,
        "frame_index": frame_index,
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "start_time_us": None,
        "end_time_us": None,
        "evidence_type": evidence_type,
        "source_artifact_ref": source_artifact_ref,
        "source_fingerprint": source_fingerprint,
        "observed_value_ref": None,
        "score": score,
        "reliability_tier": reliability_tier,
        "polarity": polarity,
        "review_status": "unreviewed",
        "producer": "identity_contract_synth",
        "producer_version": "0.0.0",
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "leakage_class": leakage_class,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def assignment_row(
    run_id: str,
    video_id: str,
    assignment_id: str,
    *,
    track_id: int,
    target_player_id: str,
    assignment_status: str,
    evidence_ids: Sequence[str],
    supporting: int,
    conflicting: int = 0,
    metric_eligibility: str,
    policy_fingerprint: str,
    start_frame_index: int = 0,
    end_frame_index: int = 10,
    target_scope: str = "target",
    manual_review_required: bool = False,
    assignment_version: int = 1,
    supersedes_assignment_id: str | None = None,
    revoked_by_assignment_id: str | None = None,
    reason_codes: Sequence[str] | None = None,
    leakage_class: str = "synthetic",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "assignment_id": assignment_id,
        "track_id": track_id,
        "target_player_id": target_player_id,
        "assignment_status": assignment_status,
        "target_scope": target_scope,
        "evidence_ids": list(evidence_ids),
        "supporting_evidence_count": supporting,
        "conflicting_evidence_count": conflicting,
        "confidence": None,
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "start_time_us": None,
        "end_time_us": None,
        "metric_eligibility": metric_eligibility,
        "manual_review_required": manual_review_required,
        "assignment_version": assignment_version,
        "supersedes_assignment_id": supersedes_assignment_id,
        "revoked_by_assignment_id": revoked_by_assignment_id,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "producer": "identity_contract_synth",
        "producer_version": "0.0.0",
        "policy_fingerprint": policy_fingerprint,
        "leakage_class": leakage_class,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def link_row(
    run_id: str,
    video_id: str,
    link_id: str,
    *,
    source_track_id: int,
    target_track_id: int,
    policy_fingerprint: str,
    source_video_id: str | None = None,
    target_video_id: str | None = None,
    time_gap_us: int = 0,
    decision_status: str = "candidate",
    evidence_ids: Sequence[str] | None = None,
    conflict_flag: bool = False,
    manual_review_required: bool = False,
    reason_codes: Sequence[str] | None = None,
    similarity_score: float | None = None,
    cut_or_window_ref: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "link_id": link_id,
        "source_track_id": source_track_id,
        "target_track_id": target_track_id,
        "source_video_id": source_video_id or video_id,
        "target_video_id": target_video_id or video_id,
        "time_gap_us": time_gap_us,
        "cut_or_window_ref": cut_or_window_ref,
        "evidence_ids": list(evidence_ids or []),
        "similarity_score": similarity_score,
        "decision_status": decision_status,
        "conflict_flag": conflict_flag,
        "manual_review_required": manual_review_required,
        "producer": "identity_contract_synth",
        "config_fingerprint": policy_fingerprint,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def summary_row(run_id: str, video_id: str, track_id: int) -> dict[str, Any]:
    """Minimal track_summaries-compatible stub fields used for FK checks."""
    # Full cast via contract when needed by callers.
    return {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": track_id,
    }


def manual_anchor_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_manual_01",
            evidence_type="manual_track_anchor",
            reliability_tier="manual_verified",
            track_id=0,
            start_frame_index=0,
            end_frame_index=20,
        )
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="confirmed",
            evidence_ids=["ev_manual_01"],
            supporting=1,
            metric_eligibility="eligible",
            policy_fingerprint=policy_fingerprint,
            end_frame_index=20,
            reason_codes=["MANUAL_VERIFIED_IN_SCOPE"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def two_supporting_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_jersey_01",
            evidence_type="jersey_number",
            reliability_tier="supporting",
            source_artifact_ref="jersey_observations",
            source_fingerprint="aabc7642" + "0" * 56,
        ),
        evidence_row(
            rid,
            vid,
            "ev_team_01",
            evidence_type="team_assignment",
            reliability_tier="supporting",
            source_artifact_ref="team_assignments",
            source_fingerprint="759aa9b7" + "0" * 56,
        ),
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_prov_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="provisional",
            evidence_ids=["ev_jersey_01", "ev_team_01"],
            supporting=2,
            metric_eligibility="provisional_only",
            policy_fingerprint=policy_fingerprint,
            reason_codes=["MULTI_SUPPORTING_PROVISIONAL"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def alone_insufficient_bundle(
    policy_fingerprint: str, *, evidence_type: str, evidence_id: str = "ev_alone_01"
) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    reason_map = {
        "jersey_number": "JERSEY_ALONE_INSUFFICIENT",
        "team_assignment": "TEAM_ALONE_INSUFFICIENT",
        "appearance_similarity": "APPEARANCE_ALONE_INSUFFICIENT",
        "role_consistency": "ROLE_ALONE_INSUFFICIENT",
    }
    ev = [
        evidence_row(
            rid,
            vid,
            evidence_id,
            evidence_type=evidence_type,
            reliability_tier="supporting",
            score=0.91 if evidence_type == "appearance_similarity" else None,
        )
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_cand_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="candidate",
            evidence_ids=[evidence_id],
            supporting=1,
            metric_eligibility="not_eligible",
            policy_fingerprint=policy_fingerprint,
            reason_codes=[reason_map.get(evidence_type, "SINGLE_WEAK_CANNOT_CONFIRM")],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def conflicting_jersey_team_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_jersey_c",
            evidence_type="jersey_number",
            reliability_tier="supporting",
            polarity="supports",
        ),
        evidence_row(
            rid,
            vid,
            "ev_team_c",
            evidence_type="team_assignment",
            reliability_tier="conflicting",
            polarity="conflicts",
        ),
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_rej_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="rejected",
            evidence_ids=["ev_jersey_c", "ev_team_c"],
            supporting=1,
            conflicting=1,
            metric_eligibility="not_eligible",
            policy_fingerprint=policy_fingerprint,
            manual_review_required=True,
            reason_codes=["HARD_EVIDENCE_CONFLICT"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def two_target_candidates_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid, vid, "ev_a", evidence_type="jersey_number", reliability_tier="supporting"
        ),
        evidence_row(
            rid, vid, "ev_b", evidence_type="team_assignment", reliability_tier="supporting"
        ),
        evidence_row(
            rid,
            vid,
            "ev_c",
            evidence_type="appearance_similarity",
            reliability_tier="supporting",
            track_id=1,
        ),
        evidence_row(
            rid,
            vid,
            "ev_d",
            evidence_type="role_consistency",
            reliability_tier="supporting",
            track_id=1,
        ),
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_t0",
            track_id=0,
            target_player_id=tid,
            assignment_status="provisional",
            evidence_ids=["ev_a", "ev_b"],
            supporting=2,
            metric_eligibility="provisional_only",
            policy_fingerprint=policy_fingerprint,
            manual_review_required=True,
        ),
        assignment_row(
            rid,
            vid,
            "asn_t1",
            track_id=1,
            target_player_id=tid,
            assignment_status="provisional",
            evidence_ids=["ev_c", "ev_d"],
            supporting=2,
            metric_eligibility="provisional_only",
            policy_fingerprint=policy_fingerprint,
            manual_review_required=True,
        ),
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def cross_shot_link_bundle(policy_fingerprint: str, *, long_gap: bool = False) -> dict[str, Any]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    gap = 5_000_000 if long_gap else 200_000
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_temp_01",
            evidence_type="temporal_continuity",
            reliability_tier="weak",
        ),
        evidence_row(
            rid,
            vid,
            "ev_spat_01",
            evidence_type="spatial_motion_continuity",
            reliability_tier="weak",
            track_id=1,
        ),
    ]
    links = [
        link_row(
            rid,
            vid,
            "link_cut_01",
            source_track_id=0,
            target_track_id=1,
            policy_fingerprint=policy_fingerprint,
            time_gap_us=gap,
            decision_status="review_required" if long_gap else "candidate",
            evidence_ids=["ev_temp_01", "ev_spat_01"],
            manual_review_required=True,
            cut_or_window_ref="shot_cut_03",
            reason_codes=["LONG_GAP"] if long_gap else ["CROSS_SHOT_CANDIDATE"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", []),
        "reid_candidate_links": _cast("reid_candidate_links", links),
        "evidence_rows": ev,
        "assignment_rows": [],
        "link_rows": links,
    }


def cross_video_auto_link_rows(policy_fingerprint: str) -> list[dict[str, Any]]:
    ids = base_ids()
    rid, vid = ids["run_id"], ids["video_id"]
    return [
        link_row(
            rid,
            vid,
            "link_xvid_01",
            source_track_id=0,
            target_track_id=0,
            policy_fingerprint=policy_fingerprint,
            source_video_id=vid,
            target_video_id="video_other_99",
            decision_status="candidate",
            reason_codes=[],
        )
    ]


def leakage_negative_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_eval_01",
            evidence_type="manual_track_anchor",
            reliability_tier="manual_verified",
            leakage_class="evaluation",
            quality_flags=["frozen_eval_gt_misuse_candidate"],
        )
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_leak_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="confirmed",
            evidence_ids=["ev_eval_01"],
            supporting=1,
            metric_eligibility="eligible",
            policy_fingerprint=policy_fingerprint,
            leakage_class="production",
            reason_codes=["MANUAL_VERIFIED_IN_SCOPE"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def unknown_low_coverage_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    ev = [
        evidence_row(
            rid,
            vid,
            "ev_unk_01",
            evidence_type="unknown",
            reliability_tier="unavailable",
            polarity="neutral",
        )
    ]
    asn = [
        assignment_row(
            rid,
            vid,
            "asn_unk_01",
            track_id=0,
            target_player_id=tid,
            assignment_status="unknown",
            evidence_ids=["ev_unk_01"],
            supporting=0,
            metric_eligibility="not_evaluable",
            policy_fingerprint=policy_fingerprint,
            end_frame_index=0,
            reason_codes=["INSUFFICIENT_COVERAGE"],
        )
    ]
    return {
        **ids,
        "identity_evidence": _cast("identity_evidence", ev),
        "track_identity_assignments": _cast("track_identity_assignments", asn),
        "reid_candidate_links": _cast("reid_candidate_links", []),
        "evidence_rows": ev,
        "assignment_rows": asn,
        "link_rows": [],
    }


def audit_entry(
    *,
    run_id: str,
    video_id: str,
    target_player_id: str,
    audit_id: str = "audit_01",
    action: str = "confirm",
    previous_decision: str | None = "provisional",
    new_decision: str = "confirmed",
    assignment_id: str | None = "asn_01",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_id": audit_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "assignment_id": assignment_id,
        "track_id": 0,
        "actor_id": "reviewer_01",
        "acted_at_utc": "2026-07-23T12:00:00.000000Z",
        "action": action,
        "previous_decision": previous_decision,
        "new_decision": new_decision,
        "reason": "manual_scope_confirm",
        "artifact_hashes": ["b" * 64],
        "linked_evidence_ids": ["ev_manual_01"],
        "provenance": {"stage": "7A", "append_only": True, "notes": None},
    }


__all__ = [
    "base_ids",
    "evidence_row",
    "assignment_row",
    "link_row",
    "summary_row",
    "manual_anchor_bundle",
    "two_supporting_bundle",
    "alone_insufficient_bundle",
    "conflicting_jersey_team_bundle",
    "two_target_candidates_bundle",
    "cross_shot_link_bundle",
    "cross_video_auto_link_rows",
    "leakage_negative_bundle",
    "unknown_low_coverage_bundle",
    "audit_entry",
    "_cast",
]
