"""Appearance ReID candidate matching (Stage 7B) — candidates only, no merge."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.identity.appearance_descriptor import cosine_similarity
from football_analytics.identity.appearance_profiles import (
    PRODUCER,
    PRODUCER_VERSION,
    AppearanceProfile,
)
from football_analytics.identity.policy import decide_assignment_status
from football_analytics.identity.types import (
    CONTRACT_VERSION,
    EvidencePolarity,
    EvidenceType,
    LeakageClass,
    LinkDecisionStatus,
    ReliabilityTier,
    ReviewStatus,
)


class AppearanceMatchingError(ValueError):
    """Matching failure."""


@dataclass(frozen=True)
class MatchPair:
    source_track_id: int
    target_track_id: int
    similarity: float
    decision_status: str
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    manual_review_required: bool
    conflict_flag: bool
    time_gap_us: int
    evidence_tier: str


def _intervals_overlap(a0: int | None, a1: int | None, b0: int | None, b1: int | None) -> bool:
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return False
    return not (a1 < b0 or b1 < a0)


def _time_gap_us(a0: int | None, a1: int | None, b0: int | None, b1: int | None) -> int:
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return 0
    if _intervals_overlap(a0, a1, b0, b1):
        return 0
    if a1 < b0:
        return int(b0 - a1)
    return int(a0 - b1)


def score_profile_pair(
    left: AppearanceProfile,
    right: AppearanceProfile,
    *,
    config: Mapping[str, Any],
    same_video: bool,
    left_entity: str = "human",
    right_entity: str = "human",
    role_conflict: bool = False,
) -> MatchPair | None:
    """Score one ordered pair; returns None if skipped (self)."""
    if left.track_id == right.track_id:
        return None
    matching = config["matching"]
    reasons: list[str] = []
    flags: list[str] = ["same_kit_false_match_risk", "no_physical_merge"]
    conflict = False
    review = False

    if matching["human_ball_forbidden"] and (left_entity == "ball" or right_entity == "ball"):
        return MatchPair(
            source_track_id=left.track_id,
            target_track_id=right.track_id,
            similarity=0.0,
            decision_status=LinkDecisionStatus.REJECTED.value,
            reason_codes=("HUMAN_BALL_LINK_FORBIDDEN",),
            quality_flags=tuple(flags),
            manual_review_required=False,
            conflict_flag=True,
            time_gap_us=0,
            evidence_tier=ReliabilityTier.UNAVAILABLE.value,
        )

    if matching["cross_video_forbidden"] and not same_video:
        return MatchPair(
            source_track_id=left.track_id,
            target_track_id=right.track_id,
            similarity=0.0,
            decision_status=LinkDecisionStatus.REJECTED.value,
            reason_codes=("CROSS_VIDEO_AUTO_LINK_FORBIDDEN",),
            quality_flags=tuple(flags),
            manual_review_required=False,
            conflict_flag=True,
            time_gap_us=0,
            evidence_tier=ReliabilityTier.UNAVAILABLE.value,
        )

    if left.status != "ok" or right.status != "ok":
        return MatchPair(
            source_track_id=left.track_id,
            target_track_id=right.track_id,
            similarity=0.0,
            decision_status=LinkDecisionStatus.REJECTED.value,
            reason_codes=("insufficient_appearance_evidence",),
            quality_flags=tuple(flags),
            manual_review_required=True,
            conflict_flag=False,
            time_gap_us=0,
            evidence_tier=ReliabilityTier.UNAVAILABLE.value,
        )

    overlap = _intervals_overlap(
        left.start_frame_index,
        left.end_frame_index,
        right.start_frame_index,
        right.end_frame_index,
    )
    gap = _time_gap_us(
        left.start_time_us or (left.start_frame_index or 0),
        left.end_time_us or (left.end_frame_index or 0),
        right.start_time_us or (right.start_frame_index or 0),
        right.end_time_us or (right.end_frame_index or 0),
    )
    if matching["temporal_overlap_forbidden"] and overlap:
        return MatchPair(
            source_track_id=left.track_id,
            target_track_id=right.track_id,
            similarity=0.0,
            decision_status=LinkDecisionStatus.REJECTED.value,
            reason_codes=("TEMPORAL_OVERLAP_FORBIDDEN",),
            quality_flags=tuple(flags),
            manual_review_required=False,
            conflict_flag=True,
            time_gap_us=0,
            evidence_tier=ReliabilityTier.UNAVAILABLE.value,
        )

    sim = cosine_similarity(left.embedding, right.embedding)
    # Prefer raw cosine clamped to [0,1] for candidate ranking when L2-normalized.
    rank_score = float(max(0.0, min(1.0, sim)))

    if role_conflict:
        reasons.append("ROLE_CONFLICT_WARNING")
        flags.append("role_conflict")
        conflict = True
        review = True

    thr = float(matching["similarity_threshold"])
    reject_below = float(matching["reject_below"])
    margin = float(matching["ambiguity_margin"])

    if rank_score < reject_below:
        status = LinkDecisionStatus.REJECTED.value
        reasons.append("BELOW_REJECT_THRESHOLD")
        tier = ReliabilityTier.WEAK.value
    elif rank_score < thr:
        status = LinkDecisionStatus.REJECTED.value
        reasons.append("BELOW_CANDIDATE_THRESHOLD")
        tier = ReliabilityTier.WEAK.value
    else:
        status = LinkDecisionStatus.CANDIDATE.value
        reasons.append("APPEARANCE_SIMILARITY_CANDIDATE")
        tier = (
            ReliabilityTier.SUPPORTING.value
            if rank_score >= thr + margin
            else ReliabilityTier.WEAK.value
        )
        # Cross-shot gap → still candidate evidence only.
        if gap > 0:
            flags.append("cross_shot_or_gap_candidate")

    if matching["auto_confirm"] is not False:
        raise AppearanceMatchingError("auto_confirm must be false")

    return MatchPair(
        source_track_id=min(left.track_id, right.track_id),
        target_track_id=max(left.track_id, right.track_id),
        similarity=rank_score,
        decision_status=status,
        reason_codes=tuple(dict.fromkeys(reasons)),
        quality_flags=tuple(dict.fromkeys(flags)),
        manual_review_required=review,
        conflict_flag=conflict,
        time_gap_us=int(max(0, gap)),
        evidence_tier=tier,
    )


def propose_reid_candidates(
    profiles: Sequence[AppearanceProfile],
    *,
    config: Mapping[str, Any],
    video_id: str,
    entity_by_track: Mapping[int, str] | None = None,
    role_conflict_pairs: set[tuple[int, int]] | None = None,
) -> list[MatchPair]:
    """Deterministic pairwise ranking with mutual-nearest + ambiguity margin."""
    matching = config["matching"]
    entities = entity_by_track or {}
    conflicts = role_conflict_pairs or set()
    ok_profiles = [p for p in profiles if p.status == "ok"]
    ok_profiles = sorted(ok_profiles, key=lambda p: int(p.track_id))

    # All unordered pairs scored once (canonical track order).
    raw: list[MatchPair] = []
    for i, left in enumerate(ok_profiles):
        for right in ok_profiles[i + 1 :]:
            pair_key = (min(left.track_id, right.track_id), max(left.track_id, right.track_id))
            role_c = pair_key in conflicts
            m = score_profile_pair(
                left,
                right,
                config=config,
                same_video=True,
                left_entity=entities.get(left.track_id, "human"),
                right_entity=entities.get(right.track_id, "human"),
                role_conflict=role_c,
            )
            if m is not None:
                raw.append(m)

    # Also emit explicit rejects for non-ok / entity issues when requested via full list
    for p in profiles:
        if entities.get(p.track_id) == "ball":
            for other in profiles:
                if other.track_id == p.track_id:
                    continue
                m = score_profile_pair(
                    p,
                    other,
                    config=config,
                    same_video=True,
                    left_entity="ball",
                    right_entity=entities.get(other.track_id, "human"),
                )
                if m is not None and m.decision_status == LinkDecisionStatus.REJECTED.value:
                    raw.append(m)

    # Mutual nearest filtering for candidates
    by_track: dict[int, list[MatchPair]] = {}
    for m in raw:
        if m.decision_status != LinkDecisionStatus.CANDIDATE.value:
            continue
        by_track.setdefault(m.source_track_id, []).append(m)
        by_track.setdefault(m.target_track_id, []).append(m)

    def best_for(track_id: int) -> MatchPair | None:
        cands = by_track.get(track_id, [])
        if not cands:
            return None
        return sorted(
            cands,
            key=lambda x: (-x.similarity, x.source_track_id, x.target_track_id),
        )[0]

    margin = float(matching["ambiguity_margin"])
    cap = int(matching["candidate_cap_per_track"])
    out: list[MatchPair] = []
    seen: set[tuple[int, int]] = set()

    for m in sorted(raw, key=lambda x: (-x.similarity, x.source_track_id, x.target_track_id)):
        key = (m.source_track_id, m.target_track_id)
        if key in seen:
            continue
        seen.add(key)
        status = m.decision_status
        reasons = list(m.reason_codes)
        review = m.manual_review_required
        flags = list(m.quality_flags)

        if status == LinkDecisionStatus.CANDIDATE.value:
            # Ambiguity: second-best within margin
            for tid in (m.source_track_id, m.target_track_id):
                cands = sorted(
                    by_track.get(tid, []),
                    key=lambda x: (-x.similarity, x.source_track_id, x.target_track_id),
                )
                if len(cands) >= 2 and (cands[0].similarity - cands[1].similarity) < margin:
                    status = LinkDecisionStatus.REVIEW_REQUIRED.value
                    reasons.append("AMBIGUOUS_MARGIN")
                    review = True
                    flags.append("ambiguous")
                    break
            if matching["mutual_nearest"]:
                b_src = best_for(m.source_track_id)
                b_tgt = best_for(m.target_track_id)
                if b_src is None or b_tgt is None:
                    continue
                src_partner = (
                    b_src.target_track_id
                    if b_src.source_track_id == m.source_track_id
                    else b_src.source_track_id
                )
                tgt_partner = (
                    b_tgt.target_track_id
                    if b_tgt.source_track_id == m.target_track_id
                    else b_tgt.source_track_id
                )
                if (
                    not (
                        {src_partner, m.source_track_id} == {m.source_track_id, m.target_track_id}
                        or {tgt_partner, m.target_track_id}
                        == {m.source_track_id, m.target_track_id}
                    )
                    and status == LinkDecisionStatus.CANDIDATE.value
                ):
                    status = LinkDecisionStatus.REVIEW_REQUIRED.value
                    reasons.append("NOT_MUTUAL_NEAREST")
                    review = True

        # Cap per track counted on output candidates/review
        if status in {
            LinkDecisionStatus.CANDIDATE.value,
            LinkDecisionStatus.REVIEW_REQUIRED.value,
        }:
            src_n = sum(
                1
                for o in out
                if o.source_track_id == m.source_track_id or o.target_track_id == m.source_track_id
            )
            tgt_n = sum(
                1
                for o in out
                if o.source_track_id == m.target_track_id or o.target_track_id == m.target_track_id
            )
            if src_n >= cap or tgt_n >= cap:
                status = LinkDecisionStatus.REJECTED.value
                reasons.append("CANDIDATE_CAP")

        out.append(
            MatchPair(
                source_track_id=m.source_track_id,
                target_track_id=m.target_track_id,
                similarity=m.similarity,
                decision_status=status,
                reason_codes=tuple(dict.fromkeys(reasons)),
                quality_flags=tuple(dict.fromkeys(flags)),
                manual_review_required=review or status == LinkDecisionStatus.REVIEW_REQUIRED.value,
                conflict_flag=m.conflict_flag,
                time_gap_us=m.time_gap_us,
                evidence_tier=m.evidence_tier,
            )
        )

    # Ensure cross-video reject helper is available to callers via score_profile_pair
    _ = video_id
    return out


def match_to_evidence_and_link_rows(
    matches: Sequence[MatchPair],
    profiles: Sequence[AppearanceProfile],
    *,
    run_id: str,
    video_id: str,
    config_fingerprint: str,
    policy: Mapping[str, Any],
    leakage_class: str = LeakageClass.SYNTHETIC.value,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit identity_evidence + reid_candidate_links; appearance alone → candidate max."""
    profile_by_track = {p.track_id: p for p in profiles}
    evidence_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for idx, m in enumerate(matches):
        eid = f"ev_app_{video_id}_{idx:04d}"
        lid = f"reid_{video_id}_{m.source_track_id}_{m.target_track_id}_{idx:04d}"
        tier = m.evidence_tier
        if tier not in {
            ReliabilityTier.SUPPORTING.value,
            ReliabilityTier.WEAK.value,
            ReliabilityTier.UNAVAILABLE.value,
            ReliabilityTier.CONFLICTING.value,
        }:
            tier = ReliabilityTier.WEAK.value
        # Never emit strong/manual from appearance alone.
        if tier in {ReliabilityTier.STRONG.value, ReliabilityTier.MANUAL_VERIFIED.value}:
            tier = ReliabilityTier.SUPPORTING.value

        polarity = EvidencePolarity.SUPPORTS.value
        if m.decision_status == LinkDecisionStatus.REJECTED.value:
            polarity = EvidencePolarity.NEUTRAL.value
        if m.conflict_flag:
            polarity = EvidencePolarity.CONFLICTS.value
            tier = ReliabilityTier.CONFLICTING.value

        review_status = ReviewStatus.UNREVIEWED.value
        if m.manual_review_required:
            review_status = ReviewStatus.NEEDS_REVIEW.value

        src_prof = profile_by_track.get(m.source_track_id)
        evidence_rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "evidence_id": eid,
                "track_id": int(m.source_track_id),
                "frame_index": src_prof.start_frame_index if src_prof else None,
                "start_frame_index": src_prof.start_frame_index if src_prof else None,
                "end_frame_index": src_prof.end_frame_index if src_prof else None,
                "start_time_us": src_prof.start_time_us if src_prof else None,
                "end_time_us": src_prof.end_time_us if src_prof else None,
                "evidence_type": EvidenceType.APPEARANCE_SIMILARITY.value,
                "source_artifact_ref": (
                    src_prof.profile_id if src_prof else f"profile_t{m.source_track_id}"
                ),
                "source_fingerprint": src_prof.profile_fingerprint if src_prof else None,
                "observed_value_ref": f"track:{m.target_track_id}",
                "score": float(m.similarity),
                "reliability_tier": tier,
                "polarity": polarity,
                "review_status": review_status,
                "producer": PRODUCER,
                "producer_version": PRODUCER_VERSION,
                "reason_codes": list(m.reason_codes),
                "quality_flags": list(m.quality_flags),
                "leakage_class": leakage_class,
                "provenance_json": json.dumps(
                    {
                        "pair": [m.source_track_id, m.target_track_id],
                        "decision_status": m.decision_status,
                    },
                    sort_keys=True,
                ),
                "contract_version": CONTRACT_VERSION,
            }
        )

        # Stage 7A alone-insufficient: appearance cannot confirm.
        status, reasons = decide_assignment_status(
            [
                {
                    "evidence_type": EvidenceType.APPEARANCE_SIMILARITY.value,
                    "reliability_tier": tier,
                    "polarity": polarity,
                }
            ],
            policy=policy,
            within_manual_anchor_scope=False,
        )
        if status not in {"candidate", "rejected", "unknown"}:
            raise AppearanceMatchingError(
                f"appearance-only assignment status must be candidate-class, got {status}"
            )

        link_rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "link_id": lid,
                "source_track_id": int(m.source_track_id),
                "target_track_id": int(m.target_track_id),
                "source_video_id": video_id,
                "target_video_id": video_id,
                "time_gap_us": int(m.time_gap_us),
                "cut_or_window_ref": None,
                "evidence_ids": [eid],
                "similarity_score": float(m.similarity),
                "decision_status": m.decision_status,
                "conflict_flag": bool(m.conflict_flag),
                "manual_review_required": bool(m.manual_review_required),
                "producer": PRODUCER,
                "config_fingerprint": config_fingerprint,
                "reason_codes": list(dict.fromkeys([*m.reason_codes, *reasons])),
                "quality_flags": list(m.quality_flags),
                "provenance_json": json.dumps(
                    {"assignment_status_ceiling": status, "auto_confirm": False},
                    sort_keys=True,
                ),
                "contract_version": CONTRACT_VERSION,
            }
        )

    return evidence_rows, link_rows


__all__ = [
    "AppearanceMatchingError",
    "MatchPair",
    "score_profile_pair",
    "propose_reid_candidates",
    "match_to_evidence_and_link_rows",
]
