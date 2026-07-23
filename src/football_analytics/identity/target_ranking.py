"""Deterministic target-candidate ranking (review aid only; Stage 7E)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.identity.types import AssignmentStatus, ReliabilityTier


def _tier_weight(tier: str) -> float:
    return {
        ReliabilityTier.MANUAL_VERIFIED.value: 1.0,
        ReliabilityTier.STRONG.value: 0.75,
        ReliabilityTier.SUPPORTING.value: 0.55,
        ReliabilityTier.WEAK.value: 0.25,
        ReliabilityTier.CONFLICTING.value: -0.8,
        ReliabilityTier.UNAVAILABLE.value: 0.0,
    }.get(tier, 0.0)


def rank_score_for_candidate(
    *,
    proposed_status: str,
    supporting_count: int,
    conflicting_count: int,
    reliability_tiers: Sequence[str],
    appearance_margin: float | None,
    team_jersey_consistent: bool,
    coverage_frames: int,
    has_manual_scope: bool,
    review_required: bool,
) -> float:
    score = 0.0
    if has_manual_scope:
        score += 2.0
    score += 0.35 * float(supporting_count)
    score -= 0.6 * float(conflicting_count)
    score += sum(_tier_weight(t) for t in reliability_tiers)
    if appearance_margin is not None:
        score += max(0.0, min(0.4, float(appearance_margin)))
    if team_jersey_consistent:
        score += 0.25
    score += min(0.5, 0.01 * float(max(0, coverage_frames)))
    if proposed_status == AssignmentStatus.PROVISIONAL.value:
        score += 0.4
    elif proposed_status == AssignmentStatus.REJECTED.value:
        score -= 1.0
    if review_required:
        score += 0.05  # prioritize review visibility slightly
    return round(score, 6)


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    ambiguity_margin: float = 0.05,
    max_candidates: int = 64,
) -> list[dict[str, Any]]:
    """Sort candidates deterministically; mark near ties as ambiguous."""
    ordered = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            -float(c.get("rank_score", 0.0)),
            int(c.get("track_id", 0)),
            str(c.get("candidate_id", "")),
        ),
    )
    capped = ordered[: max(0, int(max_candidates))]
    for i, c in enumerate(capped):
        c["rank"] = i + 1
        if i + 1 < len(capped):
            gap = float(c["rank_score"]) - float(capped[i + 1]["rank_score"])
            if gap <= float(ambiguity_margin):
                c["ambiguous"] = True
                reasons = list(c.get("reason_codes") or [])
                if "AMBIGUOUS_NEAR_TIE" not in reasons:
                    reasons.append("AMBIGUOUS_NEAR_TIE")
                c["reason_codes"] = reasons
                c["manual_review_required"] = True
        else:
            c.setdefault("ambiguous", bool(c.get("ambiguous", False)))
    return capped


__all__ = [
    "rank_score_for_candidate",
    "rank_candidates",
]
