"""Canonical team_assignments + team identity evidence builders (Stage 7C)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.identity.appearance_profiles import AppearanceProfile
from football_analytics.identity.team_clustering import (
    ANONYMOUS_TEAM_IDS,
    TeamClusterModel,
    collect_seed_tracks,
    score_against_clusters,
    team_feature_vector,
)
from football_analytics.identity.types import (
    CONTRACT_VERSION,
    EvidencePolarity,
    EvidenceType,
    LeakageClass,
    ReliabilityTier,
    ReviewStatus,
)

PRODUCER = "team_assignment_baseline"
PRODUCER_VERSION = "0.1.0"

ALLOWED_TEAM_ROLES = frozenset(
    {"home", "away", "official", "goalkeeper_home", "goalkeeper_away", "unknown"}
)


class TeamAssignmentError(ValueError):
    """Team assignment construction failure."""


@dataclass(frozen=True)
class TrackTeamDecision:
    track_id: int
    team_id: str
    team_role: str
    status: str
    confidence: float | None
    distance: float | None
    margin: float | None
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    review_required: bool
    start_frame_index: int
    end_frame_index: int
    role: str


def _conf_from_distance(distance: float | None, margin: float | None) -> float | None:
    if distance is None:
        return None
    base = max(0.0, min(1.0, 1.0 - float(distance)))
    if margin is not None:
        base = max(0.0, min(1.0, 0.7 * base + 0.3 * min(1.0, float(margin) * 5.0)))
    return float(base)


def decide_track_team(
    profile: AppearanceProfile,
    *,
    role: str,
    model: TeamClusterModel,
    config: Mapping[str, Any],
    prior_team_id: str | None = None,
) -> TrackTeamDecision:
    """Role-aware anonymous team decision for one tracklet."""
    asn = config["assignment"]
    role_l = str(role).lower()
    start = int(profile.start_frame_index if profile.start_frame_index is not None else 0)
    end = int(profile.end_frame_index if profile.end_frame_index is not None else start)
    flags: list[str] = ["anonymous_team_labels", "no_home_away", "no_real_team_name"]
    reasons: list[str] = []

    if role_l in {"referee", "staff"}:
        return TrackTeamDecision(
            track_id=profile.track_id,
            team_id=str(asn["referee_staff_team_id"]),
            team_role=str(asn["referee_staff_team_role"]),
            status="not_eligible",
            confidence=None,
            distance=None,
            margin=None,
            reason_codes=("ROLE_NOT_ELIGIBLE", f"ROLE:{role_l}"),
            quality_flags=tuple([*flags, "not_eligible", f"role_{role_l}"]),
            review_required=False,
            start_frame_index=start,
            end_frame_index=end,
            role=role_l,
        )

    if role_l == "goalkeeper":
        # Never auto-bind GK from kit alone.
        return TrackTeamDecision(
            track_id=profile.track_id,
            team_id=str(asn["goalkeeper_default_team_id"]),
            team_role="unknown",
            status="unknown",
            confidence=None,
            distance=None,
            margin=None,
            reason_codes=("GOALKEEPER_NO_AUTO_TEAM_FROM_KIT",),
            quality_flags=tuple([*flags, "goalkeeper_kit_unbound", "review_recommended"]),
            review_required=True,
            start_frame_index=start,
            end_frame_index=end,
            role=role_l,
        )

    if profile.status != "ok":
        return TrackTeamDecision(
            track_id=profile.track_id,
            team_id="unknown",
            team_role=str(asn["player_team_role"]),
            status="unknown",
            confidence=None,
            distance=None,
            margin=None,
            reason_codes=("INSUFFICIENT_APPEARANCE",),
            quality_flags=tuple([*flags, "insufficient_appearance"]),
            review_required=False,
            start_frame_index=start,
            end_frame_index=end,
            role=role_l,
        )

    if model.status != "ok":
        return TrackTeamDecision(
            track_id=profile.track_id,
            team_id="unknown",
            team_role=str(asn["player_team_role"]),
            status="unknown",
            confidence=None,
            distance=None,
            margin=None,
            reason_codes=tuple(model.reason_codes) or ("INSUFFICIENT_TEAM_EVIDENCE",),
            quality_flags=tuple([*flags, "insufficient_team_evidence"]),
            review_required=False,
            start_frame_index=start,
            end_frame_index=end,
            role=role_l,
        )

    vec = team_feature_vector(profile.embedding, config=config)
    outlier_ids = set(int(x) for x in (model.provenance.get("outlier_track_ids") or []))
    if profile.track_id in outlier_ids:
        return TrackTeamDecision(
            track_id=profile.track_id,
            team_id="unknown",
            team_role=str(asn["player_team_role"]),
            status="unknown",
            confidence=None,
            distance=None,
            margin=None,
            reason_codes=("THIRD_COLOR_OUTLIER", "OUTLIER_SEED_REJECTED"),
            quality_flags=tuple([*flags, "third_color_outlier", "status_unknown"]),
            review_required=True,
            start_frame_index=start,
            end_frame_index=end,
            role=role_l,
        )
    scored = score_against_clusters(vec, model, config=config)
    team_id = str(scored["team_id"])
    if team_id not in ANONYMOUS_TEAM_IDS:
        raise TeamAssignmentError(f"non-anonymous team_id: {team_id}")
    status = str(scored["status"])
    reasons.extend(str(x) for x in scored["reason_codes"])
    review = status == "ambiguous"
    if status == "ambiguous":
        flags.append("ambiguous")
        flags.append("similar_kit_risk")

    # Unknown role: at most candidate after clusters exist.
    if role_l == "unknown":
        if status == "assigned":
            status = "candidate"
            reasons.append("UNKNOWN_ROLE_CANDIDATE_ONLY")
            flags.append("unknown_role_candidate")
            review = True
        team_role = str(asn["player_team_role"])
    else:
        team_role = str(asn["player_team_role"])

    # Track team-switch conflict vs prior interval assignment.
    if (
        prior_team_id is not None
        and prior_team_id in {"team_a", "team_b"}
        and team_id in {"team_a", "team_b"}
        and prior_team_id != team_id
    ):
        status = "conflict"
        team_id = "unknown"
        reasons.append("TEAM_SWITCH_CONFLICT")
        flags.append("team_switch_conflict")
        review = True

    conf = _conf_from_distance(scored.get("distance"), scored.get("margin"))
    if status in {"unknown", "ambiguous", "conflict", "candidate"} and team_id == "unknown":
        conf = None if status != "candidate" else conf

    if status == "assigned":
        flags.append("status_assigned")
    elif status == "candidate":
        flags.append("status_candidate")
    elif status == "ambiguous":
        flags.append("status_ambiguous")
    elif status == "conflict":
        flags.append("status_conflict")
    else:
        flags.append("status_unknown")

    return TrackTeamDecision(
        track_id=profile.track_id,
        team_id=team_id if status != "ambiguous" else "unknown",
        team_role=team_role,
        status=status,
        confidence=conf if status in {"assigned", "candidate"} else None,
        distance=scored.get("distance"),
        margin=scored.get("margin"),
        reason_codes=tuple(dict.fromkeys(reasons)),
        quality_flags=tuple(dict.fromkeys(flags)),
        review_required=review,
        start_frame_index=start,
        end_frame_index=end,
        role=role_l,
    )


def decisions_to_assignment_rows(
    decisions: Sequence[TrackTeamDecision],
    *,
    run_id: str,
    video_id: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Canonical team_assignments rows (schema-compatible; anonymous IDs only)."""
    source = str(config["assignment"]["source"])
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(sorted(decisions, key=lambda x: x.track_id)):
        if d.team_id not in ANONYMOUS_TEAM_IDS:
            raise TeamAssignmentError(f"non-anonymous team_id forbidden: {d.team_id}")
        if d.team_role not in ALLOWED_TEAM_ROLES:
            raise TeamAssignmentError(f"invalid team_role: {d.team_role}")
        # Never invent home/away or GK home/away from kit.
        if d.team_role in {"home", "away", "goalkeeper_home", "goalkeeper_away"}:
            raise TeamAssignmentError("home/away / GK home/away auto forbidden in Stage 7C")
        qf = list(d.quality_flags) + [f"reason:{r}" for r in d.reason_codes[:6]]
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "assignment_id": int(i),
                "track_id": int(d.track_id),
                "start_frame_index": int(d.start_frame_index),
                "end_frame_index": int(d.end_frame_index),
                "team_id": d.team_id,
                "team_role": d.team_role,
                "confidence": d.confidence,
                "source": source,
                "quality_flags": qf,
            }
        )
    return rows


def decisions_to_evidence_rows(
    decisions: Sequence[TrackTeamDecision],
    *,
    run_id: str,
    video_id: str,
    config_fingerprint: str,
    cluster_fingerprint: str | None,
    leakage_class: str = LeakageClass.SYNTHETIC.value,
) -> list[dict[str, Any]]:
    """Stage 7A identity_evidence rows — team_assignment supporting only."""
    rows: list[dict[str, Any]] = []
    for d in sorted(decisions, key=lambda x: x.track_id):
        if d.status == "not_eligible":
            polarity = EvidencePolarity.NEUTRAL.value
            tier = ReliabilityTier.UNAVAILABLE.value
            reasons = list(d.reason_codes) + ["TEAM_ALONE_INSUFFICIENT"]
        elif d.status in {"conflict"}:
            polarity = EvidencePolarity.CONFLICTS.value
            tier = ReliabilityTier.CONFLICTING.value
            reasons = list(d.reason_codes) + ["TEAM_ALONE_INSUFFICIENT"]
        elif d.status in {"assigned", "candidate"} and d.team_id in {"team_a", "team_b"}:
            polarity = EvidencePolarity.SUPPORTS.value
            tier = ReliabilityTier.SUPPORTING.value
            reasons = list(d.reason_codes) + ["TEAM_ALONE_INSUFFICIENT"]
        else:
            polarity = EvidencePolarity.NEUTRAL.value
            tier = ReliabilityTier.WEAK.value
            reasons = list(d.reason_codes) + ["TEAM_ALONE_INSUFFICIENT"]

        # Never emit strong/manual_verified from team evidence alone.
        if tier in {ReliabilityTier.STRONG.value, ReliabilityTier.MANUAL_VERIFIED.value}:
            raise TeamAssignmentError("team evidence tier too strong")

        eid = f"tev_{video_id}_t{d.track_id}"
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "evidence_id": eid,
                "track_id": int(d.track_id),
                "frame_index": None,
                "start_frame_index": int(d.start_frame_index),
                "end_frame_index": int(d.end_frame_index),
                "start_time_us": None,
                "end_time_us": None,
                "evidence_type": EvidenceType.TEAM_ASSIGNMENT.value,
                "source_artifact_ref": "team_assignments",
                "source_fingerprint": cluster_fingerprint,
                "observed_value_ref": d.team_id,
                "score": d.confidence,
                "reliability_tier": tier,
                "polarity": polarity,
                "review_status": (
                    ReviewStatus.NEEDS_REVIEW.value
                    if d.review_required
                    else ReviewStatus.UNREVIEWED.value
                ),
                "producer": PRODUCER,
                "producer_version": PRODUCER_VERSION,
                "reason_codes": reasons,
                "quality_flags": list(d.quality_flags),
                "leakage_class": leakage_class,
                "provenance_json": json.dumps(
                    {
                        "status": d.status,
                        "distance": d.distance,
                        "margin": d.margin,
                        "role": d.role,
                        "config_fingerprint": config_fingerprint,
                        "auto_confirm": False,
                    },
                    sort_keys=True,
                ),
                "contract_version": CONTRACT_VERSION,
            }
        )
    return rows


def build_team_decisions(
    profiles: Sequence[AppearanceProfile],
    *,
    config: Mapping[str, Any],
    role_by_track: Mapping[int, str],
    model: TeamClusterModel,
    prior_team_by_track: Mapping[int, str] | None = None,
) -> tuple[list[TrackTeamDecision], list[dict[str, Any]]]:
    seeds, rejected = collect_seed_tracks(profiles, config=config, role_by_track=role_by_track)
    _ = seeds
    decisions: list[TrackTeamDecision] = []
    for p in sorted(profiles, key=lambda x: x.track_id):
        role = str(role_by_track.get(p.track_id, "unknown")).lower()
        prior = (prior_team_by_track or {}).get(p.track_id)
        decisions.append(
            decide_track_team(p, role=role, model=model, config=config, prior_team_id=prior)
        )
    return decisions, rejected


__all__ = [
    "PRODUCER",
    "PRODUCER_VERSION",
    "TeamAssignmentError",
    "TrackTeamDecision",
    "decide_track_team",
    "decisions_to_assignment_rows",
    "decisions_to_evidence_rows",
    "build_team_decisions",
]
