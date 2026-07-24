"""Human-ball interaction eligibility helpers (Stage 10A — contracts only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.interaction.types import InteractionContractError


def pitch_distance_usable(candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Pitch-space distance is usable only with valid calibration and known grounded ball."""
    reasons: list[str] = []
    if str(candidate.get("calibration_status", "")) != "valid":
        reasons.append("INVALID_CALIBRATION")
    air = str(candidate.get("ball_air_state", "unknown"))
    if air == "unknown" or air == "airborne":
        reasons.append("AIRBORNE_UNKNOWN_BLOCKS_PITCH")
    if str(candidate.get("ball_observation_state", "")) != "observed":
        reasons.append("PREDICTED_SOLE_EVIDENCE")
    if str(candidate.get("human_observation_state", "")) != "observed":
        reasons.append("PREDICTED_SOLE_EVIDENCE")
    return (len(reasons) == 0, reasons)


def proximity_eligible_as_evidence(candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(candidate.get("playability_status", "")) in {"replay", "non_playable"}:
        reasons.append("REPLAY_OR_CUT_TERMINATES")
    if str(candidate.get("ball_candidate_status", "")) == "missing":
        reasons.append("MISSING_BALL_NOT_NO_POSSESSION")
    if str(candidate.get("ball_candidate_status", "")) == "ambiguous":
        reasons.append("AMBIGUOUS_PRIMARY_BALL")
    if str(candidate.get("human_observation_state", "")) in {"predicted", "interpolated"}:
        reasons.append("PREDICTED_SOLE_EVIDENCE")
    if str(candidate.get("ball_observation_state", "")) in {"predicted", "interpolated", "missing"}:
        reasons.append("PREDICTED_SOLE_EVIDENCE")
    return (len(reasons) == 0, reasons)


def automatic_state_allowed(state: str, *, max_state: str = "provisional") -> bool:
    order = ["candidate", "provisional", "confirmed"]
    if state not in order:
        return state in {
            "contested",
            "unknown",
            "not_evaluable",
            "rejected",
        }
    return order.index(state) <= order.index(max_state)


def assert_no_automatic_confirmed(state: str, *, via_manual_review: bool = False) -> None:
    if state == "confirmed" and not via_manual_review:
        raise InteractionContractError("AUTOMATIC_CONFIRMED_FORBIDDEN")


def missing_ball_means_no_possession(*, policy: Mapping[str, Any]) -> bool:
    """Must always be False under Stage 10A policy."""
    return policy.get("eligibility", {}).get("missing_ball_is_not_no_possession") is not True


def low_joint_coverage_status(*, joint_coverage_ratio: float | None, threshold: float = 0.2) -> str:
    if joint_coverage_ratio is None:
        return "not_evaluable"
    if joint_coverage_ratio < threshold:
        return "not_evaluable"
    return "evaluable"


def target_relationship_eligible(rel: str) -> bool:
    return rel in {"confirmed_target", "candidate_target", "anonymous", "non_target", "unknown"}


def classify_input_kinds(candidate: Mapping[str, Any]) -> dict[str, str]:
    return {
        "human": str(candidate.get("human_observation_state", "unknown")),
        "ball": str(candidate.get("ball_observation_state", "unknown")),
        "evidence_space": str(candidate.get("evidence_space", "none")),
        "ball_air_state": str(candidate.get("ball_air_state", "unknown")),
        "ball_candidate_status": str(candidate.get("ball_candidate_status", "missing")),
        "target_relationship": str(candidate.get("target_relationship", "unknown")),
        "playability": str(candidate.get("playability_status", "unknown")),
        "calibration": str(candidate.get("calibration_status", "unknown")),
    }


def eligibility_reasons_summary(reasons: Sequence[str]) -> str:
    if not reasons:
        return "eligible"
    if "LOW_JOINT_COVERAGE_NOT_EVALUABLE" in reasons:
        return "not_evaluable"
    return "ineligible"


__all__ = [
    "pitch_distance_usable",
    "proximity_eligible_as_evidence",
    "automatic_state_allowed",
    "assert_no_automatic_confirmed",
    "missing_ball_means_no_possession",
    "low_joint_coverage_status",
    "target_relationship_eligible",
    "classify_input_kinds",
    "eligibility_reasons_summary",
]
