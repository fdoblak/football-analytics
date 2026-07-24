"""Passing eligibility helpers (Stage 11A)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def cut_replay_blocks_pass(row: Mapping[str, Any]) -> bool:
    return bool(row.get("cut_or_replay")) or str(row.get("playability_status")) in {
        "replay",
        "non_playable",
    }


def hard_gap_blocks_pass(row: Mapping[str, Any]) -> bool:
    return bool(row.get("hard_gap"))


def calibration_usable(row: Mapping[str, Any]) -> bool:
    return str(row.get("calibration_status")) == "valid"


def pass_candidate_eligible(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if cut_replay_blocks_pass(row):
        reasons.append("CUT_REPLAY_GAP_NO_PASS")
    if hard_gap_blocks_pass(row):
        reasons.append("HARD_GAP_NO_PASS")
    if bool(row.get("owner_change_alone")) and not row.get("evidence_refs"):
        reasons.append("OWNER_CHANGE_ALONE_NOT_PASS")
    if (
        str(row.get("candidate_state")) == "confirmed"
        and str(row.get("review_status")) != "reviewed"
    ):
        reasons.append("AUTOMATIC_CONFIRMED_FORBIDDEN")
    return (len(reasons) == 0), reasons


def box_touch_eligible_status(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if bool(row.get("penalty_presence_alone")) and not bool(row.get("has_possession_or_contact")):
        reasons.append("PENALTY_PRESENCE_NOT_BOX_TOUCH")
    if not bool(row.get("has_possession_or_contact")):
        reasons.append("MISSING_CONTACT_OR_POSSESSION")
    if not bool(row.get("has_pitch_mapping")):
        reasons.append("INVALID_CALIBRATION")
    if str(row.get("playability_status")) != "playable":
        reasons.append("NON_PLAYABLE")
    return (len(reasons) == 0 and bool(row.get("is_box_touch_candidate"))), reasons


def low_coverage_not_evaluable(
    *, joint_coverage_ratio: float | None, threshold: float = 0.2
) -> bool:
    if joint_coverage_ratio is None:
        return True
    return float(joint_coverage_ratio) < float(threshold)


__all__ = [
    "cut_replay_blocks_pass",
    "hard_gap_blocks_pass",
    "calibration_usable",
    "pass_candidate_eligible",
    "box_touch_eligible_status",
    "low_coverage_not_evaluable",
]
