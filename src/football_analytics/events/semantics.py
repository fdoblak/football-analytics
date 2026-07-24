"""Stage 13 events semantics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.events.eligibility import live_event_eligible

SOURCE_FAMILY_MAP: Mapping[str, str] = {
    "pass_candidates": "pass",
    "pass_outcomes": "pass",
    "reception_candidates": "reception",
    "ball_progression_segments": "progression",
    "target_ball_touches": "touch",
    "take_on_attempts": "take_on",
    "ground_duel_candidates": "ground_duel",
    "aerial_duel_candidates": "aerial_duel",
    "tackle_events": "tackle",
    "recovery_events": "recovery",
    "turnover_events": "turnover",
    "clearance_events": "clearance",
    "possession_hypotheses": "possession",
    "ball_contact_candidates": "contact",
}


def family_for_source_contract(source_contract: str) -> str:
    return str(SOURCE_FAMILY_MAP.get(source_contract, "other"))


def source_event_id_field(source_contract: str) -> str:
    mapping = {
        "pass_candidates": "pass_candidate_id",
        "pass_outcomes": "outcome_id",
        "reception_candidates": "reception_candidate_id",
        "ball_progression_segments": "segment_id",
        "target_ball_touches": "touch_id",
        "take_on_attempts": "take_on_attempt_id",
        "ground_duel_candidates": "ground_duel_candidate_id",
        "aerial_duel_candidates": "aerial_duel_candidate_id",
        "tackle_events": "tackle_event_id",
        "recovery_events": "recovery_event_id",
        "turnover_events": "turnover_event_id",
        "clearance_events": "clearance_event_id",
        "possession_hypotheses": "possession_hypothesis_id",
        "ball_contact_candidates": "contact_candidate_id",
    }
    return mapping.get(source_contract, "event_id")


def extract_source_event_id(source_contract: str, row: Mapping[str, Any]) -> str:
    field = source_event_id_field(source_contract)
    val = row.get(field) or row.get("event_id") or row.get("id")
    return str(val)


def replay_status_from_source(row: Mapping[str, Any]) -> str:
    if row.get("cut_or_replay") is True:
        return "unknown"
    status = row.get("replay_status")
    if status:
        return str(status)
    play = str(row.get("playability_status") or "")
    if play == "replay":
        return "replay"
    if play == "playable":
        return "live"
    return "unknown"


def event_live_eligible(row: Mapping[str, Any]) -> bool:
    return live_event_eligible(
        replay_status=replay_status_from_source(row),
        cut_or_replay=bool(row.get("cut_or_replay")),
        hard_gap=bool(row.get("hard_gap")),
        playability=str(row.get("playability_status") or ""),
    )


def evaluation_leakage_guard(payload: Mapping[str, Any]) -> list[str]:
    """Reject accuracy claims / GT labels leaking into production ledger."""
    violations: list[str] = []
    if payload.get("opta_accuracy_validated") is True:
        violations.append("EVALUATION_LEAKAGE_OPTA")
    if payload.get("real_football_accuracy_validated") is True:
        violations.append("EVALUATION_LEAKAGE_REAL_FOOTBALL")
    if payload.get("ground_truth_labels") is not None:
        violations.append("EVALUATION_LEAKAGE_GT")
    attrs = payload.get("attributes_json")
    if isinstance(attrs, str) and (
        '"ground_truth"' in attrs or '"opta_official"' in attrs or '"accuracy_claim"' in attrs
    ):
        violations.append("EVALUATION_LEAKAGE_ATTRIBUTES")
    return violations


__all__ = [
    "SOURCE_FAMILY_MAP",
    "family_for_source_contract",
    "source_event_id_field",
    "extract_source_event_id",
    "replay_status_from_source",
    "event_live_eligible",
    "evaluation_leakage_guard",
]
