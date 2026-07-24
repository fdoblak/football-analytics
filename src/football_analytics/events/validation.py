"""Bundle validation for Stage 13 synthetic events artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.events.eligibility import implies_live, live_event_eligible
from football_analytics.events.semantics import evaluation_leakage_guard
from football_analytics.events.types import METRIC_ORIGIN, EventsContractError


def validate_events_bundle(
    *,
    ledger: Sequence[Mapping[str, Any]] | None = None,
    revisions: Sequence[Mapping[str, Any]] | None = None,
    replays: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    ledger = list(ledger or [])
    revisions = list(revisions or [])
    replays = list(replays or [])

    seen: set[tuple[str, str, str]] = set()
    for row in ledger:
        key = (str(row["run_id"]), str(row["video_id"]), str(row["ledger_event_id"]))
        if key in seen:
            raise EventsContractError(f"duplicate ledger PK: {key}")
        seen.add(key)
        if row.get("metric_origin") != METRIC_ORIGIN:
            raise EventsContractError("ledger metric_origin must be project_generated")
        if row.get("automatic_ceiling") not in {"candidate", "provisional"}:
            raise EventsContractError("automatic_ceiling must be candidate|provisional")
        if row.get("event_state") == "confirmed" and row.get("review_status") == "unreviewed":
            raise EventsContractError("automatic confirmed without review forbidden")
        leaks = evaluation_leakage_guard(row)
        if leaks:
            raise EventsContractError(f"evaluation leakage: {leaks}")
        status = str(row.get("replay_status"))
        eligible = bool(row.get("live_event_eligible"))
        if status == "unknown" and eligible:
            raise EventsContractError("unknown replay cannot be live_event_eligible")
        if status in {"replay", "replay_transition"} and eligible:
            raise EventsContractError("confirmed replay cannot be live_event_eligible")
        if eligible and not live_event_eligible(replay_status=status):
            raise EventsContractError("live_event_eligible inconsistent with replay_status")

    rev_seen: set[tuple[str, str, str]] = set()
    for row in revisions:
        key = (str(row["run_id"]), str(row["video_id"]), str(row["revision_id"]))
        if key in rev_seen:
            raise EventsContractError(f"duplicate revision PK: {key}")
        rev_seen.add(key)

    for row in replays:
        if row.get("implies_live") is True and str(row.get("replay_status")) != "live":
            raise EventsContractError("implies_live only allowed for live status")
        if str(row.get("replay_status")) == "unknown" and (
            row.get("implies_live") is True or row.get("live_event_eligible") is True
        ):
            raise EventsContractError("uncertain replay must not invent live")
        if row.get("live_event_eligible") is True and not implies_live(str(row["replay_status"])):
            raise EventsContractError("live_event_eligible requires live replay_status")


__all__ = ["validate_events_bundle"]
