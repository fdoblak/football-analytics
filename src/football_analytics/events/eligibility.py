"""Live vs replay eligibility helpers for events (Stage 13A).

Never invent live when replay is uncertain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.broadcast.types import ReplayStatus


def live_event_eligible(
    *,
    replay_status: str,
    cut_or_replay: bool = False,
    hard_gap: bool = False,
    playability: str | None = None,
) -> bool:
    """True only when replay is explicitly live and window is otherwise usable."""
    if cut_or_replay or hard_gap:
        return False
    if playability in {"non_playable", "uncertain"}:
        return False
    return str(replay_status) == ReplayStatus.LIVE.value


def implies_live(replay_status: str) -> bool:
    """Never treat unknown/transition/replay as live."""
    return str(replay_status) == ReplayStatus.LIVE.value


def normalize_replay_status(
    *,
    candidate_status: str | None,
    confidence: float | None,
    live_confidence_min: float = 0.70,
    replay_confidence_min: float = 0.55,
) -> str:
    """Conservative status: uncertain → unknown (blocks live)."""
    allowed = {e.value for e in ReplayStatus}
    status = str(candidate_status or ReplayStatus.UNKNOWN.value)
    if status not in allowed:
        return ReplayStatus.UNKNOWN.value
    conf = float(confidence) if confidence is not None else 0.0
    if status == ReplayStatus.LIVE.value and conf < live_confidence_min:
        return ReplayStatus.UNKNOWN.value
    if status == ReplayStatus.REPLAY.value and conf < replay_confidence_min:
        return ReplayStatus.UNKNOWN.value
    if status == ReplayStatus.REPLAY_TRANSITION.value and conf < replay_confidence_min:
        return ReplayStatus.UNKNOWN.value
    return status


def eligibility_reason_codes(row: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    status = str(row.get("replay_status", ReplayStatus.UNKNOWN.value))
    if status == ReplayStatus.UNKNOWN.value:
        codes.append("REPLAY_UNCERTAIN_BLOCKS_LIVE")
    elif status in {ReplayStatus.REPLAY.value, ReplayStatus.REPLAY_TRANSITION.value}:
        codes.append("REPLAY_NOT_LIVE")
    if row.get("cut_or_replay") is True:
        codes.append("CUT_OR_REPLAY")
    if row.get("hard_gap") is True:
        codes.append("HARD_GAP")
    if (
        not live_event_eligible(
            replay_status=status,
            cut_or_replay=bool(row.get("cut_or_replay")),
            hard_gap=bool(row.get("hard_gap")),
            playability=str(row.get("playability_status") or row.get("playability") or ""),
        )
        and "REPLAY_UNCERTAIN_BLOCKS_LIVE" not in codes
        and "REPLAY_NOT_LIVE" not in codes
    ):
        codes.append("LIVE_EVENT_INELIGIBLE")
    return codes


__all__ = [
    "live_event_eligible",
    "implies_live",
    "normalize_replay_status",
    "eligibility_reason_codes",
]
