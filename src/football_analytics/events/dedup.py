"""Duplicate event suppression helpers (Stage 13A/13C)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def intervals_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
    *,
    tolerance_us: int = 0,
) -> bool:
    """Half-open interval overlap with optional tolerance expansion."""
    a0 = int(a_start) - int(tolerance_us)
    a1 = int(a_end) + int(tolerance_us)
    b0 = int(b_start)
    b1 = int(b_end)
    return a0 < b1 and b0 < a1


def _conf(row: Mapping[str, Any]) -> float:
    c = row.get("confidence")
    if c is None:
        return 0.0
    return float(c)


def _target_rank(row: Mapping[str, Any]) -> int:
    rel = str(row.get("target_relationship") or "unknown")
    order = {
        "confirmed_target": 3,
        "candidate_target": 2,
        "anonymous": 1,
        "non_target": 0,
        "unknown": 0,
    }
    return order.get(rel, 0)


def prefer_row(a: Mapping[str, Any], b: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prefer confirmed target, then higher confidence, then earlier start."""
    if _target_rank(a) != _target_rank(b):
        return a if _target_rank(a) > _target_rank(b) else b
    if _conf(a) != _conf(b):
        return a if _conf(a) > _conf(b) else b
    return a if int(a.get("start_time_us", 0)) <= int(b.get("start_time_us", 0)) else b


def suppress_duplicate_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str = "event_family",
    overlap_us: int = 500_000,
    id_key: str = "ledger_event_id",
) -> list[dict[str, Any]]:
    """Mark overlapping same-family candidates; preserve all rows (append-only)."""
    out: list[dict[str, Any]] = [dict(r) for r in rows]
    n = len(out)
    for i in range(n):
        out[i].setdefault("suppressed_duplicate", False)
        out[i].setdefault("suppressed_by_id", None)
        out[i].setdefault("overlap_group_id", None)
    for i in range(n):
        for j in range(i + 1, n):
            if str(out[i].get(family_key)) != str(out[j].get(family_key)):
                continue
            if not intervals_overlap(
                int(out[i]["start_time_us"]),
                int(out[i]["end_time_us"]),
                int(out[j]["start_time_us"]),
                int(out[j]["end_time_us"]),
                tolerance_us=overlap_us,
            ):
                continue
            group = f"ov_{min(i, j):04d}_{max(i, j):04d}"
            out[i]["overlap_group_id"] = out[i].get("overlap_group_id") or group
            out[j]["overlap_group_id"] = out[j].get("overlap_group_id") or group
            winner = prefer_row(out[i], out[j])
            loser = out[j] if winner is out[i] else out[i]
            if loser.get("suppressed_duplicate") is True:
                continue
            loser["suppressed_duplicate"] = True
            loser["suppressed_by_id"] = str(winner.get(id_key))
            reasons = list(loser.get("reason_codes") or [])
            if "DUPLICATE_SUPPRESSED" not in reasons:
                reasons.append("DUPLICATE_SUPPRESSED")
            loser["reason_codes"] = reasons
    return out


__all__ = [
    "intervals_overlap",
    "prefer_row",
    "suppress_duplicate_events",
]
