"""Machine-checkable lifecycle transition table (Stage 6A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.tracking.types import LifecycleState, TransitionError

# Default birth uses previous=None represented as the string "null" in policy.
DEFAULT_ALLOWED: Mapping[str, frozenset[str]] = {
    "null": frozenset({LifecycleState.TENTATIVE.value}),
    LifecycleState.TENTATIVE.value: frozenset(
        {
            LifecycleState.CONFIRMED.value,
            LifecycleState.LOST.value,
            LifecycleState.TERMINATED.value,
        }
    ),
    LifecycleState.CONFIRMED.value: frozenset(
        {LifecycleState.LOST.value, LifecycleState.TERMINATED.value}
    ),
    LifecycleState.LOST.value: frozenset(
        {LifecycleState.CONFIRMED.value, LifecycleState.TERMINATED.value}
    ),
    LifecycleState.TERMINATED.value: frozenset(),
}


def _as_state(value: str | LifecycleState | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, LifecycleState):
        return value.value
    return str(value)


def allowed_transitions_from_policy(policy: Mapping[str, Any] | None) -> dict[str, frozenset[str]]:
    if policy is None:
        return {k: frozenset(v) for k, v in DEFAULT_ALLOWED.items()}
    raw = policy.get("allowed_transitions")
    if not isinstance(raw, Mapping):
        return {k: frozenset(v) for k, v in DEFAULT_ALLOWED.items()}
    out: dict[str, frozenset[str]] = {}
    for src, dsts in raw.items():
        if not isinstance(dsts, (list, tuple)):
            raise TransitionError(f"allowed_transitions[{src}] must be a list")
        out[str(src)] = frozenset(str(d) for d in dsts)
    return out


def assert_transition_allowed(
    previous: str | LifecycleState | None,
    nxt: str | LifecycleState,
    *,
    policy: Mapping[str, Any] | None = None,
    allow_birth_confirmed: bool | None = None,
) -> None:
    """Raise TransitionError if previous → nxt is illegal under policy."""
    src = _as_state(previous)
    dst = _as_state(nxt)
    table = allowed_transitions_from_policy(policy)
    if src not in table:
        raise TransitionError(f"unknown previous lifecycle state: {src}")
    if dst not in {s.value for s in LifecycleState}:
        raise TransitionError(f"unknown next lifecycle state: {dst}")
    if dst not in table[src]:
        raise TransitionError(f"illegal transition {src} -> {dst}")
    birth_ok = allow_birth_confirmed
    if birth_ok is None and policy is not None:
        life = policy.get("lifecycle")
        if isinstance(life, Mapping):
            birth_ok = bool(life.get("allow_birth_confirmed", False))
    if birth_ok is False and src == "null" and dst == LifecycleState.CONFIRMED.value:
        raise TransitionError("birth directly to confirmed is forbidden by policy")
    if src == LifecycleState.TERMINATED.value:
        raise TransitionError("terminated track cannot transition (no reopen)")


def validate_lifecycle_sequence(
    events: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate ordered lifecycle events for one track; return error messages."""
    errors: list[str] = []
    if not events:
        return ["lifecycle sequence empty"]
    ordered = sorted(events, key=lambda e: int(e["event_index"]))
    prev_state: str | None = None
    prev_index = -1
    entity: str | None = None
    for ev in ordered:
        idx = int(ev["event_index"])
        if idx <= prev_index:
            errors.append(f"non-increasing event_index at {idx}")
            break
        prev_index = idx
        state = str(ev["lifecycle_state"])
        declared_prev = ev.get("previous_state")
        expected_prev = prev_state
        if declared_prev != expected_prev:
            errors.append(
                f"previous_state mismatch at event {idx}: "
                f"declared={declared_prev!r} expected={expected_prev!r}"
            )
            break
        try:
            assert_transition_allowed(expected_prev, state, policy=policy)
        except TransitionError as exc:
            errors.append(str(exc))
            break
        et = str(ev["entity_type"])
        if entity is None:
            entity = et
        elif et != entity:
            errors.append(f"entity_type changed within track at event {idx}")
            break
        prev_state = state
    return errors


__all__ = [
    "DEFAULT_ALLOWED",
    "allowed_transitions_from_policy",
    "assert_transition_allowed",
    "validate_lifecycle_sequence",
]
