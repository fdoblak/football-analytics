"""Cross-namespace mixing hard-fail guards."""

from __future__ import annotations

from football_analytics.acceptance.namespaces import (
    AUTHORITATIVE_SOCCERTRACK_TARGET,
    DEPRECATED_INVALID_TARGET,
    NAMESPACE_SELF_CONTAINED,
    NAMESPACE_SOCCERTRACK_REFERENCE,
    NAMESPACE_TEAMTRACK_REAL_VIDEO,
    TEAMTRACK_PILOT_TARGET,
)


class NamespaceIsolationError(ValueError):
    """Raised when evidence namespaces are mixed incorrectly."""


def assert_namespaces_isolated(
    *,
    soccertrack_player_id: str | None = None,
    teamtrack_track_id: int | None = None,
    claim_same_person: bool = False,
) -> None:
    if claim_same_person:
        raise NamespaceIsolationError("TeamTrack and SoccerTrack targets must not be merged")
    # coexistence allowed only without identity merge
    if (
        soccertrack_player_id
        and teamtrack_track_id is not None
        and str(soccertrack_player_id) == str(TEAMTRACK_PILOT_TARGET["anonymous_track_id"])
    ):
        raise NamespaceIsolationError("numeric identity collision misuse")
    if soccertrack_player_id == str(DEPRECATED_INVALID_TARGET["player_id"]):
        raise NamespaceIsolationError("deprecated SoccerTrack target refused")
    if (
        soccertrack_player_id
        and soccertrack_player_id != AUTHORITATIVE_SOCCERTRACK_TARGET["player_id"]
    ):
        raise NamespaceIsolationError("non-authoritative SoccerTrack target refused")


def assert_prediction_not_reference(path_parts: list[str]) -> None:
    lowered = {p.lower() for p in path_parts}
    if NAMESPACE_SOCCERTRACK_REFERENCE in lowered and "predictions" in lowered:
        raise NamespaceIsolationError("reference namespace must not live under predictions/")
    if NAMESPACE_TEAMTRACK_REAL_VIDEO in lowered and NAMESPACE_SOCCERTRACK_REFERENCE in lowered:
        raise NamespaceIsolationError("cannot colocate TeamTrack and SoccerTrack evidence roots")


KNOWN_NAMESPACES = {
    NAMESPACE_TEAMTRACK_REAL_VIDEO,
    NAMESPACE_SOCCERTRACK_REFERENCE,
    NAMESPACE_SELF_CONTAINED,
}

__all__ = [
    "KNOWN_NAMESPACES",
    "NamespaceIsolationError",
    "assert_namespaces_isolated",
    "assert_prediction_not_reference",
]
