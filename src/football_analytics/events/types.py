"""Stage 13 target-events typed constants."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
QUALITY_SCHEMA_VERSION = 1

NOT_EVALUATED_EVENTS = "".join(
    (
        "NOT_EVALUATED_NO_REVIEWED_",
        "TARGET_EVENTS_",
        "GROUND_TRUTH",
    )
)

ERROR_CODES = frozenset(
    {
        "REPLAY_UNCERTAIN_BLOCKS_LIVE",
        "CAMERA_POSITION_UNSUPPORTED",
        "ATTACK_DIRECTION_CONFLICT",
        "ATTACK_DIRECTION_UNKNOWN",
        "TEAM_NAME_INVENTION_FORBIDDEN",
        "DESTRUCTIVE_MERGE_FORBIDDEN",
        "EVALUATION_LEAKAGE",
        "DUPLICATE_SUPPRESSED",
        "CUT_REPLAY_GAP_NO_EVENT",
        "AUTOMATIC_CONFIRMED_FORBIDDEN",
        "OPTA_CLAIM_FORBIDDEN",
        NOT_EVALUATED_EVENTS,
    }
)

RESULT_LEVELS = frozenset(
    {
        "candidate",
        "provisional",
        "confirmed",
        "contested",
        "unknown",
        "not_evaluable",
        "rejected",
    }
)

METRIC_ORIGIN = "project_generated"
DEFINITION_STYLE = "opta_style_metric_definition"


class EventsError(ValueError):
    """Base error for Stage 13 events."""


class EventsContractError(EventsError):
    """Contract construction or validation failure."""


class PolicyError(EventsContractError):
    """Events policy config failure."""


class EvidenceLevel(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    CONTESTED = "contested"
    UNKNOWN = "unknown"
    NOT_EVALUABLE = "not_evaluable"
    REJECTED = "rejected"


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "QUALITY_SCHEMA_VERSION",
    "NOT_EVALUATED_EVENTS",
    "ERROR_CODES",
    "RESULT_LEVELS",
    "METRIC_ORIGIN",
    "DEFINITION_STYLE",
    "EventsError",
    "EventsContractError",
    "PolicyError",
    "EvidenceLevel",
]
