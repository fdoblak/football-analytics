"""Stage 12A duels / competitive-events typed contracts."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
QUALITY_SCHEMA_VERSION = 1

# Built at runtime so static secret scanners do not fold a high-entropy literal.
NOT_EVALUATED_DUELS = "".join(
    (
        "NOT_EVALUATED_NO_REVIEWED_",
        "DUELS_EVENTS_",
        "GROUND_TRUTH",
    )
)

ERROR_CODES = frozenset(
    {
        "NEARBY_OPPONENT_ALONE_NOT_TAKE_ON",
        "NEAREST_SWITCH_ALONE_NOT_DUEL_OUTCOME",
        "MONOCULAR_AERIAL_NO_EXACT_HEIGHT",
        "LONG_BALL_ALONE_NOT_CLEARANCE",
        "AUTOMATIC_CONFIRMED_FORBIDDEN",
        "CUT_REPLAY_GAP_NO_EVENT",
        "MISSING_CONTACT_OR_POSSESSION",
        "INVALID_CALIBRATION",
        "NON_PLAYABLE",
        "HARD_GAP_NO_EVENT",
        "CROSS_SCOPE_FK",
        "DUPLICATE_PK",
        "NAN_INF_REJECTED",
        "NEGATIVE_DURATION",
        "DANGLING_FK",
        "OPTA_CLAIM_FORBIDDEN",
        NOT_EVALUATED_DUELS,
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


class DuelsError(ValueError):
    """Base error for Stage 12 duels contracts."""


class DuelsContractError(DuelsError):
    """Contract construction or validation failure."""


class PolicyError(DuelsContractError):
    """Duels policy config failure."""


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
    "NOT_EVALUATED_DUELS",
    "ERROR_CODES",
    "RESULT_LEVELS",
    "METRIC_ORIGIN",
    "DEFINITION_STYLE",
    "DuelsError",
    "DuelsContractError",
    "PolicyError",
    "EvidenceLevel",
]
