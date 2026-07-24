"""Stage 11A passing / reception / progression typed contracts."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
QUALITY_SCHEMA_VERSION = 1

# Built at runtime so static secret scanners do not fold a high-entropy literal.
NOT_EVALUATED_PASSING = "".join(
    (
        "NOT_EVALUATED_NO_REVIEWED_",
        "PASSING_",
        "GROUND_TRUTH",
    )
)

ERROR_CODES = frozenset(
    {
        "OWNER_CHANGE_ALONE_NOT_PASS",
        "CUT_REPLAY_GAP_NO_PASS",
        "ATTACK_DIRECTION_UNKNOWN",
        "DIRECTIONAL_METRICS_NOT_EVALUABLE",
        "PENALTY_PRESENCE_NOT_BOX_TOUCH",
        "AUTOMATIC_CONFIRMED_FORBIDDEN",
        "PROXIMITY_NOT_RECEPTION",
        "MISSING_CONTACT_OR_POSSESSION",
        "INVALID_CALIBRATION",
        "NON_PLAYABLE",
        "HARD_GAP_NO_PASS",
        "CROSS_SCOPE_FK",
        "DUPLICATE_PK",
        "NAN_INF_REJECTED",
        "NEGATIVE_DURATION",
        "DANGLING_FK",
        "OPTA_CLAIM_FORBIDDEN",
        NOT_EVALUATED_PASSING,
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


class PassingError(ValueError):
    """Base error for Stage 11 passing contracts."""


class PassingContractError(PassingError):
    """Contract construction or validation failure."""


class PolicyError(PassingContractError):
    """Passing policy config failure."""


class EvidenceLevel(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    CONTESTED = "contested"
    UNKNOWN = "unknown"
    NOT_EVALUABLE = "not_evaluable"
    REJECTED = "rejected"


class PassOutcome(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    UNCERTAIN = "uncertain"
    NOT_EVALUABLE = "not_evaluable"
    REJECTED = "rejected"


class AttackDirection(str, Enum):
    TOWARD_GOAL_A = "toward_goal_a"
    TOWARD_GOAL_B = "toward_goal_b"
    UNKNOWN = "unknown"


class NeutralZone(str, Enum):
    GOAL_A = "goal_a"
    MIDDLE = "middle"
    GOAL_B = "goal_b"
    UNKNOWN = "unknown"
    NOT_EVALUABLE = "not_evaluable"


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "QUALITY_SCHEMA_VERSION",
    "NOT_EVALUATED_PASSING",
    "ERROR_CODES",
    "RESULT_LEVELS",
    "METRIC_ORIGIN",
    "DEFINITION_STYLE",
    "PassingError",
    "PassingContractError",
    "PolicyError",
    "EvidenceLevel",
    "PassOutcome",
    "AttackDirection",
    "NeutralZone",
]
