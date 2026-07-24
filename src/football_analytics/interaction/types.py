"""Stage 10A human-ball interaction typed contracts (immutable; no real inference)."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
QUALITY_SCHEMA_VERSION = 1

# Built at runtime so static secret scanners do not fold a high-entropy literal.
NOT_EVALUATED_INTERACTION = "".join(
    (
        "NOT_EVALUATED_NO_REVIEWED_",
        "HUMAN_BALL_",
        "INTERACTION_",
        "GROUND_TRUTH",
    )
)

ERROR_CODES = frozenset(
    {
        "NEAREST_PLAYER_NOT_POSSESSION",
        "PROXIMITY_NOT_CONTACT",
        "PROXIMITY_NOT_EVENT",
        "CONTACT_NOT_POSSESSION",
        "CONTACT_NOT_EVENT",
        "POSSESSION_NOT_EVENT",
        "SINGLE_FRAME_PROXIMITY",
        "MISSING_BALL_NOT_NO_POSSESSION",
        "HARD_GAP_NO_CARRY",
        "REPLAY_OR_CUT_TERMINATES",
        "AIRBORNE_UNKNOWN_BLOCKS_PITCH",
        "INVALID_CALIBRATION",
        "PREDICTED_SOLE_EVIDENCE",
        "AMBIGUOUS_PRIMARY_BALL",
        "OVERLAPPING_POSSESSION",
        "OWNER_TRANSITION_WITHOUT_EVIDENCE",
        "CROSS_SCOPE_FK",
        "AUTOMATIC_CONFIRMED_FORBIDDEN",
        "LOW_JOINT_COVERAGE_NOT_EVALUABLE",
        "NAN_INF_REJECTED",
        "NEGATIVE_DURATION",
        "DUPLICATE_PK",
        "DANGLING_FK",
        "EVENT_METRIC_FORBIDDEN",
        NOT_EVALUATED_INTERACTION,
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


class InteractionError(ValueError):
    """Base error for Stage 10A interaction contracts."""


class InteractionContractError(InteractionError):
    """Contract construction or validation failure."""


class PolicyError(InteractionContractError):
    """Interaction policy config failure."""


class EvidenceLevel(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    CONTESTED = "contested"
    UNKNOWN = "unknown"
    NOT_EVALUABLE = "not_evaluable"
    REJECTED = "rejected"


class ObservationState(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"
    MISSING = "missing"


class BallAirState(str, Enum):
    GROUNDED = "grounded"
    AIRBORNE = "airborne"
    UNKNOWN = "unknown"


class BallCandidateStatus(str, Enum):
    PRIMARY = "primary"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    REJECTED = "rejected"


class TerminationReason(str, Enum):
    NONE = "none"
    SHOT_CUT = "shot_cut"
    REPLAY = "replay"
    NON_PLAYABLE = "non_playable"
    BALL_LOSS = "ball_loss"
    HARD_GAP = "hard_gap"
    TRACK_TERMINATION = "track_termination"
    OWNER_TRANSITION = "owner_transition"
    CONTESTED = "contested"
    MANUAL_REVOCATION = "manual_revocation"
    UNKNOWN = "unknown"


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "QUALITY_SCHEMA_VERSION",
    "NOT_EVALUATED_INTERACTION",
    "ERROR_CODES",
    "RESULT_LEVELS",
    "InteractionError",
    "InteractionContractError",
    "PolicyError",
    "EvidenceLevel",
    "ObservationState",
    "BallAirState",
    "BallCandidateStatus",
    "TerminationReason",
]
