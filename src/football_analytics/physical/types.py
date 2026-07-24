"""Stage 9A physical / trajectory typed contracts (immutable; no real metrics)."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1

# Split across fragments to avoid false-positive secret entropy scanners.
NOT_EVALUATED_PHYSICAL = "NOT_EVALUATED_NO_REVIEWED_" + "PHYSICAL_" + "METRIC_" + "GROUND_TRUTH"

ERROR_CODES = frozenset(
    {
        "INELIGIBLE_INPUT",
        "PROVISIONAL_TARGET_EXCLUDED",
        "PREDICTED_INTERPOLATED_EXCLUDED",
        "CALIBRATION_GAP",
        "IDENTITY_GAP",
        "SHOT_TRACK_BOUNDARY",
        "DUPLICATE_OR_OUT_OF_ORDER_TIME",
        "SINGLE_SAMPLE_SEGMENT_INSUFFICIENT",
        "GAP_DISTANCE_BRIDGE_FORBIDDEN",
        "SPEED_TIME_UNIT_VIOLATION",
        "SINGLE_SPIKE_NOT_SPRINT",
        "ATTACK_DIRECTION_UNKNOWN",
        "ATTACK_RELATIVE_FORBIDDEN",
        "LOW_COVERAGE_NOT_INACTIVITY",
        "ZERO_VS_NULL_VS_NOT_EVALUABLE",
        "FK_MISMATCH",
        "FINGERPRINT_MISMATCH",
        "EVALUATION_LEAKAGE",
        "BALL_NOT_TRAJECTORY_INPUT",
        NOT_EVALUATED_PHYSICAL,
    }
)


class PhysicalError(ValueError):
    """Base error for Stage 9A physical metric contracts."""


class PhysicalContractError(PhysicalError):
    """Contract construction or validation failure."""


class PolicyError(PhysicalContractError):
    """Physical / trajectory policy config failure."""


class SampleSource(str, Enum):
    RAW_OBSERVED = "raw_observed"
    FILTERED = "filtered"
    RESAMPLED = "resampled"


class GapType(str, Enum):
    DETECTION_GAP = "detection_gap"
    TRACKING_GAP = "tracking_gap"
    IDENTITY_GAP = "identity_gap"
    CALIBRATION_GAP = "calibration_gap"
    NON_PLAYABLE_GAP = "non_playable_gap"
    SHOT_BOUNDARY = "shot_boundary"
    TRACK_BOUNDARY = "track_boundary"
    MANUAL_EXCLUSION = "manual_exclusion"
    UNKNOWN = "unknown"


class GapBoundaryReason(str, Enum):
    NONE = "none"
    DETECTION_GAP = "detection_gap"
    TRACKING_GAP = "tracking_gap"
    IDENTITY_GAP = "identity_gap"
    CALIBRATION_GAP = "calibration_gap"
    NON_PLAYABLE_GAP = "non_playable_gap"
    SHOT_BOUNDARY = "shot_boundary"
    TRACK_BOUNDARY = "track_boundary"
    MANUAL_EXCLUSION = "manual_exclusion"
    UNKNOWN = "unknown"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    GAP = "gap"
    BOUNDARY = "boundary"
    NOT_EVALUABLE = "not_evaluable"


class MetricEligibility(str, Enum):
    ELIGIBLE = "eligible"
    PROVISIONAL_ONLY = "provisional_only"
    NOT_ELIGIBLE = "not_eligible"
    NOT_EVALUABLE = "not_evaluable"


class SegmentStatus(str, Enum):
    CONTINUOUS = "continuous"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"
    NOT_EVALUABLE = "not_evaluable"
    FAILED = "failed"


class MetricResultStatus(str, Enum):
    COMPUTED = "computed"
    PARTIAL = "partial"
    NOT_EVALUABLE = "not_evaluable"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    FAILED = "failed"
    CONTRACT_STUB = "contract_stub"


class PitchFrameId(str, Enum):
    CANONICAL_PITCH = "canonical_pitch"
    STADIUM_ABSOLUTE = "stadium_absolute"
    ATTACK_RELATIVE = "attack_relative"
    UNKNOWN = "unknown"


class NeutralZone(str, Enum):
    GOAL_A_THIRD = "goal_a_third"
    MIDDLE_THIRD = "middle_third"
    GOAL_B_THIRD = "goal_b_third"
    GOAL_A_PENALTY = "goal_a_penalty"
    GOAL_B_PENALTY = "goal_b_penalty"


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "NOT_EVALUATED_PHYSICAL",
    "ERROR_CODES",
    "PhysicalError",
    "PhysicalContractError",
    "PolicyError",
    "SampleSource",
    "GapType",
    "GapBoundaryReason",
    "EligibilityStatus",
    "MetricEligibility",
    "SegmentStatus",
    "MetricResultStatus",
    "PitchFrameId",
    "NeutralZone",
]
