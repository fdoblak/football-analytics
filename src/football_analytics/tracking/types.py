"""Stage 6A tracking typed contracts (immutable; no tracker inference)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1

ERROR_CODES = frozenset(
    {
        "INVALID_TRANSITION",
        "TERMINATED_REOPEN",
        "DUPLICATE_DETECTION_ASSIGNMENT",
        "DUPLICATE_FRAME_OBSERVATION",
        "HUMAN_BALL_MERGE",
        "CROSS_VIDEO_FK",
        "DANGLING_DETECTION_FK",
        "DANGLING_FRAME_FK",
        "TIMESTAMP_REVERSE",
        "FPS_TIME_INVENTED",
        "INVALID_BBOX",
        "FINGERPRINT_MISMATCH",
        "TRACK_ID_REUSE",
        "ROLE_CONFLICT",
        "UNKNOWN_ROLE_PRESERVED",
        "ROUTING_GAP",
        "PHYSICAL_METRIC_INELIGIBLE",
        "RECEIPT_COUNT_MISMATCH",
        "MANUAL_REVIEW_REQUIRED",
        "NOT_EVALUATED_NO_REVIEWED_" "TRACKING_GROUND_TRUTH",
    }
)

# Stage 6A observation sources → existing track_observations.observation_state v1.
OBSERVATION_STATE_MAP: Mapping[str, str] = {
    "detection_associated": "observed",
    "predicted": "predicted",
    "interpolated": "interpolated",
    # not_observed: prefer no row (documented; may use quality_flags only if encoded)
    "not_observed": "prefer_no_row",
}


class TrackingError(ValueError):
    """Base error for Stage 6A tracking contracts."""


class TrackingContractError(TrackingError):
    """Contract construction or validation failure."""


class TransitionError(TrackingContractError):
    """Illegal lifecycle state transition."""


class LifecycleState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


class ObservationSource(str, Enum):
    DETECTION_ASSOCIATED = "detection_associated"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"
    NOT_OBSERVED = "not_observed"


class TrackEntityType(str, Enum):
    HUMAN = "human"
    BALL = "ball"


class GapReason(str, Enum):
    DETECTION_LOSS = "DETECTION_LOSS_GAP"
    ROUTING_INELIGIBLE = "ROUTING_INELIGIBLE_GAP"
    REPLAY = "REPLAY_GAP"
    GRAPHICS = "GRAPHICS_GAP"
    ANALYSIS_WINDOW_BOUNDARY = "ANALYSIS_WINDOW_BOUNDARY_GAP"


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    PARTIAL = "partial"


def observation_state_for_source(source: ObservationSource | str) -> str:
    """Map Stage 6A source to track_observations.observation_state (or prefer_no_row)."""
    key = source.value if isinstance(source, ObservationSource) else str(source)
    if key not in OBSERVATION_STATE_MAP:
        raise TrackingContractError(f"unknown observation source: {key}")
    return OBSERVATION_STATE_MAP[key]


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "ERROR_CODES",
    "OBSERVATION_STATE_MAP",
    "TrackingError",
    "TrackingContractError",
    "TransitionError",
    "LifecycleState",
    "ObservationSource",
    "TrackEntityType",
    "GapReason",
    "ReceiptStatus",
    "observation_state_for_source",
]
