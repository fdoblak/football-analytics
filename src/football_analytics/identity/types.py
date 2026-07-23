"""Stage 7A identity typed contracts (immutable; no ReID inference)."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_IDENTITY = "NOT_EVALUATED_NO_REVIEWED_" "IDENTITY_GROUND_TRUTH"

ERROR_CODES = frozenset(
    {
        "SINGLE_WEAK_CANNOT_CONFIRM",
        "JERSEY_ALONE_INSUFFICIENT",
        "TEAM_ALONE_INSUFFICIENT",
        "ROLE_ALONE_INSUFFICIENT",
        "APPEARANCE_ALONE_INSUFFICIENT",
        "HARD_EVIDENCE_CONFLICT",
        "CROSS_VIDEO_AUTO_LINK_FORBIDDEN",
        "PHYSICAL_MERGE_FORBIDDEN",
        "FACE_BIOMETRIC_FORBIDDEN",
        "DUPLICATE_CONFIRMED_IDENTITY",
        "TRACK_MULTI_IDENTITY",
        "REVOKED_NOT_METRIC_ELIGIBLE",
        "LEAKAGE_SEPARATION_VIOLATION",
        "APPEND_ONLY_VIOLATION",
        "FINGERPRINT_MISMATCH",
        "FK_MISMATCH",
        "HASH_VERSION_MISMATCH",
        "INSUFFICIENT_COVERAGE",
        "MANUAL_REVIEW_REQUIRED",
        "AUTO_TARGET_SELECTION_FORBIDDEN",
        NOT_EVALUATED_IDENTITY,
    }
)


class IdentityError(ValueError):
    """Base error for Stage 7A identity contracts."""


class IdentityContractError(IdentityError):
    """Contract construction or validation failure."""


class PolicyError(IdentityContractError):
    """Identity policy config failure."""


class EvidenceType(str, Enum):
    MANUAL_TRACK_ANCHOR = "manual_track_anchor"
    APPEARANCE_SIMILARITY = "appearance_similarity"
    JERSEY_NUMBER = "jersey_number"
    TEAM_ASSIGNMENT = "team_assignment"
    ROLE_CONSISTENCY = "role_consistency"
    TEMPORAL_CONTINUITY = "temporal_continuity"
    SPATIAL_MOTION_CONTINUITY = "spatial_motion_continuity"
    CAMERA_VIEW_SUITABILITY = "camera_view_suitability"
    NEGATIVE_EXCLUSION = "negative_exclusion"
    UNKNOWN = "unknown"


class ReliabilityTier(str, Enum):
    MANUAL_VERIFIED = "manual_verified"
    STRONG = "strong"
    SUPPORTING = "supporting"
    WEAK = "weak"
    CONFLICTING = "conflicting"
    UNAVAILABLE = "unavailable"


class EvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    NEUTRAL = "neutral"


class AssignmentStatus(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


class TargetScope(str, Enum):
    TARGET = "target"
    NON_TARGET = "non_target"
    UNKNOWN = "unknown"


class MetricEligibility(str, Enum):
    ELIGIBLE = "eligible"
    PROVISIONAL_ONLY = "provisional_only"
    NOT_ELIGIBLE = "not_eligible"
    NOT_EVALUABLE = "not_evaluable"


class LinkDecisionStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"
    UNKNOWN = "unknown"


class LeakageClass(str, Enum):
    PRODUCTION = "production"
    TUNING = "tuning"
    MANUAL_LABEL = "manual_label"
    EVALUATION = "evaluation"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    PARTIAL = "partial"


# Alone-insufficient evidence types (cannot confirm by themselves).
ALONE_INSUFFICIENT_TYPES = frozenset(
    {
        EvidenceType.JERSEY_NUMBER.value,
        EvidenceType.TEAM_ASSIGNMENT.value,
        EvidenceType.ROLE_CONSISTENCY.value,
        EvidenceType.APPEARANCE_SIMILARITY.value,
        EvidenceType.TEMPORAL_CONTINUITY.value,
        EvidenceType.SPATIAL_MOTION_CONTINUITY.value,
        EvidenceType.CAMERA_VIEW_SUITABILITY.value,
        EvidenceType.UNKNOWN.value,
    }
)


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "NOT_EVALUATED_IDENTITY",
    "ERROR_CODES",
    "IdentityError",
    "IdentityContractError",
    "PolicyError",
    "EvidenceType",
    "ReliabilityTier",
    "EvidencePolarity",
    "AssignmentStatus",
    "TargetScope",
    "MetricEligibility",
    "LinkDecisionStatus",
    "LeakageClass",
    "ReviewStatus",
    "ReceiptStatus",
    "ALONE_INSUFFICIENT_TYPES",
]
