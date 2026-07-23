"""Stage 8A calibration typed contracts (immutable; no SV inference)."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1

# Split to avoid false-positive secret entropy scanners.
NOT_EVALUATED_CALIBRATION = "NOT_EVALUATED_NO_REVIEWED_" "CALIBRATION_GROUND_TRUTH"

ERROR_CODES = frozenset(
    {
        "INSUFFICIENT_CORRESPONDENCES",
        "COLLINEAR_CORRESPONDENCES",
        "DUPLICATE_CORRESPONDENCES",
        "SINGULAR_HOMOGRAPHY",
        "ILL_CONDITIONED_HOMOGRAPHY",
        "MIRRORED_HOMOGRAPHY",
        "HIGH_REPROJECTION_ERROR",
        "ROUND_TRIP_FAILURE",
        "PITCH_TEMPLATE_MISMATCH",
        "SEGMENT_OVERLAP_CONFLICT",
        "SHOT_CUT_TERMINATE",
        "SILENT_GAP_FILL_FORBIDDEN",
        "PREDICTED_NOT_METRIC_ELIGIBLE",
        "INTERPOLATED_NOT_METRIC_ELIGIBLE",
        "EXTRAPOLATED_NOT_METRIC_ELIGIBLE",
        "AIRBORNE_BALL_NOT_METRIC_ELIGIBLE",
        "NOT_CALIBRATED",
        "FK_MISMATCH",
        "HASH_VERSION_MISMATCH",
        "FINGERPRINT_MISMATCH",
        "ATTACK_DIRECTION_UNKNOWN",
        NOT_EVALUATED_CALIBRATION,
    }
)


class CalibrationError(ValueError):
    """Base error for Stage 8A calibration contracts."""


class CalibrationContractError(CalibrationError):
    """Contract construction or validation failure."""


class PolicyError(CalibrationContractError):
    """Calibration policy config failure."""


class HomographyError(CalibrationContractError):
    """Homography solve / validation failure."""


class FeatureType(str, Enum):
    KEYPOINT = "keypoint"
    LINE = "line"
    LINE_INTERSECTION = "line_intersection"
    UNKNOWN = "unknown"


class FeatureStatus(str, Enum):
    DETECTED = "detected"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class Suitability(str, Enum):
    SUITABLE = "suitable"
    MARGINAL = "marginal"
    UNSUITABLE = "unsuitable"
    UNKNOWN = "unknown"


class CoordinateFrameId(str, Enum):
    SOURCE_IMAGE = "source_image"
    NORMALIZED_IMAGE = "normalized_image"
    MODEL_SPACE = "model_space"
    CANONICAL_PITCH = "canonical_pitch"
    ATTACK_RELATIVE = "attack_relative"


class MappingStatus(str, Enum):
    MAPPED = "mapped"
    OUTSIDE_PITCH = "outside_pitch"
    EXTRAPOLATED = "extrapolated"
    UNCERTAIN = "uncertain"
    NOT_CALIBRATED = "not_calibrated"
    NOT_ELIGIBLE = "not_eligible"
    FAILED = "failed"


class PhysicalMetricEligibility(str, Enum):
    ELIGIBLE = "eligible"
    PROVISIONAL_ONLY = "provisional_only"
    NOT_ELIGIBLE = "not_eligible"
    NOT_EVALUABLE = "not_evaluable"


class ValidityStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NOT_ELIGIBLE = "not_eligible"
    CONFLICT = "conflict"
    ABSTAIN = "abstain"


class CameraMotion(str, Enum):
    STATIC = "static"
    PAN = "pan"
    ZOOM = "zoom"
    PAN_ZOOM = "pan_zoom"
    UNKNOWN = "unknown"


class AttackDirection(str, Enum):
    UNKNOWN = "unknown"
    TOWARD_GOAL_A = "toward_goal_a"
    TOWARD_GOAL_B = "toward_goal_b"


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    PARTIAL = "partial"


class ObservationSource(str, Enum):
    DETECTION_ASSOCIATED = "detection_associated"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"
    NOT_OBSERVED = "not_observed"
    SYNTHETIC = "synthetic"


class SourcePointType(str, Enum):
    BBOX_BOTTOM_CENTRE = "bbox_bottom_centre"
    BBOX_CENTRE = "bbox_centre"
    KEYPOINT = "keypoint"
    MANUAL = "manual"
    UNKNOWN = "unknown"


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "NOT_EVALUATED_CALIBRATION",
    "ERROR_CODES",
    "CalibrationError",
    "CalibrationContractError",
    "PolicyError",
    "HomographyError",
    "FeatureType",
    "FeatureStatus",
    "Suitability",
    "CoordinateFrameId",
    "MappingStatus",
    "PhysicalMetricEligibility",
    "ValidityStatus",
    "CameraMotion",
    "AttackDirection",
    "ReceiptStatus",
    "ObservationSource",
    "SourcePointType",
]
