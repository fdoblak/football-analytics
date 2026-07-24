"""Stage 16 acceptance contracts — namespaces, provenance, partitions."""

from __future__ import annotations

from enum import Enum

# Camera / domain
CAMERA_DOMAIN_PANORAMIC = "panoramic_full_pitch"
BROADCAST_ACCEPTANCE_STATUS = "not_covered_by_this_dataset"
FULL_PITCH_TRACKING_ACCEPTANCE = "covered"

# Provenance — never present as user-manual production confirmation
EXTERNAL_CC_BY_REFERENCE_GT = "external_cc_by_reference_gt"
EXTERNAL_REFERENCE_CONFIRMATION = "external_reference_confirmation"

# Hard namespace separation
NAMESPACE_PREDICTIONS = "predictions"
NAMESPACE_REFERENCE_GT = "reference_ground_truth"
NAMESPACE_MANUAL_OR_EXTERNAL_ANCHORS = "manual_or_external_anchors"
NAMESPACE_EVALUATION = "evaluation"

LEAKAGE_SEPARATION_VIOLATION = "LEAKAGE_SEPARATION_VIOLATION"

# Partitions (half-1 tuning / half-2 held-out preferred)
PARTITION_TUNING = "tuning_calibration"
PARTITION_HELD_OUT = "held_out_acceptance"

# Metric evaluability taxonomy
EVALUATED_AGAINST_EXTERNAL_GT = "evaluated_against_external_gt"
PIPELINE_DERIVED_NOT_DIRECTLY_GT_SUPPORTED = "pipeline_derived_not_directly_gt_supported"
MANUAL_REVIEW_REQUIRED = "manual_review_required"
NOT_EVALUABLE = "not_evaluable"

# Explicit non-zero missing codes
NOT_OBSERVED = "not_observed"
INSUFFICIENT_COVERAGE = "insufficient_coverage"
REFERENCE_GT_NOT_AVAILABLE = "reference_gt_not_available"

FPS_DEFAULT = 25
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

BAS_LABELS_CANONICAL: tuple[str, ...] = (
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
)


class RoleName(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    OTHER = "other"


def normalize_bas_label(raw: str) -> str:
    """Map Drive uppercase labels to canonical title-case BAS names."""
    key = " ".join(str(raw).strip().upper().split())
    mapping = {
        "PASS": "Pass",
        "DRIVE": "Drive",
        "HEADER": "Header",
        "HIGH PASS": "High Pass",
        "OUT": "Out",
        "CROSS": "Cross",
        "THROW IN": "Throw In",
        "SHOT": "Shot",
        "BALL PLAYER BLOCK": "Ball Player Block",
        "PLAYER SUCCESSFUL TACKLE": "Player Successful Tackle",
        "FREE KICK": "Free Kick",
        "GOAL": "Goal",
    }
    if key not in mapping:
        raise ValueError(f"Unknown BAS label: {raw!r}")
    return mapping[key]
