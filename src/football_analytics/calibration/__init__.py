"""Stage 8A pitch calibration / homography / coordinate contracts (no SV inference)."""

from __future__ import annotations

from football_analytics.calibration.contracts import (
    CALIBRATION_ARROW_CONTRACTS,
    CALIBRATION_FEATURES_CONTRACT,
    CALIBRATION_SEGMENTS_CONTRACT,
    CALIBRATIONS_CONTRACT,
    EXPECTED_CALIBRATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    PROJECTED_POSITIONS_CONTRACT,
    assert_calibration_contracts_registered,
    assert_calibrations_fingerprint_frozen,
    calibration_schema_fingerprints,
)
from football_analytics.calibration.evaluation import (
    NOT_EVALUATED_CALIBRATION,
    evaluate_calibration,
)
from football_analytics.calibration.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "CALIBRATIONS_CONTRACT",
    "CALIBRATION_FEATURES_CONTRACT",
    "CALIBRATION_SEGMENTS_CONTRACT",
    "PROJECTED_POSITIONS_CONTRACT",
    "CALIBRATION_ARROW_CONTRACTS",
    "EXPECTED_CALIBRATIONS_FP",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_CALIBRATION",
    "assert_calibration_contracts_registered",
    "assert_calibrations_fingerprint_frozen",
    "calibration_schema_fingerprints",
    "evaluate_calibration",
]
