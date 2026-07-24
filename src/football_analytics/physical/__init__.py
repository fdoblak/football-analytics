"""Target trajectory and physical metric contracts / baselines (Stages 9A–9C).

Importing this package must NOT load models/videos. Stage 9C motion compute is
explicit via `motion_service.compute_physical_motion` (not on import).
"""

from __future__ import annotations

from football_analytics.physical.contracts import (
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    PHYSICAL_ARROW_CONTRACTS,
    assert_frozen_upstream_fingerprints,
    assert_physical_contracts_registered,
    physical_schema_fingerprints,
)
from football_analytics.physical.evaluation import (
    NOT_EVALUATED_PHYSICAL,
    evaluate_physical_metrics,
)
from football_analytics.physical.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "PHYSICAL_ARROW_CONTRACTS",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_PHYSICAL",
    "assert_physical_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "physical_schema_fingerprints",
    "evaluate_physical_metrics",
]
