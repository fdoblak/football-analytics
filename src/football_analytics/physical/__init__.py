"""Target trajectory and physical metric contracts / baselines (Stages 9A–9E).

Importing this package must NOT load models/videos. Explicit compute entrypoints:
`motion_service`, `spatial_service`, `pipeline_service.integrate_physical_metrics`.
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
