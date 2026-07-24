"""Duels / competitive events contracts (Stage 12A).

Importing this package must NOT load models/videos or run real inference.
"""

from __future__ import annotations

from football_analytics.duels.contracts import (
    DUELS_ARROW_CONTRACTS,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_duels_contracts_registered,
    assert_frozen_upstream_fingerprints,
    duels_schema_fingerprints,
)
from football_analytics.duels.evaluation import (
    NOT_EVALUATED_DUELS,
    evaluate_duels,
)
from football_analytics.duels.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "DUELS_ARROW_CONTRACTS",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_DUELS",
    "assert_duels_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "duels_schema_fingerprints",
    "evaluate_duels",
]
