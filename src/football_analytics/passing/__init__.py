"""Passing / reception / progression contracts (Stage 11A).

Importing this package must NOT load models/videos or run real inference.
"""

from __future__ import annotations

from football_analytics.passing.contracts import (
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    PASSING_ARROW_CONTRACTS,
    assert_frozen_upstream_fingerprints,
    assert_passing_contracts_registered,
    passing_schema_fingerprints,
)
from football_analytics.passing.evaluation import (
    NOT_EVALUATED_PASSING,
    evaluate_passing,
)
from football_analytics.passing.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "PASSING_ARROW_CONTRACTS",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_PASSING",
    "assert_passing_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "passing_schema_fingerprints",
    "evaluate_passing",
]
