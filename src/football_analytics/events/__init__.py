"""Target events package (Stage 13).

Importing this package must NOT load models/videos or run real inference.
"""

from __future__ import annotations

from football_analytics.events.contracts import (
    EVENTS_ARROW_CONTRACTS,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_events_contracts_registered,
    assert_frozen_upstream_fingerprints,
    events_schema_fingerprints,
)
from football_analytics.events.evaluation import NOT_EVALUATED_EVENTS, evaluate_events
from football_analytics.events.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "EVENTS_ARROW_CONTRACTS",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_EVENTS",
    "assert_events_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "events_schema_fingerprints",
    "evaluate_events",
]
