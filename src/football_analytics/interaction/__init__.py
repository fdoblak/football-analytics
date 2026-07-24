"""Human-ball interaction / possession contracts (Stage 10A).

Importing this package must NOT load models/videos or run interaction inference.
"""

from __future__ import annotations

from football_analytics.interaction.contracts import (
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    INTERACTION_ARROW_CONTRACTS,
    assert_frozen_upstream_fingerprints,
    assert_interaction_contracts_registered,
    interaction_schema_fingerprints,
)
from football_analytics.interaction.evaluation import (
    NOT_EVALUATED_INTERACTION,
    evaluate_human_ball_interaction,
)
from football_analytics.interaction.types import CONTRACT_VERSION

__all__ = [
    "CONTRACT_VERSION",
    "INTERACTION_ARROW_CONTRACTS",
    "EXPECTED_REGISTRY_CONTRACT_COUNT",
    "NOT_EVALUATED_INTERACTION",
    "assert_interaction_contracts_registered",
    "assert_frozen_upstream_fingerprints",
    "interaction_schema_fingerprints",
    "evaluate_human_ball_interaction",
]
