"""Runtime foundation package (Stage 2B).

Side-effect free on import: no torch/GPU/network/dataset loading.
"""

from __future__ import annotations

from football_analytics.core.config import ConfigError
from football_analytics.core.hashing import HashError
from football_analytics.core.records import RecordError
from football_analytics.core.run_context import RunContextError
from football_analytics.core.run_id import RunIdError

FoundationError = RunContextError

__all__ = [
    "ConfigError",
    "FoundationError",
    "HashError",
    "RecordError",
    "RunContextError",
    "RunIdError",
]
