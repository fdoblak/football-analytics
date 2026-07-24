"""Concurrency lock helper re-export for Stage 15 hardening."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy
from football_analytics.pipeline.cache import acquire_key_lock


@contextmanager
def cache_key_lock(
    cache_root: Path,
    cache_key: str,
    *,
    policy: HardeningPolicy | None = None,
) -> Iterator[None]:
    """Acquire the standard cache key flock with hardening timeout."""
    pol = policy or load_hardening_policy()
    timeout = float(pol.raw["concurrency"]["lock_timeout_seconds"])
    with acquire_key_lock(cache_root, cache_key, timeout_seconds=timeout):
        yield
