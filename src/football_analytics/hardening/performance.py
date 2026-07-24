"""15F performance / bounded memory / deterministic repeat helpers."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any, TypeVar

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy

T = TypeVar("T")


class PerformanceGateError(ValueError):
    """Performance / timeout gate failure."""


def run_with_timeout_budget(
    fn: Callable[[], T],
    *,
    timeout_sec: float,
    label: str = "operation",
) -> tuple[T, dict[str, Any]]:
    """Run callable and record duration; raise if over budget (cooperative)."""
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started
    meta = {"label": label, "elapsed_sec": elapsed, "timeout_sec": timeout_sec}
    if elapsed > timeout_sec:
        raise PerformanceGateError(
            f"{label}: elapsed_sec={elapsed:.3f} exceeds timeout_sec={timeout_sec}"
        )
    return result, meta


def bounded_memory_probe(
    fn: Callable[[], T],
    *,
    soft_limit_mb: int | None = None,
    policy: HardeningPolicy | None = None,
    label: str = "probe",
) -> tuple[T, dict[str, Any]]:
    """Run under tracemalloc; record peak; warn-style soft limit (raise if far over)."""
    pol = policy or load_hardening_policy()
    limit = (
        soft_limit_mb
        if soft_limit_mb is not None
        else int(pol.raw["performance"]["memory_soft_limit_mb"])
    )
    tracemalloc.start()
    try:
        result = fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    meta = {
        "label": label,
        "peak_mb": round(peak_mb, 3),
        "soft_limit_mb": limit,
        "within_soft_limit": peak_mb <= float(limit) * 2.0,  # 2x headroom for interpreter noise
    }
    if peak_mb > float(limit) * 4.0:
        raise PerformanceGateError(
            f"{label}: peak_mb={peak_mb:.1f} far above soft_limit_mb={limit}"
        )
    return result, meta


def deterministic_repeat(fn: Callable[[], T], *, runs: int = 2) -> tuple[T, dict[str, Any]]:
    """Run callable twice (or more) and require equal results for comparable outputs."""
    if runs < 2:
        raise ValueError("runs must be >= 2")
    first = fn()
    for i in range(1, runs):
        again = fn()
        if again != first:
            raise PerformanceGateError(f"nondeterministic result at repeat={i}")
    return first, {"runs": runs, "deterministic": True}


def streaming_parquet_notes() -> dict[str, Any]:
    """Document streaming parquet write path used to bound memory."""
    return {
        "api": "football_analytics.data.parquet.write_contract_parquet_streaming",
        "materialize_policy": "football_analytics.hardening.materialize",
        "notes": [
            "Prefer streaming ParquetWriter batches over full-table to_pylist",
            "RISK-029 closed machine-locally via max_pylist_rows + chunked iteration",
        ],
    }
