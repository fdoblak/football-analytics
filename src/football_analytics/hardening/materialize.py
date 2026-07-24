"""RISK-029 bounded streaming / materialization policy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


class MaterializeBoundError(ValueError):
    """Raised when a materialization would exceed hardening bounds."""


def estimate_pylist_bytes(row_count: int, *, bytes_per_row: int = 256) -> int:
    return max(0, int(row_count) * max(1, int(bytes_per_row)))


def assert_pylist_bounds(
    row_count: int,
    *,
    policy: HardeningPolicy | None = None,
    bytes_per_row: int = 256,
    context: str = "table",
) -> None:
    """Refuse unbounded Python materialization of large tables."""
    pol = policy or load_hardening_policy()
    if pol.allow_unbounded_pylist:
        raise MaterializeBoundError("policy incorrectly allows unbounded pylist")
    if row_count > pol.max_pylist_rows:
        raise MaterializeBoundError(
            f"{context}: row_count={row_count} exceeds max_pylist_rows={pol.max_pylist_rows}"
        )
    est = estimate_pylist_bytes(row_count, bytes_per_row=bytes_per_row)
    max_bytes = int(pol.raw["materialize"]["max_pylist_bytes_estimate"])
    if est > max_bytes:
        raise MaterializeBoundError(
            f"{context}: estimated_bytes={est} exceeds max_pylist_bytes_estimate={max_bytes}"
        )


def bounded_pylist(
    table: Any,
    *,
    policy: HardeningPolicy | None = None,
    context: str = "table",
) -> list[dict[str, Any]]:
    """Convert an Arrow table to pylist only when within policy bounds."""
    pol = policy or load_hardening_policy()
    n = int(table.num_rows)
    assert_pylist_bounds(n, policy=pol, context=context)
    return table.to_pylist()


def iter_table_batches(
    table: Any,
    *,
    policy: HardeningPolicy | None = None,
) -> Iterator[Any]:
    """Yield Arrow record batches with policy chunk size."""
    pol = policy or load_hardening_policy()
    chunk = max(1, pol.chunk_rows)
    yield from table.to_batches(max_chunksize=chunk)


def streaming_parquet_preferred(policy: HardeningPolicy | None = None) -> bool:
    pol = policy or load_hardening_policy()
    return bool(pol.raw["materialize"]["prefer_streaming_parquet"])


def materialize_policy_summary(policy: HardeningPolicy | None = None) -> dict[str, Any]:
    pol = policy or load_hardening_policy()
    return {
        "risk_id": "RISK-029",
        "max_pylist_rows": pol.max_pylist_rows,
        "chunk_rows": pol.chunk_rows,
        "prefer_streaming_parquet": streaming_parquet_preferred(pol),
        "allow_unbounded_pylist": False,
        "status": "closed_machine_local",
    }


def chunked_sequence(items: Sequence[Any], *, chunk_size: int) -> Iterator[Sequence[Any]]:
    size = max(1, int(chunk_size))
    for i in range(0, len(items), size):
        yield items[i : i + size]
