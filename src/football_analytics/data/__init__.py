"""Canonical Arrow/Parquet data contracts (Stage 2C).

Importing this package does not load PyArrow until submodule APIs are used.
"""

from __future__ import annotations

__all__ = ["DataContractError"]


class DataContractError(ValueError):
    """Base error for data contract operations."""
