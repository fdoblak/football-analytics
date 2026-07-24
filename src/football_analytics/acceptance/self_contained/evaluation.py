"""Evaluation helpers for self-contained acceptance (contract arithmetic only)."""

from __future__ import annotations

from typing import Any


def compare_metric_vectors(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    mismatches: list[str] = []
    for key, exp in expected.items():
        act = actual.get(key)
        if isinstance(exp, float) and isinstance(act, (int, float)):
            if abs(float(act) - float(exp)) > rtol * max(1.0, abs(float(exp))):
                mismatches.append(key)
        elif act != exp:
            mismatches.append(key)
    return {"equal": not mismatches, "mismatches": mismatches}


__all__ = ["compare_metric_vectors"]
