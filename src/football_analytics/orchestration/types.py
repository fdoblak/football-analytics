"""Orchestration error types (Stage 14)."""

from __future__ import annotations


class OrchestrationError(RuntimeError):
    """Single-player orchestration / review / report failure."""


class StaleArtifactError(OrchestrationError):
    """Fingerprint / CAS stale rejection."""


class OverwriteForbiddenError(OrchestrationError):
    """No-overwrite policy violation."""


class CancellationError(OrchestrationError):
    """Run cancelled with receipt."""


__all__ = [
    "OrchestrationError",
    "StaleArtifactError",
    "OverwriteForbiddenError",
    "CancellationError",
]
