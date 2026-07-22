"""Pipeline exception hierarchy (Stage 2D)."""

from __future__ import annotations


class PipelineError(ValueError):
    """Base error for stage / cache / artifact pipeline operations."""


class StageError(PipelineError):
    """Stage identity, request, or execution failure."""


class CacheError(PipelineError):
    """Cache key, layout, publish, read, or quarantine failure."""


class ArtifactError(PipelineError):
    """Artifact path, hash, size, or media-type validation failure."""
