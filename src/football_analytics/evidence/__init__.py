"""Evidence retention package (small GitHub-safe artifacts only)."""

from __future__ import annotations

from football_analytics.evidence.collector import (
    MAX_BACKFILL_TOTAL_BYTES,
    MAX_FILE_BYTES,
    backfill_from_workspace,
    evidence_root,
    is_safe_evidence_file,
)

__all__ = [
    "MAX_FILE_BYTES",
    "MAX_BACKFILL_TOTAL_BYTES",
    "evidence_root",
    "is_safe_evidence_file",
    "backfill_from_workspace",
]
