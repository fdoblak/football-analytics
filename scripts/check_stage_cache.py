#!/usr/bin/env python3
"""Validate Stage 2D stage interface and content-addressed cache."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.pipeline.stage_cache_check import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_FINDING,
    EXIT_INTEGRITY,
    EXIT_PASS,
    main,
    run_stage_cache_checks,
)

__all__ = [
    "EXIT_PASS",
    "EXIT_FINDING",
    "EXIT_CONFIG",
    "EXIT_INTEGRITY",
    "main",
    "run_stage_cache_checks",
]

if __name__ == "__main__":
    raise SystemExit(main())
