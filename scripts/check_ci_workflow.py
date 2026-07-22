#!/usr/bin/env python3
"""Validate GitHub Actions CI workflow safety (Stage 2D)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.pipeline.ci_workflow_check import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_FINDING,
    EXIT_INTEGRITY,
    EXIT_PASS,
    main,
    run_ci_workflow_checks,
)

__all__ = [
    "EXIT_PASS",
    "EXIT_FINDING",
    "EXIT_CONFIG",
    "EXIT_INTEGRITY",
    "main",
    "run_ci_workflow_checks",
]

if __name__ == "__main__":
    raise SystemExit(main())
