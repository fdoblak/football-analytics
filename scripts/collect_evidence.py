#!/usr/bin/env python3
"""Collect/backfill small safe evidence artifacts into artifacts/evidence/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/home/fdoblak/workspace"),
        help="Workspace root to scan (read-only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from football_analytics.evidence.collector import (
        backfill_from_workspace,
        mark_missing_stages,
    )

    summary = backfill_from_workspace(
        workspace_root=Path(args.workspace),
        project_root=REPO_ROOT,
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        # Stages that may have been cleaned before retention existed.
        early = [f"stage_{i:02d}" for i in range(0, 10)]
        summary["marked_missing"] = mark_missing_stages(early, project_root=REPO_ROOT)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "evidence_backfill "
            f"copied={summary['copied']} skipped={summary['skipped']} "
            f"not_available_cleaned={summary['not_available_cleaned']} "
            f"total_bytes={summary['total_bytes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
