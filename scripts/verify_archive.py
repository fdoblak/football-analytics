#!/usr/bin/env python3
"""Read-only archive verifier."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.utils import archive_safety as safety  # noqa: E402


def verify(
    *,
    run_id: Optional[str],
    policy_path: Path,
    archive_path: Optional[Path],
    json_out: Optional[Path],
) -> safety.OpResult:
    result = safety.OpResult()
    policy = safety.load_policy(policy_path)
    if run_id:
        safety.validate_run_id(run_id, policy)
    archive_root = Path(policy["paths"]["archive_root"])
    if archive_path is None:
        if not run_id:
            return result.fail("run-id or archive-path required", safety.EXIT_CONFIG).finalize()
        archive_path = archive_root / run_id
    archive_path = Path(archive_path)
    try:
        archive_path = safety.assert_contained(archive_path, archive_root, label="archive")
        manifest, records = safety.verify_archive_tree(
            archive_path, expected_run_id=run_id, policy=policy
        )
    except safety.ArchiveError as exc:
        return result.fail(str(exc), exc.exit_code).finalize()

    result.extras.update(
        {
            "run_id": manifest.get("run_id"),
            "archive_path": str(archive_path),
            "archive_id": manifest.get("archive_id"),
            "total_files": manifest.get("total_files"),
            "total_bytes": manifest.get("total_bytes"),
            "independent_backup": manifest.get("independent_backup"),
            "failure_domain": manifest.get("failure_domain"),
            "archive_manifest_sha256": safety.sha256_file(archive_path / "archive_manifest.json"),
            "files_verified": len(records),
        }
    )
    if manifest.get("independent_backup") is not False:
        return result.fail("manifest claims independent_backup", safety.EXIT_INTEGRITY).finalize()

    # optional receipt check beside source
    if run_id:
        receipt = Path(policy["paths"]["runs_root"]) / run_id / "archive_receipt.json"
        if receipt.is_file():
            data = safety.load_json(receipt)
            if data.get("archive_path") != str(archive_path):
                result.warn("receipt archive_path differs from verified path")
            if data.get("source_manifest_sha256") != manifest.get("source_manifest_sha256"):
                return result.fail("receipt source_manifest_sha256 mismatch", safety.EXIT_INTEGRITY).finalize()

    if json_out:
        safety.write_json_atomic(json_out, result.finalize().to_dict())
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify an archived run")
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--policy",
        default=str(REPO_ROOT / "configs" / "system" / "archive_policy.yaml"),
    )
    p.add_argument("--archive-path", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(
            run_id=args.run_id,
            policy_path=Path(args.policy),
            archive_path=Path(args.archive_path) if args.archive_path else None,
            json_out=Path(args.json_out) if args.json_out else None,
        )
    except safety.ArchiveError as exc:
        result = safety.OpResult().fail(str(exc), exc.exit_code).finalize()
    except Exception as exc:  # noqa: BLE001
        result = safety.OpResult().fail(f"unhandled: {exc}", safety.EXIT_CONFIG).finalize()
        result.extras["traceback"] = traceback.format_exc(limit=10)
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
