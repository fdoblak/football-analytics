#!/usr/bin/env python3
"""Quarantine cleanup for archived runs. Default: dry-run. No permanent delete."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.utils import archive_safety as safety  # noqa: E402


def cleanup_run(
    *,
    run_id: str,
    confirm_run_id: str | None,
    policy_path: Path,
    execute: bool,
    json_out: Path | None,
) -> safety.OpResult:
    result = safety.OpResult()
    policy = safety.load_policy(policy_path)
    paths = policy["paths"]
    pol = policy["policy"]
    safety.validate_run_id(run_id, policy)

    result.extras["mode"] = "execute" if execute else "dry_run"
    result.extras["cleanup_mode"] = pol.get("cleanup_mode", "quarantine")
    result.extras["independent_backup"] = False

    if execute and confirm_run_id != run_id:
        return result.fail(
            "execute requires --confirm-run-id matching --run-id exactly",
            safety.EXIT_SECURITY,
        ).finalize()

    runs_root = Path(paths["runs_root"])
    archive_root = Path(paths["archive_root"])
    quarantine_root = Path(paths["quarantine_root"])
    source_raw = runs_root / run_id
    if source_raw.is_symlink():
        return result.fail("source is symlink", safety.EXIT_SECURITY).finalize()
    source = safety.assert_contained(source_raw, runs_root, label="cleanup_source")
    safety.assert_not_dangerous_operation_root(source)

    if not source.is_dir():
        return result.fail(f"source missing: {source}", safety.EXIT_CONFIG).finalize()

    if safety.current_points_to_run(policy, source):
        return result.fail(
            "refusing cleanup: run is target of workspace current symlink",
            safety.EXIT_SECURITY,
        ).finalize()

    receipt_path = source / "archive_receipt.json"
    if not receipt_path.is_file():
        return result.fail("archive_receipt.json missing", safety.EXIT_INTEGRITY).finalize()
    receipt = safety.load_json(receipt_path)
    if receipt.get("run_id") != run_id:
        return result.fail("receipt run_id mismatch", safety.EXIT_INTEGRITY).finalize()

    archive_path = Path(str(receipt.get("archive_path") or ""))
    if not archive_path:
        return result.fail("receipt missing archive_path", safety.EXIT_INTEGRITY).finalize()
    archive_path = safety.assert_contained(archive_path, archive_root, label="archive")

    try:
        manifest, _ = safety.verify_archive_tree(
            archive_path, expected_run_id=run_id, policy=policy
        )
    except safety.ArchiveError as exc:
        return result.fail(f"archive re-verify failed: {exc}", exc.exit_code).finalize()

    if receipt.get("source_manifest_sha256") != manifest.get("source_manifest_sha256"):
        return result.fail(
            "receipt↔archive source_manifest_sha256 mismatch", safety.EXIT_INTEGRITY
        ).finalize()
    if receipt.get("archive_id") and receipt.get("archive_id") != manifest.get("archive_id"):
        return result.fail("receipt↔archive archive_id mismatch", safety.EXIT_INTEGRITY).finalize()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    q_name = f"{run_id}_{ts}"
    quarantine_target = quarantine_root / q_name
    try:
        quarantine_target = safety.assert_contained(
            quarantine_target, quarantine_root, label="quarantine"
        )
    except safety.ArchiveError as exc:
        return result.fail(str(exc), exc.exit_code).finalize()

    if quarantine_target.exists():
        return result.fail("quarantine target collision", safety.EXIT_INTEGRITY).finalize()

    if not safety.same_filesystem(source, quarantine_root):
        return result.fail(
            "cross-device move refused (source and quarantine must share filesystem)",
            safety.EXIT_SECURITY,
        ).finalize()

    result.extras.update(
        {
            "source_path": str(source),
            "archive_path": str(archive_path),
            "quarantine_path": str(quarantine_target),
            "retention_days": pol.get("quarantine_retention_days"),
            "purge_status": "not_performed",
        }
    )

    if not execute:
        result.warn("dry-run only; no quarantine move")
        if json_out:
            safety.write_json_atomic(json_out, result.finalize().to_dict())
        return result.finalize()

    quarantine_root.mkdir(parents=True, exist_ok=True)
    q_receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "original_path": str(source),
        "quarantine_path": str(quarantine_target),
        "archive_path": str(archive_path),
        "archive_verified_at": safety.utc_now(),
        "moved_at": safety.utc_now(),
        "retention_days": pol.get("quarantine_retention_days", 30),
        "purge_status": "not_performed",
        "cleanup_mode": "quarantine",
        "independent_backup": False,
        "tool_version": policy.get("tool_version", safety.TOOL_VERSION),
    }
    try:
        # write receipt into source before move so it lands in quarantine
        safety.write_json_atomic(source / "quarantine_receipt.json", q_receipt)
        os.rename(str(source), str(quarantine_target))
        safety.fsync_dir(quarantine_root)
        safety.fsync_dir(runs_root)
    except Exception as exc:  # noqa: BLE001
        return result.fail(f"quarantine move failed: {exc}", safety.EXIT_INTEGRITY).finalize()

    if source.exists():
        return result.fail(
            "source still exists after quarantine move", safety.EXIT_INTEGRITY
        ).finalize()
    if not quarantine_target.is_dir():
        return result.fail("quarantine target missing after move", safety.EXIT_INTEGRITY).finalize()

    # archive still intact
    try:
        safety.verify_archive_tree(archive_path, expected_run_id=run_id, policy=policy)
    except safety.ArchiveError as exc:
        return result.fail(f"archive broken after cleanup: {exc}", exc.exit_code).finalize()

    result.extras["quarantine_receipt"] = str(quarantine_target / "quarantine_receipt.json")
    if json_out:
        safety.write_json_atomic(json_out, result.finalize().to_dict())
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quarantine cleanup for archived runs")
    p.add_argument("--run-id", required=True)
    p.add_argument("--confirm-run-id", default=None)
    p.add_argument(
        "--policy",
        default=str(REPO_ROOT / "configs" / "system" / "archive_policy.yaml"),
    )
    p.add_argument("--json-out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execute = bool(args.execute) and not bool(args.dry_run)
    if not args.execute:
        execute = False
    try:
        result = cleanup_run(
            run_id=args.run_id,
            confirm_run_id=args.confirm_run_id,
            policy_path=Path(args.policy),
            execute=execute,
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
