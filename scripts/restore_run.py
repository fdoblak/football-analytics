#!/usr/bin/env python3
"""Restore a verified archive into runs_root. Default: dry-run."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.utils import archive_safety as safety  # noqa: E402


def restore_run(
    *,
    run_id: str,
    policy_path: Path,
    execute: bool,
    json_out: Optional[Path],
) -> safety.OpResult:
    result = safety.OpResult()
    tmp: Optional[Path] = None
    policy = safety.load_policy(policy_path)
    paths = policy["paths"]
    safety.validate_run_id(run_id, policy)
    runs_root = Path(paths["runs_root"])
    archive_root = Path(paths["archive_root"])
    archive_path = safety.assert_contained(archive_root / run_id, archive_root, label="archive")
    target = safety.assert_contained(runs_root / run_id, runs_root, label="restore_target")
    safety.assert_not_dangerous_operation_root(target)

    try:
        manifest, records = safety.verify_archive_tree(
            archive_path, expected_run_id=run_id, policy=policy
        )
    except safety.ArchiveError as exc:
        return result.fail(str(exc), exc.exit_code).finalize()

    result.extras["archive_path"] = str(archive_path)
    result.extras["restore_target"] = str(target)
    result.extras["independent_backup"] = False
    result.extras["mode"] = "execute" if execute else "dry_run"
    result.extras["planned_files"] = len(records)

    if target.exists():
        return result.fail(
            f"restore target already exists (no overwrite): {target}",
            safety.EXIT_INTEGRITY,
        ).finalize()

    if not execute:
        result.warn("dry-run only; no restore written")
        if json_out:
            safety.write_json_atomic(json_out, result.finalize().to_dict())
        return result.finalize()

    runs_root.mkdir(parents=True, exist_ok=True)
    tmp = runs_root / f".restore_tmp_{run_id}_{os.getpid()}"
    if tmp.exists():
        return result.fail("temporary restore path collision", safety.EXIT_INTEGRITY).finalize()
    try:
        tmp.mkdir(parents=False)
        (tmp / ".stage1d_restore_tmp").write_text("stage1d-tmp\n", encoding="utf-8")
        for rec in records:
            src = archive_path / rec.relative_path
            dst = tmp / rec.relative_path
            safety.copy_file_verified(src, dst, rec)
        (tmp / ".stage1d_restore_tmp").unlink()
        # do NOT copy archive_manifest.json into the run as a normal artifact
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "restored_at": safety.utc_now(),
            "archive_path": str(archive_path),
            "archive_id": manifest.get("archive_id"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "total_files": manifest.get("total_files"),
            "total_bytes": manifest.get("total_bytes"),
            "tool_version": policy.get("tool_version", safety.TOOL_VERSION),
        }
        safety.write_json_atomic(tmp / "restore_receipt.json", receipt)
        # verify hashes of restored payload files (exclude receipt)
        for rec in records:
            p = tmp / rec.relative_path
            if p.stat().st_size != rec.size_bytes or safety.sha256_file(p).lower() != rec.sha256.lower():
                raise safety.ArchiveError(f"restore hash mismatch: {rec.relative_path}")
        os.rename(str(tmp), str(target))
        tmp = None
        safety.fsync_dir(runs_root)
        # final compare against archive records
        restored = safety.inventory_regular_files(target)
        restored_payload = [r for r in restored if r.relative_path != "restore_receipt.json"]
        if {(r.relative_path, r.sha256, r.size_bytes) for r in restored_payload} != {
            (r.relative_path, r.sha256, r.size_bytes) for r in records
        }:
            raise safety.ArchiveError("final restore inventory mismatch vs archive")
        # archive unchanged: re-verify
        safety.verify_archive_tree(archive_path, expected_run_id=run_id, policy=policy)
        result.extras["restored_files"] = len(records)
        result.extras["restore_receipt"] = str(target / "restore_receipt.json")
    except Exception as exc:  # noqa: BLE001
        if tmp is not None and tmp.exists():
            try:
                safety.remove_exact_tree(tmp, must_be_under=runs_root)
            except Exception as cleanup_exc:  # noqa: BLE001
                result.warn(f"tmp cleanup issue: {cleanup_exc}")
        code = exc.exit_code if isinstance(exc, safety.ArchiveError) else safety.EXIT_INTEGRITY
        return result.fail(str(exc), code).finalize()

    if json_out:
        safety.write_json_atomic(json_out, result.finalize().to_dict())
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restore archived run into runs_root")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--policy",
        default=str(REPO_ROOT / "configs" / "system" / "archive_policy.yaml"),
    )
    p.add_argument("--json-out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    execute = bool(args.execute) and not bool(args.dry_run)
    if not args.execute:
        execute = False
    try:
        result = restore_run(
            run_id=args.run_id,
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
