#!/usr/bin/env python3
"""Archive a completed run (copy → verify → receipt). Default: dry-run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.utils import archive_safety as safety  # noqa: E402


def archive_run(
    *,
    run_id: str,
    policy_path: Path,
    execute: bool,
    json_out: Optional[Path],
) -> safety.OpResult:
    result = safety.OpResult()
    tmp_archive: Optional[Path] = None
    policy = safety.load_policy(policy_path)
    paths = policy["paths"]
    pol = policy["policy"]
    result.extras["independent_backup"] = False
    result.extras["failure_domain"] = pol.get("failure_domain")
    result.extras["archive_backend"] = pol.get("active_archive_backend")

    safety.validate_run_id(run_id, policy)
    runs_root = Path(paths["runs_root"])
    archive_root = Path(paths["archive_root"])
    source_raw = runs_root / run_id
    if source_raw.is_symlink():
        return result.fail("source run is a symlink", safety.EXIT_SECURITY).finalize()
    source = safety.assert_contained(source_raw, runs_root, label="source_run")
    safety.assert_not_dangerous_operation_root(source)

    if not source.is_dir():
        return result.fail(f"source run missing: {source}", safety.EXIT_CONFIG).finalize()

    manifest_name = (policy.get("required_source_files") or ["run_manifest.json"])[0]
    manifest_path = source / manifest_name
    if not manifest_path.is_file():
        return result.fail("run_manifest.json missing", safety.EXIT_INTEGRITY).finalize()

    run_manifest = safety.parse_run_manifest(manifest_path)
    if run_manifest["run_id"] != run_id:
        return result.fail("manifest run_id mismatch", safety.EXIT_INTEGRITY).finalize()
    if pol.get("require_completed_status", True) and run_manifest["status"] != safety.ARCHIVEABLE_STATUS:
        return result.fail(
            f"run status must be completed, got {run_manifest['status']!r}",
            safety.EXIT_INTEGRITY,
        ).finalize()

    for rel in run_manifest.get("required_artifacts") or []:
        art = source / rel
        if not art.is_file() or art.is_symlink():
            return result.fail(f"required artifact missing: {rel}", safety.EXIT_INTEGRITY).finalize()

    try:
        safety.scan_tree_for_unsafe(source)
        records = safety.inventory_regular_files(source)
    except safety.ArchiveError as exc:
        return result.fail(str(exc), exc.exit_code).finalize()

    # Exclude prior receipts from archive content? Spec says inventory of source files.
    # archive_receipt may exist from prior attempt — include if present as regular file.
    source_manifest_sha = safety.sha256_file(manifest_path)
    final_archive = safety.assert_contained(
        archive_root / run_id, archive_root, label="archive_target"
    )

    min_free = int(pol.get("minimum_archive_free_bytes") or 0)
    if free := safety.free_bytes(archive_root):
        result.extras["archive_free_bytes"] = free
        if free < min_free:
            return result.fail(
                f"insufficient archive free space: {free} < {min_free}",
                safety.EXIT_CONFIG,
            ).finalize()

    if final_archive.exists():
        # safe no-op only if fully verified and checksums match
        try:
            existing_manifest, existing_recs = safety.verify_archive_tree(
                final_archive, expected_run_id=run_id, policy=policy
            )
            if existing_manifest.get("source_manifest_sha256") == source_manifest_sha and {
                (r.relative_path, r.sha256, r.size_bytes) for r in existing_recs
            } == {(r.relative_path, r.sha256, r.size_bytes) for r in records}:
                result.extras["mode"] = "idempotent_noop"
                result.extras["archive_path"] = str(final_archive)
                result.warn("archive already present and verified; no-op")
                if json_out:
                    safety.write_json_atomic(json_out, result.finalize().to_dict())
                return result.finalize()
        except safety.ArchiveError:
            pass
        return result.fail(
            f"archive target exists and is not an identical verified archive: {final_archive}",
            safety.EXIT_INTEGRITY,
        ).finalize()

    result.extras["planned_files"] = len(records)
    result.extras["planned_bytes"] = sum(r.size_bytes for r in records)
    result.extras["source_run_path"] = str(source)
    result.extras["archive_path"] = str(final_archive)
    result.extras["mode"] = "execute" if execute else "dry_run"

    if not execute:
        result.warn("dry-run only; no archive written")
        if json_out:
            safety.write_json_atomic(json_out, result.finalize().to_dict())
        return result.finalize()

    archive_root.mkdir(parents=True, exist_ok=True)
    tmp_name = f".archive_tmp_{run_id}_{os.getpid()}"
    tmp_archive = archive_root / tmp_name
    if tmp_archive.exists():
        return result.fail("temporary archive path collision", safety.EXIT_INTEGRITY).finalize()
    marker = tmp_archive / ".stage1d_archive_tmp"
    try:
        tmp_archive.mkdir(parents=False)
        marker.write_text("stage1d-tmp\n", encoding="utf-8")
        for rec in records:
            src = source / rec.relative_path
            dst = tmp_archive / rec.relative_path
            safety.copy_file_verified(src, dst, rec)
        # drop marker before building final manifest inventory semantics
        marker.unlink()
        # re-inventory tmp without marker
        copied = safety.inventory_regular_files(tmp_archive)
        if {(r.relative_path, r.sha256, r.size_bytes) for r in copied} != {
            (r.relative_path, r.sha256, r.size_bytes) for r in records
        }:
            raise safety.ArchiveError("copied inventory mismatch", safety.EXIT_INTEGRITY)
        archive_manifest = safety.build_archive_manifest(
            run_id=run_id,
            source_run_path=source,
            archive_path=final_archive,
            records=records,
            source_manifest_sha256=source_manifest_sha,
            policy=policy,
        )
        safety.write_json_atomic(tmp_archive / "archive_manifest.json", archive_manifest)
        safety.fsync_dir(tmp_archive)
        os.rename(str(tmp_archive), str(final_archive))
        tmp_archive = None
        safety.fsync_dir(archive_root)
        verified_manifest, _ = safety.verify_archive_tree(
            final_archive, expected_run_id=run_id, policy=policy
        )
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "archived_at": safety.utc_now(),
            "source_run_path": str(source),
            "archive_path": str(final_archive),
            "archive_id": verified_manifest["archive_id"],
            "source_manifest_sha256": source_manifest_sha,
            "archive_manifest_sha256": safety.sha256_file(final_archive / "archive_manifest.json"),
            "total_files": verified_manifest["total_files"],
            "total_bytes": verified_manifest["total_bytes"],
            "independent_backup": False,
            "failure_domain": pol.get("failure_domain"),
            "tool_version": policy.get("tool_version", safety.TOOL_VERSION),
        }
        safety.write_json_atomic(source / "archive_receipt.json", receipt)
        result.extras["receipt_path"] = str(source / "archive_receipt.json")
        result.extras["archive_id"] = verified_manifest["archive_id"]
        result.extras["total_files"] = verified_manifest["total_files"]
        result.extras["total_bytes"] = verified_manifest["total_bytes"]
    except Exception as exc:  # noqa: BLE001
        if tmp_archive is not None and tmp_archive.exists():
            try:
                safety.remove_exact_tree(
                    tmp_archive, must_be_under=archive_root, marker_name=".stage1d_archive_tmp"
                )
            except Exception as cleanup_exc:  # noqa: BLE001
                result.warn(f"tmp cleanup issue: {cleanup_exc}")
        code = exc.exit_code if isinstance(exc, safety.ArchiveError) else safety.EXIT_INTEGRITY
        return result.fail(str(exc), code).finalize()

    if json_out:
        safety.write_json_atomic(json_out, result.finalize().to_dict())
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Archive a completed run")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--policy",
        default=str(REPO_ROOT / "configs" / "system" / "archive_policy.yaml"),
    )
    p.add_argument("--json-out", default=None)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--execute", action="store_true", default=False)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    execute = bool(args.execute) and not bool(args.dry_run)
    # default dry-run when neither flag: dry-run
    if not args.execute:
        execute = False
    try:
        result = archive_run(
            run_id=args.run_id,
            policy_path=Path(args.policy),
            execute=execute,
            json_out=Path(args.json_out) if args.json_out else None,
        )
    except safety.ArchiveError as exc:
        result = safety.OpResult().fail(str(exc), exc.exit_code).finalize()
    except Exception as exc:  # noqa: BLE001
        result = safety.OpResult().fail(f"unhandled: {exc}", safety.EXIT_CONFIG)
        result.extras["traceback"] = traceback.format_exc(limit=15)
        result.finalize()
        if args.json_out:
            try:
                safety.write_json_atomic(Path(args.json_out), result.to_dict())
            except Exception:
                pass
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
