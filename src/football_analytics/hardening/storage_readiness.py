"""15C storage / backup readiness (do not pretend /mnt/d exists)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


class StorageReadinessError(ValueError):
    """Storage readiness validation failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def load_paths_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "system" / "paths.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StorageReadinessError("paths.yaml invalid")
    return raw


def load_archive_policy(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "system" / "archive_policy.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StorageReadinessError("archive_policy.yaml invalid")
    return raw


def validate_storage_readiness(
    repo_root: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    """Validate storage contracts without claiming D: / independent backup."""
    pol = policy or load_hardening_policy()
    if bool(pol.raw["storage"]["pretend_mnt_d_exists"]):
        raise StorageReadinessError("pretend_mnt_d_exists must remain false")
    paths = load_paths_config(repo_root)
    archive = load_archive_policy(repo_root)
    storage = paths.get("storage") or {}
    planned = str(storage.get("planned_archive_root") or "")
    status = str(storage.get("planned_archive_status") or "")
    independent = bool((archive.get("policy") or {}).get("independent_backup"))
    mnt_d_exists = Path("/mnt/d").exists()
    findings: list[str] = []
    # Allow unverified; do not require verified for Stage 15.
    if status == "verified" and not mnt_d_exists:
        findings.append("planned archive marked verified but /mnt/d missing")
    if independent:
        findings.append("independent_backup must remain false until Stage 16 verified migration")
    if bool(pol.raw["storage"]["independent_backup_claimed"]):
        findings.append("policy must not claim independent backup")
    if findings:
        raise StorageReadinessError("; ".join(findings))
    return {
        "active_root": storage.get("active_root"),
        "planned_archive_root": planned,
        "planned_archive_status": status,
        "mnt_d_exists": mnt_d_exists,
        "mnt_d_claimed_ready": False,
        "independent_backup": False,
        "classification": storage.get("classification"),
        "status": "PASS",
    }


def scan_large_files(
    root: Path,
    *,
    policy: HardeningPolicy | None = None,
    max_files: int = 200,
) -> dict[str, Any]:
    """Scan for large files; classify video/model exclusions (read-only)."""
    pol = policy or load_hardening_policy()
    threshold = int(pol.raw["storage"]["large_file_scan_bytes"])
    exclude_suffixes = {
        str(s).lower() for s in pol.raw["storage"]["exclude_suffixes_from_backup_claims"]
    }
    hits: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    if not root.exists():
        return {
            "root": str(root),
            "threshold_bytes": threshold,
            "large_files": [],
            "excluded_video_or_model": [],
            "scanned": False,
            "reason": "root_missing",
        }
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if count >= max_files and not hits:
                break
            fp = Path(dirpath) / name
            if fp.is_symlink():
                continue
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            count += 1
            suffix = fp.suffix.lower()
            if size >= threshold:
                item = {"path": str(fp), "size_bytes": size, "suffix": suffix}
                if suffix in exclude_suffixes:
                    excluded.append(item)
                else:
                    hits.append(item)
        if len(hits) + len(excluded) >= max_files:
            break
    return {
        "root": str(root),
        "threshold_bytes": threshold,
        "files_seen": count,
        "large_files": hits[:50],
        "excluded_video_or_model": excluded[:50],
        "scanned": True,
    }


def write_cleanup_receipt(
    path: Path,
    *,
    action: str,
    items: list[str],
    contain_root: Path | None = None,
) -> dict[str, Any]:
    """Write a cleanup receipt that never claims permanent delete by default."""
    receipt = {
        "schema_version": 1,
        "action": action,
        "items": list(items),
        "permanent_delete_performed": False,
        "mode": "quarantine_or_dry_run",
        "written_at_utc": _utc_now(),
    }
    receipt["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in receipt.items() if k != "receipt_fingerprint"}
    )
    target = Path(path)
    write_json_record(
        target,
        receipt,
        contain_root=contain_root or target.parent,
        overwrite=True,
    )
    return receipt


def quarantine_path_safe(src: Path, quarantine_root: Path) -> dict[str, Any]:
    """Move a path into quarantine (no permanent delete)."""
    import secrets
    import shutil

    qroot = Path(quarantine_root)
    qroot.mkdir(parents=True, mode=0o700, exist_ok=True)
    dest = qroot / src.name
    if dest.exists():
        dest = qroot / f"{src.name}_{secrets.token_hex(4)}"
    shutil.move(str(src), str(dest))
    receipt = {
        "schema_version": 1,
        "original_path": str(src),
        "quarantine_path": str(dest),
        "permanent_delete_performed": False,
        "moved_at_utc": _utc_now(),
    }
    receipt["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in receipt.items() if k != "receipt_fingerprint"}
    )
    write_json_record(
        (
            Path(dest).parent / f"{Path(dest).name}.quarantine_receipt.json"
            if Path(dest).is_file()
            else Path(dest) / "quarantine_receipt.json"
        ),
        receipt,
        contain_root=qroot,
        overwrite=False,
    )
    return receipt
