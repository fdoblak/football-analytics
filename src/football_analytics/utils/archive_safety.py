"""Shared path safety, inventory, checksum, and archive helpers (Stage 1D)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TOOL_VERSION = "1.0.0-stage1d"

EXIT_OK = 0
EXIT_INTEGRITY = 1
EXIT_CONFIG = 2
EXIT_SECURITY = 3

RUN_STATUSES = frozenset({"pending", "running", "failed", "completed", "archived"})
ARCHIVEABLE_STATUS = "completed"

DANGEROUS_ROOTS = frozenset(
    {
        "/",
        "/home",
        "/home/fdoblak",
        "/home/fdoblak/workspace",
        "/home/fdoblak/football_data",
        "/home/fdoblak/projects",
        "/home/fdoblak/projects/football-analytics",
    }
)

ENV_VAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
REL_UNSAFE_RE = re.compile(r"(^|/)\.\.(/|$)")


class ArchiveError(Exception):
    def __init__(self, message: str, exit_code: int = EXIT_INTEGRITY) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class FileRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    file_type: str = "regular"

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "file_type": self.file_type,
        }


@dataclass
class OpResult:
    status: str = "PASS"
    exit_code: int = EXIT_OK
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str, code: int = EXIT_INTEGRITY) -> OpResult:
        self.errors.append(msg)
        self.exit_code = code
        self.status = "FAIL"
        return self

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self) -> OpResult:
        if self.errors:
            self.status = "FAIL"
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_OK
        else:
            self.status = "PASS"
            self.exit_code = EXIT_OK
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "exit_code": self.exit_code,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "extras": self.extras,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArchiveError(f"policy missing: {path}", EXIT_CONFIG)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(f"policy parse failed: {exc}", EXIT_CONFIG) from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ArchiveError("policy schema_version must be 1", EXIT_CONFIG)
    paths = data.get("paths") or {}
    policy = data.get("policy") or {}
    for key in ("runs_root", "archive_root", "quarantine_root"):
        if key not in paths:
            raise ArchiveError(f"policy missing paths.{key}", EXIT_CONFIG)
        raw = paths[key]
        if not isinstance(raw, str) or not raw:
            raise ArchiveError(f"paths.{key} must be non-empty string", EXIT_CONFIG)
        if ENV_VAR_RE.search(raw):
            raise ArchiveError(f"paths.{key} contains unresolved env var", EXIT_SECURITY)
        if not os.path.isabs(raw):
            raise ArchiveError(f"paths.{key} must be absolute", EXIT_SECURITY)
    if policy.get("independent_backup") is not False:
        raise ArchiveError("policy.independent_backup must be false", EXIT_CONFIG)
    if policy.get("checksum_algorithm") != "sha256":
        raise ArchiveError("only sha256 supported", EXIT_CONFIG)
    pattern = (data.get("run_id") or {}).get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ArchiveError("run_id.pattern required", EXIT_CONFIG)
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ArchiveError(f"invalid run_id.pattern: {exc}", EXIT_CONFIG) from exc
    return data


def validate_run_id(run_id: str, policy: dict[str, Any]) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise ArchiveError("run_id empty", EXIT_SECURITY)
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ArchiveError("run_id contains path separators", EXIT_SECURITY)
    pattern = (policy.get("run_id") or {}).get("pattern")
    if not isinstance(pattern, str) or not re.fullmatch(pattern, run_id):
        raise ArchiveError(f"run_id does not match policy pattern: {run_id}", EXIT_SECURITY)


def resolve_strict(path: Path) -> Path:
    # Do not follow final component if missing; resolve parents.
    return path if path.is_absolute() else path.resolve()


def assert_not_dangerous_operation_root(path: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ArchiveError(f"cannot resolve path: {exc}", EXIT_SECURITY) from exc
    if str(resolved) in DANGEROUS_ROOTS:
        raise ArchiveError(f"refusing operation on dangerous root: {resolved}", EXIT_SECURITY)
    if not str(resolved):
        raise ArchiveError("empty resolved path", EXIT_SECURITY)


def assert_contained(path: Path, root: Path, *, label: str) -> Path:
    try:
        root_r = root.resolve()
        path_r = path.resolve()
    except OSError as exc:
        raise ArchiveError(f"{label}: resolve failed: {exc}", EXIT_SECURITY) from exc
    try:
        path_r.relative_to(root_r)
    except ValueError as exc:
        raise ArchiveError(
            f"{label}: path escape {path_r} not under {root_r}",
            EXIT_SECURITY,
        ) from exc
    if path_r == root_r:
        raise ArchiveError(f"{label}: refusing root itself as target: {path_r}", EXIT_SECURITY)
    return path_r


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(str(path if path.exists() else path.parent))
    return int(usage.free)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_special_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return True
    if stat.S_ISFIFO(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISSOCK(mode):
        return True
    is_door = getattr(stat, "S_ISDOOR", None)
    return bool(callable(is_door) and is_door(mode))


def normalize_rel_path(rel: str) -> str:
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        raise ArchiveError(f"absolute relative_path forbidden: {rel}", EXIT_SECURITY)
    if "\\" in rel:
        raise ArchiveError(f"backslash in relative_path forbidden: {rel}", EXIT_SECURITY)
    if REL_UNSAFE_RE.search(rel) or rel in {".", ".."}:
        raise ArchiveError(f"unsafe relative_path: {rel}", EXIT_SECURITY)
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise ArchiveError(f"unsafe relative_path: {rel}", EXIT_SECURITY)
    return "/".join(parts)


def scan_tree_for_unsafe(root: Path) -> None:
    """Reject symlinks and special files under root (do not follow links)."""
    if root.is_symlink():
        raise ArchiveError(f"root is symlink: {root}", EXIT_SECURITY)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        # prune symlink dirs
        keep_dirs = []
        for name in dirnames:
            p = base / name
            if p.is_symlink():
                raise ArchiveError(f"symlink directory rejected: {p}", EXIT_SECURITY)
            keep_dirs.append(name)
        dirnames[:] = keep_dirs
        for name in filenames:
            p = base / name
            if p.is_symlink():
                raise ArchiveError(f"symlink file rejected: {p}", EXIT_SECURITY)
            if is_special_file(p):
                raise ArchiveError(f"special file rejected: {p}", EXIT_SECURITY)


def inventory_regular_files(root: Path) -> list[FileRecord]:
    scan_tree_for_unsafe(root)
    records: list[FileRecord] = []
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in sorted(filenames):
            p = base / name
            if not p.is_file() or p.is_symlink():
                raise ArchiveError(f"non-regular file: {p}", EXIT_SECURITY)
            rel = normalize_rel_path(str(p.relative_to(root)).replace(os.sep, "/"))
            records.append(
                FileRecord(
                    relative_path=rel,
                    size_bytes=p.stat().st_size,
                    sha256=sha256_file(p),
                    file_type="regular",
                )
            )
    records.sort(key=lambda r: r.relative_path)
    # duplicate check
    seen: set[str] = set()
    for rec in records:
        if rec.relative_path in seen:
            raise ArchiveError(f"duplicate relative_path: {rec.relative_path}", EXIT_SECURITY)
        seen.add(rec.relative_path)
    return records


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    parent = path.parent
    if not parent.is_dir():
        raise ArchiveError(f"json parent missing: {parent}", EXIT_CONFIG)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
        fsync_dir(parent)
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def copy_file_verified(src: Path, dst: Path, expected: FileRecord) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as s, open(dst, "wb") as d:
        while True:
            chunk = s.read(1024 * 1024)
            if not chunk:
                break
            d.write(chunk)
        d.flush()
        os.fsync(d.fileno())
    if dst.stat().st_size != expected.size_bytes:
        raise ArchiveError(f"size mismatch after copy: {dst}", EXIT_INTEGRITY)
    digest = sha256_file(dst)
    if digest.lower() != expected.sha256.lower():
        raise ArchiveError(f"hash mismatch after copy: {dst}", EXIT_INTEGRITY)


def remove_exact_tree(path: Path, *, must_be_under: Path, marker_name: str | None = None) -> None:
    """Remove only an exact temporary tree we created. No rm -rf of broad roots."""
    path = path.resolve()
    must = must_be_under.resolve()
    try:
        path.relative_to(must)
    except ValueError as exc:
        raise ArchiveError(f"refuse cleanup outside bound: {path}", EXIT_SECURITY) from exc
    if path == must:
        raise ArchiveError("refuse removing bound root", EXIT_SECURITY)
    if path.is_symlink():
        raise ArchiveError(f"refuse removing symlink tree: {path}", EXIT_SECURITY)
    allowed_prefix = path.name.startswith(".archive_tmp_") or path.name.startswith(".restore_tmp_")
    if marker_name and not (path / marker_name).exists() and not allowed_prefix:
        raise ArchiveError(f"refuse removing unmarked tree: {path}", EXIT_SECURITY)
    if not path.is_dir():
        if path.is_file():
            path.unlink()
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.is_symlink():
                p.unlink()
            else:
                p.unlink()
        for name in dirnames:
            p = base / name
            if p.is_symlink():
                p.unlink()
            else:
                p.rmdir()
    path.rmdir()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(f"json load failed {path}: {exc}", EXIT_CONFIG) from exc
    if not isinstance(data, dict):
        raise ArchiveError(f"json must be object: {path}", EXIT_CONFIG)
    return data


def parse_run_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("schema_version") != 1:
        raise ArchiveError("run_manifest schema_version must be 1", EXIT_CONFIG)
    if data.get("status") not in RUN_STATUSES:
        raise ArchiveError(f"invalid run status: {data.get('status')}", EXIT_CONFIG)
    if not isinstance(data.get("run_id"), str):
        raise ArchiveError("run_manifest.run_id missing", EXIT_CONFIG)
    return data


def validate_archive_manifest_structure(manifest: dict[str, Any]) -> list[FileRecord]:
    if manifest.get("schema_version") != 1:
        raise ArchiveError("archive_manifest schema_version must be 1", EXIT_INTEGRITY)
    for key in (
        "archive_id",
        "run_id",
        "source_run_path",
        "archive_path",
        "checksum_algorithm",
        "source_manifest_sha256",
        "files",
    ):
        if key not in manifest:
            raise ArchiveError(f"archive_manifest missing {key}", EXIT_INTEGRITY)
    if manifest.get("checksum_algorithm") != "sha256":
        raise ArchiveError("checksum_algorithm must be sha256", EXIT_INTEGRITY)
    if manifest.get("independent_backup") is not False:
        raise ArchiveError("independent_backup must be false", EXIT_INTEGRITY)
    if not SHA256_RE.match(str(manifest.get("source_manifest_sha256") or "")):
        raise ArchiveError("source_manifest_sha256 invalid", EXIT_INTEGRITY)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ArchiveError("files must be list", EXIT_INTEGRITY)
    records: list[FileRecord] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ArchiveError("file entry must be object", EXIT_INTEGRITY)
        rel = normalize_rel_path(str(item.get("relative_path") or ""))
        if rel == "archive_manifest.json":
            raise ArchiveError("archive_manifest.json must not be in files list", EXIT_INTEGRITY)
        if rel in seen:
            raise ArchiveError(f"duplicate relative_path in manifest: {rel}", EXIT_SECURITY)
        seen.add(rel)
        digest = str(item.get("sha256") or "")
        if not SHA256_RE.match(digest):
            raise ArchiveError(f"bad sha256 for {rel}", EXIT_INTEGRITY)
        size = item.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArchiveError(f"bad size for {rel}", EXIT_INTEGRITY)
        if item.get("file_type") != "regular":
            raise ArchiveError(f"non-regular file_type for {rel}", EXIT_SECURITY)
        records.append(FileRecord(rel, size, digest.lower(), "regular"))
    expected_order = sorted(r.relative_path for r in records)
    actual_order = [r.relative_path for r in records]
    if actual_order != expected_order:
        raise ArchiveError("archive manifest files not deterministically sorted", EXIT_INTEGRITY)
    total_files = manifest.get("total_files")
    total_bytes = manifest.get("total_bytes")
    if total_files != len(records):
        raise ArchiveError("total_files mismatch", EXIT_INTEGRITY)
    if total_bytes != sum(r.size_bytes for r in records):
        raise ArchiveError("total_bytes mismatch", EXIT_INTEGRITY)
    return records


def verify_archive_tree(
    archive_path: Path,
    *,
    expected_run_id: str | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[FileRecord]]:
    if archive_path.is_symlink():
        raise ArchiveError(f"archive path is symlink: {archive_path}", EXIT_SECURITY)
    if not archive_path.is_dir():
        raise ArchiveError(f"archive missing: {archive_path}", EXIT_INTEGRITY)
    if policy:
        assert_contained(
            archive_path,
            Path(policy["paths"]["archive_root"]),
            label="archive",
        )
    manifest_path = archive_path / "archive_manifest.json"
    if not manifest_path.is_file():
        raise ArchiveError("archive_manifest.json missing", EXIT_INTEGRITY)
    manifest = load_json(manifest_path)
    records = validate_archive_manifest_structure(manifest)
    if expected_run_id and manifest.get("run_id") != expected_run_id:
        raise ArchiveError("manifest run_id mismatch", EXIT_INTEGRITY)
    scan_tree_for_unsafe(archive_path)
    on_disk: dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(archive_path, followlinks=False):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            rel = normalize_rel_path(str(p.relative_to(archive_path)).replace(os.sep, "/"))
            if rel == "archive_manifest.json":
                continue
            on_disk[rel] = p
    expected = {r.relative_path: r for r in records}
    missing = sorted(set(expected) - set(on_disk))
    extra = sorted(set(on_disk) - set(expected))
    if missing:
        raise ArchiveError(f"missing archive files: {missing}", EXIT_INTEGRITY)
    if extra:
        raise ArchiveError(f"unexpected archive files: {extra}", EXIT_INTEGRITY)
    for rel, rec in expected.items():
        p = on_disk[rel]
        size = p.stat().st_size
        if size != rec.size_bytes:
            raise ArchiveError(f"size mismatch {rel}", EXIT_INTEGRITY)
        digest = sha256_file(p)
        if digest.lower() != rec.sha256.lower():
            raise ArchiveError(f"hash mismatch {rel}", EXIT_INTEGRITY)
    # totals already checked vs records; recheck against disk
    disk_bytes = sum(p.stat().st_size for p in on_disk.values())
    if disk_bytes != manifest["total_bytes"]:
        raise ArchiveError("disk total_bytes mismatch", EXIT_INTEGRITY)
    return manifest, records


def current_points_to_run(policy: dict[str, Any], run_path: Path) -> bool:
    current = Path((policy.get("paths") or {}).get("current_symlink") or "")
    if not current or not str(current):
        return False
    if not current.exists() and not current.is_symlink():
        return False
    try:
        run_resolved = run_path.resolve()
        if current.is_symlink() or current.exists():
            try:
                target = current.resolve(strict=False)
            except OSError:
                return False
            workspace = Path((policy.get("paths") or {}).get("workspace_root") or current.parent)
            runs_root = Path((policy.get("paths") or {}).get("runs_root") or "")
            allowed_roots = []
            for root in (workspace, runs_root):
                if not root:
                    continue
                try:
                    allowed_roots.append(root.resolve())
                except OSError:
                    continue
            under_allowed = False
            for root in allowed_roots:
                try:
                    target.relative_to(root)
                    under_allowed = True
                    break
                except ValueError:
                    continue
            if not under_allowed:
                return False
            try:
                return target.resolve() == run_resolved
            except OSError:
                return False
        return False
    except OSError:
        return False


def build_archive_manifest(
    *,
    run_id: str,
    source_run_path: Path,
    archive_path: Path,
    records: Sequence[FileRecord],
    source_manifest_sha256: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    pol = policy.get("policy") or {}
    files = [r.to_dict() for r in sorted(records, key=lambda x: x.relative_path)]
    return {
        "schema_version": 1,
        "archive_id": f"arch_{run_id}_{uuid.uuid4().hex[:8]}",
        "run_id": run_id,
        "created_at": utc_now(),
        "source_run_path": str(source_run_path),
        "archive_path": str(archive_path),
        "archive_backend": pol.get("active_archive_backend", "wsl_local"),
        "failure_domain": pol.get("failure_domain", "same_wsl_vhdx"),
        "independent_backup": False,
        "checksum_algorithm": "sha256",
        "source_manifest_sha256": source_manifest_sha256,
        "total_files": len(files),
        "total_bytes": sum(r.size_bytes for r in records),
        "files": files,
        "verification": {
            "status": "verified_at_archive_time",
            "verified_at": utc_now(),
        },
        "tool_version": policy.get("tool_version", TOOL_VERSION),
    }


def same_filesystem(a: Path, b: Path) -> bool:
    try:
        return (
            os.stat(a if a.exists() else a.parent).st_dev
            == os.stat(b if b.exists() else b.parent).st_dev
        )
    except OSError:
        return False
