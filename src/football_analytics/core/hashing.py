"""SHA-256 helpers and deterministic directory manifests (Stage 2B)."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_CHUNK_SIZE = 16 * 1024 * 1024
MIN_CHUNK_SIZE = 4096
CANONICALIZATION_VERSION = 1


class HashError(ValueError):
    """Hashing or path safety failure."""


@dataclass(frozen=True)
class FileHashRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DirectoryHashManifest:
    algorithm: str
    canonicalization_version: int
    files: tuple[FileHashRecord, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "canonicalization_version": self.canonicalization_version,
            "files": [f.to_dict() for f in self.files],
            "digest": self.digest,
        }


def _reject_non_regular(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise HashError(f"{label}: symlink rejected: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise HashError(f"{label}: cannot stat: {path}") from exc
    if not stat.S_ISREG(mode):
        raise HashError(f"{label}: not a regular file: {path}")


def _bounded_chunk_size(chunk_size: int) -> int:
    if (
        not isinstance(chunk_size, int)
        or chunk_size < MIN_CHUNK_SIZE
        or chunk_size > MAX_CHUNK_SIZE
    ):
        raise HashError(f"chunk_size out of bounds [{MIN_CHUNK_SIZE}, {MAX_CHUNK_SIZE}]")
    return chunk_size


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        raise HashError("sha256_bytes requires bytes")
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Stream SHA-256 of a regular file; reject symlink/special; detect mid-hash mutation."""
    target = Path(path)
    chunk = _bounded_chunk_size(chunk_size)
    if not target.exists():
        raise HashError(f"file missing: {target}")
    _reject_non_regular(target, label="sha256_file")
    before = target.lstat()
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    after = target.lstat()
    if (before.st_mtime_ns, before.st_size, before.st_ino) != (
        after.st_mtime_ns,
        after.st_size,
        after.st_ino,
    ):
        raise HashError(f"file changed during hash: {target}")
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """UTF-8 JSON with sorted keys, compact separators, allow_nan=False."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HashError(f"canonical JSON failed: {exc}") from exc
    return text.encode("utf-8")


def hash_canonical_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _normalize_rel(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise HashError(f"path escapes root: {path}") from exc
    text = rel.as_posix()
    if text.startswith("/") or ".." in text.split("/"):
        raise HashError(f"unsafe relative path: {text}")
    return text


def hash_directory_tree(
    path: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    include_hidden: bool = False,
) -> DirectoryHashManifest:
    """Deterministic directory manifest hashed via canonical JSON."""
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise HashError(f"directory missing: {root}")
    if root.is_symlink():
        raise HashError(f"directory symlink rejected: {root}")
    chunk = _bounded_chunk_size(chunk_size)
    records: list[FileHashRecord] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        # prune symlink dirs and optionally hidden
        keep: list[str] = []
        for name in sorted(dirnames):
            child = base / name
            if child.is_symlink():
                raise HashError(f"symlink directory rejected: {child}")
            if not include_hidden and name.startswith("."):
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            if not include_hidden and name.startswith("."):
                continue
            fp = base / name
            if fp.is_symlink():
                raise HashError(f"symlink file rejected: {fp}")
            _reject_non_regular(fp, label="hash_directory_tree")
            rel = _normalize_rel(fp, root)
            digest = sha256_file(fp, chunk_size=chunk)
            size = fp.stat().st_size
            records.append(FileHashRecord(relative_path=rel, size_bytes=size, sha256=digest))

    records.sort(key=lambda r: r.relative_path)
    payload = {
        "algorithm": "sha256",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "files": [r.to_dict() for r in records],
    }
    digest = hash_canonical_json(payload)
    return DirectoryHashManifest(
        algorithm="sha256",
        canonicalization_version=CANONICALIZATION_VERSION,
        files=tuple(records),
        digest=digest,
    )
