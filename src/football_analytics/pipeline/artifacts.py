"""Artifact path safety, hashing, copy, and on-disk verification (Stage 2D)."""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.redaction import is_sensitive_key
from football_analytics.pipeline.exceptions import ArtifactError
from football_analytics.pipeline.types import ArtifactRef, validate_safe_relative_path

# Re-export for callers / tests.
SAFE_REL_PATH = validate_safe_relative_path


def _reject_non_regular(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ArtifactError(f"{label}: symlink rejected: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArtifactError(f"{label}: cannot stat: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ArtifactError(f"{label}: not a regular file: {path}")


def is_hardlinked(path: Path) -> bool:
    """Return True when the path has more than one hard link."""
    try:
        return os.stat(path, follow_symlinks=False).st_nlink > 1
    except OSError as exc:
        raise ArtifactError(f"cannot stat for hardlink check: {path}") from exc


def build_artifact_ref(
    logical_name: str,
    path: Path | str,
    *,
    root: Path,
    media_type: str,
    contract_name: str | None = None,
    contract_version: int | None = None,
    schema_fingerprint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Hash a regular file under ``root`` and build an :class:`ArtifactRef`."""
    target = Path(path)
    base = Path(root)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        _reject_non_regular(target, label="build_artifact_ref")
    if not target.exists():
        raise ArtifactError(f"artifact missing: {target}")
    _reject_non_regular(target, label="build_artifact_ref")
    try:
        rel = target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactError(f"artifact escapes root: {target}") from exc
    rel = validate_safe_relative_path(rel)
    digest = sha256_file(target)
    size = target.stat().st_size
    meta = dict(metadata or {})
    for key in meta:
        if is_sensitive_key(key):
            raise ArtifactError(f"secret-bearing metadata key forbidden: {key}")
    return ArtifactRef(
        logical_name=logical_name,
        relative_path=rel,
        media_type=media_type,
        size_bytes=size,
        sha256=digest,
        contract_name=contract_name,
        contract_version=contract_version,
        schema_fingerprint=schema_fingerprint,
        metadata=meta,
    )


def verify_artifact_on_disk(
    ref: ArtifactRef,
    *,
    root: Path,
    reject_hardlinks: bool = False,
) -> None:
    """Verify size and SHA-256; reject symlinks/special files (and hardlinks if policy)."""
    base = Path(root)
    rel = validate_safe_relative_path(ref.relative_path)
    target = (base / rel).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise ArtifactError(f"artifact escapes root: {rel}") from exc
    if not target.exists():
        raise ArtifactError(f"artifact missing: {rel}")
    _reject_non_regular(target, label="verify_artifact_on_disk")
    if reject_hardlinks and is_hardlinked(target):
        raise ArtifactError(f"hardlink rejected: {rel}")
    size = target.stat().st_size
    if size != ref.size_bytes:
        raise ArtifactError(f"size mismatch for {rel}: expected {ref.size_bytes}, got {size}")
    digest = sha256_file(target)
    if digest != ref.sha256:
        raise ArtifactError(f"sha256 mismatch for {rel}")
    if "parquet" in ref.media_type.lower() or rel.lower().endswith(".parquet"):
        _verify_parquet_schema_fingerprint(target, ref)


def _verify_parquet_schema_fingerprint(path: Path, ref: ArtifactRef) -> None:
    """Lazy PyArrow metadata check when Parquet contract fingerprint is present."""
    if not ref.schema_fingerprint:
        return
    import pyarrow.parquet as pq  # lazy: only when validating parquet artifacts

    try:
        meta = pq.read_metadata(path)
        schema = meta.schema.to_arrow_schema()
        md = schema.metadata or {}
        raw = md.get(b"football_analytics.schema_fingerprint")
        if raw is None:
            raise ArtifactError("parquet missing schema_fingerprint metadata")
        found = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if found != ref.schema_fingerprint:
            raise ArtifactError("parquet schema_fingerprint mismatch")
    except ArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"parquet metadata validation failed: {type(exc).__name__}") from exc


def copy_artifact_file(src: Path | str, dst: Path | str) -> Path:
    """Copy a regular file with ``shutil.copy2`` (no hardlink). Reject if destination exists."""
    source = Path(src)
    dest = Path(dst)
    if not source.exists():
        raise ArtifactError(f"copy source missing: {source}")
    _reject_non_regular(source, label="copy_artifact_file")
    if dest.exists() or dest.is_symlink():
        raise ArtifactError(f"copy destination already exists: {dest}")
    dest.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if dest.parent.is_symlink():
        raise ArtifactError("copy destination parent must not be a symlink")
    shutil.copy2(source, dest)
    _reject_non_regular(dest, label="copy_artifact_file.dest")
    with contextlib.suppress(OSError):
        os.chmod(dest, 0o600)
    return dest
