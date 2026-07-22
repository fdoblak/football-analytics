"""Atomic JSON record writer (Stage 2B)."""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import canonical_json_bytes
from football_analytics.core.redaction import is_sensitive_key, redact_value


class RecordError(ValueError):
    """Safe record write failure."""


def _ensure_parent_safe(parent: Path, *, contain_root: Path | None) -> Path:
    parent = parent.resolve()
    if parent.is_symlink():
        raise RecordError("parent directory must not be a symlink")
    if not parent.exists():
        if contain_root is not None:
            contain = contain_root.resolve()
            try:
                parent.relative_to(contain)
            except ValueError as exc:
                raise RecordError("parent escapes containment root") from exc
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise RecordError("parent is not a safe directory")
    if contain_root is not None:
        contain = contain_root.resolve()
        try:
            parent.relative_to(contain)
        except ValueError as exc:
            raise RecordError("parent escapes containment root") from exc
    return parent


def _reject_secret_payload(payload: Any) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if is_sensitive_key(k):
                raise RecordError("secret-bearing key forbidden in record payload")
            _reject_secret_payload(v)
    elif isinstance(payload, list):
        for v in payload:
            _reject_secret_payload(v)


def write_json_record(
    path: Path | str,
    payload: dict[str, Any],
    *,
    contain_root: Path | None = None,
    overwrite: bool = False,
    pretty: bool = True,
    mode: int = 0o600,
) -> Path:
    """Write JSON atomically; default no-overwrite; fsync; mode 0600 when possible."""
    target = Path(path)
    if target.exists():
        if not overwrite:
            raise RecordError(f"target already exists: {target}")
        if target.is_symlink():
            raise RecordError("refusing to overwrite symlink target")
        mode_bits = target.lstat().st_mode
        if not stat.S_ISREG(mode_bits):
            raise RecordError("refusing to overwrite non-regular file")

    if not isinstance(payload, dict):
        raise RecordError("payload must be a dict")
    _reject_secret_payload(payload)
    safe_payload = redact_value(payload)
    if not isinstance(safe_payload, dict):
        raise RecordError("payload redaction failed")

    parent = _ensure_parent_safe(target.parent, contain_root=contain_root)
    if contain_root is not None:
        try:
            (parent / target.name).resolve().relative_to(contain_root.resolve())
        except ValueError as exc:
            raise RecordError("target escapes containment root") from exc

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if pretty:
                json.dump(
                    safe_payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")
            else:
                handle.write(canonical_json_bytes(safe_payload).decode("utf-8"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        if target.exists() and not overwrite:
            raise RecordError(f"target already exists: {target}")
        os.replace(str(tmp_path), str(target))
        with contextlib.suppress(OSError):
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        with contextlib.suppress(OSError):
            os.chmod(target, mode)
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise
    return target
