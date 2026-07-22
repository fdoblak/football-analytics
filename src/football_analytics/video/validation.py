"""Path/source safety and policy validation for Stage 3A (no FFmpeg execution)."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from football_analytics.core.hashing import HashError, sha256_file
from football_analytics.utils.archive_safety import (
    DANGEROUS_ROOTS,
    assert_contained,
    assert_not_dangerous_operation_root,
    resolve_strict,
)
from football_analytics.video.types import (
    IngestRequest,
    SourceKind,
    VideoPolicyError,
    VideoSource,
    VideoSourceError,
)

ENV_VAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
URL_SCHEMES = frozenset({"http", "https", "ftp", "s3", "gs", "rtmp", "rtsp"})


@dataclass(frozen=True)
class SourceIntegrity:
    path: str
    size_bytes: int
    sha256: str
    mutated: bool = False


def reject_unsafe_path_string(path: str, *, label: str) -> str:
    if not isinstance(path, str) or not path:
        raise VideoSourceError(f"{label} empty")
    if "\x00" in path:
        raise VideoSourceError(f"{label} contains null byte")
    if "~" in path:
        raise VideoSourceError(f"{label} home shorthand forbidden")
    if ENV_VAR_RE.search(path):
        raise VideoSourceError(f"{label} environment expansion forbidden")
    if ".." in Path(path).parts:
        raise VideoSourceError(f"{label} relative traversal forbidden")
    parsed = urlparse(path)
    if parsed.scheme.lower() in URL_SCHEMES:
        raise VideoSourceError(f"{label} network URL scheme forbidden")
    if path.startswith(("http://", "https://", "ftp://", "s3://", "gs://")):
        raise VideoSourceError(f"{label} network URI forbidden")
    return path


def require_absolute_path(path: str, *, label: str) -> Path:
    reject_unsafe_path_string(path, label=label)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise VideoSourceError(f"{label} must be absolute")
    return candidate


def classify_file_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise VideoSourceError(f"cannot stat: {path}") from exc
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "char_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "special"


def assert_safe_source_path(
    path: str,
    *,
    contain_root: str | Path,
    policy: Mapping[str, Any],
) -> Path:
    """Validate source path string and filesystem kind under containment root."""
    if policy.get("network_sources_allowed") is not False:
        raise VideoPolicyError("network_sources_allowed must be false")
    abs_path = require_absolute_path(path, label="source_path")
    root = require_absolute_path(str(contain_root), label="contain_root")
    assert_not_dangerous_operation_root(root)
    if str(root.resolve()) in DANGEROUS_ROOTS or str(root) in DANGEROUS_ROOTS:
        # still allow workspace/video_contract_checks and football_data leaves via assert_contained
        pass
    resolved = resolve_strict(abs_path)
    assert_contained(resolved, resolve_strict(root), label="source_path")
    kind = classify_file_kind(abs_path)
    if kind == "symlink" and not policy.get("symlinks_allowed", False):
        raise VideoSourceError("symlink source rejected")
    if kind != "regular":
        if kind == "directory":
            raise VideoSourceError("directory pretending to be video rejected")
        if not policy.get("special_files_allowed", False):
            raise VideoSourceError(f"special file rejected: {kind}")
        raise VideoSourceError(f"unsupported source kind: {kind}")
    return resolved


def assert_safe_output_root(
    output_root: str,
    *,
    contain_root: str | Path,
    source_path: str,
    overwrite_allowed: bool = False,
) -> Path:
    if overwrite_allowed:
        raise VideoSourceError("overwrite_allowed must be false")
    out = require_absolute_path(output_root, label="output_root")
    root = require_absolute_path(str(contain_root), label="contain_root")
    assert_not_dangerous_operation_root(root)
    resolved_out = resolve_strict(out) if out.exists() else out.resolve()
    assert_contained(resolved_out, resolve_strict(root), label="output_root")
    src = require_absolute_path(source_path, label="source_path")
    if resolve_strict(src) == resolved_out:
        raise VideoSourceError("source/output path collision")
    return resolved_out


def extension_hint(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else ""


def assert_extension_allowed(path: Path, policy: Mapping[str, Any]) -> str:
    ext = extension_hint(path)
    allowed = {str(x).lower() for x in policy["allowed_file_extensions"]}
    if ext not in allowed:
        raise VideoSourceError(f"extension not allowed: {ext or '<none>'}")
    return ext


def verify_source_integrity(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> SourceIntegrity:
    """Stream SHA-256 with mutation detection (TOCTOU residual risk documented)."""
    kind = classify_file_kind(path)
    if kind != "regular":
        raise VideoSourceError(f"integrity requires regular file, got {kind}")
    before = path.lstat()
    try:
        digest = sha256_file(path)
    except HashError as exc:
        raise VideoSourceError(str(exc)) from exc
    after = path.lstat()
    mutated = (before.st_mtime_ns, before.st_size, before.st_ino) != (
        after.st_mtime_ns,
        after.st_size,
        after.st_ino,
    )
    size = int(after.st_size)
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise VideoSourceError(f"source size mismatch: expected {expected_size_bytes} got {size}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise VideoSourceError("source sha256 mismatch (mutation or wrong file)")
    if mutated:
        raise VideoSourceError("source mutated during hash (TOCTOU)")
    return SourceIntegrity(path=str(path), size_bytes=size, sha256=digest, mutated=False)


def assert_request_source_compatibility(
    request: IngestRequest,
    source: VideoSource,
    policy: Mapping[str, Any],
) -> None:
    if request.source_id != source.source_id:
        raise VideoSourceError("request.source_id mismatch")
    if request.expected_source_sha256 != source.source_sha256:
        raise VideoSourceError("expected_source_sha256 mismatch")
    if request.expected_source_size_bytes != source.source_size_bytes:
        raise VideoSourceError("expected_source_size_bytes mismatch")
    if request.policy_version != policy["policy_version"]:
        raise VideoPolicyError("request.policy_version mismatch")
    if request.fixture_mode:
        allowed = set(policy["fixture_policy"]["allowed_kinds"])
        if source.source_kind.value not in allowed:
            raise VideoSourceError("fixture_mode forbidden on production user source")
    if source.source_kind == SourceKind.USER_LOCAL_VIDEO and request.fixture_mode:
        raise VideoSourceError("fixture_mode cannot target user_local_video")
    if policy.get("network_sources_allowed") is not False:
        raise VideoPolicyError("network ingest must remain closed")


def assert_size_within_policy(size_bytes: int, policy: Mapping[str, Any]) -> None:
    maximum = int(policy["maximum_source_size_bytes"])
    if size_bytes < 0 or size_bytes > maximum:
        raise VideoPolicyError(f"source size out of policy bounds: {size_bytes}")


def assert_dimensions_within_policy(width: int, height: int, policy: Mapping[str, Any]) -> None:
    if width < int(policy["minimum_width"]) or height < int(policy["minimum_height"]):
        raise VideoPolicyError("dimensions below policy minimum")
    if width > int(policy["maximum_width"]) or height > int(policy["maximum_height"]):
        raise VideoPolicyError("dimensions above policy maximum")


def assert_duration_within_policy(duration_us: int | None, policy: Mapping[str, Any]) -> None:
    if duration_us is None:
        if not policy.get("unknown_duration_allowed", True):
            raise VideoPolicyError("unknown duration forbidden by policy")
        return
    if duration_us < int(policy["minimum_duration_us"]):
        raise VideoPolicyError("duration below policy minimum")
    if duration_us > int(policy["maximum_duration_us"]):
        raise VideoPolicyError("duration above policy maximum")


def same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a.resolve() == b.resolve()
