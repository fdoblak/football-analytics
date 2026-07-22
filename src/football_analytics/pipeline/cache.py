"""Content-addressed local cache: policy, publish, verify, restore, quarantine (Stage 2D)."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.redaction import redact_text
from football_analytics.pipeline.artifacts import (
    copy_artifact_file,
    is_hardlinked,
    verify_artifact_on_disk,
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.exceptions import ArtifactError, CacheError
from football_analytics.pipeline.types import (
    ArtifactRef,
    CacheManifest,
    StageIdentity,
    StageResult,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_MAX_POLICY_BYTES = 64 * 1024


@dataclass(frozen=True)
class CachePolicyConfig:
    """Loaded system cache policy (see configs/system/cache_policy.yaml)."""

    schema_version: int
    enabled: bool
    algorithm: str
    layout_version: int
    verify_on_read: bool
    verify_on_publish: bool
    reject_symlinks: bool
    reject_special_files: bool
    reject_hardlinks: bool
    lock_timeout_seconds: float
    max_manifest_bytes: int
    max_entry_files: int
    max_entry_bytes: int
    quarantine_corrupt_entries: bool
    automatic_purge: bool


def load_cache_policy(path: Path) -> CachePolicyConfig:
    """Load and validate cache policy YAML."""
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise CacheError("cache policy path must be a regular file")
    size = target.stat().st_size
    if size > _MAX_POLICY_BYTES:
        raise CacheError("cache policy file too large")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CacheError("cache policy root must be a mapping")
    required = {
        "schema_version",
        "enabled",
        "algorithm",
        "layout_version",
        "verify_on_read",
        "verify_on_publish",
        "reject_symlinks",
        "reject_special_files",
        "reject_hardlinks",
        "lock_timeout_seconds",
        "max_manifest_bytes",
        "max_entry_files",
        "max_entry_bytes",
        "quarantine_corrupt_entries",
        "automatic_purge",
    }
    missing = required - set(raw)
    if missing:
        raise CacheError(f"cache policy missing fields: {sorted(missing)}")
    if int(raw["schema_version"]) != 1:
        raise CacheError("unsupported cache policy schema_version")
    if str(raw["algorithm"]) != "sha256":
        raise CacheError("only sha256 cache algorithm supported")
    if int(raw["layout_version"]) != 1:
        raise CacheError("only layout_version=1 supported")
    if bool(raw.get("automatic_purge")):
        raise CacheError("automatic_purge must be false in Stage 2D")
    return CachePolicyConfig(
        schema_version=1,
        enabled=bool(raw["enabled"]),
        algorithm="sha256",
        layout_version=1,
        verify_on_read=bool(raw["verify_on_read"]),
        verify_on_publish=bool(raw["verify_on_publish"]),
        reject_symlinks=bool(raw["reject_symlinks"]),
        reject_special_files=bool(raw["reject_special_files"]),
        reject_hardlinks=bool(raw["reject_hardlinks"]),
        lock_timeout_seconds=float(raw["lock_timeout_seconds"]),
        max_manifest_bytes=int(raw["max_manifest_bytes"]),
        max_entry_files=int(raw["max_entry_files"]),
        max_entry_bytes=int(raw["max_entry_bytes"]),
        quarantine_corrupt_entries=bool(raw["quarantine_corrupt_entries"]),
        automatic_purge=False,
    )


def resolve_cache_root(paths_yaml: Path) -> Path:
    """Resolve ``system.cache`` from paths.yaml."""
    target = Path(paths_yaml)
    if not target.is_file() or target.is_symlink():
        raise CacheError("paths.yaml must be a regular file")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CacheError("paths.yaml root must be a mapping")
    system = raw.get("system")
    if not isinstance(system, dict) or "cache" not in system:
        raise CacheError("paths.yaml missing system.cache")
    root = Path(str(system["cache"]))
    if not root.is_absolute():
        raise CacheError("system.cache must be absolute")
    return root


def entry_dir(cache_root: Path, cache_key: str) -> Path:
    """Return ``<cache_root>/v1/sha256/<ab>/<remaining62>/``."""
    if not _SHA256_RE.fullmatch(cache_key):
        raise CacheError("cache_key must be 64 lowercase hex")
    return Path(cache_root) / "v1" / "sha256" / cache_key[:2] / cache_key[2:]


def _lock_path(cache_root: Path, cache_key: str) -> Path:
    if not _SHA256_RE.fullmatch(cache_key):
        raise CacheError("cache_key must be 64 lowercase hex")
    return Path(cache_root) / "v1" / "locks" / cache_key[:2] / f"{cache_key[2:]}.lock"


@contextlib.contextmanager
def acquire_key_lock(
    cache_root: Path,
    cache_key: str,
    timeout_seconds: float,
) -> Iterator[None]:
    """Acquire an exclusive ``fcntl.flock`` for ``cache_key`` with bounded timeout."""
    if timeout_seconds < 0:
        raise CacheError("lock timeout must be non-negative")
    lock_file = _lock_path(cache_root, cache_key)
    lock_file.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(lock_file.parent, 0o700)
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CacheError("cache key lock timeout") from None
                time.sleep(0.05)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


def _chmod_tree(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        with contextlib.suppress(OSError):
            os.chmod(base, 0o700)
        for name in dirnames:
            child = base / name
            if child.is_symlink():
                raise CacheError(f"symlink rejected under cache entry: {child}")
        for name in filenames:
            child = base / name
            if child.is_symlink():
                raise CacheError(f"symlink rejected under cache entry: {child}")
            with contextlib.suppress(OSError):
                os.chmod(child, 0o600)


def _reject_unsafe_file(path: Path, *, policy: CachePolicyConfig) -> None:
    if policy.reject_symlinks and path.is_symlink():
        raise CacheError(f"symlink rejected: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CacheError(f"cannot stat: {path}") from exc
    if policy.reject_special_files and not stat.S_ISREG(mode):
        raise CacheError(f"special file rejected: {path}")
    if policy.reject_hardlinks and is_hardlinked(path):
        raise CacheError(f"hardlink rejected: {path}")


def _artifact_from_dict(data: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        logical_name=str(data["logical_name"]),
        relative_path=str(data["relative_path"]),
        media_type=str(data["media_type"]),
        size_bytes=int(data["size_bytes"]),
        sha256=str(data["sha256"]),
        contract_name=data.get("contract_name"),
        contract_version=data.get("contract_version"),
        schema_fingerprint=data.get("schema_fingerprint"),
        metadata=data.get("metadata") or {},
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def publish_cache_entry(
    *,
    cache_root: Path,
    cache_key: str,
    stage_identity: StageIdentity,
    config_fingerprint: str,
    artifacts: Mapping[str, ArtifactRef],
    artifact_root: Path,
    stage_result: StageResult,
    policy: CachePolicyConfig,
    source_run_id: str,
) -> Path:
    """Atomically publish a verified cache entry (copy, never hardlink/overwrite)."""
    if not policy.enabled:
        raise CacheError("cache publish refused: policy disabled")
    if not _SHA256_RE.fullmatch(cache_key):
        raise CacheError("cache_key must be 64 lowercase hex")
    final = entry_dir(cache_root, cache_key)
    if final.exists():
        # Concurrent winner already published; caller re-verifies.
        return final

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    prefix = cache_key[:8]
    tmp = cache_root / f".tmp_publish_{prefix}_{secrets.token_hex(4)}"
    if tmp.exists():
        raise CacheError("temporary publish directory collision")
    tmp.mkdir(mode=0o700)
    arts_dir = tmp / "artifacts"
    arts_dir.mkdir(mode=0o700)

    try:
        total_bytes = 0
        published: list[ArtifactRef] = []
        for name in sorted(artifacts.keys()):
            ref = artifacts[name]
            if not isinstance(ref, ArtifactRef):
                raise CacheError("artifact values must be ArtifactRef")
            src = Path(artifact_root) / ref.relative_path
            _reject_unsafe_file(src, policy=policy)
            verify_artifact_on_disk(
                ref, root=artifact_root, reject_hardlinks=policy.reject_hardlinks
            )
            dst = arts_dir / ref.relative_path
            copy_artifact_file(src, dst)
            if policy.verify_on_publish:
                verify_artifact_on_disk(
                    ref, root=arts_dir, reject_hardlinks=policy.reject_hardlinks
                )
            total_bytes += ref.size_bytes
            published.append(ref)

        if len(published) > policy.max_entry_files:
            raise CacheError("cache entry exceeds max_entry_files")
        if total_bytes > policy.max_entry_bytes:
            raise CacheError("cache entry exceeds max_entry_bytes")

        manifest = CacheManifest(
            cache_key=cache_key,
            layout_version=policy.layout_version,
            stage_name=stage_identity.name,
            stage_version=stage_identity.version,
            config_fingerprint=config_fingerprint,
            artifacts=tuple(r.to_dict() for r in published),
            created_at_utc=_utc_now(),
            source_run_id=source_run_id,
        )
        manifest_path = write_json_record(
            tmp / "cache_manifest.json",
            manifest.to_dict(),
            contain_root=tmp,
            overwrite=False,
        )
        if manifest_path.stat().st_size > policy.max_manifest_bytes:
            raise CacheError("cache manifest exceeds max_manifest_bytes")
        write_json_record(
            tmp / "stage_result.json",
            stage_result.to_dict(),
            contain_root=tmp,
            overwrite=False,
        )
        _chmod_tree(tmp)

        with acquire_key_lock(cache_root, cache_key, policy.lock_timeout_seconds):
            if final.exists():
                shutil.rmtree(tmp, ignore_errors=True)
                return final
            final.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.rename(str(tmp), str(final))
            with contextlib.suppress(OSError):
                os.chmod(final, 0o700)
        return final
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise CacheError(f"symlink rejected: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise CacheError(f"json too large: {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CacheError(f"json root must be object: {path.name}")
    return raw


def verify_cache_entry(
    cache_root: Path,
    cache_key: str,
    *,
    expected_stage: StageIdentity,
    expected_config_fp: str,
    expected_inputs: Mapping[str, ArtifactRef],
    expected_compatibility_fp: str,
    policy: CachePolicyConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify an existing cache entry; return (manifest_dict, stage_result_dict)."""
    entry = entry_dir(cache_root, cache_key)
    if not entry.is_dir() or entry.is_symlink():
        raise CacheError("cache entry missing or unsafe")

    manifest_path = entry / "cache_manifest.json"
    result_path = entry / "stage_result.json"
    arts_dir = entry / "artifacts"
    if not manifest_path.is_file() or not result_path.is_file() or not arts_dir.is_dir():
        raise CacheError("cache entry incomplete")

    manifest = _load_json_object(manifest_path, max_bytes=policy.max_manifest_bytes)
    result = _load_json_object(result_path, max_bytes=policy.max_manifest_bytes)

    if manifest.get("cache_key") != cache_key:
        raise CacheError("manifest cache_key mismatch")
    if manifest.get("stage_name") != expected_stage.name:
        raise CacheError("manifest stage_name mismatch")
    if int(manifest.get("stage_version", -1)) != expected_stage.version:
        raise CacheError("manifest stage_version mismatch")
    if manifest.get("config_fingerprint") != expected_config_fp:
        raise CacheError("manifest config_fingerprint mismatch")

    recomputed = compute_cache_key(
        stage=expected_stage,
        config_fingerprint=expected_config_fp,
        compatibility_fingerprint=expected_compatibility_fp,
        inputs=expected_inputs,
    )
    if recomputed != cache_key:
        raise CacheError("recomputed cache key mismatch")

    art_dicts = manifest.get("artifacts")
    if not isinstance(art_dicts, list):
        raise CacheError("manifest artifacts must be a list")
    if len(art_dicts) > policy.max_entry_files:
        raise CacheError("cache entry exceeds max_entry_files")

    total = 0
    for item in art_dicts:
        if not isinstance(item, dict):
            raise CacheError("artifact entry must be object")
        ref = _artifact_from_dict(item)
        total += ref.size_bytes
        if policy.verify_on_read:
            try:
                verify_artifact_on_disk(
                    ref, root=arts_dir, reject_hardlinks=policy.reject_hardlinks
                )
            except ArtifactError as exc:
                raise CacheError(str(exc)) from exc
        else:
            path = arts_dir / ref.relative_path
            _reject_unsafe_file(path, policy=policy)

    if total > policy.max_entry_bytes:
        raise CacheError("cache entry exceeds max_entry_bytes")

    # Reject unexpected files under artifacts/
    expected_rels = {str(a["relative_path"]) for a in art_dicts}
    for dirpath, dirnames, filenames in os.walk(arts_dir, followlinks=False):
        base = Path(dirpath)
        for name in dirnames:
            child = base / name
            if child.is_symlink():
                raise CacheError(f"symlink directory rejected: {child}")
        for name in filenames:
            child = base / name
            rel = child.relative_to(arts_dir).as_posix()
            if rel not in expected_rels:
                raise CacheError(f"unexpected cache artifact file: {rel}")
            _reject_unsafe_file(child, policy=policy)

    return manifest, result


def restore_cache_entry(
    cache_root: Path,
    cache_key: str,
    *,
    output_directory: Path,
    policy: CachePolicyConfig,
    expected_stage: StageIdentity,
    expected_config_fp: str,
    expected_inputs: Mapping[str, ArtifactRef],
    expected_compatibility_fp: str,
) -> Mapping[str, ArtifactRef]:
    """Copy verified cache artifacts into ``output_directory`` (no overwrite)."""
    manifest, _result = verify_cache_entry(
        cache_root,
        cache_key,
        expected_stage=expected_stage,
        expected_config_fp=expected_config_fp,
        expected_inputs=expected_inputs,
        expected_compatibility_fp=expected_compatibility_fp,
        policy=policy,
    )
    arts_dir = entry_dir(cache_root, cache_key) / "artifacts"
    out_root = Path(output_directory)
    out_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    restored: dict[str, ArtifactRef] = {}
    try:
        for item in manifest["artifacts"]:
            ref = _artifact_from_dict(item)
            src = arts_dir / ref.relative_path
            dst = out_root / ref.relative_path
            if dst.exists() or dst.is_symlink():
                raise CacheError(f"restore would overwrite existing output: {ref.relative_path}")
            copy_artifact_file(src, dst)
            verify_artifact_on_disk(ref, root=out_root, reject_hardlinks=policy.reject_hardlinks)
            restored[ref.logical_name] = ref
    except Exception:
        # Best-effort partial restore cleanup of files we created.
        for ref in restored.values():
            with contextlib.suppress(OSError):
                (out_root / ref.relative_path).unlink()
        raise
    return restored


def quarantine_cache_entry(
    cache_root: Path,
    cache_key: str,
    *,
    quarantine_root: Path,
    reason: str,
) -> dict[str, Any]:
    """Move a corrupt entry to quarantine; never permanently delete."""
    if not _SHA256_RE.fullmatch(cache_key):
        raise CacheError("cache_key must be 64 lowercase hex")
    entry = entry_dir(cache_root, cache_key)
    if not entry.exists():
        raise CacheError("cache entry missing for quarantine")
    qroot = Path(quarantine_root)
    qroot.mkdir(parents=True, mode=0o700, exist_ok=True)
    dest = qroot / cache_key
    if dest.exists():
        dest = qroot / f"{cache_key}_{secrets.token_hex(4)}"

    manifest_hash: str | None = None
    manifest_path = entry / "cache_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        with contextlib.suppress(Exception):
            manifest_hash = sha256_file(manifest_path)

    with acquire_key_lock(cache_root, cache_key, timeout_seconds=30.0):
        if not entry.exists():
            raise CacheError("cache entry disappeared before quarantine")
        dest.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.rename(str(entry), str(dest))

    base_receipt = {
        "schema_version": 1,
        "original_cache_key": cache_key,
        "original_path": str(entry),
        "quarantine_path": str(dest),
        "reason": redact_text(str(reason)),
        "detected_at_utc": _utc_now(),
        "manifest_hash": manifest_hash,
        "permanent_delete_performed": False,
    }
    receipt = {
        **base_receipt,
        "receipt_fingerprint": hash_canonical_json(base_receipt),
    }
    write_json_record(
        dest / "quarantine_receipt.json",
        receipt,
        contain_root=dest,
        overwrite=False,
    )
    return receipt
