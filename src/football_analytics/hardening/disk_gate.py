"""Disk-space gates for pipeline / cache / temp operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


class DiskSpaceError(ValueError):
    """Insufficient free disk space for a gated operation."""


def free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem containing path (creates parents if needed)."""
    target = Path(path)
    probe = target if target.exists() else target.parent
    if not probe.exists():
        probe.mkdir(parents=True, mode=0o700, exist_ok=True)
    usage = shutil.disk_usage(str(probe if probe.is_dir() else probe.parent))
    return int(usage.free)


def assert_disk_space(
    path: Path,
    *,
    minimum_free_bytes: int,
    context: str = "operation",
) -> dict[str, Any]:
    """Raise DiskSpaceError when free space is below the gate."""
    free = free_bytes(path)
    ok = free >= int(minimum_free_bytes)
    report = {
        "context": context,
        "path": str(path),
        "free_bytes": free,
        "minimum_free_bytes": int(minimum_free_bytes),
        "ok": ok,
    }
    if not ok:
        raise DiskSpaceError(
            f"{context}: free_bytes={free} < minimum_free_bytes={minimum_free_bytes}"
        )
    return report


def gate_pipeline_disk(
    path: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    pol = policy or load_hardening_policy()
    return assert_disk_space(
        path,
        minimum_free_bytes=pol.min_free_pipeline_bytes,
        context="pipeline",
    )


def gate_cache_disk(
    path: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    pol = policy or load_hardening_policy()
    return assert_disk_space(
        path,
        minimum_free_bytes=int(pol.raw["disk"]["minimum_free_bytes_cache"]),
        context="cache",
    )


def disk_gate_summary(path: Path, *, policy: HardeningPolicy | None = None) -> dict[str, Any]:
    pol = policy or load_hardening_policy()
    free = free_bytes(path)
    warn = int(pol.raw["disk"]["warning_free_bytes"])
    return {
        "path": str(path),
        "free_bytes": free,
        "minimum_free_bytes_pipeline": pol.min_free_pipeline_bytes,
        "minimum_free_bytes_cache": int(pol.raw["disk"]["minimum_free_bytes_cache"]),
        "warning_free_bytes": warn,
        "below_warning": free < warn,
        "pipeline_ok": free >= pol.min_free_pipeline_bytes,
        "cache_ok": free >= int(pol.raw["disk"]["minimum_free_bytes_cache"]),
    }
