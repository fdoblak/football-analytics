"""Environment / provenance record without heavy imports (Stage 2B)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from football_analytics.core.redaction import sanitize_remote_url

GPU_CLASSIFICATION = "AGENT_CONTEXT_GPU_UNVERIFIABLE"

PACKAGE_ALLOWLIST = (
    "football-analytics",
    "PyYAML",
    "pyarrow",
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "opencv-python",
    "opencv-python-headless",
    "ultralytics",
    "SoccerNet",
)

GIT_TIMEOUT_SEC = 5.0


class EnvironmentError(ValueError):
    """Environment record failure."""


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _wsl_detected() -> bool:
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SEC,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def collect_git_metadata(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None or not (repo_root / ".git").exists():
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "remote_sanitized": None,
        }
    commit = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    branch = _run_git(["branch", "--show-current"], cwd=repo_root)
    porcelain = _run_git(["status", "--porcelain"], cwd=repo_root)
    dirty = None if porcelain is None else bool(porcelain.strip())
    remote = _run_git(["remote", "get-url", "origin"], cwd=repo_root)
    remote_sanitized = sanitize_remote_url(remote) if remote else None
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "remote_sanitized": remote_sanitized,
    }


def _conda_info() -> dict[str, Any]:
    name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    sanitized = None
    if prefix:
        # Keep path but do not include unrelated secrets; prefix is a filesystem path.
        sanitized = str(Path(prefix))
    return {"environment_name": name, "prefix_sanitized": sanitized}


def build_environment_record(
    *,
    project_version: str,
    config_fingerprint: Mapping[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build schema_version=1 environment record. Does not import torch/GPU."""
    packages = {name: _package_version(name) for name in PACKAGE_ALLOWLIST}
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "project_version": project_version,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "wsl_detected": _wsl_detected(),
        },
        "conda": _conda_info(),
        "packages": packages,
        "git": collect_git_metadata(repo_root),
        "gpu_validation": {
            "classification": GPU_CLASSIFICATION,
            "torch_imported": False,
            "cuda_initialized": False,
        },
        "config_fingerprint": dict(config_fingerprint),
    }
