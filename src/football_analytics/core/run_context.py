"""Run context initialization (Stage 2B foundation).

Distinct from Stage 1 archive ``run_manifest``:
- run_context: identity + provenance at initialization
- run_manifest: artifact/archive inventory
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics import __version__
from football_analytics.core.config import (
    config_fingerprint,
    default_defaults_path,
    load_resolved_config,
    resolved_config_as_dict,
)
from football_analytics.core.environment import build_environment_record
from football_analytics.core.records import RecordError, write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.core.structured_logging import configure_logger, log_event


class RunContextError(ValueError):
    """Run context initialization failure."""


@dataclass(frozen=True)
class InitializedRun:
    run_id: str
    run_dir: Path
    resolved_config_path: Path
    environment_path: Path
    run_context_path: Path
    log_path: Path
    config_fingerprint: dict[str, Any]
    run_context: dict[str, Any]


def _safe_cleanup(run_dir: Path) -> None:
    if run_dir.exists() and run_dir.is_dir() and not run_dir.is_symlink():
        with contextlib.suppress(OSError):
            shutil.rmtree(run_dir)


def initialize_run_context(
    *,
    runs_root: Path,
    run_id: str | None = None,
    defaults_path: Path | None = None,
    user_config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
    console_logging: bool = False,
) -> InitializedRun:
    """Create a transactional foundation run directory under ``runs_root``.

    Does not open video, search datasets/models, start GPU, or import external repos.
    """
    root = Path(runs_root)
    if root.exists() and root.is_symlink():
        raise RunContextError("runs_root must not be a symlink")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise RunContextError("runs_root is not a safe directory")

    rid = validate_run_id(run_id) if run_id is not None else generate_run_id()
    run_dir = root / rid
    if run_dir.exists():
        raise RunContextError(f"run directory already exists: {rid}")

    defaults = defaults_path or default_defaults_path()
    config = load_resolved_config(
        defaults_path=defaults,
        user_config_path=user_config_path,
        environ=environ,
        overrides=overrides,
    )
    fp = config_fingerprint(config)
    env_rec = build_environment_record(
        project_version=__version__,
        config_fingerprint=fp,
        repo_root=repo_root,
    )

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    resolved_name = "resolved_config.json"
    env_name = "environment.json"
    ctx_name = "run_context.json"
    log_rel = "logs/events.jsonl"

    run_context: dict[str, Any] = {
        "schema_version": 1,
        "run_id": rid,
        "created_at_utc": created_at,
        "project_version": __version__,
        "git_commit": env_rec["git"].get("commit"),
        "config_fingerprint": fp,
        "environment_record_path": env_name,
        "resolved_config_path": resolved_name,
        "log_path": log_rel,
        "status": "initialized",
    }

    try:
        run_dir.mkdir(mode=0o700, exist_ok=False)
        (run_dir / "logs").mkdir(mode=0o700, exist_ok=False)

        resolved_path = write_json_record(
            run_dir / resolved_name,
            resolved_config_as_dict(config),
            contain_root=root,
            overwrite=False,
        )
        environment_path = write_json_record(
            run_dir / env_name,
            env_rec,
            contain_root=root,
            overwrite=False,
        )
        log_path = run_dir / log_rel
        logger = configure_logger(
            level=str(config["logging"]["level"]),
            console=console_logging,
            jsonl_path=log_path,
            contain_root=root,
            max_bytes=int(config["logging"]["max_bytes"]),
            backup_count=int(config["logging"]["backup_count"]),
            run_id=rid,
        )
        log_event(
            logger,
            "INFO",
            "run context initialized",
            event="run_context_initialized",
            run_id=rid,
            stage="foundation",
            context={"status": "initialized"},
        )
        run_context_path = write_json_record(
            run_dir / ctx_name,
            run_context,
            contain_root=root,
            overwrite=False,
        )
    except Exception as exc:
        _safe_cleanup(run_dir)
        if isinstance(exc, (RunContextError, RecordError)):
            raise
        raise RunContextError(f"initialization failed: {type(exc).__name__}") from exc

    return InitializedRun(
        run_id=rid,
        run_dir=run_dir,
        resolved_config_path=resolved_path,
        environment_path=environment_path,
        run_context_path=run_context_path,
        log_path=log_path,
        config_fingerprint=fp,
        run_context=run_context,
    )
