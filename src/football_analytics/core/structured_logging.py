"""Secret-safe structured logging (stdlib) — Stage 2B.

Module name avoids shadowing the standard library ``logging`` package.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from football_analytics.core.redaction import redact_text, redact_value

LOGGER_NAME = "football_analytics"
LOG_SCHEMA_VERSION = 1
_VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LoggingError(ValueError):
    """Logger setup failure."""


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "schema_version": LOG_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.funcName or "log"),
            "message": redact_text(record.getMessage()),
            "run_id": getattr(record, "run_id", None),
            "stage": getattr(record, "stage", None),
            "context": redact_value(getattr(record, "context", {}) or {}),
            "exception": None,
        }
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        try:
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise LoggingError(f"log JSON encode failed: {exc}") from exc
        return line.replace("\n", "\\n")


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = redact_text(super().format(record))
        return msg.replace("\n", "\\n").replace("\r", "\\r")


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    key = str(level).upper()
    if key not in _VALID_LEVELS:
        raise LoggingError(f"invalid log level: {level!r}")
    return _VALID_LEVELS[key]


def _ensure_log_parent(path: Path, contain_root: Path | None) -> None:
    parent = path.parent
    if contain_root is not None:
        if not path.is_absolute():
            raise LoggingError("jsonl_path must be absolute when contain_root is set")
        contain = contain_root.resolve()
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            parent.resolve().relative_to(contain)
            if path.exists():
                path.resolve().relative_to(contain)
        except ValueError as exc:
            raise LoggingError("log path escapes containment") from exc
    else:
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)

    if parent.is_symlink():
        raise LoggingError("log parent must not be a symlink")


def configure_logger(
    *,
    level: str = "INFO",
    console: bool = True,
    jsonl_path: Path | None = None,
    contain_root: Path | None = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 3,
    run_id: str | None = None,
) -> logging.Logger:
    """Idempotent setup for the package logger; does not mutate the root logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_parse_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(_parse_level(level))
        ch.setFormatter(HumanFormatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(ch)

    if jsonl_path is not None:
        path = Path(jsonl_path)
        if path.exists() and path.is_symlink():
            raise LoggingError("jsonl path must not be a symlink")
        _ensure_log_parent(path, contain_root)
        if path.exists() and not path.is_file():
            raise LoggingError("jsonl path exists and is not a regular file")
        if path.exists():
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise LoggingError("jsonl path is not a regular file")

        fh = RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(_parse_level(level))
        fh.setFormatter(JsonlFormatter())
        logger.addHandler(fh)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)

    logger.run_id = run_id  # type: ignore[attr-defined]
    return logger


def log_event(
    logger: logging.Logger,
    level: str,
    message: str,
    *,
    event: str = "event",
    run_id: str | None = None,
    stage: str | None = None,
    context: dict[str, Any] | None = None,
    exc_info: bool = False,
) -> None:
    lvl = _parse_level(level)
    rid = run_id if run_id is not None else getattr(logger, "run_id", None)
    logger.log(
        lvl,
        message,
        extra={
            "event": event,
            "run_id": rid,
            "stage": stage,
            "context": context or {},
        },
        exc_info=exc_info,
    )
