"""Canonical Run ID: generate / validate / parse (Stage 2B)."""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# run_YYYYMMDDTHHMMSSffffffZ_<12_hex>
RUN_ID_PATTERN = re.compile(r"^run_([0-9]{8}T[0-9]{12}Z)_([a-f0-9]{12})$")
MAX_RUN_ID_LENGTH = 48
SHELL_SPECIAL = frozenset("$`;|&><*?[]{}'\"")


class RunIdError(ValueError):
    """Invalid or unsafe run identifier."""


@dataclass(frozen=True)
class ParsedRunId:
    value: str
    timestamp_token: str
    suffix: str
    created_at_utc: datetime


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_suffix() -> str:
    return secrets.token_hex(6)


def generate_run_id(
    *,
    now: Callable[[], datetime] | None = None,
    suffix_factory: Callable[[], str] | None = None,
) -> str:
    """Generate ``run_<UTC_compact>_<12_hex>`` using cryptographic randomness."""
    clock = now or _default_now
    make_suffix = suffix_factory or _default_suffix
    instant = clock()
    if instant.tzinfo is None:
        raise RunIdError("clock must be timezone-aware UTC")
    instant = instant.astimezone(timezone.utc)
    ts = instant.strftime("%Y%m%dT%H%M%S%f") + "Z"
    suffix = make_suffix()
    if not re.fullmatch(r"[a-f0-9]{12}", suffix):
        raise RunIdError("suffix_factory must return 12 lowercase hex chars")
    value = f"run_{ts}_{suffix}"
    validate_run_id(value)
    return value


def validate_run_id(value: Any) -> str:
    """Validate canonical Stage 2B run ID. Does not accept arbitrary paths."""
    if not isinstance(value, str):
        raise RunIdError("run_id must be a string")
    if not value:
        raise RunIdError("run_id empty")
    if len(value) > MAX_RUN_ID_LENGTH:
        raise RunIdError("run_id exceeds maximum length")
    if "/" in value or "\\" in value or ".." in value or " " in value or "\n" in value:
        raise RunIdError("run_id contains unsafe path or whitespace characters")
    if any(ch in SHELL_SPECIAL for ch in value):
        raise RunIdError("run_id contains shell-special character")
    if not RUN_ID_PATTERN.fullmatch(value):
        raise RunIdError("run_id does not match canonical Stage 2B pattern")
    return value


def parse_run_id(value: str) -> ParsedRunId:
    """Parse a validated run ID into an immutable dataclass."""
    validated = validate_run_id(value)
    match = RUN_ID_PATTERN.fullmatch(validated)
    assert match is not None
    ts_token, suffix = match.group(1), match.group(2)
    created = datetime.strptime(ts_token, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    return ParsedRunId(
        value=validated,
        timestamp_token=ts_token,
        suffix=suffix,
        created_at_utc=created,
    )
