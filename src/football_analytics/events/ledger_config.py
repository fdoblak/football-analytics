"""13C config loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.registry import default_project_root
from football_analytics.events.types import EventsContractError

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024


class LedgerConfigError(EventsContractError):
    """Config failure."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_deep_unfreeze(v) for v in value]
    return value


def default_ledger_config_path(*, project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "configs" / "events" / "ledger_baseline.yaml"


def load_ledger_config(
    path: Path | None = None, *, project_root: Path | None = None
) -> Mapping[str, Any]:
    p = path or default_ledger_config_path(project_root=project_root)
    if p.is_symlink():
        raise LedgerConfigError(f"symlink rejected: {p}")
    raw = p.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise LedgerConfigError(f"config too large: {p}")
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise LedgerConfigError("config root must be mapping")
    if int(data.get("schema_version", 0)) != CONFIG_VERSION:
        raise LedgerConfigError("schema_version mismatch")

    if data.get("append_only") is not True:
        raise LedgerConfigError("append_only must be true")
    if data.get("no_destructive_merge") is not True:
        raise LedgerConfigError("no_destructive_merge must be true")
    if data.get("evaluation_leakage_guard") is not True:
        raise LedgerConfigError("evaluation_leakage_guard must be true")

    return _deep_freeze(data)


def ledger_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hash_canonical_json(_deep_unfreeze(config))


__all__ = [
    "LedgerConfigError",
    "default_ledger_config_path",
    "load_ledger_config",
    "ledger_config_fingerprint",
]
