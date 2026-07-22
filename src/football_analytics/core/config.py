"""Safe YAML config load, merge, fingerprint (Stage 2B)."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.core.hashing import CANONICALIZATION_VERSION, hash_canonical_json
from football_analytics.core.redaction import REDACTED, is_sensitive_key, redact_value

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024
ENV_PREFIX = "FOOTBALL_ANALYTICS_"

ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "project",
        "runtime",
        "logging",
        "hashing",
        "environment_record",
        "storage_config",
    }
)

# Flat allowlist: env var name -> (path tuple, caster)
ENV_OVERRIDE_ALLOWLIST: dict[str, tuple[tuple[str, ...], type]] = {
    "FOOTBALL_ANALYTICS_LOG_LEVEL": (("logging", "level"), str),
    "FOOTBALL_ANALYTICS_LOG_FORMAT": (("logging", "format"), str),
}

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
VALID_LOG_FORMATS = frozenset({"human", "json"})


class ConfigError(ValueError):
    """Config load / merge / validation failure (messages must not leak secrets)."""


def _reject_non_json_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ConfigError(f"non-finite float forbidden at {path}")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _reject_non_json_numbers(v, path=f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_non_json_numbers(v, path=f"{path}[{i}]")


def _reject_secrets_in_mapping(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if is_sensitive_key(k):
                raise ConfigError(f"secret-bearing key forbidden in config at {path}.{k}")
            _reject_secrets_in_mapping(v, path=f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_secrets_in_mapping(v, path=f"{path}[{i}]")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive deterministic merge; overlay wins. Type confusion is rejected."""
    out: dict[str, Any] = dict(base)
    for key in sorted(overlay.keys(), key=lambda x: str(x)):
        ov = overlay[key]
        if key not in out:
            if isinstance(ov, Mapping):
                out[key] = deep_merge({}, ov)
            elif isinstance(ov, list):
                out[key] = list(ov)
            else:
                out[key] = ov
            continue
        bv = out[key]
        if isinstance(bv, Mapping) and isinstance(ov, Mapping):
            out[key] = deep_merge(bv, ov)
        elif type(bv) is type(ov) or (
            isinstance(bv, (int, float))
            and isinstance(ov, (int, float))
            and not isinstance(bv, bool)
            and not isinstance(ov, bool)
        ):
            if isinstance(ov, list):
                out[key] = list(ov)
            elif isinstance(ov, Mapping):
                out[key] = deep_merge({}, ov)
            else:
                out[key] = ov
        else:
            raise ConfigError(f"type mismatch merging key {key!r}")
    return out


def load_yaml_mapping(path: Path, *, max_bytes: int = MAX_CONFIG_BYTES) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ConfigError("config path must be a regular non-symlink file")
    size = target.stat().st_size
    if size > max_bytes:
        raise ConfigError("config exceeds maximum byte size")
    raw = target.read_bytes()
    if len(raw) > max_bytes:
        raise ConfigError("config exceeds maximum byte size")
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"YAML parse failed: {type(exc).__name__}") from exc
    if data is None:
        raise ConfigError("config root must be a mapping")
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    _reject_non_json_numbers(data)
    _reject_secrets_in_mapping(data)
    return data


def _validate_resolved(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError("schema_version must be 1")
    unknown = set(data.keys()) - ALLOWED_TOP_LEVEL
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    for key in ALLOWED_TOP_LEVEL:
        if key not in data:
            raise ConfigError(f"missing required top-level key: {key}")
        if key != "schema_version" and not isinstance(data[key], Mapping):
            raise ConfigError(f"{key} must be a mapping")

    logging_cfg = data["logging"]
    level = logging_cfg.get("level")
    if level not in VALID_LOG_LEVELS:
        raise ConfigError("logging.level invalid")
    fmt = logging_cfg.get("format")
    if fmt not in VALID_LOG_FORMATS:
        raise ConfigError("logging.format invalid")
    if data["hashing"].get("algorithm") != "sha256":
        raise ConfigError("hashing.algorithm must be sha256")
    gpu = data["environment_record"].get("gpu_classification")
    if gpu != "AGENT_CONTEXT_GPU_UNVERIFIABLE":
        raise ConfigError("environment_record.gpu_classification invalid")
    _reject_non_json_numbers(data)
    _reject_secrets_in_mapping(data)


def apply_env_overrides(
    config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    out = deep_merge({}, config)
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        if name not in ENV_OVERRIDE_ALLOWLIST:
            raise ConfigError(f"forbidden environment override: {name}")
        path, caster = ENV_OVERRIDE_ALLOWLIST[name]
        try:
            casted = caster(value)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"invalid environment override type for {name}") from exc
        if name.endswith("LOG_LEVEL") and casted not in VALID_LOG_LEVELS:
            raise ConfigError("invalid FOOTBALL_ANALYTICS_LOG_LEVEL")
        if name.endswith("LOG_FORMAT") and casted not in VALID_LOG_FORMATS:
            raise ConfigError("invalid FOOTBALL_ANALYTICS_LOG_FORMAT")
        cursor: dict[str, Any] = out
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                raise ConfigError(f"env override path broken for {name}")
            cursor = nxt
        cursor[path[-1]] = casted
    return out


def load_resolved_config(
    *,
    defaults_path: Path,
    user_config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> MappingProxyType:
    """Load and merge config layers; return immutable mapping proxy."""
    base = load_yaml_mapping(defaults_path)
    merged = deep_merge({}, base)
    if user_config_path is not None:
        user = load_yaml_mapping(user_config_path)
        unknown = set(user.keys()) - ALLOWED_TOP_LEVEL
        if unknown:
            raise ConfigError(f"unknown top-level keys in user config: {sorted(unknown)}")
        merged = deep_merge(merged, user)
    merged = apply_env_overrides(merged, environ=environ)
    if overrides:
        if any(is_sensitive_key(k) for k in overrides):
            raise ConfigError("secret-bearing override key forbidden")
        unknown = set(overrides.keys()) - ALLOWED_TOP_LEVEL
        if unknown:
            raise ConfigError(f"unknown top-level keys in overrides: {sorted(unknown)}")
        merged = deep_merge(merged, overrides)
    _validate_resolved(merged)
    frozen = _deep_freeze(merged)
    assert isinstance(frozen, MappingProxyType)
    return frozen


def resolved_config_as_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize a plain dict copy suitable for JSON (secrets already rejected)."""

    def _plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: _plain(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [_plain(v) for v in value]
        if isinstance(value, list):
            return [_plain(v) for v in value]
        return value

    return _plain(config)


def config_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    plain = resolved_config_as_dict(config)
    # Belt-and-suspenders: never fingerprint secret-shaped payloads.
    safe = redact_value(plain)
    if REDACTED in str(safe) and any(is_sensitive_key(k) for k in _walk_keys(plain)):
        raise ConfigError("secret values must not appear in resolved config")
    digest = hash_canonical_json(plain)
    return {
        "algorithm": "sha256",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "digest": digest,
    }


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for k, v in value.items():
            keys.append(str(k))
            keys.extend(_walk_keys(v))
    elif isinstance(value, list):
        for v in value:
            keys.extend(_walk_keys(v))
    return keys


def default_defaults_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "project" / "defaults.yaml"
