"""Immutable pipeline request/result/artifact types (Stage 2D)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from football_analytics.core.redaction import is_sensitive_key, redact_value
from football_analytics.core.run_id import RunIdError, validate_run_id
from football_analytics.pipeline.exceptions import ArtifactError, PipelineError, StageError

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
STAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{1,63}")
LOGICAL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

STAGE_STATUSES = frozenset({"succeeded", "failed", "cache_hit", "skipped", "cancelled"})
StageStatus = Literal["succeeded", "failed", "cache_hit", "skipped", "cancelled"]

CACHE_MANIFEST_SCHEMA_VERSION = 1
STAGE_RESULT_SCHEMA_VERSION = 1


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PipelineError(f"{label} must be 64 lowercase hex chars")
    return value


def _reject_secrets_in_mapping(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_sensitive_key(key):
                raise PipelineError(f"secret-bearing key forbidden at {path}.{key}")
            _reject_secrets_in_mapping(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_secrets_in_mapping(item, path=f"{path}[{idx}]")


def _assert_json_compatible(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PipelineError(f"non-finite float forbidden at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineError(f"non-string map key at {path}")
            _assert_json_compatible(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_json_compatible(item, path=f"{path}[{idx}]")
        return
    raise PipelineError(f"non-JSON-compatible value at {path}: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PipelineError("mapping keys must be strings")
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _to_plain(value: Any) -> Any:
    """Convert MappingProxyType / nested mappings to plain JSON-ready structures."""
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def validate_safe_relative_path(rel: str) -> str:
    """Validate a POSIX relative path (no absolute, ``..``, or backslash)."""
    if not isinstance(rel, str) or not rel:
        raise ArtifactError("relative_path empty")
    if rel.startswith("/") or rel.startswith("\\"):
        raise ArtifactError(f"absolute relative_path forbidden: {rel}")
    if "\\" in rel:
        raise ArtifactError(f"backslash in relative_path forbidden: {rel}")
    if rel in {".", ".."}:
        raise ArtifactError(f"unsafe relative_path: {rel}")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts:
        raise ArtifactError(f"unsafe relative_path: {rel}")
    if ".." in parts:
        raise ArtifactError(f"unsafe relative_path: {rel}")
    if any(ord(ch) < 32 for ch in rel):
        raise ArtifactError("relative_path contains control characters")
    return "/".join(parts)


@dataclass(frozen=True)
class ContractRef:
    """Named contract version reference."""

    name: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise StageError("contract name empty")
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise StageError("contract name contains path-like content")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise StageError("contract version must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


def _normalize_contract_refs(
    refs: tuple[ContractRef | Mapping[str, Any], ...] | list[ContractRef | Mapping[str, Any]],
) -> tuple[ContractRef, ...]:
    out: list[ContractRef] = []
    for item in refs:
        if isinstance(item, ContractRef):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(ContractRef(name=str(item["name"]), version=int(item["version"])))
        else:
            raise StageError("contract ref must be ContractRef or mapping")
    return tuple(out)


@dataclass(frozen=True)
class StageIdentity:
    """Immutable stage identity included in cache keys."""

    name: str
    version: int
    code_fingerprint: str
    input_contracts: tuple[ContractRef, ...]
    output_contracts: tuple[ContractRef, ...]
    deterministic: bool
    cacheable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not STAGE_NAME_RE.fullmatch(self.name):
            raise StageError("stage name fails safe identifier policy")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise StageError("stage version must be a positive int")
        object.__setattr__(
            self,
            "code_fingerprint",
            _require_sha256(self.code_fingerprint, label="code_fingerprint"),
        )
        object.__setattr__(self, "input_contracts", _normalize_contract_refs(self.input_contracts))
        object.__setattr__(
            self, "output_contracts", _normalize_contract_refs(self.output_contracts)
        )
        if not isinstance(self.deterministic, bool) or not isinstance(self.cacheable, bool):
            raise StageError("deterministic/cacheable must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "code_fingerprint": self.code_fingerprint,
            "input_contracts": [c.to_dict() for c in self.input_contracts],
            "output_contracts": [c.to_dict() for c in self.output_contracts],
            "deterministic": self.deterministic,
            "cacheable": self.cacheable,
        }


@dataclass(frozen=True)
class CachePolicyRequest:
    """Request-level cache enable flag (not system policy)."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise StageError("cache_policy.enabled must be bool")


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed artifact reference relative to a controlled root."""

    logical_name: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    contract_name: str | None = None
    contract_version: int | None = None
    schema_fingerprint: str | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.logical_name, str) or not LOGICAL_NAME_RE.fullmatch(
            self.logical_name
        ):
            raise ArtifactError("logical_name fails safe identifier policy")
        object.__setattr__(self, "relative_path", validate_safe_relative_path(self.relative_path))
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ArtifactError("media_type empty")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ArtifactError("size_bytes must be a non-negative int")
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, label="sha256"))

        needs_contract = (
            "parquet" in self.media_type.lower() or self.relative_path.lower().endswith(".parquet")
        )
        if needs_contract and (
            not self.contract_name or self.contract_version is None or not self.schema_fingerprint
        ):
            raise ArtifactError(
                "parquet artifacts require contract_name, contract_version, schema_fingerprint"
            )
        if self.contract_name is not None and (
            not isinstance(self.contract_name, str) or not self.contract_name
        ):
            raise ArtifactError("contract_name empty")
        if self.contract_version is not None and (
            not isinstance(self.contract_version, int)
            or isinstance(self.contract_version, bool)
            or self.contract_version < 0
        ):
            raise ArtifactError("contract_version must be a non-negative int")
        if self.schema_fingerprint is not None:
            object.__setattr__(
                self,
                "schema_fingerprint",
                _require_sha256(self.schema_fingerprint, label="schema_fingerprint"),
            )

        meta = dict(self.metadata) if self.metadata is not None else {}
        _reject_secrets_in_mapping(meta, path="metadata")
        _assert_json_compatible(meta, path="metadata")
        object.__setattr__(self, "metadata", _freeze_mapping(meta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "schema_fingerprint": self.schema_fingerprint,
            "metadata": _to_plain(self.metadata),
        }


def _freeze_artifact_inputs(
    inputs: Mapping[str, ArtifactRef],
) -> Mapping[str, ArtifactRef]:
    if not isinstance(inputs, Mapping):
        raise StageError("inputs must be a mapping")
    frozen: dict[str, ArtifactRef] = {}
    for key, ref in inputs.items():
        if not isinstance(key, str) or not key:
            raise StageError("input keys must be non-empty strings")
        if not isinstance(ref, ArtifactRef):
            raise StageError("input values must be ArtifactRef")
        if ref.logical_name != key:
            raise StageError("input key must match ArtifactRef.logical_name")
        frozen[key] = ref
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class StageRequest:
    """Immutable single-stage execution request."""

    run_id: str
    stage_identity: StageIdentity
    config_fingerprint: str
    compatibility_fingerprint: str
    inputs: Mapping[str, ArtifactRef]
    working_directory: Path
    output_directory: Path
    requested_at_utc: str
    cache_policy_enabled: bool = True

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        except RunIdError as exc:
            raise StageError(str(exc)) from exc
        if not isinstance(self.stage_identity, StageIdentity):
            raise StageError("stage_identity must be StageIdentity")
        object.__setattr__(
            self,
            "config_fingerprint",
            _require_sha256(self.config_fingerprint, label="config_fingerprint"),
        )
        object.__setattr__(
            self,
            "compatibility_fingerprint",
            _require_sha256(self.compatibility_fingerprint, label="compatibility_fingerprint"),
        )
        object.__setattr__(self, "inputs", _freeze_artifact_inputs(self.inputs))
        work = Path(self.working_directory)
        out = Path(self.output_directory)
        object.__setattr__(self, "working_directory", work)
        object.__setattr__(self, "output_directory", out)
        if work.resolve() == out.resolve():
            raise StageError("working_directory and output_directory must differ")
        if not isinstance(self.requested_at_utc, str) or not self.requested_at_utc:
            raise StageError("requested_at_utc empty")
        if not isinstance(self.cache_policy_enabled, bool):
            raise StageError("cache_policy_enabled must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Secret-safe request dictionary (paths as strings; no absolute expansion required)."""
        payload = {
            "run_id": self.run_id,
            "stage_identity": self.stage_identity.to_dict(),
            "config_fingerprint": self.config_fingerprint,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "inputs": {k: v.to_dict() for k, v in sorted(self.inputs.items())},
            "working_directory": str(self.working_directory),
            "output_directory": str(self.output_directory),
            "requested_at_utc": self.requested_at_utc,
            "cache_policy_enabled": self.cache_policy_enabled,
        }
        redacted = redact_value(_to_plain(payload))
        if not isinstance(redacted, dict):
            raise StageError("request redaction failed")
        return redacted


def _validate_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metrics, Mapping):
        raise StageError("metrics must be a mapping")
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise StageError("metric keys must be non-empty strings")
        if is_sensitive_key(key):
            raise StageError(f"secret-bearing metric key forbidden: {key}")
        if isinstance(value, (bool, str, int)):
            out[key] = value
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise StageError(f"non-finite metric forbidden: {key}")
            out[key] = value
        else:
            raise StageError(f"metric {key} must be finite float/int/str/bool")
    return MappingProxyType(out)


def _validate_error(error: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if error is None:
        return None
    if not isinstance(error, Mapping):
        raise StageError("error must be a mapping or None")
    allowed = {"class", "message"}
    if set(error.keys()) != allowed:
        raise StageError("error must have exactly keys class/message")
    cls = error["class"]
    msg = error["message"]
    if not isinstance(cls, str) or not cls:
        raise StageError("error.class must be a non-empty string")
    if not isinstance(msg, str):
        raise StageError("error.message must be a string")
    if "traceback" in msg.lower():
        raise StageError("error.message must not contain traceback")
    safe = {"class": cls, "message": str(redact_value(msg))}
    return MappingProxyType(safe)


@dataclass(frozen=True)
class StageResult:
    """Immutable stage execution result."""

    run_id: str
    stage_name: str
    stage_version: int
    status: StageStatus
    cache_key: str
    cache_hit: bool
    started_at_utc: str
    finished_at_utc: str
    duration_ms: int
    inputs: Mapping[str, Mapping[str, Any]]
    outputs: Mapping[str, Mapping[str, Any]]
    metrics: Mapping[str, Any]
    warnings: tuple[str, ...]
    error: Mapping[str, Any] | None
    execution_fingerprint: str
    schema_version: int = STAGE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_RESULT_SCHEMA_VERSION:
            raise StageError("unsupported stage_result schema_version")
        try:
            object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        except RunIdError as exc:
            raise StageError(str(exc)) from exc
        if not isinstance(self.stage_name, str) or not STAGE_NAME_RE.fullmatch(self.stage_name):
            raise StageError("stage_name fails safe identifier policy")
        if (
            not isinstance(self.stage_version, int)
            or isinstance(self.stage_version, bool)
            or self.stage_version < 1
        ):
            raise StageError("stage_version must be a positive int")
        if self.status not in STAGE_STATUSES:
            raise StageError(f"invalid stage status: {self.status}")
        object.__setattr__(self, "cache_key", _require_sha256(self.cache_key, label="cache_key"))
        if not isinstance(self.cache_hit, bool):
            raise StageError("cache_hit must be bool")
        if self.status == "cache_hit" and not self.cache_hit:
            raise StageError("cache_hit status requires cache_hit=True")
        if not isinstance(self.started_at_utc, str) or not self.started_at_utc:
            raise StageError("started_at_utc empty")
        if not isinstance(self.finished_at_utc, str) or not self.finished_at_utc:
            raise StageError("finished_at_utc empty")
        if self.finished_at_utc < self.started_at_utc:
            raise StageError("finished_at_utc must be >= started_at_utc")
        if (
            not isinstance(self.duration_ms, int)
            or isinstance(self.duration_ms, bool)
            or self.duration_ms < 0
        ):
            raise StageError("duration_ms must be a non-negative int")
        object.__setattr__(self, "inputs", _freeze_mapping(dict(self.inputs)))
        object.__setattr__(self, "outputs", _freeze_mapping(dict(self.outputs)))
        object.__setattr__(self, "metrics", _validate_metrics(self.metrics))
        warnings = tuple(str(w) for w in self.warnings)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "error", _validate_error(self.error))
        object.__setattr__(
            self,
            "execution_fingerprint",
            _require_sha256(self.execution_fingerprint, label="execution_fingerprint"),
        )
        if self.status == "failed" and self.error is None:
            raise StageError("failed status requires error")
        if self.status in {"succeeded", "cache_hit"} and self.error is not None:
            raise StageError("success/cache_hit must not include error")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "stage_version": self.stage_version,
            "status": self.status,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_ms": self.duration_ms,
            "inputs": _to_plain(self.inputs),
            "outputs": _to_plain(self.outputs),
            "metrics": _to_plain(self.metrics),
            "warnings": list(self.warnings),
            "error": _to_plain(self.error) if self.error is not None else None,
            "execution_fingerprint": self.execution_fingerprint,
        }
        redacted = redact_value(payload)
        if not isinstance(redacted, dict):
            raise StageError("result redaction failed")
        return redacted


@dataclass(frozen=True)
class StageExecutionOutput:
    """Return value from Stage.execute (pre-result packaging)."""

    outputs: Mapping[str, ArtifactRef]
    metrics: Mapping[str, Any] = MappingProxyType({})
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", _freeze_artifact_inputs(self.outputs))
        object.__setattr__(self, "metrics", _validate_metrics(self.metrics))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))


@dataclass(frozen=True)
class CacheManifest:
    """Cache entry integrity manifest."""

    cache_key: str
    layout_version: int
    stage_name: str
    stage_version: int
    config_fingerprint: str
    artifacts: tuple[dict[str, Any], ...]
    created_at_utc: str
    source_run_id: str
    schema_version: int = CACHE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_MANIFEST_SCHEMA_VERSION:
            raise PipelineError("unsupported cache_manifest schema_version")
        object.__setattr__(self, "cache_key", _require_sha256(self.cache_key, label="cache_key"))
        if not isinstance(self.layout_version, int) or self.layout_version < 1:
            raise PipelineError("layout_version must be a positive int")
        if not isinstance(self.stage_name, str) or not STAGE_NAME_RE.fullmatch(self.stage_name):
            raise PipelineError("stage_name fails safe identifier policy")
        if not isinstance(self.stage_version, int) or self.stage_version < 1:
            raise PipelineError("stage_version must be a positive int")
        object.__setattr__(
            self,
            "config_fingerprint",
            _require_sha256(self.config_fingerprint, label="config_fingerprint"),
        )
        arts = tuple(dict(a) for a in self.artifacts)
        for art in arts:
            ArtifactRef(
                logical_name=str(art["logical_name"]),
                relative_path=str(art["relative_path"]),
                media_type=str(art["media_type"]),
                size_bytes=int(art["size_bytes"]),
                sha256=str(art["sha256"]),
                contract_name=art.get("contract_name"),
                contract_version=art.get("contract_version"),
                schema_fingerprint=art.get("schema_fingerprint"),
                metadata=art.get("metadata") or {},
            )
        object.__setattr__(self, "artifacts", arts)
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc:
            raise PipelineError("created_at_utc empty")
        try:
            object.__setattr__(self, "source_run_id", validate_run_id(self.source_run_id))
        except RunIdError as exc:
            raise PipelineError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cache_key": self.cache_key,
            "layout_version": self.layout_version,
            "stage_name": self.stage_name,
            "stage_version": self.stage_version,
            "config_fingerprint": self.config_fingerprint,
            "artifacts": [dict(a) for a in self.artifacts],
            "created_at_utc": self.created_at_utc,
            "source_run_id": self.source_run_id,
        }
