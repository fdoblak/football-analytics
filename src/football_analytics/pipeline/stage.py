"""Stage protocol, registry, and synthetic echo fixture (Stage 2D)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.pipeline.artifacts import build_artifact_ref
from football_analytics.pipeline.exceptions import StageError
from football_analytics.pipeline.types import (
    ContractRef,
    StageExecutionOutput,
    StageIdentity,
    StageRequest,
)

SYNTHETIC_ECHO_CODE_IDENTITY = (
    "football_analytics.pipeline.stage.SyntheticEchoStage/v1/"
    "echo.bin=sha256(input_bytes||config_fingerprint)"
)


def make_stage_identity(
    *,
    name: str,
    version: int,
    code_fingerprint: str,
    input_contracts: tuple[ContractRef, ...] | list[ContractRef] = (),
    output_contracts: tuple[ContractRef, ...] | list[ContractRef] = (),
    deterministic: bool = True,
    cacheable: bool = True,
) -> StageIdentity:
    """Helper to construct a validated :class:`StageIdentity`."""
    return StageIdentity(
        name=name,
        version=version,
        code_fingerprint=code_fingerprint,
        input_contracts=tuple(input_contracts),
        output_contracts=tuple(output_contracts),
        deterministic=deterministic,
        cacheable=cacheable,
    )


@runtime_checkable
class Stage(Protocol):
    """Canonical single-stage interface (no DAG scheduling)."""

    @property
    def identity(self) -> StageIdentity: ...

    def execute(self, request: StageRequest) -> StageExecutionOutput: ...


class StageRegistry:
    """Explicit in-process stage registry (no dynamic import/eval/entry points)."""

    def __init__(self) -> None:
        self._stages: dict[tuple[str, int], Stage] = {}

    def register(self, stage: Stage) -> None:
        identity = stage.identity
        key = (identity.name, identity.version)
        if key in self._stages:
            raise StageError(f"duplicate stage registration: {identity.name} v{identity.version}")
        self._stages[key] = stage

    def get(self, name: str, version: int | None = None) -> Stage:
        if version is not None:
            key = (name, version)
            if key not in self._stages:
                raise StageError(f"stage not found: {name} v{version}")
            return self._stages[key]
        matches = sorted(
            (((n, v), s) for (n, v), s in self._stages.items() if n == name),
            key=lambda item: item[0],
        )
        if not matches:
            raise StageError(f"stage not found: {name}")
        if len(matches) > 1:
            # Prefer highest version when unambiguous selection is not requested.
            return matches[-1][1]
        return matches[0][1]

    def list(self) -> list[StageIdentity]:
        return [
            self._stages[key].identity
            for key in sorted(self._stages.keys(), key=lambda kv: (kv[0], kv[1]))
        ]


class SyntheticEchoStage:
    """Synthetic fixture stage — NOT a product pipeline stage.

    Takes one input artifact (any media type), writes deterministic ``echo.bin``
    containing the hex SHA-256 of ``input_bytes || config_fingerprint``.
    Tracks ``_executions`` for cache-hit proofs in tests.
    """

    def __init__(self) -> None:
        self._executions = 0
        self._identity = make_stage_identity(
            name="synthetic_echo",
            version=1,
            code_fingerprint=hash_canonical_json({"code_identity": SYNTHETIC_ECHO_CODE_IDENTITY}),
            input_contracts=(),
            output_contracts=(),
            deterministic=True,
            cacheable=True,
        )

    @property
    def identity(self) -> StageIdentity:
        return self._identity

    @property
    def executions(self) -> int:
        return self._executions

    def execute(self, request: StageRequest) -> StageExecutionOutput:
        if request.stage_identity.name != self._identity.name:
            raise StageError("stage identity mismatch")
        if len(request.inputs) != 1:
            raise StageError("synthetic_echo requires exactly one input")
        logical_name, ref = next(iter(request.inputs.items()))
        src = (request.working_directory / ref.relative_path).resolve()
        try:
            src.relative_to(request.working_directory.resolve())
        except ValueError as exc:
            raise StageError("input escapes working_directory") from exc
        if not src.is_file() or src.is_symlink():
            raise StageError("input must be a regular file under working_directory")
        digest = sha256_file(src)
        # Content = SHA-256(input_bytes || config_fingerprint) as lowercase hex.
        # Re-hash bytes with config fingerprint for deterministic echo payload.
        payload_digest = _echo_digest(src, request.config_fingerprint)
        out_path = request.output_directory / "echo.bin"
        out_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if out_path.exists():
            raise StageError("echo.bin already exists in output_directory")
        out_path.write_bytes(payload_digest.encode("ascii"))
        with contextlib.suppress(OSError):
            out_path.chmod(0o600)
        self._executions += 1
        art = build_artifact_ref(
            "echo",
            out_path,
            root=request.output_directory,
            media_type="application/octet-stream",
            metadata={"source_logical_name": logical_name, "source_sha256": digest},
        )
        return StageExecutionOutput(
            outputs={"echo": art},
            metrics={"bytes_written": art.size_bytes, "executions": self._executions},
            warnings=(),
        )


def _echo_digest(path: Path, config_fingerprint: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    h.update(config_fingerprint.encode("utf-8"))
    return h.hexdigest()
