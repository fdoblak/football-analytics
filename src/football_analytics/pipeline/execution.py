"""Single-stage execution lifecycle with optional cache (Stage 2D)."""

from __future__ import annotations

import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.redaction import redact_text
from football_analytics.pipeline.artifacts import verify_artifact_on_disk
from football_analytics.pipeline.cache import (
    CachePolicyConfig,
    entry_dir,
    publish_cache_entry,
    quarantine_cache_entry,
    restore_cache_entry,
    verify_cache_entry,
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.exceptions import ArtifactError, CacheError, StageError
from football_analytics.pipeline.receipts import write_stage_execution_receipt
from football_analytics.pipeline.stage import Stage
from football_analytics.pipeline.types import (
    ArtifactRef,
    StageExecutionOutput,
    StageRequest,
    StageResult,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _safe_error(exc: BaseException) -> dict[str, str]:
    return {
        "class": type(exc).__name__,
        "message": str(redact_text(str(exc))),
    }


def _execution_fingerprint(
    *,
    status: str,
    cache_key: str,
    outputs: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    return hash_canonical_json(
        {
            "status": status,
            "cache_key": cache_key,
            "outputs": outputs,
            "metrics": metrics,
        }
    )


def _zero_key() -> str:
    return "0" * 64


def execute_stage(
    stage: Stage,
    request: StageRequest,
    *,
    cache_root: Path,
    policy: CachePolicyConfig,
    quarantine_root: Path | None = None,
    force_miss: bool = False,
    write_receipt: bool = True,
) -> StageResult:
    """Run one stage with optional content-addressed cache lookup/publish."""
    started = _utc_now()
    t0 = time.perf_counter()
    identity = stage.identity
    warnings: list[str] = []

    if (
        request.stage_identity.name != identity.name
        or request.stage_identity.version != identity.version
    ):
        raise StageError("request stage_identity does not match stage")

    cacheable = (
        identity.cacheable
        and identity.deterministic
        and policy.enabled
        and request.cache_policy_enabled
        and not force_miss
    )

    try:
        cache_key = compute_cache_key(
            stage=identity,
            config_fingerprint=request.config_fingerprint,
            compatibility_fingerprint=request.compatibility_fingerprint,
            inputs=request.inputs,
        )
    except CacheError:
        cache_key = _zero_key()

    # Cache hit path
    if cacheable:
        entry = entry_dir(cache_root, cache_key)
        if entry.exists():
            try:
                verify_cache_entry(
                    cache_root,
                    cache_key,
                    expected_stage=identity,
                    expected_config_fp=request.config_fingerprint,
                    expected_inputs=request.inputs,
                    expected_compatibility_fp=request.compatibility_fingerprint,
                    policy=policy,
                )
                outputs = restore_cache_entry(
                    cache_root,
                    cache_key,
                    output_directory=request.output_directory,
                    policy=policy,
                    expected_stage=identity,
                    expected_config_fp=request.config_fingerprint,
                    expected_inputs=request.inputs,
                    expected_compatibility_fp=request.compatibility_fingerprint,
                )
                finished = _utc_now()
                duration_ms = max(0, int((time.perf_counter() - t0) * 1000))
                out_dicts = {k: v.to_dict() for k, v in sorted(outputs.items())}
                in_dicts = {k: v.to_dict() for k, v in sorted(request.inputs.items())}
                result = StageResult(
                    run_id=request.run_id,
                    stage_name=identity.name,
                    stage_version=identity.version,
                    status="cache_hit",
                    cache_key=cache_key,
                    cache_hit=True,
                    started_at_utc=started,
                    finished_at_utc=finished,
                    duration_ms=duration_ms,
                    inputs=in_dicts,
                    outputs=out_dicts,
                    metrics={"restored_artifacts": len(outputs)},
                    warnings=tuple(warnings),
                    error=None,
                    execution_fingerprint=_execution_fingerprint(
                        status="cache_hit",
                        cache_key=cache_key,
                        outputs=out_dicts,
                        metrics={"restored_artifacts": len(outputs)},
                    ),
                )
                if write_receipt:
                    write_stage_execution_receipt(
                        request.output_directory / "stage_execution_receipt.json",
                        result,
                        contain_root=request.output_directory,
                    )
                return result
            except (CacheError, ArtifactError) as exc:
                warnings.append(f"cache corruption detected: {type(exc).__name__}")
                if policy.quarantine_corrupt_entries and quarantine_root is not None:
                    try:
                        quarantine_cache_entry(
                            cache_root,
                            cache_key,
                            quarantine_root=quarantine_root,
                            reason=str(exc),
                        )
                        warnings.append("corrupt cache entry quarantined")
                    except CacheError as qexc:
                        warnings.append(f"quarantine failed: {type(qexc).__name__}")
                # Treat as miss and continue.

    # Miss / non-cacheable execution
    work = Path(request.working_directory)
    out = Path(request.output_directory)
    work.mkdir(parents=True, mode=0o700, exist_ok=True)
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    temp_exec = work / f".tmp_exec_{secrets.token_hex(6)}"
    temp_exec.mkdir(mode=0o700)

    exec_output: StageExecutionOutput | None = None
    try:
        try:
            exec_output = stage.execute(request)
        except Exception as exc:  # noqa: BLE001 — isolate stage failures
            finished = _utc_now()
            duration_ms = max(0, int((time.perf_counter() - t0) * 1000))
            in_dicts = {k: v.to_dict() for k, v in sorted(request.inputs.items())}
            result = StageResult(
                run_id=request.run_id,
                stage_name=identity.name,
                stage_version=identity.version,
                status="failed",
                cache_key=cache_key,
                cache_hit=False,
                started_at_utc=started,
                finished_at_utc=finished,
                duration_ms=duration_ms,
                inputs=in_dicts,
                outputs={},
                metrics={},
                warnings=tuple(warnings),
                error=_safe_error(exc),
                execution_fingerprint=_execution_fingerprint(
                    status="failed",
                    cache_key=cache_key,
                    outputs={},
                    metrics={},
                ),
            )
            if write_receipt:
                with_context = out if out.exists() else work
                write_stage_execution_receipt(
                    with_context / "stage_execution_receipt.json",
                    result,
                    contain_root=with_context,
                )
            return result

        # Verify outputs under output_directory
        verified: dict[str, ArtifactRef] = {}
        for name, ref in exec_output.outputs.items():
            verify_artifact_on_disk(ref, root=out, reject_hardlinks=policy.reject_hardlinks)
            verified[name] = ref

        finished = _utc_now()
        duration_ms = max(0, int((time.perf_counter() - t0) * 1000))
        out_dicts = {k: v.to_dict() for k, v in sorted(verified.items())}
        in_dicts = {k: v.to_dict() for k, v in sorted(request.inputs.items())}
        metrics = dict(exec_output.metrics)
        warnings.extend(exec_output.warnings)

        result = StageResult(
            run_id=request.run_id,
            stage_name=identity.name,
            stage_version=identity.version,
            status="succeeded",
            cache_key=cache_key,
            cache_hit=False,
            started_at_utc=started,
            finished_at_utc=finished,
            duration_ms=duration_ms,
            inputs=in_dicts,
            outputs=out_dicts,
            metrics=metrics,
            warnings=tuple(warnings),
            error=None,
            execution_fingerprint=_execution_fingerprint(
                status="succeeded",
                cache_key=cache_key,
                outputs=out_dicts,
                metrics=metrics,
            ),
        )

        if (
            identity.cacheable
            and identity.deterministic
            and policy.enabled
            and request.cache_policy_enabled
        ):
            try:
                publish_cache_entry(
                    cache_root=cache_root,
                    cache_key=cache_key,
                    stage_identity=identity,
                    config_fingerprint=request.config_fingerprint,
                    artifacts=MappingProxyType(verified),
                    artifact_root=out,
                    stage_result=result,
                    policy=policy,
                    source_run_id=request.run_id,
                )
            except CacheError as exc:
                warnings.append(f"cache publish skipped: {type(exc).__name__}")
                result = StageResult(
                    run_id=result.run_id,
                    stage_name=result.stage_name,
                    stage_version=result.stage_version,
                    status=result.status,
                    cache_key=result.cache_key,
                    cache_hit=result.cache_hit,
                    started_at_utc=result.started_at_utc,
                    finished_at_utc=result.finished_at_utc,
                    duration_ms=result.duration_ms,
                    inputs=dict(result.inputs),
                    outputs=dict(result.outputs),
                    metrics=dict(result.metrics),
                    warnings=tuple(warnings),
                    error=None,
                    execution_fingerprint=result.execution_fingerprint,
                )

        if write_receipt:
            write_stage_execution_receipt(
                out / "stage_execution_receipt.json",
                result,
                contain_root=out,
            )
        return result
    finally:
        if temp_exec.exists():
            shutil.rmtree(temp_exec, ignore_errors=True)
