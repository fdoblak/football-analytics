"""Stage 3C normalization service: probe → plan → safe FFmpeg → conformance → receipt."""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.utils.archive_safety import assert_contained, resolve_strict
from football_analytics.video.ffmpeg import (
    FfmpegError,
    assert_libx264_available,
    get_ffmpeg_version,
    resolve_ffmpeg_binary,
    run_ffmpeg_normalize,
)
from football_analytics.video.ffprobe import (
    ProbeError,
    decode_ffprobe_json,
    get_ffprobe_version,
    resolve_ffprobe_binary,
    run_ffprobe,
)
from football_analytics.video.media_validation import validate_probe_against_policy
from football_analytics.video.normalization import (
    PlannedNormalization,
    compute_timeout_seconds,
    estimate_output_bytes,
    plan_normalization,
)
from football_analytics.video.normalization_validation import (
    ConformanceResult,
    validate_normalized_output,
)
from football_analytics.video.probe_parser import map_ffprobe_json_to_video_probe
from football_analytics.video.probe_service import (
    SourceSnapshot,
    assert_snapshots_equal,
    snapshot_source,
)
from football_analytics.video.types import (
    AudioTransformInfo,
    FrameRateConversionInfo,
    Issue,
    NormalizationCleanup,
    NormalizationProvenance,
    NormalizationReceipt,
    NormalizationSelectedStreams,
    NormalizationStatus,
    ResizeTransformInfo,
    RotationTransformInfo,
    VideoProbe,
    VideoSourceError,
)
from football_analytics.video.validation import (
    assert_extension_allowed,
    assert_safe_source_path,
    reject_unsafe_path_string,
    require_absolute_path,
)


@dataclass
class NormalizationServiceResult:
    accepted: bool
    exit_code: int
    status: NormalizationStatus
    receipt: NormalizationReceipt
    plan: PlannedNormalization | None
    source_probe: VideoProbe | None
    output_probe: VideoProbe | None
    conformance: ConformanceResult | None
    output_path: str | None
    error_code: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "status": self.status.value,
            "error_code": self.error_code,
            "output_path": self.output_path,
            "receipt_id": self.receipt.receipt_id,
            "required": None if self.plan is None else self.plan.plan.required,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _path_safe_id(prefix: str) -> str:
    token = generate_run_id().split("_")[-1]
    return f"{prefix}_{token}"


def _sanitize_message(msg: str) -> str:
    return msg.replace("/home/fdoblak/", "~/").replace("\x00", "")


def _assert_safe_output_file(
    output: str,
    *,
    contain_root: Path,
    source_path: Path,
    overwrite_allowed: bool = False,
) -> Path:
    reject_unsafe_path_string(output, label="output")
    out = require_absolute_path(output, label="output")
    root = require_absolute_path(str(contain_root), label="contain_root")
    parent = out.parent
    if parent.exists() and parent.is_symlink():
        raise VideoSourceError("output parent must not be a symlink")
    if out.exists():
        if not overwrite_allowed:
            raise VideoSourceError("output already exists (overwrite forbidden)")
        raise VideoSourceError("overwrite_allowed must be false")
    resolved_parent = resolve_strict(parent) if parent.exists() else parent.resolve()
    assert_contained(resolved_parent, resolve_strict(root), label="output_parent")
    # Ensure final path would be contained
    candidate = resolved_parent / out.name
    assert_contained(candidate, resolve_strict(root), label="output")
    if resolve_strict(source_path) == candidate:
        raise VideoSourceError("source/output path collision")
    if source_path.resolve() == out.resolve() if out.exists() else False:
        raise VideoSourceError("inplace normalization forbidden")
    return candidate


def _probe_media(
    path: Path,
    *,
    policy: Mapping[str, Any],
    source_id: str,
    source_sha256: str,
    file_size_bytes: int,
) -> VideoProbe:
    ff = policy["ffprobe_policy"]
    binary = resolve_ffprobe_binary(
        ff["ffprobe_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
    )
    version = get_ffprobe_version(binary)
    raw = run_ffprobe(path, policy=policy, binary=binary, version=version)
    data = decode_ffprobe_json(raw.stdout, max_depth=int(ff["maximum_json_depth"]))
    return map_ffprobe_json_to_video_probe(
        data,
        source_id=source_id,
        source_sha256=source_sha256,
        file_size_bytes=file_size_bytes,
        probe_tool_version=version.version_token,
        probed_at_utc=_utc_now(),
        max_stream_count=int(ff["maximum_stream_count"]),
    )


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _release_lock(lock_path: Path | None) -> bool:
    if lock_path is None:
        return True
    try:
        lock_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _remove_temp(temp_path: Path | None) -> bool:
    if temp_path is None:
        return True
    try:
        temp_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _write_receipt(path: Path, receipt: NormalizationReceipt, *, contain_root: Path) -> None:
    write_json_record(path, receipt.to_dict(), contain_root=contain_root, overwrite=False)


def _empty_transforms(
    *,
    planned: PlannedNormalization | None,
    audio_policy: str,
) -> tuple[
    FrameRateConversionInfo,
    RotationTransformInfo,
    ResizeTransformInfo,
    AudioTransformInfo,
    NormalizationSelectedStreams,
]:
    if planned is None:
        fr = FrameRateConversionInfo(
            performed=False,
            source_mode="unknown",
            target_mode="unchanged",
            notes="not planned",
            requires_stage3d_mapping=False,
        )
        rot = RotationTransformInfo(performed=False, source_degrees=0, output_degrees=0)
        resize = ResizeTransformInfo(
            performed=False,
            source_width=1,
            source_height=1,
            target_width=None,
            target_height=None,
        )
        audio = AudioTransformInfo(policy=audio_policy, action="absent")
        streams = NormalizationSelectedStreams(video_stream_index=0, audio_stream_index=None)
        return fr, rot, resize, audio, streams
    if planned.frame_rate_conversion:
        target_mode = "cfr"
        notes = "forced CFR for VFR source"
        requires_3d = True
        source_mode = "vfr"
    else:
        target_mode = "unchanged"
        notes = "frame rate preserved"
        requires_3d = False
        source_mode = "unknown"
    fr = FrameRateConversionInfo(
        performed=planned.frame_rate_conversion,
        source_mode=source_mode,
        target_mode=target_mode,
        notes=notes,
        requires_stage3d_mapping=requires_3d,
    )
    rot = RotationTransformInfo(
        performed=planned.bake_rotation,
        source_degrees=planned.source_rotation_degrees,
        output_degrees=0 if planned.bake_rotation else planned.source_rotation_degrees,
    )
    resize = ResizeTransformInfo(
        performed=planned.resize_performed,
        source_width=planned.source_width,
        source_height=planned.source_height,
        target_width=planned.plan.target_width,
        target_height=planned.plan.target_height,
    )
    audio = AudioTransformInfo(policy=audio_policy, action=planned.audio_action)
    streams = NormalizationSelectedStreams(
        video_stream_index=planned.video_stream_index,
        audio_stream_index=planned.audio_stream_index,
    )
    return fr, rot, resize, audio, streams


def run_video_normalization(
    *,
    source: str,
    output: str,
    policy: Mapping[str, Any],
    expected_source_sha256: str | None = None,
    execute: bool = False,
    contain_root: str | Path | None = None,
    receipt_dir: str | Path | None = None,
    source_id: str | None = None,
    run_id: str | None = None,
    plan_id: str | None = None,
    ffmpeg_runner: Callable[..., Any] | None = None,
    disk_usage_fn: Callable[[str], Any] | None = None,
) -> NormalizationServiceResult:
    """Execute Stage 3C normalization pipeline (default dry-run)."""
    started = _utc_now()
    receipt_id = _path_safe_id("nrcpt")
    sid = source_id or _path_safe_id("src")
    pid = plan_id or _path_safe_id("plan")
    rid = run_id or generate_run_id()
    nd = policy["normalization_defaults"]
    ff = policy["ffmpeg_policy"]
    audio_policy = str(nd.get("target_audio_policy", "copy_if_present_else_drop"))
    root = Path(contain_root) if contain_root else Path(str(ff["runtime_root"]))
    root.mkdir(parents=True, exist_ok=True)

    lock_path: Path | None = None
    temp_path: Path | None = None
    planned: PlannedNormalization | None = None
    source_probe: VideoProbe | None = None
    output_probe: VideoProbe | None = None
    conformance: ConformanceResult | None = None
    ffmpeg_path = str(ff.get("ffmpeg_binary", "/usr/bin/ffmpeg"))
    ffmpeg_version = "unknown"
    before: SourceSnapshot | None = None

    def finish(
        *,
        status: NormalizationStatus,
        exit_code: int,
        code: str | None,
        message: str | None = None,
        accepted: bool = False,
        output_artifact: str | None = None,
        output_sha: str | None = None,
        output_size: int | None = None,
        output_probe_fp: str | None = None,
        duration_drift: int | None = None,
        warnings: tuple[Issue, ...] = (),
        errors: tuple[Issue, ...] = (),
        transforms: tuple[str, ...] = (),
        argv_summary: str | None = None,
    ) -> NormalizationServiceResult:
        nonlocal lock_path, temp_path
        temp_removed = _remove_temp(temp_path)
        lock_released = _release_lock(lock_path)
        temp_path = None
        lock_path = None
        err_list = list(errors)
        if code and message and not any(e.code == code for e in err_list):
            err_list.insert(0, Issue(code=code, message=_sanitize_message(message)))
        if status in {NormalizationStatus.REJECTED, NormalizationStatus.FAILED} and not err_list:
            err_list = [Issue(code=code or "FAILED", message=message or "failed")]
        fr, rot, resize, audio, streams = _empty_transforms(
            planned=planned, audio_policy=audio_policy
        )
        # Fix frame rate source mode from probe when available
        if source_probe is not None and planned is not None:
            for stream in source_probe.streams:
                if getattr(stream, "stream_index", None) == planned.video_stream_index:
                    mode = getattr(stream, "frame_rate_mode", None)
                    if mode is not None:
                        fr = FrameRateConversionInfo(
                            performed=planned.frame_rate_conversion,
                            source_mode=mode.value,
                            target_mode=("cfr" if planned.frame_rate_conversion else "unchanged"),
                            notes=(
                                "forced CFR; Stage 3D mapping required"
                                if planned.frame_rate_conversion
                                else "preserved"
                            ),
                            requires_stage3d_mapping=planned.frame_rate_conversion,
                        )
                    break
        source_sha = (
            before.sha256
            if before is not None
            else (expected_source_sha256 if expected_source_sha256 else "0" * 64)
        )
        if len(source_sha) != 64:
            source_sha = "0" * 64
        plan_fp = planned.plan.plan_fingerprint if planned else ("0" * 64)
        plan_id_out = planned.plan.plan_id if planned else pid
        receipt = NormalizationReceipt(
            receipt_id=receipt_id,
            run_id=rid,
            plan_id=plan_id_out,
            plan_fingerprint=plan_fp,
            source_id=sid,
            source_sha256=source_sha,
            source_probe_fingerprint=(None if source_probe is None else source_probe.fingerprint()),
            output_artifact=output_artifact,
            output_sha256=output_sha,
            output_size_bytes=output_size,
            output_probe_fingerprint=output_probe_fp,
            status=status,
            started_at_utc=started,
            completed_at_utc=_utc_now(),
            ffmpeg_path=ffmpeg_path,
            ffmpeg_version=ffmpeg_version,
            execution_profile=str(nd.get("execution_profile", "cpu_libx264_crf23")),
            selected_streams=streams,
            applied_transforms=(
                transforms if transforms else (planned.applied_transforms if planned else ())
            ),
            frame_rate_conversion=fr,
            rotation_transform=rot,
            resize_transform=resize,
            audio_transform=audio,
            duration_drift_us=duration_drift,
            warnings=warnings,
            errors=tuple(err_list),
            cleanup=NormalizationCleanup(temp_removed=temp_removed, lock_released=lock_released),
            provenance=NormalizationProvenance(
                stage="3C",
                label="stage3c_normalize",
                notes="safe video normalization",
                sanitized_argv_summary=argv_summary,
            ),
        )
        # Persist receipt
        try:
            if receipt_dir is not None:
                rdir = Path(receipt_dir)
                rdir.mkdir(parents=True, exist_ok=True)
                _write_receipt(rdir / "normalization_receipt.json", receipt, contain_root=root)
            elif status == NormalizationStatus.SUCCEEDED and output_artifact:
                _write_receipt(
                    Path(output_artifact).parent / "normalization_receipt.json",
                    receipt,
                    contain_root=root,
                )
            elif receipt_dir is None:
                # dry-run / skip: write beside intended output parent if under root
                out_parent = Path(output).parent
                if out_parent.exists():
                    with contextlib.suppress(Exception):
                        _write_receipt(
                            out_parent / "normalization_receipt.json",
                            receipt,
                            contain_root=root,
                        )
        except Exception:  # noqa: BLE001
            pass
        return NormalizationServiceResult(
            accepted=accepted,
            exit_code=exit_code,
            status=status,
            receipt=receipt,
            plan=planned,
            source_probe=source_probe,
            output_probe=output_probe,
            conformance=conformance,
            output_path=output_artifact,
            error_code=code,
        )

    if execute and not expected_source_sha256:
        return finish(
            status=NormalizationStatus.REJECTED,
            exit_code=2,
            code="USAGE_EXPECTED_SHA_REQUIRED",
            message="--execute requires --expected-source-sha256",
        )

    # Path safety
    try:
        reject_unsafe_path_string(source, label="source")
        src_path = assert_safe_source_path(source, contain_root=root, policy=policy)
        out_path = _assert_safe_output_file(
            output, contain_root=root, source_path=src_path, overwrite_allowed=False
        )
        if nd.get("inplace_forbidden", True) and src_path.resolve() == out_path:
            raise VideoSourceError("inplace normalization forbidden")
        assert_extension_allowed(src_path, policy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except (VideoSourceError, ProbeError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or "SOURCE_NOT_REGULAR_FILE"
        return finish(
            status=NormalizationStatus.REJECTED,
            exit_code=3,
            code=str(code),
            message=str(exc),
        )

    # Expected SHA verify
    try:
        before = snapshot_source(src_path)
        if expected_source_sha256 and before.sha256 != expected_source_sha256:
            return finish(
                status=NormalizationStatus.REJECTED,
                exit_code=3,
                code="SOURCE_HASH_MISMATCH",
                message="source sha256 does not match expected",
            )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="SOURCE_HASH_MISMATCH",
            message=str(exc),
        )

    # Probe source
    try:
        source_probe = _probe_media(
            src_path,
            policy=policy,
            source_id=sid,
            source_sha256=before.sha256,
            file_size_bytes=before.size_bytes,
        )
        validation = validate_probe_against_policy(
            source_probe, policy, source_size_bytes=before.size_bytes
        )
        if not validation.accepted:
            return finish(
                status=NormalizationStatus.REJECTED,
                exit_code=1,
                code=validation.errors[0].code if validation.errors else "PROBE_REJECTED",
                message=validation.errors[0].message if validation.errors else "probe rejected",
                errors=tuple(validation.errors),
                warnings=tuple(validation.warnings),
            )
    except ProbeError as exc:
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=4 if exc.code.startswith("PROBE_") else 3,
            code=exc.code,
            message=exc.message,
        )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=4,
            code="PROBE_UNEXPECTED_STRUCTURE",
            message=str(exc),
        )

    # Plan
    try:
        planned = plan_normalization(
            probe=source_probe,
            policy=policy,
            output_path=str(out_path),
            plan_id=pid,
            source_id=sid,
        )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=1,
            code="PLAN_FAILED",
            message=str(exc),
        )

    # Dry-run
    if not execute:
        status = (
            NormalizationStatus.PLANNED if planned.plan.required else NormalizationStatus.SKIPPED
        )
        return finish(
            status=status,
            exit_code=0,
            code=None,
            accepted=True,
            transforms=planned.applied_transforms,
        )

    # Execute + not required → skipped (no copy/hardlink)
    if not planned.plan.required:
        return finish(
            status=NormalizationStatus.SKIPPED,
            exit_code=0,
            code=None,
            accepted=True,
            transforms=planned.applied_transforms,
        )

    # FFmpeg binary / capabilities
    try:
        binary = resolve_ffmpeg_binary(
            ff["ffmpeg_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
        )
        ver = get_ffmpeg_version(binary)
        ffmpeg_path = ver.path
        ffmpeg_version = ver.version_token
        assert_libx264_available(binary)
    except FfmpegError as exc:
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=4,
            code=exc.code,
            message=exc.message,
        )

    # Disk preflight
    try:
        usage_fn = disk_usage_fn or shutil.disk_usage
        usage = usage_fn(str(out_path.parent))
        free_raw = getattr(usage, "free", None)
        if free_raw is None and isinstance(usage, tuple) and len(usage) >= 3:
            free_raw = usage[2]
        if free_raw is None:
            raise TypeError("disk usage missing free bytes")
        free = int(free_raw)
        estimate = estimate_output_bytes(source_probe, policy)
        min_free = int(ff["minimum_free_space_bytes"])
        headroom = int(ff["minimum_free_space_after_estimate_bytes"])
        if free < min_free or free < (estimate + headroom):
            return finish(
                status=NormalizationStatus.REJECTED,
                exit_code=3,
                code="INSUFFICIENT_OUTPUT_SPACE",
                message=f"free={free} estimate={estimate} min={min_free}",
            )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="DISK_PREFLIGHT_FAILED",
            message=str(exc),
        )

    # Exclusive lock
    lock_path = Path(str(out_path) + ".norm.lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        lock_path = None
        return finish(
            status=NormalizationStatus.REJECTED,
            exit_code=3,
            code="NORMALIZATION_LOCK_HELD",
            message="exclusive lock already held",
        )
    except OSError as exc:
        lock_path = None
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="NORMALIZATION_LOCK_FAILED",
            message=str(exc),
        )

    # Temp file
    token = secrets.token_hex(8)
    temp_path = out_path.parent / f"{out_path.name}.tmp.{token}.mp4"
    if temp_path.exists():
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="TEMP_OUTPUT_EXISTS",
            message="temp output collision",
        )

    timeout = compute_timeout_seconds(source_probe.duration_us, policy)
    runner = ffmpeg_runner or run_ffmpeg_normalize
    argv_summary = None
    try:
        raw = runner(
            src_path,
            temp_path,
            planned,
            policy,
            binary=binary,
            version=ver,
            timeout_seconds=timeout,
        )
        argv_summary = " ".join(raw.argv[:12]) + "…"
    except FfmpegError as exc:
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=4,
            code=exc.code,
            message=exc.message,
        )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=4,
            code="FFMPEG_PROCESS_FAILED",
            message=str(exc),
        )

    # Probe temp
    try:
        temp_size = int(temp_path.stat().st_size)
        temp_sha = sha256_file(temp_path)
        output_probe = _probe_media(
            temp_path,
            policy=policy,
            source_id=sid,
            source_sha256=temp_sha,
            file_size_bytes=temp_size,
        )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=5,
            code="OUTPUT_PROBE_FAILED",
            message=str(exc),
            argv_summary=argv_summary,
        )

    conformance = validate_normalized_output(
        plan=planned,
        source_probe=source_probe,
        output_probe=output_probe,
        policy=policy,
    )
    if not conformance.ok:
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=5,
            code=conformance.errors[0].code if conformance.errors else "CONFORMANCE_FAILED",
            message=(conformance.errors[0].message if conformance.errors else "conformance failed"),
            errors=tuple(conformance.errors),
            warnings=tuple(conformance.warnings),
            duration_drift=conformance.duration_drift_us,
            argv_summary=argv_summary,
        )

    # Source post snapshot
    try:
        after = snapshot_source(src_path)
        try:
            assert_snapshots_equal(before, after)
        except ProbeError:
            return finish(
                status=NormalizationStatus.FAILED,
                exit_code=3,
                code="SOURCE_MUTATED_DURING_NORMALIZATION",
                message="source changed during normalization",
                argv_summary=argv_summary,
            )
    except Exception as exc:  # noqa: BLE001
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="SOURCE_MUTATED_DURING_NORMALIZATION",
            message=str(exc),
            argv_summary=argv_summary,
        )

    # Publish: fsync temp, replace, fsync parent
    try:
        _fsync_file(temp_path)
        os.replace(str(temp_path), str(out_path))
        temp_path = None  # replaced
        _fsync_dir(out_path.parent)
        final_sha = sha256_file(out_path)
        final_size = int(out_path.stat().st_size)
    except OSError as exc:
        return finish(
            status=NormalizationStatus.FAILED,
            exit_code=3,
            code="OUTPUT_PUBLISH_FAILED",
            message=str(exc),
            argv_summary=argv_summary,
        )

    return finish(
        status=NormalizationStatus.SUCCEEDED,
        exit_code=0,
        code=None,
        accepted=True,
        output_artifact=str(out_path),
        output_sha=final_sha,
        output_size=final_size,
        output_probe_fp=output_probe.fingerprint(),
        duration_drift=conformance.duration_drift_us if conformance else None,
        warnings=tuple(conformance.warnings) if conformance else (),
        transforms=planned.applied_transforms,
        argv_summary=argv_summary,
    )
