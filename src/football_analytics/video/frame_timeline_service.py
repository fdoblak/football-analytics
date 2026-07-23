"""Stage 3D frame timeline service: stream map → Parquet + optional materialize."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.parquet import write_contract_parquet_streaming
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.utils.archive_safety import assert_contained, resolve_strict
from football_analytics.video.ffprobe import (
    ProbeError,
    decode_ffprobe_json,
    get_ffprobe_version,
    resolve_ffprobe_binary,
    run_ffprobe,
)
from football_analytics.video.frame_extraction import materialize_frames
from football_analytics.video.frame_timeline import (
    FrameTimelineError,
    iter_ffprobe_frame_lines,
    iter_mapped_frames_from_lines,
    mapped_frames_to_record_batches,
)
from football_analytics.video.probe_parser import map_ffprobe_json_to_video_probe
from football_analytics.video.probe_service import snapshot_source
from football_analytics.video.time_mapping import (
    MappingEvidence,
    assert_no_index_fps_invention,
    classify_mapping_quality,
    empty_mapping_evidence,
)
from football_analytics.video.types import (
    FrameRateMode,
    FrameTimelineCleanup,
    FrameTimelineMode,
    FrameTimelineProvenance,
    FrameTimelineReceipt,
    FrameTimelineStatus,
    Issue,
    MappingQuality,
    Rational,
    VideoProbe,
    VideoSourceError,
    VideoStreamInfo,
)
from football_analytics.video.validation import (
    assert_extension_allowed,
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
    require_absolute_path,
)

_TIME_REWRITE_TRANSFORMS = frozenset({"force_cfr"})


@dataclass
class FrameTimelineServiceResult:
    accepted: bool
    exit_code: int
    status: FrameTimelineStatus
    receipt: FrameTimelineReceipt
    frames_parquet: str | None
    artifact_manifest: str | None
    error_code: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "status": self.status.value,
            "error_code": self.error_code,
            "frames_parquet": self.frames_parquet,
            "artifact_manifest": self.artifact_manifest,
            "receipt_id": self.receipt.receipt_id,
            "mapping_quality": self.receipt.mapping_quality.value,
            "frame_count": self.receipt.frame_count,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _path_safe_id(prefix: str) -> str:
    token = generate_run_id().split("_")[-1]
    return f"{prefix}_{token}"


def _sanitize_message(msg: str) -> str:
    return msg.replace("/home/fdoblak/", "~/").replace("\x00", "")


def _compute_timeout(policy: Mapping[str, Any], duration_us: int | None) -> float:
    ftp = policy["frame_timeline_policy"]
    base = float(ftp["timeout_base_seconds"])
    per = float(ftp["timeout_per_media_second"])
    cap = float(ftp["maximum_timeout_seconds"])
    seconds = 0.0 if duration_us is None else max(0.0, duration_us / 1_000_000.0)
    return min(cap, base + per * seconds)


def _probe_for_timeline(
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


def _selected_video_stream(probe: VideoProbe) -> VideoStreamInfo:
    for stream in probe.streams:
        if (
            isinstance(stream, VideoStreamInfo)
            and stream.stream_index == probe.selected_video_stream_index
        ):
            return stream
    raise FrameTimelineError("NO_VIDEO_STREAM", "selected video stream missing")


def _evidence_from_norm_data(data: Mapping[str, Any]) -> MappingEvidence:
    status = data.get("status")
    status_s = str(status) if status is not None else None
    fr_raw = data.get("frame_rate_conversion")
    fr = fr_raw if isinstance(fr_raw, dict) else {}
    transforms = tuple(str(t) for t in (data.get("applied_transforms") or ()))
    performed = fr.get("performed")
    performed_b = bool(performed) if isinstance(performed, bool) else None
    requires = fr.get("requires_stage3d_mapping")
    requires_b = bool(requires) if isinstance(requires, bool) else None
    src_mode = fr.get("source_mode")
    tgt_mode = fr.get("target_mode")
    drift = data.get("duration_drift_us")
    drift_i = int(drift) if isinstance(drift, int) and not isinstance(drift, bool) else None
    time_rewrite = performed_b is True or any(t in _TIME_REWRITE_TRANSFORMS for t in transforms)
    identity = status_s == "skipped" and not time_rewrite and requires_b is not True
    return MappingEvidence(
        has_normalization_receipt=True,
        normalization_status=status_s,
        frame_rate_conversion_performed=performed_b,
        frame_rate_conversion_source_mode=str(src_mode) if src_mode is not None else None,
        frame_rate_conversion_target_mode=str(tgt_mode) if tgt_mode is not None else None,
        requires_stage3d_mapping=requires_b,
        duration_drift_us=drift_i,
        applied_transforms=transforms,
        constant_offset_us=None,
        identity_proven=identity,
    )


def _load_normalization_hints(
    path: str | None,
) -> tuple[str | None, int | None, FrameRateMode | None, MappingEvidence]:
    if path is None:
        return None, None, None, empty_mapping_evidence()
    reject_unsafe_path_string(path, label="normalization_receipt")
    p = require_absolute_path(path, label="normalization_receipt")
    if p.is_symlink() or not p.is_file():
        raise VideoSourceError("normalization receipt must be a regular file")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FrameTimelineError("INVALID_RECEIPT", "normalization receipt must be object")
    stream_idx = None
    sel = data.get("selected_streams")
    if isinstance(sel, dict) and sel.get("video_stream_index") is not None:
        stream_idx = int(sel["video_stream_index"])
    fr_mode = None
    fr = data.get("frame_rate_conversion")
    if isinstance(fr, dict) and fr.get("target_mode") in {"cfr", "vfr", "unknown"}:
        fr_mode = FrameRateMode(str(fr["target_mode"]))
    return str(p), stream_idx, fr_mode, _evidence_from_norm_data(data)


def _fail_receipt(
    *,
    receipt_id: str,
    run_id: str,
    video_id: str,
    source_path: str,
    source_sha256: str,
    mode: FrameTimelineMode,
    status: FrameTimelineStatus,
    started: str,
    ffprobe_path: str,
    ffprobe_version: str,
    errors: list[Issue],
    warnings: list[Issue] | None = None,
    normalization_receipt_path: str | None = None,
    sample_every: int | None = None,
) -> FrameTimelineReceipt:
    return FrameTimelineReceipt(
        receipt_id=receipt_id,
        run_id=run_id,
        video_id=video_id,
        source_path=source_path,
        source_sha256=source_sha256,
        normalization_receipt_path=normalization_receipt_path,
        mode=mode,
        status=status,
        started_at_utc=started,
        completed_at_utc=_utc_now(),
        ffprobe_path=ffprobe_path or "/usr/bin/ffprobe",
        ffprobe_version=ffprobe_version or "unknown",
        video_stream_index=0,
        time_base=Rational(1, 1),
        frame_rate_mode=FrameRateMode.UNKNOWN,
        frames_parquet=None,
        frames_parquet_sha256=None,
        frame_count=0,
        ok_count=0,
        skipped_count=0,
        failed_count=0,
        unknown_count=0,
        missing_pts_count=0,
        duplicate_pts_count=0,
        non_monotonic_pts_count=0,
        mapping_quality=MappingQuality.NOT_AVAILABLE,
        sample_every=sample_every,
        materialized=False,
        materialized_frame_count=None,
        artifact_manifest=None,
        warnings=tuple(warnings or ()),
        errors=tuple(errors),
        cleanup=FrameTimelineCleanup(temp_removed=True),
        provenance=FrameTimelineProvenance(stage="3D", label="frame_timeline", notes=None),
    )


def run_frame_timeline(
    *,
    source: str,
    output_dir: str,
    policy: Mapping[str, Any],
    mode: FrameTimelineMode = FrameTimelineMode.TIMELINE_ONLY,
    contain_root: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    expected_source_sha256: str | None = None,
    execute_materialize: bool = False,
    sample_every: int | None = None,
    normalization_receipt: str | None = None,
) -> FrameTimelineServiceResult:
    """Build frames.parquet timeline; materialize only with explicit flag."""
    started = _utc_now()
    receipt_id = _path_safe_id("ftl")
    run = run_id or generate_run_id()
    vid = video_id or _path_safe_id("vid")
    ftp = policy["frame_timeline_policy"]
    assert_no_index_fps_invention(dict(ftp))
    root = contain_root or Path(str(ftp["runtime_root"]))
    sample = int(sample_every) if sample_every is not None else int(ftp["sample_every_default"])

    warnings: list[Issue] = []
    ff_path = str(ftp["ffprobe_binary"])
    ff_ver = "unknown"
    norm_path, norm_stream, norm_mode = None, None, None
    mapping_evidence = empty_mapping_evidence()

    try:
        if mode != FrameTimelineMode.TIMELINE_ONLY and not execute_materialize:
            raise FrameTimelineError(
                "MATERIALIZE_FLAG_REQUIRED",
                "sampled/all_frames require --execute-materialize",
            )
        if execute_materialize and mode == FrameTimelineMode.TIMELINE_ONLY:
            raise FrameTimelineError(
                "MATERIALIZE_MODE_REQUIRED",
                "--execute-materialize requires sampled or all_frames mode",
            )
        if ftp.get("overwrite_allowed") is not False:
            raise FrameTimelineError("POLICY", "overwrite_allowed must be false")

        out_dir = require_absolute_path(output_dir, label="output_dir")
        src = assert_safe_source_path(source, contain_root=root, policy=policy)
        assert_extension_allowed(src, policy=policy)
        assert_safe_output_root(
            str(out_dir), contain_root=root, source_path=str(src), overwrite_allowed=False
        )
        out_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            resolve_strict(out_dir).relative_to(resolve_strict(root))
        except ValueError as exc:
            raise VideoSourceError("output_dir escapes containment") from exc

        snap = snapshot_source(src)
        if expected_source_sha256 and snap.sha256 != expected_source_sha256:
            raise FrameTimelineError(
                "SOURCE_HASH_MISMATCH",
                "source sha256 does not match expected_source_sha256",
            )

        binary = resolve_ffprobe_binary(
            ftp["ffprobe_binary"], allowed_realpaths=list(ftp["allowed_binary_realpaths"])
        )
        version = get_ffprobe_version(binary)
        ff_path = version.path
        ff_ver = version.version_token

        norm_path, norm_stream, norm_mode, mapping_evidence = _load_normalization_hints(
            normalization_receipt
        )
        probe = _probe_for_timeline(
            src,
            policy=policy,
            source_id=vid,
            source_sha256=snap.sha256,
            file_size_bytes=snap.size_bytes,
        )
        vstream = _selected_video_stream(probe)
        stream_index = norm_stream if norm_stream is not None else vstream.stream_index
        time_base = vstream.time_base
        frame_rate_mode = norm_mode if norm_mode is not None else vstream.frame_rate_mode
        timeout = _compute_timeout(policy, probe.duration_us)

        frames_path = out_dir / "frames.parquet"
        if frames_path.exists():
            raise VideoSourceError("frames.parquet already exists (overwrite forbidden)")

        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        contract = reg.load_contract("frames", 1)
        arrow_schema = compile_arrow_schema(contract)

        line_iter = iter_ffprobe_frame_lines(
            src,
            policy=policy,
            video_stream_index=stream_index,
            binary=binary,
            timeout_seconds=timeout,
        )
        mapped_iter, parse_result = iter_mapped_frames_from_lines(
            line_iter,
            time_base=time_base,
            maximum_frames=int(ftp["maximum_frames"]),
        )
        batches = mapped_frames_to_record_batches(
            mapped_iter,
            run_id=run,
            video_id=vid,
            batch_size=int(ftp["batch_size"]),
            arrow_schema=arrow_schema,
        )
        write_contract_parquet_streaming(
            batches,
            frames_path,
            contract,
            contain_root=resolve_strict(root),
            overwrite=False,
            compression="zstd",
            check_semantics=True,
        )
        # Confirm source unchanged
        snap2 = snapshot_source(src)
        if snap2.sha256 != snap.sha256:
            raise FrameTimelineError("SOURCE_MUTATED", "source mutated during timeline")

        for code, msg in parse_result.warnings[:50]:
            warnings.append(Issue(code=code, message=_sanitize_message(msg)))

        stats = parse_result.stats
        quality = classify_mapping_quality(
            stats, frame_rate_mode=frame_rate_mode, evidence=mapping_evidence
        )
        parquet_sha = sha256_file(frames_path)

        artifact_path: str | None = None
        materialized_count: int | None = None
        if execute_materialize:
            # Read only selected metadata columns via parquet (not a second full pylist
            # of raw probe frames). Tiny fixtures remain bounded by maximum_frames.
            import pyarrow.parquet as pq

            table = pq.read_table(
                frames_path,
                columns=[
                    "frame_index",
                    "pts",
                    "video_time_us",
                    "decode_status",
                ],
            )
            # to_pylist here is post-write metadata for materialize join only.
            meta_rows = table.to_pylist()
            mat = materialize_frames(
                source=src,
                output_dir=out_dir,
                contain_root=resolve_strict(root),
                policy=policy,
                mode=mode,
                sample_every=sample,
                frame_meta=meta_rows,
                run_id=run,
                video_id=vid,
                source_sha256=snap.sha256,
                manifest_id=_path_safe_id("fam"),
                created_at_utc=_utc_now(),
                timeout_seconds=timeout,
            )
            artifact_path = str(mat.manifest_path)
            materialized_count = mat.row_count
            assert_contained(
                resolve_strict(mat.manifest_path),
                resolve_strict(root),
                label="artifact_manifest",
            )

        receipt = FrameTimelineReceipt(
            receipt_id=receipt_id,
            run_id=run,
            video_id=vid,
            source_path=str(src),
            source_sha256=snap.sha256,
            normalization_receipt_path=norm_path,
            mode=mode,
            status=FrameTimelineStatus.SUCCEEDED,
            started_at_utc=started,
            completed_at_utc=_utc_now(),
            ffprobe_path=ff_path,
            ffprobe_version=ff_ver,
            video_stream_index=stream_index,
            time_base=time_base,
            frame_rate_mode=frame_rate_mode,
            frames_parquet=str(frames_path),
            frames_parquet_sha256=parquet_sha,
            frame_count=stats.frame_count,
            ok_count=stats.ok_count,
            skipped_count=stats.skipped_count,
            failed_count=stats.failed_count,
            unknown_count=stats.unknown_count,
            missing_pts_count=stats.missing_pts_count,
            duplicate_pts_count=stats.duplicate_pts_count,
            non_monotonic_pts_count=stats.non_monotonic_pts_count,
            mapping_quality=quality,
            sample_every=sample if mode == FrameTimelineMode.SAMPLED else None,
            materialized=bool(execute_materialize),
            materialized_frame_count=materialized_count,
            artifact_manifest=artifact_path,
            warnings=tuple(warnings),
            errors=(),
            cleanup=FrameTimelineCleanup(temp_removed=True),
            provenance=FrameTimelineProvenance(stage="3D", label="frame_timeline", notes=None),
        )
        write_json_record(
            out_dir / "frame_timeline_receipt.json",
            receipt.to_dict(),
            contain_root=resolve_strict(root),
            overwrite=False,
        )
        return FrameTimelineServiceResult(
            accepted=True,
            exit_code=0,
            status=FrameTimelineStatus.SUCCEEDED,
            receipt=receipt,
            frames_parquet=str(frames_path),
            artifact_manifest=artifact_path,
        )
    except (FrameTimelineError, ProbeError, VideoSourceError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        msg = _sanitize_message(str(getattr(exc, "message", exc)))
        status = (
            FrameTimelineStatus.REJECTED
            if code
            in {
                "MATERIALIZE_FLAG_REQUIRED",
                "MATERIALIZE_MODE_REQUIRED",
                "SOURCE_HASH_MISMATCH",
                "POLICY",
            }
            or isinstance(exc, VideoSourceError)
            else FrameTimelineStatus.FAILED
        )
        exit_code = 1 if status == FrameTimelineStatus.REJECTED else 3
        receipt = _fail_receipt(
            receipt_id=receipt_id,
            run_id=run,
            video_id=vid,
            source_path=str(source),
            source_sha256=expected_source_sha256 or ("0" * 64),
            mode=mode,
            status=status,
            started=started,
            ffprobe_path=ff_path,
            ffprobe_version=ff_ver,
            errors=[Issue(code=str(code), message=msg)],
            warnings=warnings,
            normalization_receipt_path=norm_path,
            sample_every=sample if mode == FrameTimelineMode.SAMPLED else None,
        )
        with contextlib.suppress(Exception):
            out = Path(output_dir)
            if out.is_absolute() and out.is_dir():
                write_json_record(
                    out / "frame_timeline_receipt.json",
                    receipt.to_dict(),
                    contain_root=resolve_strict(root),
                    overwrite=True,
                )
        return FrameTimelineServiceResult(
            accepted=False,
            exit_code=exit_code,
            status=status,
            receipt=receipt,
            frames_parquet=None,
            artifact_manifest=None,
            error_code=str(code),
        )
    except Exception as exc:  # noqa: BLE001
        receipt = _fail_receipt(
            receipt_id=receipt_id,
            run_id=run,
            video_id=vid,
            source_path=str(source),
            source_sha256=expected_source_sha256 or ("0" * 64),
            mode=mode,
            status=FrameTimelineStatus.FAILED,
            started=started,
            ffprobe_path=ff_path,
            ffprobe_version=ff_ver,
            errors=[Issue(code="UNEXPECTED", message=_sanitize_message(str(exc)))],
            warnings=warnings,
            normalization_receipt_path=norm_path,
            sample_every=sample if mode == FrameTimelineMode.SAMPLED else None,
        )
        return FrameTimelineServiceResult(
            accepted=False,
            exit_code=3,
            status=FrameTimelineStatus.FAILED,
            receipt=receipt,
            frames_parquet=None,
            artifact_manifest=None,
            error_code="UNEXPECTED",
        )
