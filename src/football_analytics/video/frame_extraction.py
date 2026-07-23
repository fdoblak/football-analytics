"""Optional frame image materialization + JSONL manifest (Stage 3D)."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.video.ffmpeg import (
    FfmpegError,
    _sanitize_env,
    get_ffmpeg_version,
    resolve_ffmpeg_binary,
)
from football_analytics.video.frame_timeline import FrameTimelineError
from football_analytics.video.types import FrameTimelineMode, VideoSourceError
from football_analytics.video.validation import reject_unsafe_path_string, require_absolute_path


@dataclass(frozen=True)
class MaterializeResult:
    manifest_path: Path
    rows_path: Path
    frames_dir: Path
    row_count: int
    image_paths: tuple[str, ...]


def _image_ext(fmt: str) -> str:
    if fmt == "jpeg":
        return "jpg"
    return fmt


def select_materialize_indices(
    frame_count: int,
    *,
    mode: FrameTimelineMode,
    sample_every: int,
) -> list[int]:
    if frame_count < 0:
        raise FrameTimelineError("INVALID_FRAME_COUNT", "frame_count must be >= 0")
    if mode == FrameTimelineMode.TIMELINE_ONLY:
        return []
    if mode == FrameTimelineMode.ALL_FRAMES:
        return list(range(frame_count))
    if mode == FrameTimelineMode.SAMPLED:
        if sample_every < 1:
            raise FrameTimelineError("INVALID_SAMPLE", "sample_every must be >= 1")
        return list(range(0, frame_count, sample_every))
    raise FrameTimelineError("INVALID_MODE", f"unsupported mode {mode}")


def build_ffmpeg_extract_argv(
    binary: Path,
    source_path: Path,
    output_pattern: Path,
    *,
    mode: FrameTimelineMode,
    sample_every: int,
    image_format: str,
) -> list[str]:
    """Extract frames. Sampled uses select+mod; all_frames dumps decode order."""
    ext = _image_ext(image_format)
    if output_pattern.suffix.lower().lstrip(".") not in {ext, "jpg", "jpeg", "png"}:
        raise FrameTimelineError("INVALID_PATTERN", "output pattern extension mismatch")
    src = source_path.as_posix()
    argv = [
        str(binary),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,crypto,data",
        "-i",
        src,
    ]
    if mode == FrameTimelineMode.SAMPLED:
        argv.extend(["-vf", f"select='not(mod(n\\,{int(sample_every)}))'", "-vsync", "vfr"])
    elif mode == FrameTimelineMode.ALL_FRAMES:
        argv.extend(["-vsync", "vfr"])
    else:
        raise FrameTimelineError("INVALID_MODE", "timeline_only cannot extract")
    argv.append(str(output_pattern))
    return argv


def run_ffmpeg_extract(
    source_path: Path,
    output_pattern: Path,
    *,
    policy: Mapping[str, Any],
    mode: FrameTimelineMode,
    sample_every: int,
    expected_count: int,
    timeout_seconds: float,
) -> tuple[str, ...]:
    ftp = policy["frame_timeline_policy"]
    binary = resolve_ffmpeg_binary(
        ftp["ffmpeg_binary"], allowed_realpaths=list(ftp["allowed_ffmpeg_realpaths"])
    )
    get_ffmpeg_version(binary)
    fmt = str(ftp["materialize_image_format"])
    argv = build_ffmpeg_extract_argv(
        binary,
        source_path,
        output_pattern,
        mode=mode,
        sample_every=sample_every,
        image_format=fmt,
    )
    max_err = int(ftp["maximum_stderr_bytes"])
    try:
        proc = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_sanitize_env(),
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg binary missing") from exc
    try:
        _, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        raise FfmpegError("FFMPEG_TIMEOUT", "ffmpeg extract timed out") from exc
    if stderr and len(stderr) > max_err:
        raise FfmpegError("FFMPEG_OUTPUT_LIMIT", "ffmpeg stderr exceeded limits")
    if proc.returncode != 0:
        msg = (stderr or b"").decode("utf-8", errors="replace")[:500]
        raise FfmpegError("FFMPEG_FAILED", f"ffmpeg extract exit {proc.returncode}: {msg}")

    parent = output_pattern.parent
    stem_fmt = output_pattern.name
    if "%06d" not in stem_fmt:
        raise FfmpegError("FFMPEG_PATTERN", "expected %06d output pattern")
    paths: list[str] = []
    for seq in range(1, expected_count + 1):
        name = stem_fmt.replace("%06d", f"{seq:06d}")
        path = parent / name
        if not path.is_file():
            raise FfmpegError("FFMPEG_MISSING_FRAME", f"missing output frame {name}")
        paths.append(str(path))
    # Reject unexpected extras
    extra = parent / stem_fmt.replace("%06d", f"{expected_count + 1:06d}")
    if extra.exists():
        raise FfmpegError("FFMPEG_EXTRA_FRAME", "unexpected extra extracted frame")
    return tuple(paths)


def write_artifact_manifest(
    *,
    output_dir: Path,
    contain_root: Path,
    manifest_id: str,
    run_id: str,
    video_id: str,
    source_sha256: str,
    mode: FrameTimelineMode,
    sample_every: int | None,
    image_format: str,
    frames_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    created_at_utc: str,
) -> tuple[Path, Path]:
    if mode == FrameTimelineMode.TIMELINE_ONLY:
        raise FrameTimelineError("INVALID_MODE", "timeline_only has no artifact manifest")
    reject_unsafe_path_string(str(output_dir), label="output_dir")
    rows_path = output_dir / "frame_artifacts.jsonl"
    header_path = output_dir / "frame_artifact_manifest.json"
    if rows_path.exists() or header_path.exists():
        raise VideoSourceError("artifact manifest outputs already exist")

    parent = output_dir
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = parent / f".frame_artifacts.{os.getpid()}.jsonl.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(rows_path))
        with contextlib.suppress(OSError):
            os.chmod(rows_path, 0o600)
    except Exception:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise

    header = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "run_id": run_id,
        "video_id": video_id,
        "source_sha256": source_sha256,
        "mode": mode.value,
        "sample_every": sample_every,
        "image_format": image_format,
        "frames_dir": str(frames_dir),
        "rows_path": str(rows_path),
        "row_count": len(rows),
        "created_at_utc": created_at_utc,
        "provenance": {
            "stage": "3D",
            "label": "frame_artifact_manifest",
            "notes": None,
        },
    }
    write_json_record(header_path, header, contain_root=contain_root, overwrite=False)
    return header_path, rows_path


def materialize_frames(
    *,
    source: Path,
    output_dir: Path,
    contain_root: Path,
    policy: Mapping[str, Any],
    mode: FrameTimelineMode,
    sample_every: int,
    frame_meta: Sequence[Mapping[str, Any]],
    run_id: str,
    video_id: str,
    source_sha256: str,
    manifest_id: str,
    created_at_utc: str,
    timeout_seconds: float,
) -> MaterializeResult:
    """Materialize images for selected frame indices; metadata comes from parquet rows."""
    ftp = policy["frame_timeline_policy"]
    if mode == FrameTimelineMode.TIMELINE_ONLY:
        raise FrameTimelineError("INVALID_MODE", "timeline_only does not materialize")
    fmt = str(ftp["materialize_image_format"])
    indices = select_materialize_indices(len(frame_meta), mode=mode, sample_every=sample_every)
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        raise VideoSourceError("frames directory already exists")
    frames_dir.mkdir(parents=True, mode=0o700)
    ext = _image_ext(fmt)
    pattern = frames_dir / f"frame_%06d.{ext}"
    if not indices:
        header, rows_path = write_artifact_manifest(
            output_dir=output_dir,
            contain_root=contain_root,
            manifest_id=manifest_id,
            run_id=run_id,
            video_id=video_id,
            source_sha256=source_sha256,
            mode=mode,
            sample_every=sample_every if mode == FrameTimelineMode.SAMPLED else None,
            image_format=fmt,
            frames_dir=frames_dir,
            rows=[],
            created_at_utc=created_at_utc,
        )
        return MaterializeResult(
            manifest_path=header,
            rows_path=rows_path,
            frames_dir=frames_dir,
            row_count=0,
            image_paths=(),
        )

    paths = run_ffmpeg_extract(
        source,
        pattern,
        policy=policy,
        mode=mode,
        sample_every=sample_every,
        expected_count=len(indices),
        timeout_seconds=timeout_seconds,
    )
    by_index = {int(r["frame_index"]): r for r in frame_meta}
    rows: list[dict[str, Any]] = []
    for seq, frame_index in enumerate(indices):
        fr = by_index[frame_index]
        img = paths[seq]
        rows.append(
            {
                "frame_index": frame_index,
                "pts": fr.get("pts"),
                "video_time_us": fr["video_time_us"],
                "decode_status": fr["decode_status"],
                "path": img,
                "sha256": sha256_file(img),
            }
        )
    header, rows_path = write_artifact_manifest(
        output_dir=output_dir,
        contain_root=contain_root,
        manifest_id=manifest_id,
        run_id=run_id,
        video_id=video_id,
        source_sha256=source_sha256,
        mode=mode,
        sample_every=sample_every if mode == FrameTimelineMode.SAMPLED else None,
        image_format=fmt,
        frames_dir=frames_dir,
        rows=rows,
        created_at_utc=created_at_utc,
    )
    return MaterializeResult(
        manifest_path=header,
        rows_path=rows_path,
        frames_dir=frames_dir,
        row_count=len(rows),
        image_paths=paths,
    )


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    require_absolute_path(str(path), label="jsonl")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise FrameTimelineError("INVALID_JSONL", "jsonl row must be object")
            yield obj
