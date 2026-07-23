"""Streaming FFprobe frame timeline parse → Arrow batches (Stage 3D)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_analytics.video.ffprobe import (
    ProbeError,
    _sanitize_env,
    get_ffprobe_version,
    resolve_ffprobe_binary,
)
from football_analytics.video.time_mapping import (
    MappingStats,
    duration_ts_to_us,
    pts_to_video_time_us,
)
from football_analytics.video.types import Rational, VideoError


class FrameTimelineError(VideoError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RawFrameLine:
    pts: int | None
    dts: int | None
    duration_ts: int | None
    key_frame: bool | None
    pict_type: str | None


@dataclass
class MappedFrame:
    frame_index: int
    pts: int | None
    video_time_us: int
    duration_us: int | None
    is_key_frame: bool | None
    decode_status: str


@dataclass
class TimelineParseResult:
    stats: MappingStats = field(default_factory=MappingStats)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def build_ffprobe_frames_argv(
    binary: Path,
    source_path: Path,
    *,
    video_stream_index: int,
) -> list[str]:
    """Line-oriented compact frame dump for FFmpeg 4.4.2 (no full JSON load)."""
    if video_stream_index < 0:
        raise FrameTimelineError("INVALID_STREAM", "video_stream_index must be >= 0")
    src = source_path.as_posix()
    argv = [
        str(binary),
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        str(video_stream_index),
        "-show_frames",
        "-show_entries",
        "frame=pkt_pts,pkt_dts,pkt_duration,key_frame,pict_type",
        "-print_format",
        "compact=nk=1:p=0",
        "-protocol_whitelist",
        "file,crypto,data",
    ]
    if source_path.name.startswith("-"):
        argv.extend(["--", src])
    else:
        argv.append(src)
    return argv


def _parse_int_field(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_compact_frame_line(line: str) -> RawFrameLine | None:
    """Parse one compact nk=1 frame line; ignore side-data noise."""
    text = line.strip()
    if not text:
        return None
    # Compact nk=1: values separated by '|', order matches show_entries.
    parts = text.split("|")
    if len(parts) < 4:
        return None
    # Expected: key_frame|pkt_pts|pkt_dts|pkt_duration|pict_type...
    key_frame = _parse_int_field(parts[0])
    pts = _parse_int_field(parts[1])
    dts = _parse_int_field(parts[2])
    duration_ts = _parse_int_field(parts[3])
    pict_type: str | None = None
    if len(parts) >= 5 and parts[4]:
        ch = parts[4][0]
        if ch in {"I", "P", "B", "S", "i", "p", "b", "s"}:
            pict_type = ch.upper()
    return RawFrameLine(
        pts=pts,
        dts=dts,
        duration_ts=duration_ts,
        key_frame=None if key_frame is None else bool(key_frame),
        pict_type=pict_type,
    )


def map_raw_frame(
    raw: RawFrameLine,
    *,
    frame_index: int,
    time_base: Rational,
    prev_pts: int | None,
    prev_video_time_us: int | None,
    stats: MappingStats,
    warnings: list[tuple[str, str]],
    seen_pts: set[int],
) -> tuple[MappedFrame, int | None, int]:
    """Map one raw frame; never invent times from index/fps."""
    decode_status = "ok"
    pts = raw.pts
    video_time_us: int

    if pts is None:
        stats.missing_pts_count += 1
        decode_status = "skipped"
        warnings.append(("MISSING_PTS", f"frame_index={frame_index} missing pkt_pts"))
        video_time_us = 0 if prev_video_time_us is None else prev_video_time_us
    else:
        if pts in seen_pts:
            stats.duplicate_pts_count += 1
            warnings.append(("DUPLICATE_PTS", f"frame_index={frame_index} pts={pts}"))
        else:
            seen_pts.add(pts)
        if prev_pts is not None and pts < prev_pts:
            stats.non_monotonic_pts_count += 1
            decode_status = "unknown"
            warnings.append(
                ("NON_MONOTONIC_PTS", f"frame_index={frame_index} pts={pts} prev={prev_pts}")
            )
            # Do not invent a new timeline from fps; carry prior mapped time.
            video_time_us = prev_video_time_us if prev_video_time_us is not None else 0
        else:
            try:
                video_time_us = pts_to_video_time_us(pts, time_base)
            except Exception as exc:  # noqa: BLE001
                decode_status = "failed"
                warnings.append(("PTS_MAP_FAILED", f"frame_index={frame_index}: {exc}"))
                video_time_us = prev_video_time_us if prev_video_time_us is not None else 0

        # Keep video_time_us non-decreasing for frames contract monotonic rule.
        if prev_video_time_us is not None and video_time_us < prev_video_time_us:
            video_time_us = prev_video_time_us
            if decode_status == "ok":
                decode_status = "unknown"

    duration_us = duration_ts_to_us(raw.duration_ts, time_base)
    stats.frame_count += 1
    stats.note_status(decode_status)
    mapped = MappedFrame(
        frame_index=frame_index,
        pts=pts,
        video_time_us=video_time_us,
        duration_us=duration_us,
        is_key_frame=raw.key_frame,
        decode_status=decode_status,
    )
    next_prev_pts = pts if pts is not None else prev_pts
    return mapped, next_prev_pts, video_time_us


def iter_mapped_frames_from_lines(
    lines: Iterator[str],
    *,
    time_base: Rational,
    maximum_frames: int,
) -> tuple[Iterator[MappedFrame], TimelineParseResult]:
    """Lazy map lines → MappedFrame; stats/warnings mutate during iteration."""
    result = TimelineParseResult()
    seen_pts: set[int] = set()

    def _gen() -> Iterator[MappedFrame]:
        prev_pts: int | None = None
        prev_t: int | None = None
        idx = 0
        for line in lines:
            raw = parse_compact_frame_line(line)
            if raw is None:
                continue
            if idx >= maximum_frames:
                raise FrameTimelineError("FRAME_LIMIT_EXCEEDED", "maximum_frames exceeded")
            mapped, prev_pts, prev_t = map_raw_frame(
                raw,
                frame_index=idx,
                time_base=time_base,
                prev_pts=prev_pts,
                prev_video_time_us=prev_t,
                stats=result.stats,
                warnings=result.warnings,
                seen_pts=seen_pts,
            )
            idx += 1
            yield mapped

    return _gen(), result


def iter_ffprobe_frame_lines(
    source_path: Path,
    *,
    policy: Mapping[str, Any],
    video_stream_index: int,
    binary: Path | None = None,
    timeout_seconds: float | None = None,
) -> Iterator[str]:
    """Stream ffprobe compact frame lines; bounded line size; process-group cleanup."""
    ftp = policy["frame_timeline_policy"]
    bin_path = binary or resolve_ffprobe_binary(
        ftp["ffprobe_binary"], allowed_realpaths=list(ftp["allowed_binary_realpaths"])
    )
    get_ffprobe_version(bin_path)
    if not source_path.is_absolute():
        raise FrameTimelineError("SOURCE_NOT_ABSOLUTE", "source path must be absolute")
    argv = build_ffprobe_frames_argv(bin_path, source_path, video_stream_index=video_stream_index)
    max_line = int(ftp["maximum_line_bytes"])
    max_err = int(ftp["maximum_stderr_bytes"])
    timeout = float(
        timeout_seconds if timeout_seconds is not None else ftp["maximum_timeout_seconds"]
    )
    try:
        proc = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sanitize_env(),
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise FrameTimelineError("FFPROBE_NOT_AVAILABLE", "ffprobe binary missing") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None
    stderr_buf = bytearray()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FrameTimelineError("PROBE_TIMEOUT", "ffprobe frames timed out")
            line = proc.stdout.readline()
            if not line:
                break
            if len(line) > max_line:
                raise FrameTimelineError("LINE_TOO_LONG", "ffprobe frame line too long")
            # Drain some stderr without loading unbounded
            while True:
                if len(stderr_buf) >= max_err:
                    raise FrameTimelineError(
                        "PROBE_OUTPUT_LIMIT_EXCEEDED", "ffprobe stderr exceeded limits"
                    )
                # Non-blocking-ish: peek via poll + read if available is hard in pure
                # Python; we accumulate only after process ends for simplicity bound.
                break
            yield line.decode("utf-8", errors="replace")
        # Finish process
        try:
            remaining = max(0.1, deadline - time.monotonic())
            _, err = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise FrameTimelineError("PROBE_TIMEOUT", "ffprobe frames timed out") from exc
        if err:
            stderr_buf.extend(err[: max(0, max_err - len(stderr_buf))])
            if len(err) > max_err:
                raise FrameTimelineError(
                    "PROBE_OUTPUT_LIMIT_EXCEEDED", "ffprobe stderr exceeded limits"
                )
        if proc.returncode != 0:
            raise FrameTimelineError(
                "PROBE_PROCESS_FAILED", f"ffprobe frames exit {proc.returncode}"
            )
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)


def mapped_frames_to_record_batches(
    frames: Iterator[MappedFrame],
    *,
    run_id: str,
    video_id: str,
    batch_size: int,
    arrow_schema: Any,
) -> Iterator[Any]:
    """Yield Arrow tables/batches sized by batch_size (no full pylist of all frames)."""
    import pyarrow as pa

    if batch_size < 1:
        raise FrameTimelineError("INVALID_BATCH", "batch_size must be >= 1")

    buf: list[dict[str, Any]] = []

    def _flush() -> Any:
        nonlocal buf
        table = pa.Table.from_pylist(buf, schema=arrow_schema)
        buf = []
        return table

    for fr in frames:
        buf.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": fr.frame_index,
                "pts": fr.pts,
                "video_time_us": fr.video_time_us,
                "duration_us": fr.duration_us,
                "is_key_frame": fr.is_key_frame,
                "decode_status": fr.decode_status,
            }
        )
        if len(buf) >= batch_size:
            yield _flush()
    if buf:
        yield _flush()


# Re-export ProbeError name for callers that need shared codes
__all__ = [
    "FrameTimelineError",
    "RawFrameLine",
    "MappedFrame",
    "TimelineParseResult",
    "build_ffprobe_frames_argv",
    "parse_compact_frame_line",
    "map_raw_frame",
    "iter_mapped_frames_from_lines",
    "iter_ffprobe_frame_lines",
    "mapped_frames_to_record_batches",
    "ProbeError",
]
