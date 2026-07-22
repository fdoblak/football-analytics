"""Synthetic fixture design helpers for Stage 3A (runtime-only, Git-excluded)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.video.types import (
    AudioStreamInfo,
    FrameCountSource,
    FrameRateMode,
    ProbeWarning,
    Rational,
    SourceKind,
    StreamDisposition,
    VideoContractError,
    VideoProbe,
    VideoStreamInfo,
    WarningSeverity,
    select_primary_video_stream,
)

RUNTIME_ROOT = Path("/home/fdoblak/workspace/video_contract_checks")
FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


@dataclass(frozen=True)
class FixtureScenario:
    name: str
    description: str
    kind: str  # media | metadata | negative
    requires_ffmpeg: bool = False


SCENARIOS: tuple[FixtureScenario, ...] = (
    FixtureScenario("cfr_tiny", "Small CFR H.264 video", "media", True),
    FixtureScenario("cfr_with_audio", "CFR video + synthetic audio", "media", True),
    FixtureScenario("rotation_metadata", "Synthetic probe with rotation=90", "metadata", False),
    FixtureScenario("vfr_metadata", "Synthetic VFR metadata fixture", "metadata", False),
    FixtureScenario("unknown_frame_count", "Null frame_count metadata", "metadata", False),
    FixtureScenario("unsupported_codec", "Unsupported codec negative metadata", "metadata", False),
    FixtureScenario("zero_byte", "Zero-byte invalid media negative", "negative", False),
    FixtureScenario("symlink_negative", "Symlink source negative", "negative", True),
    FixtureScenario("hash_mismatch", "Source mutation / hash mismatch negative", "negative", True),
)


def assert_runtime_root_owned() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/video_contract_checks"):
        raise VideoContractError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise VideoContractError("runtime root must not be symlink")
    return resolved


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_session_dir() -> Path:
    root = assert_runtime_root_owned()
    path = Path(tempfile.mkdtemp(prefix=f"stage3a_{utc_stamp()}_", dir=str(root)))
    if not str(path.resolve()).startswith(str(root)):
        raise VideoContractError("session dir escaped runtime root")
    return path


def ffmpeg_available() -> bool:
    return FFMPEG.is_file() and os.access(FFMPEG, os.X_OK)


def _run_ffmpeg(args: list[str]) -> None:
    if not ffmpeg_available():
        raise VideoContractError("ffmpeg unavailable")
    cmd = [str(FFMPEG), "-y", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoContractError(f"ffmpeg failed: {proc.stderr[-500:]}")


def generate_cfr_video(path: Path, *, with_audio: bool = False) -> Path:
    """Generate a tiny CFR MP4 under the session directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1:r=25",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    else:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    args.append(str(path))
    _run_ffmpeg(args)
    return path


def _disposition(*, attached: bool = False) -> StreamDisposition:
    return StreamDisposition(default=True, attached_pic=attached, forced=False)


def synthetic_video_stream(
    *,
    index: int = 0,
    width: int = 160,
    height: int = 120,
    rotation: int = 0,
    frame_rate_mode: FrameRateMode = FrameRateMode.CFR,
    frame_count: int | None = 25,
    frame_count_source: FrameCountSource = FrameCountSource.NB_FRAMES,
    duration_us: int | None = 1_000_000,
    start_pts: int | None = 0,
    r_num: int = 25,
    r_den: int = 1,
    avg_num: int = 25,
    avg_den: int = 1,
    attached_pic: bool = False,
    codec_name: str = "h264",
) -> VideoStreamInfo:
    return VideoStreamInfo(
        stream_index=index,
        codec_name=codec_name,
        codec_long_name="H.264",
        profile="High",
        pixel_format="yuv420p",
        width=width,
        height=height,
        coded_width=width,
        coded_height=height,
        sample_aspect_ratio=Rational(1, 1),
        display_aspect_ratio=Rational(width, height),
        rotation_degrees=rotation,
        time_base=Rational(1, 25_000),
        codec_time_base=Rational(1, 50),
        r_frame_rate=Rational(r_num, r_den),
        avg_frame_rate=Rational(avg_num, avg_den),
        nominal_frame_rate=Rational(r_num, r_den),
        frame_rate_mode=frame_rate_mode,
        start_pts=start_pts,
        duration_ts=None if duration_us is None else duration_us,
        duration_us=duration_us,
        frame_count=frame_count,
        frame_count_source=frame_count_source,
        bit_rate_bps=800_000,
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        field_order="progressive",
        disposition=_disposition(attached=attached_pic),
    )


def build_synthetic_probe(
    *,
    source_id: str,
    source_sha256: str,
    file_size_bytes: int,
    streams: tuple[VideoStreamInfo | AudioStreamInfo, ...],
    duration_us: int | None,
    start_time_us: int | None = 0,
    warnings: tuple[ProbeWarning, ...] = (),
    probed_at_utc: str = "2026-07-22T21:00:00Z",
) -> VideoProbe:
    selected = select_primary_video_stream(streams)
    audio_idx = next(
        (s.stream_index for s in streams if isinstance(s, AudioStreamInfo)),
        None,
    )
    return VideoProbe(
        source_id=source_id,
        source_sha256=source_sha256,
        probe_tool="synthetic_metadata",
        probe_tool_version="stage3a-1",
        probed_at_utc=probed_at_utc,
        container="mp4",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_us=duration_us,
        start_time_us=start_time_us,
        bit_rate_bps=900_000,
        file_size_bytes=file_size_bytes,
        streams=streams,
        selected_video_stream_index=selected,
        selected_audio_stream_index=audio_idx,
        warnings=warnings,
    )


def metadata_fixture(name: str, *, source_sha256: str) -> dict[str, Any]:
    """Return a synthetic probe dict for the named metadata scenario."""
    if name == "rotation_metadata":
        streams = (synthetic_video_stream(rotation=90),)
        probe = build_synthetic_probe(
            source_id="src_rotation",
            source_sha256=source_sha256,
            file_size_bytes=1024,
            streams=streams,
            duration_us=1_000_000,
        )
        return probe.to_dict()
    if name == "vfr_metadata":
        streams = (
            synthetic_video_stream(
                frame_rate_mode=FrameRateMode.VFR,
                r_num=30,
                r_den=1,
                avg_num=24000,
                avg_den=1001,
                frame_count=None,
                frame_count_source=FrameCountSource.UNKNOWN,
            ),
        )
        probe = build_synthetic_probe(
            source_id="src_vfr",
            source_sha256=source_sha256,
            file_size_bytes=2048,
            streams=streams,
            duration_us=2_000_000,
            warnings=(
                ProbeWarning(
                    code="vfr_detected",
                    message="avg_frame_rate differs from r_frame_rate; do not use index/fps",
                    severity=WarningSeverity.WARNING,
                ),
            ),
        )
        return probe.to_dict()
    if name == "unknown_frame_count":
        streams = (
            synthetic_video_stream(
                frame_count=None,
                frame_count_source=FrameCountSource.UNKNOWN,
            ),
        )
        probe = build_synthetic_probe(
            source_id="src_unk_frames",
            source_sha256=source_sha256,
            file_size_bytes=1024,
            streams=streams,
            duration_us=None,
        )
        return probe.to_dict()
    if name == "unsupported_codec":
        streams = (synthetic_video_stream(codec_name="bogus_codec"),)
        probe = build_synthetic_probe(
            source_id="src_bad_codec",
            source_sha256=source_sha256,
            file_size_bytes=512,
            streams=streams,
            duration_us=500_000,
            warnings=(
                ProbeWarning(
                    code="unsupported_codec",
                    message="codec not in policy allowlist",
                    severity=WarningSeverity.UNSUPPORTED,
                ),
            ),
        )
        return probe.to_dict()
    raise VideoContractError(f"unknown metadata fixture: {name}")


@dataclass
class FixtureSession:
    root: Path
    created_files: list[Path]
    cleanup_ok: bool = False

    def track(self, path: Path) -> Path:
        self.created_files.append(path)
        return path

    def cleanup(self) -> dict[str, Any]:
        remaining: list[str] = []
        # Remove tracked files then session dir
        for path in reversed(self.created_files):
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                remaining.append(str(path))
        try:
            if self.root.exists():
                shutil.rmtree(self.root)
        except OSError:
            remaining.append(str(self.root))
        self.cleanup_ok = not self.root.exists() and not remaining
        return {
            "cleanup_ok": self.cleanup_ok,
            "root": str(self.root),
            "remaining": remaining,
            "removed_count": len(self.created_files),
        }


def open_fixture_session() -> FixtureSession:
    return FixtureSession(root=make_session_dir(), created_files=[])


def hash_file(path: Path) -> tuple[int, str]:
    digest = sha256_file(path)
    return path.stat().st_size, digest


def fixture_kind_for_source(kind: SourceKind) -> str:
    return kind.value


def iter_scenarios() -> Iterator[FixtureScenario]:
    yield from SCENARIOS
