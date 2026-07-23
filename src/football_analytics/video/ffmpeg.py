"""Safe FFmpeg subprocess runner (Stage 3C). No shell; bounded capture; timeout."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_analytics.video.types import VideoError

DEFAULT_FFMPEG = Path("/usr/bin/ffmpeg")


class FfmpegError(VideoError):
    """Base FFmpeg execution error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FfmpegVersion:
    path: str
    realpath: str
    version_line: str
    version_token: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "realpath": self.realpath,
            "version_line": self.version_line,
            "version_token": self.version_token,
        }


@dataclass(frozen=True)
class FfmpegRawResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    version: FfmpegVersion

    def stderr_text(self) -> str:
        text = self.stderr.decode("utf-8", errors="replace")
        if len(text) > 2000:
            return text[:2000] + "…[truncated]"
        return text


def _sanitize_env() -> dict[str, str]:
    keep = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["LANG"] = env.get("LANG", "C")
    return env


def resolve_ffmpeg_binary(
    configured: str | Path,
    *,
    allowed_realpaths: Sequence[str],
) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg path must be absolute")
    if path.as_posix() != "/usr/bin/ffmpeg":
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg must be /usr/bin/ffmpeg")
    if not path.exists():
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg binary missing")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "cannot stat ffmpeg") from exc
    if not (stat.S_ISREG(mode) or path.is_symlink()):
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg is not a regular file")
    if not os.access(path, os.X_OK):
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg not executable")
    real = path.resolve()
    allowed = {str(Path(p).resolve()) for p in allowed_realpaths}
    if str(real) not in allowed:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg realpath not allowlisted")
    if "/projects/football-analytics" in str(real):
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg must not live in project tree")
    return path


def parse_ffmpeg_version_output(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("ffmpeg version"):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
            return line
    raise FfmpegError("FFMPEG_NOT_AVAILABLE", "unable to parse ffmpeg -version")


def get_ffmpeg_version(
    binary: Path,
    *,
    timeout_seconds: float = 5.0,
) -> FfmpegVersion:
    argv = [str(binary), "-version"]
    try:
        proc = subprocess.run(
            argv,
            check=False,
            shell=False,
            capture_output=True,
            timeout=timeout_seconds,
            env=_sanitize_env(),
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg binary missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError("FFMPEG_TIMEOUT", "ffmpeg -version timed out") from exc
    if proc.returncode != 0:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg -version failed")
    out = proc.stdout.decode("utf-8", errors="replace")
    token = parse_ffmpeg_version_output(out)
    version_line = out.splitlines()[0].strip() if out.splitlines() else token
    return FfmpegVersion(
        path=str(binary),
        realpath=str(binary.resolve()),
        version_line=version_line,
        version_token=token,
    )


def assert_libx264_available(
    binary: Path,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    argv = [str(binary), "-hide_banner", "-encoders"]
    try:
        proc = subprocess.run(
            argv,
            check=False,
            shell=False,
            capture_output=True,
            timeout=timeout_seconds,
            env=_sanitize_env(),
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg binary missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError("FFMPEG_TIMEOUT", "ffmpeg -encoders timed out") from exc
    text = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    if "libx264" not in text:
        raise FfmpegError("FFMPEG_ENCODER_MISSING", "libx264 encoder not available")


def _build_video_filters(
    *,
    rotation_degrees: int,
    bake_rotation: bool,
    target_width: int | None,
    target_height: int | None,
    resize: bool,
    force_setsar: bool,
) -> list[str]:
    filters: list[str] = []
    if bake_rotation and rotation_degrees != 0:
        if rotation_degrees == 90:
            filters.append("transpose=1")
        elif rotation_degrees == 270:
            filters.append("transpose=2")
        elif rotation_degrees == 180:
            filters.append("hflip")
            filters.append("vflip")
        else:
            raise FfmpegError("FFMPEG_INVALID_PLAN", f"unsupported rotation {rotation_degrees}")
    if resize and target_width is not None and target_height is not None:
        filters.append(f"scale={int(target_width)}:{int(target_height)}")
    if force_setsar:
        filters.append("setsar=1")
    return filters


def build_normalize_argv(
    binary: Path,
    *,
    source_path: Path,
    temp_output: Path,
    video_stream_ordinal: int,
    audio_stream_ordinal: int | None,
    audio_action: str,
    target_pixel_format: str,
    video_preset: str,
    video_crf: int,
    ffmpeg_threads: int,
    movflags_faststart: bool,
    rotation_degrees: int,
    bake_rotation: bool,
    target_width: int | None,
    target_height: int | None,
    resize: bool,
    force_setsar: bool,
    frame_rate_conversion: bool,
    target_frame_rate_num: int | None,
    target_frame_rate_den: int | None,
    target_audio_codec: str = "aac",
    target_audio_sample_rate_hz: int = 48000,
    target_audio_channels: int = 2,
) -> list[str]:
    """Build fixed argv from plan fields. No user filter/extra args."""
    if "-y" in (str(binary),):
        raise FfmpegError("FFMPEG_INVALID_PLAN", "overwrite flag forbidden")
    filters = _build_video_filters(
        rotation_degrees=rotation_degrees,
        bake_rotation=bake_rotation,
        target_width=target_width,
        target_height=target_height,
        resize=resize,
        force_setsar=force_setsar,
    )
    argv: list[str] = [
        str(binary),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
    ]
    # Paths that could look like options go after --
    src = source_path.as_posix()
    if source_path.name.startswith("-"):
        argv.append("--")
    argv.append(src)
    argv += ["-map", f"0:v:{int(video_stream_ordinal)}"]
    if audio_action in {"copy", "transcode"} and audio_stream_ordinal is not None:
        argv += ["-map", f"0:a:{int(audio_stream_ordinal)}"]
    if filters:
        argv += ["-vf", ",".join(filters)]
    argv += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        str(target_pixel_format),
        "-preset",
        str(video_preset),
        "-crf",
        str(int(video_crf)),
        "-threads",
        str(int(ffmpeg_threads)),
    ]
    if frame_rate_conversion:
        if target_frame_rate_num is None or target_frame_rate_den is None:
            raise FfmpegError("FFMPEG_INVALID_PLAN", "frame rate conversion missing target")
        argv += [
            "-r",
            f"{int(target_frame_rate_num)}/{int(target_frame_rate_den)}",
            "-vsync",
            "cfr",
        ]
    if audio_action == "copy":
        argv += ["-c:a", "copy"]
    elif audio_action == "transcode":
        argv += [
            "-c:a",
            str(target_audio_codec),
            "-ar",
            str(int(target_audio_sample_rate_hz)),
            "-ac",
            str(int(target_audio_channels)),
        ]
    elif audio_action in {"drop", "absent", "none"}:
        argv += ["-an"]
    else:
        raise FfmpegError("FFMPEG_INVALID_PLAN", f"unknown audio_action {audio_action}")
    if movflags_faststart:
        argv += ["-movflags", "+faststart"]
    out = temp_output.as_posix()
    if temp_output.name.startswith("-"):
        argv.append("--")
    argv.append(out)
    if "-y" in argv:
        raise FfmpegError("FFMPEG_INVALID_PLAN", "-y overwrite forbidden")
    return argv


def build_normalize_argv_from_plan(
    binary: Path,
    *,
    source_path: Path,
    temp_output: Path,
    planned: Any,
    policy: Mapping[str, Any],
) -> list[str]:
    """Convenience wrapper using PlannedNormalization + policy."""
    ff = policy["ffmpeg_policy"]
    nd = policy["normalization_defaults"]
    plan = planned.plan
    fr = plan.target_frame_rate
    return build_normalize_argv(
        binary,
        source_path=source_path,
        temp_output=temp_output,
        video_stream_ordinal=planned.video_stream_ordinal,
        audio_stream_ordinal=planned.audio_stream_ordinal,
        audio_action=planned.audio_action,
        target_pixel_format=plan.target_pixel_format,
        video_preset=str(ff["video_preset"]),
        video_crf=int(ff["video_crf"]),
        ffmpeg_threads=int(ff["ffmpeg_threads"]),
        movflags_faststart=bool(nd.get("movflags_faststart", True)),
        rotation_degrees=planned.source_rotation_degrees,
        bake_rotation=planned.bake_rotation,
        target_width=plan.target_width,
        target_height=plan.target_height,
        resize=planned.resize_performed,
        force_setsar=planned.force_setsar,
        frame_rate_conversion=planned.frame_rate_conversion,
        target_frame_rate_num=None if fr is None else fr.numerator,
        target_frame_rate_den=None if fr is None else fr.denominator,
        target_audio_codec=str(nd.get("target_audio_codec", "aac")),
        target_audio_sample_rate_hz=int(nd.get("target_audio_sample_rate_hz", 48000)),
        target_audio_channels=int(nd.get("target_audio_channels", 2)),
    )


def _bounded_communicate(
    proc: subprocess.Popen[bytes],
    *,
    max_stdout: int,
    max_stderr: int,
    timeout: float,
) -> tuple[bytes, bytes, bool, bool]:
    timed_out = False
    overflow = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
    if len(stdout) > max_stdout or len(stderr) > max_stderr:
        overflow = True
        stdout = stdout[:max_stdout]
        stderr = stderr[:max_stderr]
    return stdout, stderr, timed_out, overflow


def run_ffmpeg_normalize(
    source: Path,
    temp_output: Path,
    argv_or_plan: list[str] | Any,
    policy: Mapping[str, Any],
    *,
    binary: Path | None = None,
    version: FfmpegVersion | None = None,
    timeout_seconds: float | None = None,
) -> FfmpegRawResult:
    """Run FFmpeg normalize into a non-existent temp_output (never -y)."""
    ff = policy["ffmpeg_policy"]
    bin_path = binary or resolve_ffmpeg_binary(
        ff["ffmpeg_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
    )
    ver = version or get_ffmpeg_version(bin_path)
    if not source.is_absolute():
        raise FfmpegError("SOURCE_NOT_REGULAR_FILE", "source path must be absolute")
    if not temp_output.is_absolute():
        raise FfmpegError("OUTPUT_PATH_INVALID", "temp output must be absolute")
    if temp_output.exists():
        raise FfmpegError("TEMP_OUTPUT_EXISTS", "temp output must not exist")
    if isinstance(argv_or_plan, list):
        argv = list(argv_or_plan)
    else:
        argv = build_normalize_argv_from_plan(
            bin_path,
            source_path=source,
            temp_output=temp_output,
            planned=argv_or_plan,
            policy=policy,
        )
    if "-y" in argv:
        raise FfmpegError("FFMPEG_INVALID_PLAN", "-y overwrite forbidden")
    timeout = float(
        timeout_seconds if timeout_seconds is not None else float(ff["maximum_timeout_seconds"])
    )
    max_err = int(ff["maximum_stderr_bytes"])
    max_prog = int(ff["maximum_progress_bytes"])
    max_out = max(max_prog, 65536)
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
        raise FfmpegError("FFMPEG_NOT_AVAILABLE", "ffmpeg binary missing") from exc
    stdout, stderr, timed_out, overflow = _bounded_communicate(
        proc, max_stdout=max_out, max_stderr=max_err, timeout=timeout
    )
    if proc.poll() is None:
        import contextlib

        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    if timed_out:
        raise FfmpegError("FFMPEG_TIMEOUT", "ffmpeg timed out")
    if overflow:
        raise FfmpegError("FFMPEG_OUTPUT_LIMIT_EXCEEDED", "ffmpeg output exceeded limits")
    if proc.returncode != 0:
        raise FfmpegError("FFMPEG_PROCESS_FAILED", f"ffmpeg exit {proc.returncode}")
    if not temp_output.is_file():
        raise FfmpegError("FFMPEG_PROCESS_FAILED", "ffmpeg produced no output file")
    return FfmpegRawResult(
        argv=tuple(argv),
        returncode=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        version=ver,
    )
