"""Safe FFprobe subprocess runner (Stage 3B). No shell; bounded capture; timeout."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_analytics.video.types import VideoError

DEFAULT_FFPROBE = Path("/usr/bin/ffprobe")


class ProbeError(VideoError):
    """Base FFprobe execution error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FfprobeVersion:
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
class FfprobeRawResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    version: FfprobeVersion

    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    def stderr_text(self) -> str:
        # Bounded / sanitized for callers — do not dump full paths repeatedly
        text = self.stderr.decode("utf-8", errors="replace")
        if len(text) > 2000:
            return text[:2000] + "…[truncated]"
        return text


def _sanitize_env() -> dict[str, str]:
    keep = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    # Prefer a minimal PATH that still finds nothing unexpected for our absolute binary
    env.setdefault("PATH", "/usr/bin:/bin")
    env["LANG"] = env.get("LANG", "C")
    return env


def resolve_ffprobe_binary(
    configured: str | Path,
    *,
    allowed_realpaths: Sequence[str],
) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe path must be absolute")
    if path.as_posix() != "/usr/bin/ffprobe":
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe must be /usr/bin/ffprobe")
    if not path.exists():
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe binary missing")
    if path.is_symlink():
        # Allow system symlink only if realpath is allowlisted
        pass
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "cannot stat ffprobe") from exc
    if not (stat.S_ISREG(mode) or path.is_symlink()):
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe is not a regular file")
    if not os.access(path, os.X_OK):
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe not executable")
    real = path.resolve()
    allowed = {str(Path(p).resolve()) for p in allowed_realpaths}
    if str(real) not in allowed:
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe realpath not allowlisted")
    # Reject binaries under the project tree
    if "/projects/football-analytics" in str(real):
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe must not live in project tree")
    return path


def parse_ffprobe_version_output(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("ffprobe version"):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
            return line
    raise ProbeError("FFPROBE_NOT_AVAILABLE", "unable to parse ffprobe -version")


def get_ffprobe_version(
    binary: Path,
    *,
    timeout_seconds: float = 5.0,
) -> FfprobeVersion:
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
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe binary missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError("PROBE_TIMEOUT", "ffprobe -version timed out") from exc
    if proc.returncode != 0:
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe -version failed")
    out = proc.stdout.decode("utf-8", errors="replace")
    token = parse_ffprobe_version_output(out)
    version_line = out.splitlines()[0].strip() if out.splitlines() else token
    return FfprobeVersion(
        path=str(binary),
        realpath=str(binary.resolve()),
        version_line=version_line,
        version_token=token,
    )


def build_ffprobe_argv(binary: Path, source_path: Path, *, count_frames: bool = False) -> list[str]:
    if count_frames:
        raise ProbeError("PROBE_UNEXPECTED_STRUCTURE", "count_frames must remain false in Stage 3B")
    # Absolute source path; prefix ./ if name starts with '-' to avoid option injection
    src = source_path.as_posix()
    if source_path.name.startswith("-"):
        # Keep absolute path but place after --
        src_arg = src
        return [
            str(binary),
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-protocol_whitelist",
            "file,crypto,data",
            "--",
            src_arg,
        ]
    return [
        str(binary),
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-protocol_whitelist",
        "file,crypto,data",
        src,
    ]


def _bounded_communicate(
    proc: subprocess.Popen[bytes],
    *,
    max_stdout: int,
    max_stderr: int,
    timeout: float,
) -> tuple[bytes, bytes, bool, bool]:
    """Communicate with hard byte caps; kill on timeout or overflow."""
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


def run_ffprobe(
    source_path: Path,
    *,
    policy: Mapping[str, Any],
    binary: Path | None = None,
    version: FfprobeVersion | None = None,
) -> FfprobeRawResult:
    """Run FFprobe against a validated absolute local path (caller must pre-validate)."""
    ff = policy["ffprobe_policy"]
    bin_path = binary or resolve_ffprobe_binary(
        ff["ffprobe_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
    )
    ver = version or get_ffprobe_version(
        bin_path, timeout_seconds=min(5.0, float(ff["probe_timeout_seconds"]))
    )
    if not source_path.is_absolute():
        raise ProbeError("SOURCE_NOT_REGULAR_FILE", "source path must be absolute")
    argv = build_ffprobe_argv(
        bin_path, source_path, count_frames=bool(ff.get("count_frames", False))
    )
    timeout = float(ff["probe_timeout_seconds"])
    max_out = int(ff["maximum_stdout_bytes"])
    max_err = int(ff["maximum_stderr_bytes"])
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
        raise ProbeError("FFPROBE_NOT_AVAILABLE", "ffprobe binary missing") from exc
    stdout, stderr, timed_out, overflow = _bounded_communicate(
        proc, max_stdout=max_out, max_stderr=max_err, timeout=timeout
    )
    # Ensure no zombie process group
    if proc.poll() is None:
        import contextlib

        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    if timed_out:
        raise ProbeError("PROBE_TIMEOUT", "ffprobe timed out")
    if overflow:
        raise ProbeError("PROBE_OUTPUT_LIMIT_EXCEEDED", "ffprobe output exceeded limits")
    if proc.returncode != 0:
        raise ProbeError("PROBE_PROCESS_FAILED", f"ffprobe exit {proc.returncode}")
    return FfprobeRawResult(
        argv=tuple(argv),
        returncode=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        version=ver,
    )


def decode_ffprobe_json(stdout: bytes, *, max_depth: int) -> dict[str, Any]:
    if not stdout:
        raise ProbeError("PROBE_INVALID_JSON", "empty ffprobe stdout")
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError("PROBE_INVALID_JSON", "ffprobe stdout not utf-8") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProbeError("PROBE_INVALID_JSON", "malformed ffprobe JSON") from exc
    if not isinstance(data, dict):
        raise ProbeError("PROBE_UNEXPECTED_STRUCTURE", "ffprobe JSON root must be object")
    _assert_max_depth(data, max_depth=max_depth, depth=0)
    return data


def _assert_max_depth(value: Any, *, max_depth: int, depth: int) -> None:
    if depth > max_depth:
        raise ProbeError("PROBE_UNEXPECTED_STRUCTURE", "ffprobe JSON too deep")
    if isinstance(value, dict):
        for v in value.values():
            _assert_max_depth(v, max_depth=max_depth, depth=depth + 1)
    elif isinstance(value, list):
        for v in value:
            _assert_max_depth(v, max_depth=max_depth, depth=depth + 1)
