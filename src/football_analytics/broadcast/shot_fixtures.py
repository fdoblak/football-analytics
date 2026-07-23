"""Synthetic FFmpeg lavfi fixtures for shot-boundary baseline (runtime only)."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file

RUNTIME_ROOT = Path("/home/fdoblak/workspace/shot_boundary_checks")
FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


class ShotFixtureError(ValueError):
    """Fixture generation failure."""


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    split: str  # development | evaluation | negative
    description: str
    duration_us: int
    fps: int
    ground_truth: tuple[dict[str, Any], ...]


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/shot_boundary_checks"):
        raise ShotFixtureError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise ShotFixtureError("runtime root must not be a symlink")
    return resolved


def ffmpeg_available() -> bool:
    return FFMPEG.is_file() and os.access(FFMPEG, os.X_OK)


def xfade_available() -> bool:
    if not ffmpeg_available():
        return False
    proc = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "xfade" in (proc.stdout or "")


def _run_ffmpeg(args: list[str]) -> None:
    if not ffmpeg_available():
        raise ShotFixtureError("ffmpeg unavailable")
    cmd = [str(FFMPEG), "-y", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ShotFixtureError(f"ffmpeg failed: {(proc.stderr or '')[-800:]}")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def generate_hard_cut(path: Path, *, fps: int = 25, seconds_each: float = 0.6) -> FixtureSpec:
    """Black then white hard cut at known time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cut_us = int(round(seconds_each * 1_000_000))
    duration_us = cut_us * 2
    # Two segments concatenated via filter_complex
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={seconds_each}:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s=320x180:d={seconds_each}:r={fps}",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return FixtureSpec(
        name="hard_cut",
        split="evaluation",
        description="Black to white hard cut",
        duration_us=duration_us,
        fps=fps,
        ground_truth=(
            {
                "boundary_id": "gt_hard_001",
                "boundary_time_us": cut_us,
                "transition_type": "hard_cut",
            },
        ),
    )


def generate_hard_cut_dev(path: Path, *, fps: int = 25) -> FixtureSpec:
    """Development hard cut with two cuts (black→white→black)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seg = 0.5
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={seg}:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s=320x180:d={seg}:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={seg}:r={fps}",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    cut1 = int(round(seg * 1_000_000))
    cut2 = int(round(2 * seg * 1_000_000))
    return FixtureSpec(
        name="hard_cut_multi",
        split="development",
        description="Black/white/black hard cuts for threshold tuning",
        duration_us=int(round(3 * seg * 1_000_000)),
        fps=fps,
        ground_truth=(
            {
                "boundary_id": "gt_hard_d1",
                "boundary_time_us": cut1,
                "transition_type": "hard_cut",
            },
            {
                "boundary_id": "gt_hard_d2",
                "boundary_time_us": cut2,
                "transition_type": "hard_cut",
            },
        ),
    )


def generate_dissolve(path: Path, *, fps: int = 25) -> FixtureSpec:
    """Cross-dissolve between red and blue (xfade when available)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0.5
    fade_d = 0.5
    total = offset + fade_d + 0.5
    if xfade_available():
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c=red:s=320x180:d={offset + fade_d}:r={fps}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=blue:s=320x180:d={fade_d + 0.5}:r={fps}",
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=dissolve:duration={fade_d}:offset={offset}[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ]
        )
        mid = int(round((offset + fade_d / 2) * 1_000_000))
        ttype = "dissolve"
    else:
        # Approximate with short mid blend bridge
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c=red:s=320x180:d={offset}:r={fps}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=0x7F007F:s=320x180:d={fade_d}:r={fps}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=blue:s=320x180:d=0.5:r={fps}",
                "-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ]
        )
        mid = int(round((offset + fade_d / 2) * 1_000_000))
        ttype = "dissolve"
    return FixtureSpec(
        name="dissolve",
        split="evaluation",
        description="Red to blue dissolve/xfade",
        duration_us=int(round(total * 1_000_000)),
        fps=fps,
        ground_truth=(
            {
                "boundary_id": "gt_dissolve_001",
                "boundary_time_us": mid,
                "transition_type": ttype,
            },
        ),
    )


def generate_fade(path: Path, *, fps: int = 25) -> FixtureSpec:
    """Fade to black then fade from black (treated as fade boundaries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Solid green, fade out, black hold, fade in to yellow
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s=320x180:d=0.5:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d=0.4:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=yellow:s=320x180:d=0.5:r={fps}",
            "-filter_complex",
            (
                "[0:v]fade=t=out:st=0.2:d=0.3[v0];"
                "[2:v]fade=t=in:st=0:d=0.3[v2];"
                "[v0][1:v][v2]concat=n=3:v=1:a=0[v]"
            ),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    # Approximate midpoints of fade-out and fade-in regions
    fade_out = int(0.35 * 1_000_000)
    fade_in = int((0.5 + 0.4 + 0.15) * 1_000_000)
    return FixtureSpec(
        name="fade",
        split="evaluation",
        description="Fade out / fade in via black",
        duration_us=int(1.4 * 1_000_000),
        fps=fps,
        ground_truth=(
            {
                "boundary_id": "gt_fade_001",
                "boundary_time_us": fade_out,
                "transition_type": "fade",
            },
            {
                "boundary_id": "gt_fade_002",
                "boundary_time_us": fade_in,
                "transition_type": "fade",
            },
        ),
    )


def generate_flash(path: Path, *, fps: int = 25) -> FixtureSpec:
    """Brief white flash mid-shot — must NOT produce a shot boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 0.5s green, 2 frames white (~0.08s), 0.5s green
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s=320x180:d=0.5:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s=320x180:d=0.08:r={fps}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s=320x180:d=0.5:r={fps}",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return FixtureSpec(
        name="flash",
        split="negative",
        description="Brief flash without shot change",
        duration_us=int(1.08 * 1_000_000),
        fps=fps,
        ground_truth=(),
    )


def generate_static(path: Path, *, fps: int = 25) -> FixtureSpec:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x226622:s=320x180:d=1.0:r={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return FixtureSpec(
        name="static",
        split="negative",
        description="Static color — no boundaries",
        duration_us=1_000_000,
        fps=fps,
        ground_truth=(),
    )


def generate_pan(path: Path, *, fps: int = 25) -> FixtureSpec:
    """Slow horizontal move (pan-like) without a cut."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=640x180:d=1.2:r={fps}",
            "-vf",
            "crop=320:180:x='min(320,320*t/1.2)':y=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return FixtureSpec(
        name="pan",
        split="negative",
        description="Pan-like crop motion without cut",
        duration_us=1_200_000,
        fps=fps,
        ground_truth=(),
    )


def probe_frame_count(path: Path) -> int:
    if not FFPROBE.is_file():
        raise ShotFixtureError("ffprobe unavailable")
    proc = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ShotFixtureError(f"ffprobe failed: {proc.stderr}")
    text = (proc.stdout or "").strip().splitlines()[0].strip()
    return int(text)


def write_fixture_bundle(
    *,
    split: str,
    name: str,
    video_path: Path,
    spec: FixtureSpec,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    root = assert_runtime_root()
    out_dir = session_dir or (root / split / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.mp4"
    if video_path.resolve() != target.resolve():
        target.write_bytes(video_path.read_bytes())
    digest = sha256_file(target)
    try:
        frame_count = probe_frame_count(target)
    except Exception:  # noqa: BLE001
        frame_count = int(round(spec.duration_us / 1_000_000 * spec.fps))
    gt_path = out_dir / "ground_truth.json"
    gt_payload = {
        "schema_version": 1,
        "fixture": name,
        "split": split,
        "duration_us": spec.duration_us,
        "fps": spec.fps,
        "frame_count": frame_count,
        "boundaries": list(spec.ground_truth),
    }
    _write_manifest(gt_path, gt_payload)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "name": name,
        "split": split,
        "description": spec.description,
        "video_path": str(target),
        "video_sha256": digest,
        "duration_us": spec.duration_us,
        "fps": spec.fps,
        "frame_count": frame_count,
        "ground_truth_path": str(gt_path),
        "xfade_available": xfade_available(),
    }
    _write_manifest(out_dir / "fixture_manifest.json", manifest)
    return manifest


def materialize_standard_fixtures(*, session_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Generate development / evaluation / negative fixture sets."""
    root = assert_runtime_root()
    base = session_dir or root
    results: dict[str, dict[str, Any]] = {}

    # Development
    p = base / "development" / "hard_cut_multi" / "tmp.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    spec = generate_hard_cut_dev(p)
    results["hard_cut_multi"] = write_fixture_bundle(
        split="development", name="hard_cut_multi", video_path=p, spec=spec, session_dir=p.parent
    )
    p.unlink(missing_ok=True)
    # move content: write_fixture_bundle already copied to name.mp4

    generators = [
        ("evaluation", "hard_cut", generate_hard_cut),
        ("evaluation", "dissolve", generate_dissolve),
        ("evaluation", "fade", generate_fade),
        ("negative", "flash", generate_flash),
        ("negative", "static", generate_static),
        ("negative", "pan", generate_pan),
    ]
    for split, name, gen in generators:
        tmp = base / split / name / "tmp.mp4"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        spec = gen(tmp)
        # Force split from directory
        spec = FixtureSpec(
            name=spec.name,
            split=split,
            description=spec.description,
            duration_us=spec.duration_us,
            fps=spec.fps,
            ground_truth=spec.ground_truth,
        )
        results[name] = write_fixture_bundle(
            split=split, name=name, video_path=tmp, spec=spec, session_dir=tmp.parent
        )
        tmp.unlink(missing_ok=True)
    return results


__all__ = [
    "RUNTIME_ROOT",
    "ShotFixtureError",
    "FixtureSpec",
    "assert_runtime_root",
    "ffmpeg_available",
    "xfade_available",
    "generate_hard_cut",
    "generate_hard_cut_dev",
    "generate_dissolve",
    "generate_fade",
    "generate_flash",
    "generate_static",
    "generate_pan",
    "probe_frame_count",
    "write_fixture_bundle",
    "materialize_standard_fixtures",
]
