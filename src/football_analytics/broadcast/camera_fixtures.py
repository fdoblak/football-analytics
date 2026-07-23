"""Synthetic FFmpeg lavfi fixtures for camera-view baseline (runtime only)."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file

RUNTIME_ROOT = Path("/home/fdoblak/workspace/camera_view_checks")
FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


class CameraFixtureError(ValueError):
    """Fixture generation failure."""


@dataclass(frozen=True)
class CameraFixtureSpec:
    name: str
    split: str  # development | frozen_evaluation | negative_controls | out_of_distribution
    description: str
    duration_us: int
    fps: int
    ground_truth: dict[str, Any]
    is_ood: bool = False


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = RUNTIME_ROOT.resolve()
    if not str(resolved).startswith("/home/fdoblak/workspace/camera_view_checks"):
        raise CameraFixtureError(f"unsafe runtime root: {resolved}")
    if resolved.is_symlink():
        raise CameraFixtureError("runtime root must not be a symlink")
    return resolved


def ffmpeg_available() -> bool:
    return FFMPEG.is_file() and os.access(FFMPEG, os.X_OK)


def _run_ffmpeg(args: list[str]) -> None:
    if not ffmpeg_available():
        raise CameraFixtureError("ffmpeg unavailable")
    cmd = [str(FFMPEG), "-y", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CameraFixtureError(f"ffmpeg failed: {(proc.stderr or '')[-800:]}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def probe_frame_count(path: Path) -> int:
    if not FFPROBE.is_file():
        raise CameraFixtureError("ffprobe unavailable")
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
        raise CameraFixtureError(f"ffprobe failed: {proc.stderr}")
    text = (proc.stdout or "").strip().splitlines()[0].strip()
    return int(text)


def _gt(
    *,
    view_family: str,
    framing_scale: str,
    camera_motion: str,
    graphics_status: str,
    playability: str,
    calibration_suitability: str,
    tracking_suitability: str,
    target_identity_suitability: str,
    is_ood: bool = False,
) -> dict[str, Any]:
    return {
        "view_family": view_family,
        "framing_scale": framing_scale,
        "camera_position": "unknown",
        "camera_motion": camera_motion,
        "replay_status": "unknown",
        "graphics_status": graphics_status,
        "playability": playability,
        "calibration_suitability": calibration_suitability,
        "tracking_suitability": tracking_suitability,
        "target_identity_suitability": target_identity_suitability,
        "is_ood": is_ood,
    }


def _encode(path: Path, filter_complex: str, *, duration: float, fps: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={duration}:r={fps}",
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-t",
            str(duration),
            str(path),
        ]
    )


def generate_wide_pitch(path: Path, *, fps: int = 25, seconds: float = 1.2) -> CameraFixtureSpec:
    """Wide main-broadcast: large green pitch with mild noise + thin white line."""
    # geq draws green field; noise via noise filter; line via drawbox
    fc = (
        "[0:v]geq=r='20':g='140+10*sin(X/40)':b='40',"
        "noise=alls=8:allf=t+u,"
        "drawbox=x=0:y=88:w=320:h=2:color=white@0.7:t=fill[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="wide_pitch",
        split="frozen_evaluation",
        description="Wide green pitch with noise (main_broadcast/wide/static)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="static",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_medium_pitch(path: Path, *, fps: int = 25, seconds: float = 1.2) -> CameraFixtureSpec:
    """Medium framing: green band ~central third, dark surrounds."""
    fc = (
        "[0:v]geq="
        "r='if(between(Y,55,125)*between(X,40,280),25,8)':"
        "g='if(between(Y,55,125)*between(X,40,280),125+8*sin(X/30),12)':"
        "b='if(between(Y,55,125)*between(X,40,280),35,10)',"
        "noise=alls=6:allf=t+u[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="medium_pitch",
        split="frozen_evaluation",
        description="Medium green band (main_broadcast/medium)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="medium",
            camera_motion="static",
            graphics_status="none",
            playability="playable",
            calibration_suitability="conditionally_suitable",
            tracking_suitability="conditionally_suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_player_isolation(
    path: Path, *, fps: int = 25, seconds: float = 1.2
) -> CameraFixtureSpec:
    """Close-up / player isolation: skin-like ellipse on dark background."""
    # Skin-ish fill via geq in a central ellipse; rest dark
    fc = (
        "[0:v]geq="
        "r='if(lt(pow((X-160)/45,2)+pow((Y-90)/55,2),1),180,8)':"
        "g='if(lt(pow((X-160)/45,2)+pow((Y-90)/55,2),1),120,8)':"
        "b='if(lt(pow((X-160)/45,2)+pow((Y-90)/55,2),1),90,8)',"
        "noise=alls=4:allf=t+u[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="player_isolation",
        split="frozen_evaluation",
        description="Skin-like subject on dark bg (player_isolation/close_up)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="player_isolation",
            framing_scale="close_up",
            camera_motion="static",
            graphics_status="none",
            playability="partially_playable",
            calibration_suitability="unsuitable",
            tracking_suitability="unsuitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_fullscreen_graphics(
    path: Path, *, fps: int = 25, seconds: float = 1.2
) -> CameraFixtureSpec:
    """Full-screen high-contrast UI / text-like bars (no green tones)."""
    fc = (
        "[0:v]geq=r='15':g='15':b='15',"
        "drawbox=x=0:y=0:w=320:h=180:color=navy@1.0:t=fill,"
        "drawbox=x=10:y=15:w=300:h=28:color=white@1.0:t=fill,"
        "drawbox=x=10:y=55:w=260:h=22:color=magenta@1.0:t=fill,"
        "drawbox=x=10:y=90:w=280:h=22:color=yellow@1.0:t=fill,"
        "drawbox=x=10:y=125:w=240:h=22:color=white@1.0:t=fill,"
        "drawbox=x=0:y=155:w=320:h=25:color=red@1.0:t=fill[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="fullscreen_graphics",
        split="frozen_evaluation",
        description="Full-screen high-contrast graphics",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="graphics",
            framing_scale="unknown",
            camera_motion="static",
            graphics_status="full_screen",
            playability="non_playable",
            calibration_suitability="unsuitable",
            tracking_suitability="unsuitable",
            target_identity_suitability="unsuitable",
        ),
    )


def generate_partial_overlay(
    path: Path, *, fps: int = 25, seconds: float = 1.2
) -> CameraFixtureSpec:
    """Wide pitch with partial scoreboard overlay (top + side bars)."""
    fc = (
        "[0:v]geq=r='20':g='145':b='40',"
        "noise=alls=6:allf=t+u,"
        "drawbox=x=0:y=0:w=320:h=42:color=black@0.92:t=fill,"
        "drawbox=x=12:y=8:w=140:h=26:color=white@1.0:t=fill,"
        "drawbox=x=170:y=8:w=130:h=26:color=magenta@1.0:t=fill,"
        "drawbox=x=0:y=150:w=100:h=30:color=black@0.85:t=fill,"
        "drawbox=x=8:y=156:w=80:h=16:color=white@1.0:t=fill[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="partial_overlay",
        split="frozen_evaluation",
        description="Wide pitch + partial scoreboard overlay",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="static",
            graphics_status="partial_overlay",
            playability="partially_playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_dominant_overlay(
    path: Path, *, fps: int = 25, seconds: float = 1.2
) -> CameraFixtureSpec:
    """Pitch partly covered by large opaque graphics (~half frame)."""
    fc = (
        "[0:v]geq=r='20':g='140':b='40',"
        "drawbox=x=0:y=0:w=320:h=85:color=navy@0.95:t=fill,"
        "drawbox=x=20:y=18:w=280:h=26:color=white@1.0:t=fill,"
        "drawbox=x=20:y=52:w=200:h=18:color=magenta@1.0:t=fill,"
        "drawbox=x=0:y=150:w=320:h=30:color=black@0.9:t=fill[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="dominant_overlay",
        split="frozen_evaluation",
        description="Dominant overlay covering most of frame",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="graphics",
            framing_scale="unknown",
            camera_motion="static",
            graphics_status="dominant_overlay",
            playability="non_playable",
            calibration_suitability="unsuitable",
            tracking_suitability="unsuitable",
            target_identity_suitability="unsuitable",
        ),
    )


def generate_pan(path: Path, *, fps: int = 25, seconds: float = 1.2) -> CameraFixtureSpec:
    """Horizontal pan over a green strip with vertical stripes (motion cue)."""
    # Moving crop via scroll / crop with time-based x
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x148C28:s=640x180:d={seconds}:r={fps}",
            "-filter_complex",
            "[0:v]geq=r='20':g='140':b='40',"
            "drawbox=x=0:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "drawbox=x=120:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "drawbox=x=240:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "drawbox=x=360:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "drawbox=x=480:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "drawbox=x=600:y=0:w=40:h=180:color=white@0.5:t=fill,"
            "crop=320:180:'min(320,320*t/1.0)':0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-t",
            str(seconds),
            str(path),
        ]
    )
    return CameraFixtureSpec(
        name="pan_motion",
        split="frozen_evaluation",
        description="Horizontal pan over striped pitch",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="pan",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_zoom(path: Path, *, fps: int = 25, seconds: float = 1.2) -> CameraFixtureSpec:
    """Zoom-in on wide pitch (radial motion)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x148C28:s=320x180:d={seconds}:r={fps}",
            "-filter_complex",
            "[0:v]geq=r='20':g='145+5*sin(X/20)':b='40',"
            "noise=alls=5:allf=t,"
            "zoompan=z='1+0.8*on/30':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s=320x180:fps={fps}[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-t",
            str(seconds),
            str(path),
        ]
    )
    return CameraFixtureSpec(
        name="zoom_motion",
        split="frozen_evaluation",
        description="Zoom-in on green pitch",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="zoom",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_unstable(path: Path, *, fps: int = 25, seconds: float = 1.0) -> CameraFixtureSpec:
    """Unstable / shaky noisy pitch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x148C28:s=360x200:d={seconds}:r={fps}",
            "-filter_complex",
            "[0:v]geq=r='20':g='140':b='40',"
            "noise=alls=35:allf=t+u,"
            "crop=320:180:'20+10*sin(12*t)':'10+8*cos(15*t)'[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-t",
            str(seconds),
            str(path),
        ]
    )
    return CameraFixtureSpec(
        name="unstable_motion",
        split="frozen_evaluation",
        description="Shaky noisy pitch (unstable)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="unstable",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_ood_crowd(path: Path, *, fps: int = 25, seconds: float = 1.0) -> CameraFixtureSpec:
    """OOD crowd-like: grayish noisy texture, low green — must abstain."""
    # Low-saturation gray-brown noise (avoid green/skin HSV bands).
    fc = (
        "[0:v]geq="
        "r='110+25*sin(X/2)*sin(Y/3)':"
        "g='105+20*cos(X/3)*sin(Y/2)':"
        "b='100+22*sin(X/4)*cos(Y/5)',"
        "noise=alls=45:allf=t+u,"
        "eq=saturation=0.25[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="ood_crowd",
        split="out_of_distribution",
        description="Crowd-like noisy low-green OOD (abstain)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        is_ood=True,
        ground_truth=_gt(
            view_family="unknown",
            framing_scale="unknown",
            camera_motion="unknown",
            graphics_status="unknown",
            playability="uncertain",
            calibration_suitability="unknown",
            tracking_suitability="unknown",
            target_identity_suitability="unknown",
            is_ood=True,
        ),
    )


def generate_low_light_pitch(
    path: Path, *, fps: int = 25, seconds: float = 1.0
) -> CameraFixtureSpec:
    """Development: darker alt-green pitch for threshold tuning."""
    fc = "[0:v]geq=r='10':g='70+5*sin(X/50)':b='20',noise=alls=10:allf=t+u[v]"
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="low_light_pitch",
        split="development",
        description="Low-light alt green pitch (dev tuning)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="static",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_alt_green(path: Path, *, fps: int = 25, seconds: float = 1.0) -> CameraFixtureSpec:
    """Development: yellower/alt green tone."""
    fc = "[0:v]geq=r='50':g='150':b='30',noise=alls=7:allf=t+u[v]"
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="alt_green_pitch",
        split="development",
        description="Alt green tone pitch (dev tuning)",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="static",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_dev_closeup(path: Path, *, fps: int = 25, seconds: float = 1.0) -> CameraFixtureSpec:
    """Development close-up variant (slightly different skin blob)."""
    fc = (
        "[0:v]geq="
        "r='if(lt(pow((X-160)/55,2)+pow((Y-95)/60,2),1),200,5)':"
        "g='if(lt(pow((X-160)/55,2)+pow((Y-95)/60,2),1),140,5)':"
        "b='if(lt(pow((X-160)/55,2)+pow((Y-95)/60,2),1),100,5)',"
        "noise=alls=5:allf=t+u[v]"
    )
    _encode(path, fc, duration=seconds, fps=fps)
    return CameraFixtureSpec(
        name="dev_closeup",
        split="development",
        description="Dev player isolation variant",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="player_isolation",
            framing_scale="close_up",
            camera_motion="static",
            graphics_status="none",
            playability="partially_playable",
            calibration_suitability="unsuitable",
            tracking_suitability="unsuitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def generate_compound_motion(
    path: Path, *, fps: int = 25, seconds: float = 1.2
) -> CameraFixtureSpec:
    """Compound pan+zoom-ish motion for development / optional eval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x148C28:s=480x240:d={seconds}:r={fps}",
            "-filter_complex",
            "[0:v]geq=r='20':g='140':b='40',"
            "drawbox=x=0:y=0:w=40:h=240:color=white@0.4:t=fill,"
            "drawbox=x=160:y=0:w=40:h=240:color=white@0.4:t=fill,"
            "drawbox=x=320:y=0:w=40:h=240:color=white@0.4:t=fill,"
            "zoompan=z='1+0.35*on/30':x='40*on/30':y='10*on/30':"
            f"d=1:s=320x180:fps={fps}[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-t",
            str(seconds),
            str(path),
        ]
    )
    return CameraFixtureSpec(
        name="compound_motion",
        split="development",
        description="Compound pan+zoom development fixture",
        duration_us=int(round(seconds * 1_000_000)),
        fps=fps,
        ground_truth=_gt(
            view_family="main_broadcast",
            framing_scale="wide",
            camera_motion="compound",
            graphics_status="none",
            playability="playable",
            calibration_suitability="suitable",
            tracking_suitability="suitable",
            target_identity_suitability="conditionally_suitable",
        ),
    )


def write_fixture_bundle(
    *,
    split: str,
    name: str,
    video_path: Path,
    spec: CameraFixtureSpec,
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
        "fixture_id": name,
        "name": name,
        "split": split,
        "duration_us": spec.duration_us,
        "fps": spec.fps,
        "frame_count": frame_count,
        "is_ood": spec.is_ood,
        "shot_id": f"shot_{name}",
        "labels": dict(spec.ground_truth),
    }
    # Flatten labels onto top-level for evaluator convenience
    gt_payload.update(dict(spec.ground_truth))
    _write_json(gt_path, gt_payload)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "name": name,
        "fixture_id": name,
        "split": split,
        "description": spec.description,
        "video_path": str(target),
        "video_sha256": digest,
        "duration_us": spec.duration_us,
        "fps": spec.fps,
        "frame_count": frame_count,
        "is_ood": spec.is_ood,
        "ground_truth_path": str(gt_path),
    }
    _write_json(out_dir / "fixture_manifest.json", manifest)
    return manifest


def materialize_standard_fixtures(*, session_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Generate development / frozen_evaluation / negative / OOD fixture sets."""
    root = assert_runtime_root()
    base = session_dir or root
    results: dict[str, dict[str, Any]] = {}

    generators: list[tuple[str, str, Any]] = [
        ("development", "low_light_pitch", generate_low_light_pitch),
        ("development", "alt_green_pitch", generate_alt_green),
        ("development", "dev_closeup", generate_dev_closeup),
        ("development", "compound_motion", generate_compound_motion),
        ("frozen_evaluation", "wide_pitch", generate_wide_pitch),
        ("frozen_evaluation", "medium_pitch", generate_medium_pitch),
        ("frozen_evaluation", "player_isolation", generate_player_isolation),
        ("frozen_evaluation", "fullscreen_graphics", generate_fullscreen_graphics),
        ("frozen_evaluation", "partial_overlay", generate_partial_overlay),
        ("frozen_evaluation", "dominant_overlay", generate_dominant_overlay),
        ("frozen_evaluation", "pan_motion", generate_pan),
        ("frozen_evaluation", "zoom_motion", generate_zoom),
        ("frozen_evaluation", "unstable_motion", generate_unstable),
        ("negative_controls", "static_wide", generate_wide_pitch),
        ("out_of_distribution", "ood_crowd", generate_ood_crowd),
    ]
    for split, name, gen in generators:
        tmp = base / split / name / "tmp.mp4"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        spec = gen(tmp)
        spec = CameraFixtureSpec(
            name=name,
            split=split,
            description=spec.description,
            duration_us=spec.duration_us,
            fps=spec.fps,
            ground_truth=spec.ground_truth,
            is_ood=spec.is_ood or split == "out_of_distribution",
        )
        results[name] = write_fixture_bundle(
            split=split, name=name, video_path=tmp, spec=spec, session_dir=tmp.parent
        )
        tmp.unlink(missing_ok=True)
    return results


__all__ = [
    "RUNTIME_ROOT",
    "CameraFixtureError",
    "CameraFixtureSpec",
    "assert_runtime_root",
    "ffmpeg_available",
    "probe_frame_count",
    "generate_wide_pitch",
    "generate_medium_pitch",
    "generate_player_isolation",
    "generate_fullscreen_graphics",
    "generate_partial_overlay",
    "generate_dominant_overlay",
    "generate_pan",
    "generate_zoom",
    "generate_unstable",
    "generate_ood_crowd",
    "generate_low_light_pitch",
    "generate_alt_green",
    "generate_dev_closeup",
    "generate_compound_motion",
    "write_fixture_bundle",
    "materialize_standard_fixtures",
]
