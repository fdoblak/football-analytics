"""Deterministic pilot clip extraction for Stage 16."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from football_analytics.acceptance.download_manifest import sha256_file


def choose_pilot_window(
    *,
    target_frame_indices: list[int],
    fps: float = 25.0,
    duration_s: float = 240.0,
) -> tuple[float, float]:
    """Pick a contiguous window covering target visibility (~3–5 min)."""
    if not target_frame_indices:
        return 0.0, duration_s
    frames = sorted(target_frame_indices)
    mid = frames[len(frames) // 2]
    half = duration_s / 2.0
    start = max(0.0, mid / fps - half)
    end = start + duration_s
    return start, end


def extract_pilot_clip(
    *,
    source_video: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    receipt_path: Optional[Path] = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    receipt = {
        "source_video": str(source_video),
        "output_path": str(output_path),
        "start_s": start_s,
        "duration_s": duration_s,
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "git_policy": "pilot_clip_not_committed",
    }
    if receipt_path is not None:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
