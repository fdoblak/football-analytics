"""Streaming SoccerTrack v2 loaders (stdlib only; no full 2.7GiB json.load)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from football_analytics.acceptance.contracts import FPS_DEFAULT, normalize_bas_label
from football_analytics.acceptance.soccertrack_v2.formats import (
    BasEvent,
    GsrPlayerObservation,
    VideoHalfMeta,
    maybe_int,
    maybe_str,
)


def half_suffix(half: int) -> str:
    if half == 1:
        return "1st"
    if half == 2:
        return "2nd"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def gsr_path(root: Path, match_id: str, half: int) -> Path:
    name = f"{match_id}_{half_suffix(half)}.json"
    return Path(root) / "gsr" / str(match_id) / name


def bas_path(root: Path, match_id: str) -> Path:
    name = f"{match_id}_12_class_events.json"
    return Path(root) / "bas" / str(match_id) / name


def video_half_path(root: Path, match_id: str, half: int) -> Path:
    name = f"{match_id}_panorama_{half_suffix(half)}_half.mp4"
    return Path(root) / "videos" / str(match_id) / name


def _image_id_to_frame_index(image_id: str | int) -> int:
    """SoccerNet-style image_id like '3000001' → 0-based frame index within half."""
    text = str(image_id)
    if len(text) >= 7:
        # trailing 6 digits are 1-based frame number in many SoccerNet exports
        return max(0, int(text[-6:]) - 1)
    return max(0, int(text) - 1)


def _iter_json_array_after_key(path: Path, key: str) -> Iterator[Any]:
    """Yield objects from a top-level JSON array field without loading the file."""
    needle = f'"{key}"'.encode()
    with path.open("rb") as handle:
        # locate key
        chunk_size = 8 * 1024 * 1024
        overlap = len(needle) + 8
        prev = b""
        offset = 0
        key_at: int | None = None
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            buf = prev + data
            idx = buf.find(needle)
            if idx >= 0:
                key_at = offset - len(prev) + idx
                break
            prev = buf[-overlap:]
            offset += len(data)
        if key_at is None:
            raise KeyError(f"Key {key!r} not found in {path}")
        handle.seek(key_at)
        # read until '[' then stream-decode objects
        window = handle.read(1024 * 1024)
        text = window.decode("utf-8")
        bracket = text.find("[")
        if bracket < 0:
            raise ValueError(f"Array start for {key!r} not found in {path}")
        # absolute file position of first element region
        abs_pos = key_at + bracket + 1
        handle.seek(abs_pos)
        decoder = json.JSONDecoder()
        leftover = ""
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block and not leftover.strip():
                break
            leftover += block.decode("utf-8")
            while True:
                s = leftover.lstrip()
                stripped = len(leftover) - len(s)
                if not s:
                    leftover = ""
                    break
                if s[0] == "]":
                    return
                if s[0] == ",":
                    leftover = s[1:]
                    continue
                try:
                    obj, end = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    # need more data
                    leftover = leftover[stripped:]
                    break
                yield obj
                leftover = s[end:]


def iter_gsr_player_observations(
    path: Path,
    *,
    half: int,
    sample_stride: int = 1,
) -> Iterator[GsrPlayerObservation]:
    """Stream player (object) annotations from a COCO-style SoccerTrack GSR JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    seen = 0
    for obj in _iter_json_array_after_key(path, "annotations"):
        if not isinstance(obj, dict):
            continue
        if obj.get("supercategory") != "object":
            continue
        seen += 1
        if sample_stride > 1 and (seen % sample_stride) != 0:
            continue
        attrs = obj.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        role = str(attrs.get("role") or "other")
        bbox_pitch = obj.get("bbox_pitch") or {}
        if not isinstance(bbox_pitch, dict):
            continue
        x_m = bbox_pitch.get("x_bottom_middle")
        y_m = bbox_pitch.get("y_bottom_middle")
        if x_m is None or y_m is None:
            continue
        image_id_raw = obj.get("image_id")
        frame_index = _image_id_to_frame_index(image_id_raw)
        jersey = maybe_int(attrs.get("jersey"))
        team = maybe_str(attrs.get("team"))
        player_id = maybe_str(attrs.get("player_id"))
        if player_id is None:
            player_id = f"track:{obj.get('track_id')}"
        bbox_image = None
        bi = obj.get("bbox_image")
        if isinstance(bi, dict) and bi.get("x") is not None:
            bbox_image = (
                float(bi["x"]),
                float(bi["y"]),
                float(bi.get("w") or 0.0),
                float(bi.get("h") or 0.0),
            )
        yield GsrPlayerObservation(
            half=half,
            image_id=int(str(image_id_raw)[-6:]) if image_id_raw is not None else frame_index + 1,
            frame_index=frame_index,
            track_id=int(obj.get("track_id") or -1),
            player_id=player_id,
            role=role,
            jersey_number=jersey,
            team_side=team,
            x_m=float(x_m),
            y_m=float(y_m),
            bbox_image=bbox_image,
        )


def load_bas_events(path: Path) -> list[BasEvent]:
    """Load BAS events (supports Drive `actions` and docs `annotations`)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("actions")
    if rows is None:
        rows = data.get("annotations")
    if not isinstance(rows, list):
        raise ValueError(f"BAS file missing actions/annotations: {path}")
    out: list[BasEvent] = []
    for row in rows:
        game_time = str(row["gameTime"])
        half_str, clock = game_time.split(" - ", 1)
        label = normalize_bas_label(str(row["label"]))
        out.append(
            BasEvent(
                half=int(half_str),
                clock=clock.strip(),
                t_ms=int(row["position"]),
                label=label,
                team=maybe_str(row.get("team")),
                player_id=maybe_str(row.get("player_id")),
                visibility=maybe_str(row.get("visibility")),
            )
        )
    return out


def probe_video(path: Path) -> VideoHalfMeta:
    """Probe MP4 with ffprobe (no decode)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    fps = None
    rate = stream.get("r_frame_rate")
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    duration = stream.get("duration") or fmt.get("duration")
    half = 1 if "1st" in path.name else 2 if "2nd" in path.name else 0
    match_id = path.parent.name
    nb = stream.get("nb_frames")
    return VideoHalfMeta(
        match_id=match_id,
        half=half,
        path=str(path),
        width=maybe_int(stream.get("width")),
        height=maybe_int(stream.get("height")),
        fps=fps or float(FPS_DEFAULT),
        duration_s=float(duration) if duration is not None else None,
        frame_count=maybe_int(nb),
    )


def read_gsr_info(path: Path) -> dict[str, Any]:
    """Read only the top-level `info` object (small prefix parse)."""
    with path.open("r", encoding="utf-8") as handle:
        # info is at the start of SoccerTrack COCO exports
        head = handle.read(64 * 1024)
    decoder = json.JSONDecoder()
    # find {"info"
    start = head.find('"info"')
    if start < 0:
        raise KeyError("info")
    brace = head.find("{", start)
    info, _ = decoder.raw_decode(head[brace:])
    return info
