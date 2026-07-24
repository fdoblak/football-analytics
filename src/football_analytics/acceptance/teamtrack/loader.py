"""Load TeamTrack MOT-format soccer sequences (video + gt.txt + seqinfo.ini)."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MotBox:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    class_id: int
    visibility: float


@dataclass(frozen=True)
class TeamTrackSequence:
    dataset_id: str
    sport_view: str
    split: str
    sequence_id: str
    root: Path
    video_path: Path
    gt_path: Path
    seqinfo_path: Path
    fps: float
    seq_length: int
    im_width: int
    im_height: int
    boxes: tuple[MotBox, ...]

    def run_namespace(self) -> str:
        return f"teamtrack_{self.sport_view}_{self.sequence_id}"


def parse_seqinfo(path: Path) -> dict[str, Any]:
    parser = ConfigParser()
    text = path.read_text(encoding="utf-8")
    parser.read_string(text)
    sec = parser["Sequence"]
    framerate = sec.get("framerate")
    seqlength = sec.get("seqlength")
    imwidth = sec.get("imwidth")
    imheight = sec.get("imheight")
    if not framerate or not seqlength or not imwidth or not imheight:
        raise ValueError(f"incomplete seqinfo.ini: {path}")
    return {
        "name": sec.get("name"),
        "imdir": sec.get("imdir", "img1"),
        "framerate": float(framerate),
        "seqlength": int(float(seqlength)),
        "imwidth": int(float(imwidth)),
        "imheight": int(float(imheight)),
        "imext": sec.get("imext", ".jpg"),
    }


def load_mot_gt(path: Path) -> tuple[MotBox, ...]:
    """Parse MOTChallenge gt.txt rows into MotBox tuples."""
    boxes: list[MotBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"invalid MOT row: {line!r}")
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        class_id = int(float(parts[7])) if len(parts) > 7 else -1
        visibility = float(parts[8]) if len(parts) > 8 else -1.0
        boxes.append(
            MotBox(
                frame=frame,
                track_id=track_id,
                x=x,
                y=y,
                w=w,
                h=h,
                conf=conf,
                class_id=class_id,
                visibility=visibility,
            )
        )
    return tuple(boxes)


def load_sequence(
    *,
    root: Path,
    sport_view: str = "soccer_side",
    split: str = "train",
    sequence_id: str,
) -> TeamTrackSequence:
    seq_root = Path(root) / sport_view / sequence_id
    if not seq_root.is_dir():
        # allow root already pointing at sequence dir
        if (Path(root) / "img1.mp4").is_file() and Path(root).name == sequence_id:
            seq_root = Path(root)
        else:
            raise FileNotFoundError(f"sequence root missing: {seq_root}")
    video = seq_root / "img1.mp4"
    gt = seq_root / "gt" / "gt.txt"
    info = seq_root / "seqinfo.ini"
    for p in (video, gt, info):
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(f"required file missing or symlink: {p}")
    meta = parse_seqinfo(info)
    if meta["name"] and meta["name"] != sequence_id:
        raise ValueError(f"seqinfo name {meta['name']!r} != sequence_id {sequence_id!r}")
    boxes = load_mot_gt(gt)
    return TeamTrackSequence(
        dataset_id="teamtrack",
        sport_view=sport_view,
        split=split,
        sequence_id=sequence_id,
        root=seq_root,
        video_path=video,
        gt_path=gt,
        seqinfo_path=info,
        fps=float(meta["framerate"]),
        seq_length=int(meta["seqlength"]),
        im_width=int(meta["imwidth"]),
        im_height=int(meta["imheight"]),
        boxes=boxes,
    )


__all__ = [
    "MotBox",
    "TeamTrackSequence",
    "load_mot_gt",
    "load_sequence",
    "parse_seqinfo",
]
