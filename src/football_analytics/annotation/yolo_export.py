"""Export frozen human GT to Ultralytics YOLO dataset (runtime, not Git)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import yaml

from football_analytics.annotation.coordinates import SOURCE_HEIGHT, SOURCE_WIDTH
from football_analytics.annotation.independent_gt import (
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    IndependentGTError,
    atomic_write_json,
    sha256_file,
    utc_now,
)

DEFAULT_EXPORT_ROOT = Path("/home/fdoblak/workspace/training_datasets/own_video_human_v1")


def xyxy_to_yolo_line(
    xyxy: Sequence[float],
    *,
    class_id: int = 0,
    w: int = SOURCE_WIDTH,
    h: int = SOURCE_HEIGHT,
) -> str:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def assert_no_split_leakage(annotations: Mapping[str, Any]) -> None:
    seen: dict[int, str] = {}
    for fr in annotations.get("frames") or []:
        idx = int(fr["frame_idx"])
        sp = str(fr["split"])
        if idx in seen and seen[idx] != sp:
            raise IndependentGTError(f"LEAKAGE_CROSS_SPLIT_FRAME:{idx}")
        seen[idx] = sp
    # time windows already validated at freeze


def export_yolo_dataset(
    annotations: Mapping[str, Any],
    *,
    video: Path = DEFAULT_VIDEO,
    out_root: Path = DEFAULT_EXPORT_ROOT,
    make_train_tiles: bool = True,
) -> dict[str, Any]:
    if annotations.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise IndependentGTError("EXPORT_SOURCE_SHA_MISMATCH")
    if sha256_file(video) != EXPECTED_SOURCE_SHA256:
        raise IndependentGTError("VIDEO_SOURCE_SHA_MISMATCH")
    if not annotations.get("frozen"):
        raise IndependentGTError("EXPORT_REQUIRES_FROZEN_GT")
    assert_no_split_leakage(annotations)

    if out_root.exists():
        # wipe prior export contents safely under workspace only
        import shutil

        if "training_datasets" not in str(out_root):
            raise IndependentGTError("REFUSING_DELETE_OUTSIDE_TRAINING_DATASETS")
        shutil.rmtree(out_root)
    for sp in ("train", "dev", "holdout"):
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IndependentGTError(f"cannot_open_video:{video}")

    counts = {"train": 0, "dev": 0, "holdout": 0, "train_tiles": 0}
    frame_sha: dict[str, dict[int, str]] = {"train": {}, "dev": {}, "holdout": {}}

    try:
        for fr in annotations["frames"]:
            split = str(fr["split"])
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise IndependentGTError(f"frame_read_fail:{idx}")
            h, w = frame.shape[:2]
            if w != SOURCE_WIDTH or h != SOURCE_HEIGHT:
                raise IndependentGTError(f"bad_frame_size:{w}x{h}")
            stem = f"{idx:06d}"
            img_path = out_root / "images" / split / f"{stem}.jpg"
            lbl_path = out_root / "labels" / split / f"{stem}.txt"
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok2:
                raise IndependentGTError("jpeg_encode_fail")
            data = buf.tobytes()
            img_path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            frame_sha[split][idx] = digest
            lines = [
                xyxy_to_yolo_line(h["bbox_xyxy"])
                for h in (fr.get("humans") or [])
                if not fr.get("no_human_confirmed")
            ]
            # empty label file allowed for explicit no-human
            lbl_path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
            counts[split] += 1

            if (
                make_train_tiles
                and split == "train"
                and (fr.get("humans") or [])
                and len(fr.get("humans") or []) >= 10
            ):
                # controlled 2x2 overlapping tiles for crowded train frames only
                tiles = _train_tiles(frame, fr.get("humans") or [])
                for ti, (tile_img, tile_lines) in enumerate(tiles):
                    tstem = f"{stem}_t{ti}"
                    tip = out_root / "images" / "train" / f"{tstem}.jpg"
                    tlp = out_root / "labels" / "train" / f"{tstem}.txt"
                    ok3, buf3 = cv2.imencode(".jpg", tile_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if not ok3:
                        continue
                    tip.write_bytes(buf3.tobytes())
                    tlp.write_text(
                        ("\n".join(tile_lines) + ("\n" if tile_lines else "")),
                        encoding="utf-8",
                    )
                    counts["train_tiles"] += 1
    finally:
        cap.release()

    # cross-split image sha collision hard-fail
    all_sha: dict[str, tuple[str, int]] = {}
    for sp, mp in frame_sha.items():
        for idx, dig in mp.items():
            if dig in all_sha and all_sha[dig][0] != sp:
                raise IndependentGTError(f"CROSS_SPLIT_FRAME_SHA:{dig}:{all_sha[dig]}:{sp}:{idx}")
            all_sha[dig] = (sp, idx)

    dataset_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/dev",
        "test": "images/holdout",
        "names": {0: "human"},
        "nc": 1,
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema": "own_video_human_v1_export_manifest",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "frozen_fingerprint": annotations.get("canonical_fingerprint"),
        "counts": counts,
        "class": {0: "human"},
        "role_team_excluded_from_labels": True,
        "train_tiles_only": True,
        "dev_holdout_full_frame_only": True,
        "written_at_utc": utc_now(),
        "frame_sha256": {sp: frame_sha[sp] for sp in frame_sha},
    }
    atomic_write_json(out_root / "export_manifest.json", manifest, mode=0o600)
    return manifest


def _train_tiles(
    frame: Any,
    humans: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, list[str]]]:
    """2x2 tiles with overlap; labels remapped to tile coords."""
    h, w = frame.shape[:2]
    tw, th = w // 2 + 80, h // 2 + 60
    origins = [
        (0, 0),
        (w - tw, 0),
        (0, h - th),
        (w - tw, h - th),
    ]
    out: list[tuple[Any, list[str]]] = []
    for ox, oy in origins:
        ox = max(0, min(ox, w - tw))
        oy = max(0, min(oy, h - th))
        tile = frame[oy : oy + th, ox : ox + tw].copy()
        lines: list[str] = []
        for hum in humans:
            x1, y1, x2, y2 = (float(v) for v in hum["bbox_xyxy"])
            # clip to tile
            nx1 = max(x1, ox) - ox
            ny1 = max(y1, oy) - oy
            nx2 = min(x2, ox + tw) - ox
            ny2 = min(y2, oy + th) - oy
            if nx2 - nx1 < 8 or ny2 - ny1 < 8:
                continue
            # require sufficient overlap with original box
            inter = (nx2 - nx1) * (ny2 - ny1)
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if inter / area < 0.4:
                continue
            lines.append(xyxy_to_yolo_line([nx1, ny1, nx2, ny2], w=tw, h=th))
        if lines:
            out.append((tile, lines))
    return out


__all__ = [
    "DEFAULT_EXPORT_ROOT",
    "assert_no_split_leakage",
    "export_yolo_dataset",
    "xyxy_to_yolo_line",
]
