"""Train-only tile dataset helpers for small-object human detection (R1-F2-C)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import yaml

from football_analytics.annotation.coordinates import SOURCE_HEIGHT, SOURCE_WIDTH
from football_analytics.annotation.independent_gt import (
    IndependentGTError,
    atomic_write_json,
    utc_now,
)
from football_analytics.annotation.yolo_export import xyxy_to_yolo_line
from football_analytics.perception.tiling import TileSpec, generate_tiles, map_tile_bbox_to_source

DEFAULT_TILE_W = 704
DEFAULT_TILE_H = 512
DEFAULT_OVERLAP_X = 140  # ~20%
DEFAULT_OVERLAP_Y = 102  # ~20%
MIN_VISIBLE_FRAC = 0.35
MIN_CLIPPED_SIDE = 8.0


def clip_box_to_tile(
    xyxy: Sequence[float],
    tile: TileSpec,
) -> tuple[float, float, float, float] | None:
    """Clip source xyxy to tile; return tile-local xyxy or None if insufficient."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    nx1 = max(x1, float(tile.x0))
    ny1 = max(y1, float(tile.y0))
    nx2 = min(x2, float(tile.x1))
    ny2 = min(y2, float(tile.y1))
    if nx2 - nx1 < MIN_CLIPPED_SIDE or ny2 - ny1 < MIN_CLIPPED_SIDE:
        return None
    inter = (nx2 - nx1) * (ny2 - ny1)
    if inter / area < MIN_VISIBLE_FRAC:
        return None
    # tile-local
    return (nx1 - tile.x0, ny1 - tile.y0, nx2 - tile.x0, ny2 - tile.y0)


def roundtrip_ok(xyxy: Sequence[float], tile: TileSpec, *, tol: float = 1e-3) -> bool:
    local = clip_box_to_tile(xyxy, tile)
    if local is None:
        return True  # ignored path
    back = map_tile_bbox_to_source(local, tile, coordinate_space="tile_local")
    # compare to clipped source region
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx1 = max(x1, float(tile.x0))
    cy1 = max(y1, float(tile.y0))
    cx2 = min(x2, float(tile.x1))
    cy2 = min(y2, float(tile.y1))
    return all(abs(a - b) <= tol for a, b in zip(back, (cx1, cy1, cx2, cy2), strict=True))


def build_train_tile_dataset(
    annotations: Mapping[str, Any],
    *,
    video: Path,
    out_root: Path,
    tile_w: int = DEFAULT_TILE_W,
    tile_h: int = DEFAULT_TILE_H,
    overlap_x: int = DEFAULT_OVERLAP_X,
    overlap_y: int = DEFAULT_OVERLAP_Y,
) -> dict[str, Any]:
    """Export full-frame train + train tiles. Dev/holdout images are NOT written here."""
    if not annotations.get("frozen"):
        raise IndependentGTError("TILE_EXPORT_REQUIRES_FROZEN")
    if out_root.exists():
        import shutil

        if "training_datasets" not in str(out_root) and "r1_f2c" not in str(out_root):
            raise IndependentGTError("REFUSING_DELETE_UNSAFE_PATH")
        shutil.rmtree(out_root)
    for sp in ("train", "dev"):
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    tiles = generate_tiles(
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        tile_width=tile_w,
        tile_height=tile_h,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        max_tiles=24,
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IndependentGTError("cannot_open_video")

    lineage: list[dict[str, Any]] = []
    counts = {"train_full": 0, "train_tiles": 0, "dev_full": 0, "ignored_clipped": 0}
    tile_sha: set[str] = set()
    try:
        for fr in annotations["frames"]:
            split = str(fr["split"])
            if split == "holdout":
                continue
            if split not in {"train", "dev"}:
                continue
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise IndependentGTError(f"read_fail:{idx}")
            stem = f"{idx:06d}"
            # full frame for train+dev
            img_path = out_root / "images" / split / f"{stem}.jpg"
            lbl_path = out_root / "labels" / split / f"{stem}.txt"
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok2:
                raise IndependentGTError("jpeg_fail")
            data = buf.tobytes()
            img_path.write_bytes(data)
            lines = [
                xyxy_to_yolo_line(h["bbox_xyxy"])
                for h in (fr.get("humans") or [])
                if not fr.get("no_human_confirmed")
            ]
            lbl_path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
            if split == "train":
                counts["train_full"] += 1
            else:
                counts["dev_full"] += 1
            lineage.append(
                {
                    "kind": "full",
                    "split": split,
                    "frame_idx": idx,
                    "stem": stem,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )

            if split != "train":
                continue
            # tiles only for train
            for tile in tiles:
                t_lines: list[str] = []
                for h in fr.get("humans") or []:
                    if not roundtrip_ok(h["bbox_xyxy"], tile):
                        raise IndependentGTError(f"roundtrip_fail:{idx}:{tile.tile_id}")
                    local = clip_box_to_tile(h["bbox_xyxy"], tile)
                    if local is None:
                        counts["ignored_clipped"] += 1
                        continue
                    t_lines.append(xyxy_to_yolo_line(local, w=tile.width, h=tile.height))
                if not t_lines:
                    continue
                crop = frame[tile.y0 : tile.y1, tile.x0 : tile.x1]
                tstem = f"{stem}_{tile.tile_id}"
                tip = out_root / "images" / "train" / f"{tstem}.jpg"
                tlp = out_root / "labels" / "train" / f"{tstem}.txt"
                ok3, buf3 = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok3:
                    continue
                tdata = buf3.tobytes()
                dig = hashlib.sha256(tdata).hexdigest()
                if dig in tile_sha:
                    continue
                tile_sha.add(dig)
                tip.write_bytes(tdata)
                tlp.write_text(("\n".join(t_lines) + ("\n" if t_lines else "")), encoding="utf-8")
                counts["train_tiles"] += 1
                lineage.append(
                    {
                        "kind": "tile",
                        "split": "train",
                        "frame_idx": idx,
                        "tile_id": tile.tile_id,
                        "tile": {
                            "x0": tile.x0,
                            "y0": tile.y0,
                            "x1": tile.x1,
                            "y1": tile.y1,
                        },
                        "stem": tstem,
                        "sha256": dig,
                        "n_labels": len(t_lines),
                    }
                )
    finally:
        cap.release()

    dataset_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/dev",
        "names": {0: "human"},
        "nc": 1,
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema": "r1_f2c_tile_export_manifest_v1",
        "counts": counts,
        "tile_geometry": {
            "tile_w": tile_w,
            "tile_h": tile_h,
            "overlap_x": overlap_x,
            "overlap_y": overlap_y,
            "min_visible_frac": MIN_VISIBLE_FRAC,
            "n_tile_specs": len(tiles),
        },
        "dev_tiles_in_training": False,
        "holdout_excluded": True,
        "lineage_n": len(lineage),
        "written_at_utc": utc_now(),
    }
    atomic_write_json(out_root / "export_manifest.json", manifest, mode=0o600)
    atomic_write_json(out_root / "tile_lineage.json", {"items": lineage}, mode=0o600)
    return manifest


__all__ = [
    "DEFAULT_OVERLAP_X",
    "DEFAULT_OVERLAP_Y",
    "DEFAULT_TILE_H",
    "DEFAULT_TILE_W",
    "MIN_VISIBLE_FRAC",
    "build_train_tile_dataset",
    "clip_box_to_tile",
    "roundtrip_ok",
]
