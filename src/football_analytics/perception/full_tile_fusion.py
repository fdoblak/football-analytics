"""Full-frame + overlapping tile inference fusion for human detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.annotation.train_tiles import (
    DEFAULT_OVERLAP_X,
    DEFAULT_OVERLAP_Y,
    DEFAULT_TILE_H,
    DEFAULT_TILE_W,
)
from football_analytics.perception.candidate_merge import BallCandidate, class_aware_nms
from football_analytics.perception.detection_evaluation import BBoxDetection
from football_analytics.perception.tiling import (
    TileSpec,
    crop_tile,
    generate_tiles,
    map_tile_bbox_to_source,
)


@dataclass(frozen=True)
class FusionConfig:
    conf: float = 0.25
    predict_iou: float = 0.5
    merge_iou: float = 0.55
    imgsz: int = 960
    tile_w: int = DEFAULT_TILE_W
    tile_h: int = DEFAULT_TILE_H
    overlap_x: int = DEFAULT_OVERLAP_X
    overlap_y: int = DEFAULT_OVERLAP_Y
    max_tiles: int = 24
    mode: str = "hybrid"  # full_frame | tiled | hybrid
    device: str = "0"


def _yolo_to_cands(
    result: Any,
    *,
    source: str,
    tile: TileSpec | None = None,
    frame_w: int,
    frame_h: int,
) -> list[BallCandidate]:
    out: list[BallCandidate] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    names = result.names or {}
    for j, box in enumerate(xyxy):
        cname = names.get(int(cls[j]), "")
        if int(cls[j]) != 0 and cname not in {"person", "human"}:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if tile is not None:
            x1, y1, x2, y2 = map_tile_bbox_to_source(
                (x1, y1, x2, y2), tile, coordinate_space="tile_local"
            )
        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(
            BallCandidate(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(scores[j]),
                class_id=0,
                class_name="human",
                candidate_source=source,
            )
        )
    return out


def predict_full_tile_fused(
    model: Any,
    frame: np.ndarray,
    cfg: FusionConfig,
) -> list[BBoxDetection]:
    """Run full and/or tile inference; return fused source-space detections (frame_index unset)."""
    h, w = frame.shape[:2]
    full: list[BallCandidate] = []
    tiled: list[BallCandidate] = []
    if cfg.mode in {"full_frame", "hybrid"}:
        res = model.predict(
            source=frame,
            conf=cfg.conf,
            iou=cfg.predict_iou,
            imgsz=cfg.imgsz,
            verbose=False,
            device=cfg.device,
        )[0]
        full = _yolo_to_cands(res, source="full_frame", frame_w=w, frame_h=h)
    if cfg.mode in {"tiled", "hybrid"}:
        tiles = generate_tiles(
            w,
            h,
            tile_width=cfg.tile_w,
            tile_height=cfg.tile_h,
            overlap_x=cfg.overlap_x,
            overlap_y=cfg.overlap_y,
            max_tiles=cfg.max_tiles,
        )
        for tile in tiles:
            crop = crop_tile(frame, tile)
            if crop.size == 0:
                continue
            res = model.predict(
                source=crop,
                conf=cfg.conf,
                iou=cfg.predict_iou,
                imgsz=cfg.imgsz,
                verbose=False,
                device=cfg.device,
            )[0]
            tiled.extend(
                _yolo_to_cands(
                    res,
                    source=f"tile:{tile.tile_id}",
                    tile=tile,
                    frame_w=w,
                    frame_h=h,
                )
            )
    if cfg.mode == "full_frame":
        merged = class_aware_nms(full, merge_iou=cfg.merge_iou)
    elif cfg.mode == "tiled":
        merged = class_aware_nms(tiled, merge_iou=cfg.merge_iou)
    else:
        merged = class_aware_nms(full + tiled, merge_iou=cfg.merge_iou)
    return [
        BBoxDetection(
            frame_index=-1,
            entity_type="human",
            x1=c.x1,
            y1=c.y1,
            x2=c.x2,
            y2=c.y2,
            score=c.score,
        )
        for c in merged
    ]


def attach_frame_index(dets: Sequence[BBoxDetection], frame_index: int) -> list[BBoxDetection]:
    return [
        BBoxDetection(
            frame_index=frame_index,
            entity_type=d.entity_type,
            x1=d.x1,
            y1=d.y1,
            x2=d.x2,
            y2=d.y2,
            score=d.score,
        )
        for d in dets
    ]


__all__ = [
    "FusionConfig",
    "attach_frame_index",
    "predict_full_tile_fused",
]
