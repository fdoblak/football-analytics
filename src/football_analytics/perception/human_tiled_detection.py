"""Human detection with full-frame + overlap tile fusion (R1-F1-R1).

Produces source-space xyxy person proposals on 1336x744. Does NOT assign team/role.
Pitch eligibility is a post-filter on foot-point vs green-pitch mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from football_analytics.acceptance.stage18_own_video.pipeline import (
    compute_pitch_masks,
    footpoint,
    point_inside,
)
from football_analytics.perception.adapters.base import RawPersonBox
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.candidate_merge import BallCandidate, class_aware_nms
from football_analytics.perception.tiling import crop_tile, generate_tiles, map_tile_bbox_to_source

SOURCE_WIDTH = 1336
SOURCE_HEIGHT = 744

Eligibility = Literal["on_pitch_human_candidate", "off_pitch_human", "uncertain"]


@dataclass(frozen=True)
class HumanProposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    eligibility: Eligibility
    source: str
    suppressed: bool = False

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HumanDetectConfig:
    name: str
    conf: float = 0.22
    predict_iou: float = 0.5
    merge_iou: float = 0.55
    imgsz_full: int = 960
    imgsz_tile: int = 640
    mode: Literal["full_frame", "tiled", "hybrid"] = "hybrid"
    tile_width: int = 672
    tile_height: int = 420
    overlap_x: int = 112
    overlap_y: int = 84
    max_tiles: int = 12
    min_area: float = 160.0
    min_h: float = 24.0
    max_h_frac: float = 0.60
    min_aspect: float = 0.15
    max_aspect: float = 1.05
    device: str = "auto"


def _clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, *, w: int, h: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def geometry_ok(
    box: tuple[float, float, float, float], cfg: HumanDetectConfig, frame_h: int
) -> bool:
    x1, y1, x2, y2 = box
    ww, hh = x2 - x1, y2 - y1
    if ww * hh < cfg.min_area or hh < cfg.min_h or hh > cfg.max_h_frac * frame_h:
        return False
    aspect = ww / hh if hh > 1e-6 else 0.0
    return cfg.min_aspect <= aspect <= cfg.max_aspect


def raw_to_candidates(
    boxes: Sequence[RawPersonBox],
    *,
    source: str,
    cfg: HumanDetectConfig,
    frame_w: int,
    frame_h: int,
) -> list[BallCandidate]:
    out: list[BallCandidate] = []
    for det in boxes:
        clipped = _clip_xyxy(det.x1, det.y1, det.x2, det.y2, w=frame_w, h=frame_h)
        if clipped is None:
            continue
        if not geometry_ok(clipped, cfg, frame_h):
            continue
        x1, y1, x2, y2 = clipped
        out.append(
            BallCandidate(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(det.score),
                class_id=0,
                class_name="person",
                candidate_source=source,
            )
        )
    return out


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def predict_full_frame(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    cfg: HumanDetectConfig,
) -> list[BallCandidate]:
    device = resolve_device(cfg.device)
    boxes = adapter.predict_persons(
        frame,
        conf=cfg.conf,
        iou=cfg.predict_iou,
        imgsz=cfg.imgsz_full,
        device=device,
        half=False,
        class_ids=[0],
        class_names=["person"],
        channel_order="bgr",
    )
    return raw_to_candidates(
        boxes,
        source="full_frame",
        cfg=cfg,
        frame_w=frame.shape[1],
        frame_h=frame.shape[0],
    )


def predict_tiled(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    cfg: HumanDetectConfig,
) -> list[BallCandidate]:
    h, w = frame.shape[:2]
    device = resolve_device(cfg.device)
    tiles = generate_tiles(
        w,
        h,
        tile_width=cfg.tile_width,
        tile_height=cfg.tile_height,
        overlap_x=cfg.overlap_x,
        overlap_y=cfg.overlap_y,
        max_tiles=cfg.max_tiles,
    )
    out: list[BallCandidate] = []
    for tile in tiles:
        crop = crop_tile(frame, tile)
        if crop.size == 0:
            continue
        boxes = adapter.predict_persons(
            crop,
            conf=cfg.conf,
            iou=cfg.predict_iou,
            imgsz=cfg.imgsz_tile,
            device=device,
            half=False,
            class_ids=[0],
            class_names=["person"],
            channel_order="bgr",
        )
        mapped: list[RawPersonBox] = []
        for det in boxes:
            sx1, sy1, sx2, sy2 = map_tile_bbox_to_source(
                (det.x1, det.y1, det.x2, det.y2),
                tile,
                coordinate_space="tile_local",
            )
            mapped.append(
                RawPersonBox(
                    x1=sx1,
                    y1=sy1,
                    x2=sx2,
                    y2=sy2,
                    score=float(det.score),
                    class_id=0,
                    class_name="person",
                )
            )
        out.extend(
            raw_to_candidates(
                mapped,
                source=f"tile:{tile.tile_id}",
                cfg=cfg,
                frame_w=w,
                frame_h=h,
            )
        )
    return out


def classify_eligibility(
    box_xyxy: tuple[float, float, float, float],
    masks: Any,
) -> Eligibility:
    xywh = (
        box_xyxy[0],
        box_xyxy[1],
        box_xyxy[2] - box_xyxy[0],
        box_xyxy[3] - box_xyxy[1],
    )
    fp = footpoint(xywh)
    if point_inside(masks.visible, fp):
        return "on_pitch_human_candidate"
    # near pitch edge: uncertain rather than hard off
    x, y = fp
    if 0 <= y < masks.visible.shape[0] and 0 <= x < masks.visible.shape[1]:
        y0 = max(0, y - 8)
        y1 = min(masks.visible.shape[0], y + 9)
        x0 = max(0, x - 8)
        x1 = min(masks.visible.shape[1], x + 9)
        if np.count_nonzero(masks.visible[y0:y1, x0:x1]) > 0:
            return "uncertain"
    return "off_pitch_human"


def detect_humans(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    cfg: HumanDetectConfig,
    *,
    apply_pitch: bool = True,
) -> list[HumanProposal]:
    """Run configured human detection and optional pitch eligibility."""
    if frame.shape[1] != SOURCE_WIDTH or frame.shape[0] != SOURCE_HEIGHT:
        # Still allow other sizes for tests; clip against actual shape.
        pass
    full: list[BallCandidate] = []
    tiled: list[BallCandidate] = []
    if cfg.mode in {"full_frame", "hybrid"}:
        full = predict_full_frame(adapter, frame, cfg)
    if cfg.mode in {"tiled", "hybrid"}:
        tiled = predict_tiled(adapter, frame, cfg)
    if cfg.mode == "full_frame":
        merged = class_aware_nms(full, merge_iou=cfg.merge_iou)
    elif cfg.mode == "tiled":
        merged = class_aware_nms(tiled, merge_iou=cfg.merge_iou)
    else:
        merged = class_aware_nms(full + tiled, merge_iou=cfg.merge_iou)

    masks = compute_pitch_masks(frame) if apply_pitch else None
    proposals: list[HumanProposal] = []
    for cand in merged:
        box = cand.as_xyxy()
        elig: Eligibility = "on_pitch_human_candidate"
        if masks is not None:
            elig = classify_eligibility(box, masks)
        proposals.append(
            HumanProposal(
                x1=box[0],
                y1=box[1],
                x2=box[2],
                y2=box[3],
                score=cand.score,
                eligibility=elig,
                source=cand.candidate_source,
            )
        )
    return proposals


def duplicate_pairs(proposals: Sequence[HumanProposal], *, iou_thresh: float = 0.9) -> int:
    from football_analytics.perception.detection_evaluation import bbox_iou

    n = 0
    for i, a in enumerate(proposals):
        for b in proposals[i + 1 :]:
            if bbox_iou(a.as_xyxy(), b.as_xyxy()) > iou_thresh:
                n += 1
    return n


def merged_person_candidates(proposals: Sequence[HumanProposal]) -> int:
    """Heuristic: unusually wide boxes relative to height (possible two-person merge)."""
    n = 0
    for p in proposals:
        w = p.x2 - p.x1
        h = p.y2 - p.y1
        if h <= 1:
            continue
        if w / h > 0.85 and w > 55:
            n += 1
    return n


__all__ = [
    "SOURCE_HEIGHT",
    "SOURCE_WIDTH",
    "HumanDetectConfig",
    "HumanProposal",
    "classify_eligibility",
    "detect_humans",
    "duplicate_pairs",
    "merged_person_candidates",
]
