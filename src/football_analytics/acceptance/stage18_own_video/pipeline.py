"""Stage 18 own-video perception: pitch eligibility, roles, team, ball."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from football_analytics.acceptance.final_perception_repair.pipeline import (
    ConfirmedTracker,
    DetConfig,
    _iou,
    _kit_hist,
    _nms_xywh,
)
from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter

OWN_VIDEO_CFG = DetConfig(
    name="own_video_v1",
    conf=0.22,
    nms_iou=0.45,
    imgsz=960,
    tile_width=1336,
    tile_overlap=0,
    min_area=180.0,
    min_h=28.0,
    max_h_frac=0.55,
    min_aspect=0.18,
    max_aspect=0.95,
    center_merge_dist=10.0,
)


@dataclass
class PitchMasks:
    visible: np.ndarray
    interior: np.ndarray
    fence_y: int
    area_frac: float


def estimate_fence_y(frame: np.ndarray) -> int:
    """Far touchline / fence y from top of dominant green pitch blob."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (28, 20, 30), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # row-wise green fraction
    row_frac = green.mean(axis=1) / 255.0
    # find first row from top where green becomes substantial (pitch starts)
    thresh = 0.25
    ys = np.where(row_frac >= thresh)[0]
    if ys.size == 0:
        return int(0.22 * h)
    y_start = int(ys[0])
    # fence sits slightly above pitch green onset
    return max(0, y_start - 4)


def compute_pitch_masks(frame: np.ndarray) -> PitchMasks:
    """Tight playable pitch: green below far touchline + largest component + erosion."""
    h, w = frame.shape[:2]
    fence_y = estimate_fence_y(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (28, 20, 30), (95, 255, 255))
    # Drop sky/trees above far touchline
    green[: max(0, fence_y), :] = 0
    # Drop extreme bottom spectator strip unless strongly green
    bottom = int(0.90 * h)
    strong = cv2.inRange(hsv[bottom:, :], (28, 45, 40), (95, 255, 255))
    green[bottom:, :] = cv2.bitwise_and(green[bottom:, :], strong)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visible = np.zeros((h, w), np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 0.08 * h * w:
            cv2.drawContours(visible, [c], -1, 255, -1)
    # Milder erosion so on-field players near touchline keep eligibility
    interior = cv2.erode(visible, np.ones((17, 17), np.uint8), iterations=1)
    interior[: min(h, fence_y + 4), :] = 0
    area_frac = float(np.count_nonzero(visible)) / float(h * w)
    return PitchMasks(visible=visible, interior=interior, fence_y=fence_y, area_frac=area_frac)


def footpoint(box: tuple[float, float, float, float]) -> tuple[int, int]:
    x, y, w, h = box
    return int(x + w / 2), int(y + h - 1)


def point_inside(mask: np.ndarray, pt: tuple[int, int]) -> bool:
    x, y = pt
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x] > 0)


def detect_persons(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    *,
    cfg: DetConfig,
    device: str,
) -> list[tuple[tuple[float, float, float, float], float]]:
    boxes = adapter.predict_persons(
        frame,
        conf=cfg.conf,
        iou=0.5,
        imgsz=cfg.imgsz,
        device=device,
        half=False,
        class_ids=[0],
        class_names=["person"],
        channel_order="bgr",
    )
    raw: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    h = frame.shape[0]
    for det in boxes:
        x, y = float(det.x1), float(det.y1)
        w, hh = float(det.x2 - det.x1), float(det.y2 - det.y1)
        if w * hh < cfg.min_area or hh < cfg.min_h or hh > cfg.max_h_frac * h:
            continue
        aspect = w / hh if hh > 1e-6 else 0.0
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        raw.append((x, y, w, hh))
        scores.append(float(det.score))
    kept = _nms_xywh(raw, scores, iou_thresh=cfg.nms_iou)
    out: list[tuple[tuple[float, float, float, float], float]] = []
    for b in kept:
        best = 0.0
        for r, s in zip(raw, scores, strict=True):
            if _iou(b, r) > 0.9:
                best = max(best, s)
        out.append((b, best if best else cfg.conf))
    return out


def kit_brightness_and_hue(crop: np.ndarray) -> tuple[str, float, float]:
    """Return coarse kit label for white vs yellow teams on this video."""
    if crop.size == 0:
        return "unknown", 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # upper torso
    h = crop.shape[0]
    torso = hsv[int(0.15 * h) : int(0.55 * h), :]
    if torso.size == 0:
        torso = hsv
    mean_v = float(np.mean(torso[:, :, 2]))
    mean_s = float(np.mean(torso[:, :, 1]))
    mean_h = float(np.mean(torso[:, :, 0]))
    # yellow: high S, H around 20-40
    if mean_s > 70 and 15 <= mean_h <= 45 and mean_v > 80:
        return "yellow_kit", mean_v, mean_s
    if mean_v > 150 and mean_s < 70:
        return "white_kit", mean_v, mean_s
    if mean_v < 80 and mean_s < 80:
        return "dark_kit", mean_v, mean_s
    return "other_kit", mean_v, mean_s


def classify_human_role(
    box: tuple[float, float, float, float],
    frame: np.ndarray,
    masks: PitchMasks,
) -> dict[str, Any]:
    """Frame-level role proposal — track aggregation happens later."""
    fp = footpoint(box)
    in_vis = point_inside(masks.visible, fp)
    in_int = point_inside(masks.interior, fp)
    x, y, w, h = box
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
    crop = frame[y1:y2, x1:x2]
    kit, mean_v, mean_s = kit_brightness_and_hue(crop)
    # Heuristics for this clip geometry
    cy = y + h / 2
    role = "unknown_human"
    if not in_vis:
        # below bottom stand / above fence
        if cy > 0.85 * frame.shape[0]:
            role = "spectator"
        elif fp[1] < masks.fence_y + 20:
            role = "outside_play_area_human"
        else:
            role = "outside_play_area_human"
    elif in_vis and not in_int:
        # touchline buffer — staff/bench candidates
        role = "staff_or_bench_candidate"
    else:
        # interior pitch
        if kit == "dark_kit" and mean_v < 90:
            # referee often all-black; GK dark long sleeve near goal — leave ambiguous
            role = "referee_or_dark_player_candidate"
        else:
            role = "on_field_player_candidate"

    team_eligible = role == "on_field_player_candidate" and in_int and kit in {
        "white_kit",
        "yellow_kit",
    }
    return {
        "role": role,
        "kit": kit,
        "footpoint": list(fp),
        "in_visible_pitch": in_vis,
        "in_interior_pitch": in_int,
        "team_eligible": team_eligible,
        "mean_v": mean_v,
        "mean_s": mean_s,
        "bbox": list(box),
    }


def detect_balls(
    adapter: UltralyticsBallAdapter,
    frame: np.ndarray,
    masks: PitchMasks,
    *,
    device: str,
) -> list[dict[str, Any]]:
    dets = adapter.predict_balls(
        frame,
        conf=0.08,
        iou=0.3,
        imgsz=960,
        device=device,
        half=False,
        class_ids=[32],
        class_names=["sports ball"],
        channel_order="bgr",
    )
    out: list[dict[str, Any]] = []
    diag = float(np.hypot(frame.shape[1], frame.shape[0]))
    for det in dets:
        bw = float(det.x2 - det.x1)
        bh = float(det.y2 - det.y1)
        if not (2.0 <= bw <= 45.0 and 2.0 <= bh <= 45.0):
            continue
        aspect = bw / bh if bh > 1e-6 else 0.0
        if aspect < 0.45 or aspect > 2.2:
            continue
        cx = 0.5 * (det.x1 + det.x2)
        cy = 0.5 * (det.y1 + det.y2)
        if not point_inside(masks.visible, (int(cx), int(cy))):
            continue
        score = float(det.score)
        state = "observed" if score >= 0.28 else ("candidate" if score >= 0.12 else "ambiguous")
        out.append(
            {
                "bbox": [float(det.x1), float(det.y1), bw, bh],
                "center": [cx, cy],
                "score": score,
                "state": state,
                "norm_size": float(max(bw, bh) / diag),
            }
        )
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:5]


__all__ = [
    "OWN_VIDEO_CFG",
    "PitchMasks",
    "ConfirmedTracker",
    "classify_human_role",
    "compute_pitch_masks",
    "detect_balls",
    "detect_persons",
    "kit_brightness_and_hue",
    "_kit_hist",
]
