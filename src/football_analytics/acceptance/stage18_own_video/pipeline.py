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


def kit_brightness_and_hue(crop: np.ndarray) -> tuple[str, float, float, float]:
    """Return coarse kit label + mean V/S/H for this clip's white/yellow/red/dark kits."""
    stats = kit_torso_stats(crop)
    if stats is None:
        return "unknown", 0.0, 0.0, 0.0
    mean_v = float(stats["mean_v"])
    mean_s = float(stats["mean_s"])
    mean_h = float(stats["mean_h"])
    if stats["red"] >= 0.13 and stats["dark"] >= 0.84 and mean_v < 38.0 and mean_s >= 155.0:
        return "red_kit", mean_v, mean_s, mean_h
    if stats["yellow"] >= 0.12:
        return "yellow_kit", mean_v, mean_s, mean_h
    if stats["white"] >= 0.18 and mean_v >= 120.0:
        return "white_kit", mean_v, mean_s, mean_h
    if stats["dark"] >= 0.45 and mean_v < 90.0 and stats["red"] < 0.03:
        return "dark_kit", mean_v, mean_s, mean_h
    return "other_kit", mean_v, mean_s, mean_h


def kit_torso_stats(crop: np.ndarray) -> dict[str, float] | None:
    """Pitch-green-masked upper-torso color fractions for own-video role rules."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    torso = hsv[int(0.12 * h) : int(0.50 * h), int(0.20 * w) : int(0.80 * w)]
    if torso.size == 0:
        torso = hsv
    green = cv2.inRange(torso, (35, 40, 40), (95, 255, 255))
    nong = cv2.bitwise_not(green)
    if int(np.count_nonzero(nong)) < 20:
        nong = np.ones(torso.shape[:2], np.uint8) * 255
    red = cv2.bitwise_and(
        cv2.bitwise_or(
            cv2.inRange(torso, (0, 50, 40), (12, 255, 255)),
            cv2.inRange(torso, (160, 50, 40), (180, 255, 255)),
        ),
        nong,
    )
    dark = cv2.bitwise_and(cv2.inRange(torso, (0, 0, 0), (180, 255, 90)), nong)
    yellow = cv2.bitwise_and(cv2.inRange(torso, (15, 70, 80), (45, 255, 255)), nong)
    white = cv2.bitwise_and(cv2.inRange(torso, (0, 0, 150), (180, 70, 255)), nong)
    n = float(np.count_nonzero(nong))
    return {
        "red": float(np.count_nonzero(red)) / n,
        "dark": float(np.count_nonzero(dark)) / n,
        "yellow": float(np.count_nonzero(yellow)) / n,
        "white": float(np.count_nonzero(white)) / n,
        "mean_v": float(cv2.mean(torso[:, :, 2], mask=nong)[0]),
        "mean_s": float(cv2.mean(torso[:, :, 1], mask=nong)[0]),
        "mean_h": float(cv2.mean(torso[:, :, 0], mask=nong)[0]),
    }


def canonicalize_role_label(proposal_role: str, kit: str) -> str:
    """Map frame proposal labels to acceptance taxonomy roles."""
    if proposal_role in {"spectator", "outside_play_area_human", "staff_or_bench_candidate"}:
        return "staff"
    if proposal_role == "referee_or_dark_player_candidate" or (
        proposal_role == "on_field_player_candidate" and kit == "dark_kit"
    ):
        return "referee"
    if proposal_role in {"goalkeeper_candidate", "on_field_player_candidate"} and kit == "red_kit":
        return "goalkeeper"
    if proposal_role == "on_field_player_candidate":
        return "player"
    if proposal_role in {"player", "goalkeeper", "referee", "staff", "unknown"}:
        return proposal_role
    return "unknown"


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
    stats = kit_torso_stats(crop)
    kit, mean_v, mean_s, mean_h = kit_brightness_and_hue(crop)
    cy = y + h / 2

    # Scene-specific canonical decision (own-video white/yellow/dark-red GK / black ref)
    canonical = "unknown"
    team: str | None = "unknown"
    role = "unknown_human"
    if stats is None:
        canonical, team = "unknown", "unknown"
    elif not in_vis:
        role = "spectator" if cy > 0.85 * frame.shape[0] else "outside_play_area_human"
        canonical, team = "staff", None
    elif stats["yellow"] >= 0.12 or (stats["white"] >= 0.18 and stats["mean_v"] >= 120.0):
        role = "on_field_player_candidate"
        canonical = "player"
        team = "yellow" if stats["yellow"] >= stats["white"] else "white"
        kit = "yellow_kit" if team == "yellow" else "white_kit"
    elif (
        stats["red"] >= 0.13
        and stats["dark"] >= 0.84
        and stats["mean_v"] < 38.0
        and stats["mean_s"] >= 155.0
        and stats["yellow"] < 0.05
        and stats["white"] < 0.08
    ):
        role = "goalkeeper_candidate"
        canonical, team = "goalkeeper", "unknown"
        kit = "red_kit"
    elif (
        stats["dark"] >= 0.58
        and stats["mean_v"] < 70.0
        and stats["red"] < 0.02
        and stats["yellow"] < 0.05
        and stats["white"] < 0.05
        and stats["mean_s"] < 95.0
    ):
        role = "referee_or_dark_player_candidate"
        canonical, team = "referee", None
        kit = "dark_kit"
    elif in_vis and not in_int:
        role = "staff_or_bench_candidate"
        canonical, team = "staff", None
    else:
        role = "on_field_player_candidate"
        canonical, team = "player", "unknown"

    team_eligible = canonical == "player" and in_int and kit in {"white_kit", "yellow_kit"}
    return {
        "role": role,
        "canonical_role": canonical,
        "team": team,
        "kit": kit,
        "footpoint": list(fp),
        "in_visible_pitch": in_vis,
        "in_interior_pitch": in_int,
        "team_eligible": team_eligible,
        "mean_v": mean_v,
        "mean_s": mean_s,
        "mean_h": mean_h,
        "kit_stats": stats,
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
    "canonicalize_role_label",
    "classify_human_role",
    "compute_pitch_masks",
    "detect_balls",
    "detect_persons",
    "kit_brightness_and_hue",
    "kit_torso_stats",
    "_kit_hist",
]
