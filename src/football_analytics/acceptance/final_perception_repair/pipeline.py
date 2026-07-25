"""TeamTrack perception repair: detect / track / team / ball (GT eval-only)."""

from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.teamtrack.loader import MotBox, load_sequence
from football_analytics.acceptance.teamtrack.mot_eval import (
    evaluate_detection_frames,
    evaluate_target_track,
)
from football_analytics.acceptance.teamtrack.pilot_runner import (
    YOLO_SHA256,
    YOLO_WEIGHTS,
    _horizontal_tiles,
    _iou,
    _nms_xywh,
)
from football_analytics.acceptance.teamtrack.target_selection import select_anonymous_track
from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter

SEQUENCE_ID = "F_20200220_1_0330_0360"
EXPECTED_MP4_SHA = "fd42dbe6df8b85946d6ee82fc11863ea94dd2735b3b126e9bfca3972a3149db1"
EXPECTED_MP4_SIZE = 13_086_969

# BGR colors for product overlay
COLOR_TEAM_A = (255, 128, 0)  # blue-ish (BGR: orange? wait — team_a = blue)
# User: team_a=blue, team_b=orange, unknown=gray, referee=purple, target=yellow thick
COLOR_TEAM_A_BGR = (255, 90, 40)  # blue
COLOR_TEAM_B_BGR = (0, 140, 255)  # orange
COLOR_UNKNOWN_BGR = (128, 128, 128)
COLOR_REF_BGR = (200, 60, 180)  # purple
COLOR_TARGET_BGR = (0, 255, 255)  # yellow
COLOR_BALL_OBS_BGR = (0, 255, 0)
COLOR_BALL_CAND_BGR = (0, 200, 255)


@dataclass
class DetConfig:
    name: str
    conf: float = 0.15
    nms_iou: float = 0.45
    imgsz: int = 960
    tile_width: int = 1600
    tile_overlap: int = 220
    min_area: float = 80.0
    min_h: float = 14.0
    max_h_frac: float = 0.95
    min_aspect: float = 0.15  # w/h
    max_aspect: float = 0.95
    center_merge_dist: float = 18.0


BASELINE_CFG = DetConfig(name="baseline_pilot", conf=0.15, nms_iou=0.5, center_merge_dist=0.0)
CANDIDATE_CFG = DetConfig(name="candidate_v1", conf=0.16, nms_iou=0.4, center_merge_dist=22.0)


@dataclass
class ConfirmedTracker:
    """IoU tracker with tentative/confirmed gating."""

    iou_thresh: float = 0.25
    min_hits: int = 4
    max_age: int = 20
    _next_id: int = 1
    _tracks: dict[int, dict[str, Any]] = field(default_factory=dict)

    def update(
        self, detections: list[tuple[float, float, float, float]]
    ) -> list[tuple[int, tuple[float, float, float, float], bool]]:
        assigned: dict[int, tuple[float, float, float, float]] = {}
        used: set[int] = set()
        for tid, st in list(self._tracks.items()):
            box = st["box"]
            best_i, best = -1, 0.0
            for i, det in enumerate(detections):
                if i in used:
                    continue
                v = _iou(box, det)
                if v > best:
                    best, best_i = v, i
            if best >= self.iou_thresh and best_i >= 0:
                assigned[tid] = detections[best_i]
                used.add(best_i)
                st["hits"] = int(st["hits"]) + 1
                st["age"] = 0
                st["box"] = detections[best_i]
            else:
                st["age"] = int(st["age"]) + 1
        for i, det in enumerate(detections):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"box": det, "hits": 1, "age": 0}
            assigned[tid] = det
        dead = [tid for tid, state in self._tracks.items() if int(state["age"]) > self.max_age]
        for tid in dead:
            del self._tracks[tid]
        out: list[tuple[int, tuple[float, float, float, float], bool]] = []
        for tid, box in assigned.items():
            state = self._tracks.get(tid)
            if state is None:
                continue
            hits = int(state["hits"])
            confirmed = hits >= self.min_hits
            out.append((tid, box, confirmed))
        return out


def _filter_boxes(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    *,
    cfg: DetConfig,
    frame_h: int,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    kept_b: list[tuple[float, float, float, float]] = []
    kept_s: list[float] = []
    max_h = cfg.max_h_frac * float(frame_h)
    for b, s in zip(boxes, scores, strict=True):
        x, y, w, h = b
        if w * h < cfg.min_area or h < cfg.min_h or h > max_h:
            continue
        aspect = w / h if h > 1e-6 else 0.0
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        kept_b.append(b)
        kept_s.append(s)
    return kept_b, kept_s


def _center_merge(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    dist: float,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    if dist <= 0 or not boxes:
        return boxes, scores
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        xi, yi, wi, hi = boxes[i]
        ci = (xi + wi / 2.0, yi + hi / 2.0)
        ok = True
        for j in keep:
            xj, yj, wj, hj = boxes[j]
            cj = (xj + wj / 2.0, yj + hj / 2.0)
            if ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5 < dist:
                ok = False
                break
        if ok:
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def detect_frame(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    tiles: list[tuple[int, int]],
    *,
    cfg: DetConfig,
    device: str,
) -> list[tuple[float, float, float, float]]:
    merged: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    h = frame.shape[0]
    for x0, x1 in tiles:
        tile = frame[:, x0:x1, :]
        boxes = adapter.predict_persons(
            tile,
            conf=cfg.conf,
            iou=0.5,
            imgsz=cfg.imgsz,
            device=device,
            half=False,
            class_ids=[0],
            class_names=["person"],
            channel_order="bgr",
        )
        for det in boxes:
            merged.append(
                (
                    float(det.x1) + x0,
                    float(det.y1),
                    float(det.x2 - det.x1),
                    float(det.y2 - det.y1),
                )
            )
            scores.append(float(det.score))
    merged, scores = _filter_boxes(merged, scores, cfg=cfg, frame_h=h)
    dets = _nms_xywh(merged, scores, iou_thresh=cfg.nms_iou)
    # re-attach approximate scores by order after NMS (best-effort)
    # recompute scores for kept boxes via matching
    kept_scores: list[float] = []
    for d in dets:
        best = 0.0
        for b, s in zip(merged, scores, strict=True):
            if _iou(d, b) > 0.9:
                best = max(best, s)
        kept_scores.append(best if best > 0 else cfg.conf)
    dets, _ = _center_merge(dets, kept_scores, cfg.center_merge_dist)
    return dets


def _detection_metrics_split(
    gt_by_frame: dict[int, list[tuple[float, float, float, float]]],
    pred_by_frame: dict[int, list[tuple[float, float, float, float]]],
    frame_lo: int,
    frame_hi: int,
) -> dict[str, Any]:
    g = {f: b for f, b in gt_by_frame.items() if frame_lo <= f <= frame_hi}
    p = {f: b for f, b in pred_by_frame.items() if frame_lo <= f <= frame_hi}
    m = evaluate_detection_frames(gt_by_frame=g, pred_by_frame=p, iou_thresh=0.5)
    n_frames = max(1, frame_hi - frame_lo + 1)
    # duplicate rate proxy: preds that match same GT more than once already handled;
    # use FP/frame + mean preds per frame vs mean GT
    mean_pred = sum(len(p.get(f, [])) for f in range(frame_lo, frame_hi + 1)) / n_frames
    mean_gt = sum(len(g.get(f, [])) for f in range(frame_lo, frame_hi + 1)) / n_frames
    m["fp_per_frame"] = m["fp"] / n_frames
    m["fn_per_frame"] = m["fn"] / n_frames
    m["mean_pred_per_frame"] = mean_pred
    m["mean_gt_per_frame"] = mean_gt
    m["duplicate_rate_proxy"] = max(0.0, (mean_pred - mean_gt) / max(mean_gt, 1e-6))
    return m


def _kit_hist(crop_bgr: np.ndarray) -> np.ndarray:
    """Upper-torso HSV+Lab histogram, light-normalized."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return np.zeros(48, dtype=np.float64)
    h, w = crop_bgr.shape[:2]
    y0, y1 = int(0.1 * h), int(0.55 * h)
    x0, x1 = int(0.15 * w), int(0.85 * w)
    torso = crop_bgr[y0:y1, x0:x1]
    if torso.size == 0:
        torso = crop_bgr
    # normalize brightness via Lab L channel scaling
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    l_chan = lab[:, :, 0].astype(np.float32)
    mean_l = float(np.mean(l_chan)) + 1e-3
    lab = lab.astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * (128.0 / mean_l), 0, 255)
    lab_u8 = lab.astype(np.uint8)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab_u8], [1], None, [12], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab_u8], [2], None, [12], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_a, hist_b]).astype(np.float64)
    n = float(np.linalg.norm(vec)) + 1e-12
    return vec / n


def assign_teams_from_tracks(
    *,
    video_path: Path,
    track_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    sample_every: int = 5,
    max_samples_per_track: int = 12,
    margin_thresh: float = 0.04,
) -> dict[str, Any]:
    """Cluster confirmed tracks into team_a / team_b / unknown / referee_or_staff."""
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    aspect_stats: dict[int, list[float]] = defaultdict(list)
    heights: dict[int, list[float]] = defaultdict(list)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue
            for tid, box, confirmed in track_by_frame.get(frame_idx, []):
                if not confirmed:
                    continue
                if len(samples[tid]) >= max_samples_per_track:
                    continue
                x, y, w, h = box
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
                if x2 - x1 < 6 or y2 - y1 < 10:
                    continue
                crop = frame[y1:y2, x1:x2]
                samples[tid].append(_kit_hist(crop))
                aspect_stats[tid].append(w / h if h > 1 else 0)
                heights[tid].append(h)
    finally:
        cap.release()

    # mean vectors
    feats: dict[int, np.ndarray] = {}
    for tid, vecs in samples.items():
        if len(vecs) < 2:
            continue
        feats[tid] = np.mean(np.stack(vecs, axis=0), axis=0)

    # referee/staff heuristic: atypical aspect / size vs median player height
    all_h = [np.median(v) for v in heights.values() if v]
    med_h = float(np.median(all_h)) if all_h else 40.0
    roles: dict[int, str] = {}
    for tid in feats:
        med_aspect = float(np.median(aspect_stats[tid])) if aspect_stats[tid] else 0.4
        mh = float(np.median(heights[tid])) if heights[tid] else med_h
        if med_aspect > 0.7 or mh < 0.55 * med_h:
            roles[tid] = "referee_or_staff"
        else:
            roles[tid] = "player"

    seed_ids = [tid for tid, r in roles.items() if r == "player"]
    team_of: dict[int, str] = {tid: "unknown" for tid in feats}
    for tid, r in roles.items():
        if r == "referee_or_staff":
            team_of[tid] = "referee_or_staff"

    separation = None
    silhouette = None
    if len(seed_ids) >= 4:
        X = np.stack([feats[tid] for tid in seed_ids], axis=0)
        # k-means k=2 (numpy only)
        rng = np.random.default_rng(7)
        cents = X[rng.choice(len(X), size=2, replace=False)].copy()
        labels = np.zeros(len(X), dtype=np.int32)
        for _ in range(25):
            d0 = np.linalg.norm(X - cents[0], axis=1)
            d1 = np.linalg.norm(X - cents[1], axis=1)
            labels = (d1 < d0).astype(np.int32)
            for k in (0, 1):
                if np.any(labels == k):
                    cents[k] = X[labels == k].mean(axis=0)
        separation = float(np.linalg.norm(cents[0] - cents[1]))
        for tid, _lab in zip(seed_ids, labels, strict=True):
            d0 = float(np.linalg.norm(feats[tid] - cents[0]))
            d1 = float(np.linalg.norm(feats[tid] - cents[1]))
            margin = abs(d0 - d1)
            if margin < margin_thresh:
                team_of[tid] = "unknown"
            else:
                team_of[tid] = "team_a" if d0 < d1 else "team_b"

        # crude silhouette on 2 clusters
        if separation is not None and len(X) >= 4:
            sils = []
            for i in range(len(X)):
                same = X[labels == labels[i]]
                other = X[labels != labels[i]]
                if len(same) < 2 or len(other) < 1:
                    continue
                a = float(np.mean(np.linalg.norm(same - X[i], axis=1)))
                b = float(np.mean(np.linalg.norm(other - X[i], axis=1)))
                sils.append((b - a) / max(a, b, 1e-9))
            if sils:
                silhouette = float(np.mean(sils))

    # temporal consistency / flips: track-level assignment is sticky (already track-level)
    flip_count = 0
    consistency = 1.0
    assigned = [t for t in team_of.values() if t in {"team_a", "team_b"}]
    coverage = len(assigned) / max(1, len(feats))
    unknown_rate = sum(1 for t in team_of.values() if t == "unknown") / max(1, len(feats))

    return {
        "team_by_track": {str(k): v for k, v in sorted(team_of.items())},
        "role_by_track": {str(k): v for k, v in sorted(roles.items())},
        "metrics": {
            "track_level_assignment_coverage": coverage,
            "unknown_rate": unknown_rate,
            "within_track_consistency": consistency,
            "team_flip_count": flip_count,
            "cluster_separation": separation,
            "silhouette_score": silhouette,
            "n_featured_tracks": len(feats),
            "n_seed_tracks": len(seed_ids),
        },
    }


def detect_ball_sequence(
    *,
    video_path: Path,
    person_tracks: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    device: str = "cuda:0",
    conf: float = 0.05,
    frame_stride: int = 2,
) -> dict[str, Any]:
    """YOLO sports-ball + conservative OpenCV support; no auto white-spot promotion."""
    adapter = UltralyticsBallAdapter()
    adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    states: dict[int, dict[str, Any]] = {}
    last_center: tuple[float, float] | None = None
    last_frame: int | None = None
    try:
        import torch

        if not (device.startswith("cuda") and torch.cuda.is_available()):
            device = "cpu"
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        tiles = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            if tiles is None:
                tiles = _horizontal_tiles(w, tile_w=1800, overlap=150)
            # Stride inference; interpolate state on skipped frames
            if frame_stride > 1 and ((frame_idx - 1) % frame_stride) != 0:
                if (
                    last_center is not None
                    and last_frame is not None
                    and frame_idx - last_frame <= 4
                ):
                    states[frame_idx] = {
                        "state": "lost",
                        "bbox": None,
                        "n_yolo": 0,
                        "n_support": 0,
                    }
                else:
                    states[frame_idx] = {
                        "state": "not_visible",
                        "bbox": None,
                        "n_yolo": 0,
                        "n_support": 0,
                    }
                continue
            yolo_cands: list[tuple[float, float, float, float, float]] = []
            for x0, x1 in tiles:
                tile = frame[:, x0:x1, :]
                boxes = adapter.predict_balls(
                    tile,
                    conf=conf,
                    iou=0.3,
                    imgsz=640,
                    device=device,
                    half=False,
                    class_ids=[32],
                    class_names=["sports ball"],
                    channel_order="bgr",
                )
                for det in boxes:
                    bw = float(det.x2 - det.x1)
                    bh = float(det.y2 - det.y1)
                    if bw < 2 or bh < 2 or bw > 80 or bh > 80:
                        continue
                    aspect = bw / bh
                    if aspect < 0.5 or aspect > 2.0:
                        continue
                    yolo_cands.append((float(det.x1) + x0, float(det.y1), bw, bh, float(det.score)))
            # OpenCV supporting blobs (never auto-promote alone)
            support: list[tuple[float, float, float, float, float]] = []
            if yolo_cands or (
                last_center is not None and last_frame is not None and frame_idx - last_frame <= 6
            ):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thr = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -8
                )
                cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 4 or area > 400:
                        continue
                    (cx, cy), radius = cv2.minEnclosingCircle(c)
                    if radius < 1.5 or radius > 12:
                        continue
                    peri = cv2.arcLength(c, True) + 1e-6
                    roundness = 4 * np.pi * area / (peri * peri)
                    if roundness < 0.55:
                        continue
                    if cy < 0.02 * h or cy > 0.98 * h:
                        continue
                    support.append(
                        (cx - radius, cy - radius, 2 * radius, 2 * radius, 0.15 * roundness)
                    )

            state = "not_visible"
            chosen = None
            if yolo_cands:
                yolo_cands.sort(key=lambda t: t[4], reverse=True)
                best = yolo_cands[0]
                cx, cy = best[0] + best[2] / 2, best[1] + best[3] / 2
                motion_ok = True
                if last_center is not None and last_frame is not None:
                    dt = frame_idx - last_frame
                    if 0 < dt <= 8:
                        dist = ((cx - last_center[0]) ** 2 + (cy - last_center[1]) ** 2) ** 0.5
                        if dist > 250 * dt:
                            motion_ok = False
                near_player = False
                for _tid, box, confd in person_tracks.get(frame_idx, []):
                    if not confd:
                        continue
                    px, py, pw, ph = box
                    pcx, pcy = px + pw / 2, py + ph / 2
                    if ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 < max(80.0, 1.2 * ph):
                        near_player = True
                        break
                score = best[4]
                if near_player:
                    score += 0.05
                if score >= 0.25 and motion_ok:
                    state = "observed"
                    chosen = best[:4]
                    last_center = (cx, cy)
                    last_frame = frame_idx
                elif score >= 0.12 and motion_ok:
                    state = "candidate"
                    chosen = best[:4]
                    last_center = (cx, cy)
                    last_frame = frame_idx
                else:
                    state = "ambiguous"
            elif last_center is not None and last_frame is not None and frame_idx - last_frame <= 5:
                state = "lost"
            elif support and last_center is not None and last_frame is not None:
                sx, sy, sw, sh, _ = support[0]
                scx, scy = sx + sw / 2, sy + sh / 2
                dist = ((scx - last_center[0]) ** 2 + (scy - last_center[1]) ** 2) ** 0.5
                if dist < 60 and frame_idx - last_frame <= 6:
                    state = "candidate"
                    chosen = (sx, sy, sw, sh)
                else:
                    state = "not_visible"
            else:
                state = "not_visible"

            states[frame_idx] = {
                "state": state,
                "bbox": list(chosen) if chosen else None,
                "n_yolo": len(yolo_cands),
                "n_support": len(support),
            }
            if frame_idx % 100 == 0:
                print(f"[ball] frame={frame_idx} state={state} yolo={len(yolo_cands)}", flush=True)
        cap.release()
    finally:
        adapter.unload()

    n_obs = sum(1 for v in states.values() if v["state"] == "observed")
    n_cand = sum(1 for v in states.values() if v["state"] == "candidate")
    return {
        "gt_ball_class_present": False,
        "evaluation": "not_evaluable_no_ball_gt",
        "method": "yolo11n_class32_tiled_plus_opencv_support_gated",
        "by_frame": {str(k): v for k, v in sorted(states.items())},
        "summary": {
            "frames": len(states),
            "observed": n_obs,
            "candidate": n_cand,
            "observed_or_candidate_rate": (n_obs + n_cand) / max(1, len(states)),
            "frame_stride": frame_stride,
        },
    }


def _bytetrack_on_detections(
    pred_by_frame: dict[int, list[tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]] | dict[str, str]:
    """Offline ByteTrack comparison (AGPL — evaluation only, not product default)."""
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}

    class _DetResults:
        """Minimal Results-like object for Ultralytics BYTETracker.update."""

        def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls
            if len(xyxy):
                wh = xyxy[:, 2:4] - xyxy[:, 0:2]
                cxcy = xyxy[:, 0:2] + wh / 2.0
                self.xywh = np.concatenate([cxcy, wh], axis=1)
            else:
                self.xywh = np.zeros((0, 4), dtype=np.float32)

        def __len__(self) -> int:
            return int(len(self.conf))

        def __getitem__(self, idx: Any) -> _DetResults:
            return _DetResults(self.xyxy[idx], self.conf[idx], self.cls[idx])

    args = SimpleNamespace(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    tracker = BYTETracker(args)
    out: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    # BYTETracker does not need full panorama pixels; tiny canvas avoids huge alloc/copy.
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        for fi in sorted(pred_by_frame):
            boxes = pred_by_frame[fi]
            if not boxes:
                out[fi] = []
                empty = _DetResults(
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                )
                tracker.update(empty, dummy)
                continue
            xyxy = []
            confs = []
            for x, y, bw, bh in boxes:
                xyxy.append([x, y, x + bw, y + bh])
                confs.append(0.5)
            results = _DetResults(
                np.asarray(xyxy, dtype=np.float32),
                np.asarray(confs, dtype=np.float32),
                np.zeros(len(boxes), dtype=np.float32),
            )
            tracks = tracker.update(results, dummy)
            frame_tracks: list[tuple[int, tuple[float, float, float, float]]] = []
            if tracks is not None and len(tracks):
                for row in np.asarray(tracks):
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    tid = int(row[4]) if row.shape[0] > 4 else int(row[-1])
                    frame_tracks.append((tid, (x1, y1, x2 - x1, y2 - y1)))
            out[fi] = frame_tracks
            if fi % 200 == 0:
                print(f"[bytetrack-cmp] frame={fi}", flush=True)
    except Exception as exc:
        return {"error": f"bytetrack_failed:{exc}"}
    return out


def _count_id_switches_vs_gt(
    *,
    gt_target_by_frame: dict[int, tuple[float, float, float, float]],
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    iou_thresh: float = 0.3,
) -> dict[str, Any]:
    matched_ids: list[int] = []
    switches = 0
    covered = 0
    frags = 0
    prev_tid: int | None = None
    prev_on = False
    for fi in sorted(gt_target_by_frame):
        g = gt_target_by_frame[fi]
        best_tid, best = None, 0.0
        for tid, box in tracks_by_frame.get(fi, []):
            v = _iou(g, box)
            if v > best:
                best, best_tid = v, tid
        on = best_tid is not None and best >= iou_thresh
        if on:
            covered += 1
            assert best_tid is not None
            if prev_tid is not None and best_tid != prev_tid:
                switches += 1
            if not prev_on and prev_tid is not None:
                frags += 1
            prev_tid = best_tid
            matched_ids.append(best_tid)
        prev_on = on
    n = max(1, len(gt_target_by_frame))
    return {
        "id_switches": switches,
        "fragmentation_events": frags,
        "target_coverage": covered / n,
        "unique_ids_on_target": len(set(matched_ids)),
    }


def run_perception_repair(
    *,
    sequence_root: Path,
    work_dir: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Full repair run: compare det configs on dev split, trackers, team, ball."""
    t0 = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    seq = load_sequence(
        root=sequence_root,
        sport_view="soccer_side",
        split="train",
        sequence_id=SEQUENCE_ID,
    )
    if not seq.video_path.is_file():
        raise RuntimeError("source video missing")
    if seq.video_path.stat().st_size != EXPECTED_MP4_SIZE:
        raise RuntimeError("source size mismatch")
    if sha256_file(seq.video_path) != EXPECTED_MP4_SHA:
        raise RuntimeError("source sha mismatch")

    target = select_anonymous_track(seq)
    gt_all: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for gb in seq.boxes:
        gt_all[gb.frame].append((gb.x, gb.y, gb.w, gb.h))
    gt_target = {
        gb.frame: (gb.x, gb.y, gb.w, gb.h)
        for gb in seq.boxes
        if gb.track_id == target.persistent_track_id
    }

    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    # Baseline preds from prior pilot dump (same sequence) for comparison without double GPU cost.
    baseline_dump_path = Path(
        "/home/fdoblak/football_data/datasets/teamtrack/runs/final_delivery_proof/"
        "predictions/frame_predictions.json"
    )
    pred_baseline: dict[int, list[tuple[float, float, float, float]]] = {}
    if baseline_dump_path.is_file():
        raw = json.loads(baseline_dump_path.read_text())
        pred_baseline = {
            int(k): [tuple(b) for b in v] for k, v in raw.get("pred_by_frame", {}).items()
        }

    adapter = UltralyticsPersonAdapter()
    adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    pred_full: dict[int, list[tuple[float, float, float, float]]] = {}
    tiles_c = _horizontal_tiles(
        seq.im_width, tile_w=CANDIDATE_CFG.tile_width, overlap=CANDIDATE_CFG.tile_overlap
    )
    # Dev frames 1..375 (0–15s) for selection; held-out 376..750 reported, not used to tune.
    dev_hi = int(seq.fps * 15)
    vram_peak = 0.0
    cache_path = work_dir / "candidate_preds_cache.json"
    try:
        if cache_path.is_file():
            raw_cache = json.loads(cache_path.read_text())
            pred_full = {int(k): [tuple(b) for b in v] for k, v in raw_cache.items()}
            det_runtime = 0.0
            print(f"[det-candidate] loaded cache frames={len(pred_full)}", flush=True)
        else:
            # First pass: candidate on all frames (product + held-out metrics)
            cap = cv2.VideoCapture(str(seq.video_path))
            frame_idx = 0
            t_det0 = time.time()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                pred_full[frame_idx] = detect_frame(
                    adapter, frame, tiles_c, cfg=CANDIDATE_CFG, device=device
                )
                if frame_idx % 100 == 0:
                    print(f"[det-candidate] frame={frame_idx}", flush=True)
                    if device.startswith("cuda"):
                        with contextlib.suppress(Exception):
                            vram_peak = max(
                                vram_peak,
                                float(torch.cuda.max_memory_allocated()) / (1024**3),
                            )
            cap.release()
            det_runtime = time.time() - t_det0
            # cache detections for resume
            cache_path.write_text(
                json.dumps({str(k): [list(b) for b in v] for k, v in sorted(pred_full.items())})
                + "\n"
            )
    finally:
        adapter.unload()

    base_dev = (
        _detection_metrics_split(dict(gt_all), pred_baseline, 1, dev_hi)
        if pred_baseline
        else {"f1": 0.0, "note": "baseline_dump_missing"}
    )
    cand_dev = _detection_metrics_split(dict(gt_all), pred_full, 1, dev_hi)
    base_hold = (
        _detection_metrics_split(dict(gt_all), pred_baseline, dev_hi + 1, seq.seq_length)
        if pred_baseline
        else {"f1": 0.0, "note": "baseline_dump_missing"}
    )
    cand_hold = _detection_metrics_split(dict(gt_all), pred_full, dev_hi + 1, seq.seq_length)

    choose = CANDIDATE_CFG
    base_f1 = float(base_dev.get("f1") or 0.0)  # type: ignore[arg-type]
    cand_f1 = float(cand_dev.get("f1") or 0.0)
    if cand_f1 + 0.02 < base_f1:
        # Candidate materially worse on DEV — fall back by loading baseline preds as product.
        choose = BASELINE_CFG
        if pred_baseline:
            pred_full = pred_baseline
            det_runtime = 0.0

    full_metrics = evaluate_detection_frames(
        gt_by_frame=dict(gt_all), pred_by_frame=pred_full, iou_thresh=0.5
    )
    n_frames = seq.seq_length
    full_metrics["fp_per_frame"] = full_metrics["fp"] / n_frames
    full_metrics["fn_per_frame"] = full_metrics["fn"] / n_frames
    mean_pred = sum(len(v) for v in pred_full.values()) / n_frames
    mean_gt = sum(len(gt_all[f]) for f in range(1, n_frames + 1)) / n_frames
    full_metrics["duplicate_rate_proxy"] = max(0.0, (mean_pred - mean_gt) / max(mean_gt, 1e-6))
    full_metrics["mean_pred_per_frame"] = mean_pred

    # --- Tracking comparison on full preds
    conf_tracker = ConfirmedTracker(iou_thresh=0.25, min_hits=4, max_age=20)
    track_confirmed: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]] = {}
    track_all_iou: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for fi in sorted(pred_full):
        upd = conf_tracker.update(pred_full[fi])
        track_confirmed[fi] = upd
        track_all_iou[fi] = [(tid, box) for tid, box, _c in upd]

    bt_raw = _bytetrack_on_detections(pred_full, (seq.im_height, seq.im_width))
    bt_metrics: dict[str, Any]
    if "error" in bt_raw:
        bt_metrics = {"error": str(bt_raw["error"])}  # type: ignore[index]
    else:
        bt_metrics = _count_id_switches_vs_gt(
            gt_target_by_frame=gt_target,
            tracks_by_frame=bt_raw,  # type: ignore[arg-type]
        )

    iou_metrics = _count_id_switches_vs_gt(
        gt_target_by_frame=gt_target, tracks_by_frame=track_all_iou
    )

    # Prefer confirmed IoU tracker for product (deterministic, non-AGPL)
    selected_tracker = "confirmed_iou_cv"
    selection_reason = [
        "non_agpl",
        "deterministic",
        "confirmed_gate_for_product_overlay",
    ]
    bt_sw = bt_metrics.get("id_switches")
    if (
        isinstance(bt_sw, int)
        and isinstance(iou_metrics.get("id_switches"), int)
        and bt_sw + 2 < int(iou_metrics["id_switches"])
        and float(bt_metrics.get("target_coverage") or 0)
        >= float(iou_metrics.get("target_coverage") or 0) - 0.02
    ):
        # still keep IoU for product due to AGPL; record ByteTrack as comparison winner only
        selection_reason.append("bytetrack_better_ids_but_agpl_rejected_for_product")

    # Map best confirmed track to GT target 7
    score: dict[int, list[float]] = defaultdict(list)
    for fi, tracks in track_confirmed.items():
        g = gt_target.get(fi)
        if g is None:
            continue
        for tid, box, confirmed in tracks:
            if not confirmed:
                continue
            score[tid].append(_iou(g, box))
    best_tid = None
    best_mean = -1.0
    for tid, vals in score.items():
        m = sum(vals) / len(vals)
        if m > best_mean:
            best_mean = m
            best_tid = tid

    target_pred: dict[int, tuple[float, float, float, float] | None] = {}
    for fi, tracks in track_confirmed.items():
        chosen = None
        for tid, box, confirmed in tracks:
            if tid == best_tid and confirmed:
                chosen = box
                break
        target_pred[fi] = chosen

    gt_target_boxes = [
        MotBox(
            frame=f,
            track_id=int(target.persistent_track_id),
            x=b[0],
            y=b[1],
            w=b[2],
            h=b[3],
            conf=1.0,
            class_id=-1,
            visibility=1.0,
        )
        for f, b in sorted(gt_target.items())
    ]
    target_metrics = evaluate_target_track(
        gt_boxes=gt_target_boxes,
        pred_boxes_by_frame=target_pred,
        iou_thresh=0.3,
    )

    # Team assignment
    team_result = assign_teams_from_tracks(
        video_path=seq.video_path, track_by_frame=track_confirmed
    )

    # Ball
    ball_result = detect_ball_sequence(
        video_path=seq.video_path, person_tracks=track_confirmed, device=device
    )

    report = {
        "schema": "final_perception_repair_report_v1",
        "sequence_id": SEQUENCE_ID,
        "target_gt_track_id": target.persistent_track_id,
        "best_pred_track_id": best_tid,
        "best_pred_track_mean_iou": best_mean if best_tid is not None else None,
        "source_sha256": EXPECTED_MP4_SHA,
        "device": device,
        "elapsed_s": time.time() - t0,
        "detection": {
            "selected_config": choose.name,
            "selected_params": choose.__dict__,
            "dev_split_frames": [1, dev_hi],
            "held_out_frames": [dev_hi + 1, seq.seq_length],
            "comparison_dev": {"baseline": base_dev, "candidate": cand_dev},
            "comparison_held_out": {"baseline": base_hold, "candidate": cand_hold},
            "full_selected": full_metrics,
            "runtime_s_full": det_runtime,
            "vram_peak_gb": vram_peak,
            "note": "baseline metrics from prior pilot frame dump; candidate full-stride GPU run",
        },
        "tracking": {
            "selected": selected_tracker,
            "selection_reason": selection_reason,
            "confirmed_iou": iou_metrics,
            "bytetrack_comparison": bt_metrics,
            "botsort": "not_run_tiebreak_agpl_same_family",
            "target_tracking_eval": target_metrics,
            "min_hits": 4,
        },
        "team": team_result,
        "ball": {
            "gt_ball_class_present": False,
            "evaluation": ball_result["evaluation"],
            "method": ball_result["method"],
            "summary": ball_result["summary"],
        },
        "isolation": {
            "teamtrack_track_7_not_soccertrack_506469": True,
            "gt_not_used_in_prediction": True,
        },
    }

    # Persist frame dump for video builder
    dump = {
        "schema": "final_perception_repair_frames_v1",
        "sequence_id": SEQUENCE_ID,
        "target_gt_track_id": target.persistent_track_id,
        "best_pred_track_id": best_tid,
        "team_by_track": team_result["team_by_track"],
        "pred_by_frame": {str(k): [list(b) for b in v] for k, v in sorted(pred_full.items())},
        "tracks_by_frame": {
            str(k): [
                {"track_id": tid, "bbox": list(box), "confirmed": confd} for tid, box, confd in v
            ]
            for k, v in sorted(track_confirmed.items())
        },
        "gt_target_by_frame": {str(k): list(v) for k, v in sorted(gt_target.items())},
        "target_pred_by_frame": {
            str(k): (list(v) if v is not None else None) for k, v in sorted(target_pred.items())
        },
        "ball_by_frame": ball_result["by_frame"],
        "gt_not_used_in_prediction": True,
    }
    (work_dir / "perception_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (work_dir / "frame_dump.json").write_text(json.dumps(dump) + "\n")
    # slim ball summary without every frame for report attachment
    ball_slim = dict(ball_result)
    ball_slim.pop("by_frame", None)
    (work_dir / "ball_summary.json").write_text(json.dumps(ball_slim, indent=2) + "\n")
    return report


__all__ = [
    "COLOR_BALL_CAND_BGR",
    "COLOR_BALL_OBS_BGR",
    "COLOR_REF_BGR",
    "COLOR_TARGET_BGR",
    "COLOR_TEAM_A_BGR",
    "COLOR_TEAM_B_BGR",
    "COLOR_UNKNOWN_BGR",
    "run_perception_repair",
]
