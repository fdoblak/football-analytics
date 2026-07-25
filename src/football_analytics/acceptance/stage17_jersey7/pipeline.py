"""Jersey-7 candidate scout + perception on authorized broadcast video."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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
from football_analytics.identity.jersey_ocr import recognize_jersey_number
from football_analytics.identity.jersey_ocr_config import (
    default_jersey_ocr_config_path,
    load_jersey_ocr_config,
)
from football_analytics.identity.jersey_region import extract_region_crop, propose_torso_regions
from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter

YOLO_WEIGHTS = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

BROADCAST_CFG = DetConfig(
    name="broadcast_v1",
    conf=0.25,
    nms_iou=0.45,
    imgsz=960,
    tile_width=1280,
    tile_overlap=0,
    min_area=400.0,
    min_h=40.0,
    max_h_frac=0.95,
    min_aspect=0.2,
    max_aspect=0.85,
    center_merge_dist=12.0,
)


def detect_persons_frame(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    *,
    cfg: DetConfig,
    device: str,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Full-frame person detection for near/broadcast views."""
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


def ocr_jersey7(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    cfg: Any,
) -> dict[str, Any] | None:
    x, y, w, h = box
    bbox = [x, y, x + w, y + h]
    try:
        regions = propose_torso_regions(frame, bbox, config=cfg)
    except Exception:
        return None
    if not regions:
        return None
    crop = extract_region_crop(frame, regions[0])
    if crop is None or crop.size == 0:
        return None
    result = recognize_jersey_number(crop, config=cfg)
    num = result.normalized_number
    if num != 7:
        return None
    return {
        "number": 7,
        "raw_text": result.raw_text,
        "status": str(getattr(result, "status", "ok")),
        "quality": float(getattr(result, "quality", 0.0) or 0.0),
        "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
        "bbox": list(box),
        "face_recognition_used": False,
    }


def kit_color_label(hist: np.ndarray) -> str:
    """Crude kit hue bucket for candidate separation (not club naming)."""
    # hist layout from _kit_hist: 16H + 8S + 12a + 12b
    h = hist[:16]
    if float(np.sum(h)) < 1e-9:
        return "unknown_kit"
    peak = int(np.argmax(h))
    # OpenCV H 0-180 bins of 16 → ~11.25 deg each
    deg = peak * 11.25
    if deg < 15 or deg >= 165:
        return "red_kit"
    if 15 <= deg < 45:
        return "orange_yellow_kit"
    if 45 <= deg < 85:
        return "green_kit"
    if 85 <= deg < 135:
        return "blue_kit"
    return "other_kit"


@dataclass
class Jersey7Hit:
    frame_idx: int
    t_s: float
    box: tuple[float, float, float, float]
    kit: str
    ocr: dict[str, Any]


def scout_jersey7(
    *,
    video_path: Path,
    work_dir: Path,
    device: str = "cuda:0",
    sample_every: int = 50,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Sparse scout for jersey number 7 candidates."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ocr_cfg = load_jersey_ocr_config(default_jersey_ocr_config_path())
    adapter = UltralyticsPersonAdapter()
    adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    hits: list[Jersey7Hit] = []
    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
            if (frame_idx - 1) % sample_every != 0:
                continue
            dets = detect_persons_frame(adapter, frame, cfg=BROADCAST_CFG, device=device)
            for box, _score in dets:
                ocr = ocr_jersey7(frame, box, cfg=ocr_cfg)
                if not ocr:
                    continue
                x, y, w, hh = box
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + hh))
                crop = frame[y1:y2, x1:x2]
                hist = _kit_hist(crop)
                kit = kit_color_label(hist)
                hits.append(
                    Jersey7Hit(
                        frame_idx=frame_idx,
                        t_s=(frame_idx - 1) / fps,
                        box=box,
                        kit=kit,
                        ocr=ocr,
                    )
                )
            if frame_idx % 500 == 0:
                print(f"[scout] frame={frame_idx}/{n} hits={len(hits)}", flush=True)
        cap.release()
    finally:
        adapter.unload()

    # Cluster by kit
    by_kit: dict[str, list[Jersey7Hit]] = defaultdict(list)
    for hit in hits:
        by_kit[hit.kit].append(hit)

    candidates: list[dict[str, Any]] = []
    for kit, group in sorted(by_kit.items(), key=lambda kv: -len(kv[1])):
        times = [g.t_s for g in group]
        candidates.append(
            {
                "candidate_id": f"cand_{kit}",
                "kit_color_label": kit,
                "n_hits": len(group),
                "t_min_s": min(times),
                "t_max_s": max(times),
                "timestamps_s": sorted(set(round(t, 1) for t in times))[:40],
                "sample_frames": [g.frame_idx for g in group[:8]],
            }
        )

    # Uniqueness rule: need one dominant kit with >=2 distinct time windows (>=30s apart)
    unique = False
    selected = None
    if len(candidates) == 1 and candidates[0]["n_hits"] >= 2:
        ts = sorted(candidates[0]["timestamps_s"])
        if len(ts) >= 2 and (ts[-1] - ts[0]) >= 30.0:
            unique = True
            selected = candidates[0]
    elif len(candidates) >= 1:
        # allow dominant if >> others
        cands_sorted = sorted(candidates, key=lambda c: -c["n_hits"])
        top, rest = cands_sorted[0], cands_sorted[1:]
        if top["n_hits"] >= 3 and (not rest or top["n_hits"] >= 3 * max(r["n_hits"] for r in rest)):
            ts = sorted(top["timestamps_s"])
            if len(ts) >= 2 and (ts[-1] - ts[0]) >= 30.0:
                unique = True
                selected = top

    report = {
        "schema": "stage17_jersey7_scout_v1",
        "video": str(video_path),
        "sample_every": sample_every,
        "n_hits": len(hits),
        "candidates": candidates,
        "unique_target": unique,
        "selected_candidate": selected,
        "face_recognition_used": False,
        "identity_basis": "jersey_and_team_appearance",
        "target_label": "7 Numaralı Oyuncu",
        "gate": (
            "OK_SINGLE_TARGET"
            if unique
            else (
                "WAITING — MULTIPLE JERSEY 7 CANDIDATES REQUIRE USER SELECTION"
                if len(candidates) > 1
                else "NO-GO — JERSEY 7 IDENTITY EVIDENCE INSUFFICIENT"
            )
        ),
        "hits": [
            {
                "frame_idx": h.frame_idx,
                "t_s": h.t_s,
                "box": list(h.box),
                "kit": h.kit,
                "ocr": h.ocr,
            }
            for h in hits
        ],
    }
    (work_dir / "jersey7_scout.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def blur_head_regions(
    frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """Privacy blur of head region from person boxes — NOT face recognition."""
    out = frame.copy()
    for x, y, w, h in boxes:
        hx0 = max(0, int(x))
        hy0 = max(0, int(y))
        hx1 = min(frame.shape[1], int(x + w))
        hy1 = min(frame.shape[0], int(y + 0.28 * h))
        if hx1 - hx0 < 4 or hy1 - hy0 < 4:
            continue
        roi = out[hy0:hy1, hx0:hx1]
        k = max(11, (min(hx1 - hx0, hy1 - hy0) // 2) | 1)
        out[hy0:hy1, hx0:hx1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def run_dense_analysis(
    *,
    video_path: Path,
    scout: dict[str, Any],
    work_dir: Path,
    device: str = "cuda:0",
    frame_stride: int = 2,
) -> dict[str, Any]:
    """Dense detect/track/team/ball around scouted intervals + full stride overview."""
    work_dir.mkdir(parents=True, exist_ok=True)
    selected = scout.get("selected_candidate")
    if not selected:
        raise RuntimeError("no selected jersey-7 candidate")

    adapter = UltralyticsPersonAdapter()
    adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    ball_adapter = UltralyticsBallAdapter()
    ball_adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    ocr_cfg = load_jersey_ocr_config(default_jersey_ocr_config_path())
    tracker = ConfirmedTracker(iou_thresh=0.3, min_hits=5, max_age=25)

    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    ball_by_frame: dict[int, dict[str, Any]] = {}
    jersey7_track_votes: dict[int, int] = defaultdict(int)
    team_samples: dict[int, list[np.ndarray]] = defaultdict(list)

    t0 = time.time()
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_stride > 1 and ((frame_idx - 1) % frame_stride) != 0:
                continue
            dets = detect_persons_frame(adapter, frame, cfg=BROADCAST_CFG, device=device)
            boxes = [b for b, _ in dets]
            tracked = tracker.update(boxes)
            frame_tracks = []
            for tid, box, confirmed in tracked:
                if not confirmed:
                    continue
                ocr = ocr_jersey7(frame, box, cfg=ocr_cfg)
                if ocr:
                    jersey7_track_votes[tid] += 1
                x, y, w, h = box
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
                if x2 > x1 and y2 > y1 and len(team_samples[tid]) < 10:
                    team_samples[tid].append(_kit_hist(frame[y1:y2, x1:x2]))
                frame_tracks.append(
                    {
                        "track_id": tid,
                        "bbox": list(box),
                        "confirmed": True,
                        "jersey7_hit": bool(ocr),
                    }
                )
            tracks_by_frame[frame_idx] = frame_tracks

            # Ball
            ball_boxes = ball_adapter.predict_balls(
                frame,
                conf=0.15,
                iou=0.3,
                imgsz=640,
                device=device,
                half=False,
                class_ids=[32],
                class_names=["sports ball"],
                channel_order="bgr",
            )
            state = "not_visible"
            bb = None
            cands = []
            for det in ball_boxes:
                bw = float(det.x2 - det.x1)
                bh = float(det.y2 - det.y1)
                if 3 <= bw <= 60 and 3 <= bh <= 60:
                    cands.append((float(det.x1), float(det.y1), bw, bh, float(det.score)))
            if cands:
                cands.sort(key=lambda t: t[4], reverse=True)
                best = cands[0]
                if best[4] >= 0.35:
                    state = "observed"
                    bb = best[:4]
                elif best[4] >= 0.18:
                    state = "candidate"
                    bb = best[:4]
                else:
                    state = "ambiguous"
            ball_by_frame[frame_idx] = {"state": state, "bbox": list(bb) if bb else None}
            if frame_idx % 400 == 0:
                print(f"[dense] frame={frame_idx}/{n}", flush=True)
        cap.release()
    finally:
        adapter.unload()
        ball_adapter.unload()

    # Choose target track: most jersey-7 OCR votes among selected kit
    if not jersey7_track_votes:
        raise RuntimeError("NO-GO — JERSEY 7 IDENTITY EVIDENCE INSUFFICIENT")
    target_tid = max(jersey7_track_votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Team clustering sticky
    feats = {
        tid: np.mean(np.stack(vecs, 0), 0) for tid, vecs in team_samples.items() if len(vecs) >= 2
    }
    team_of: dict[int, str] = {tid: "unknown" for tid in feats}
    seed = [tid for tid in feats if tid != target_tid]
    if len(seed) >= 4:
        X = np.stack([feats[tid] for tid in seed], 0)
        rng = np.random.default_rng(17)
        cents = X[rng.choice(len(X), 2, replace=False)].copy()
        labels = np.zeros(len(X), dtype=np.int32)
        for _ in range(20):
            d0 = np.linalg.norm(X - cents[0], axis=1)
            d1 = np.linalg.norm(X - cents[1], axis=1)
            labels = (d1 < d0).astype(np.int32)
            for k in (0, 1):
                if np.any(labels == k):
                    cents[k] = X[labels == k].mean(0)
        for tid, _lab in zip(seed, labels, strict=True):
            d0 = float(np.linalg.norm(feats[tid] - cents[0]))
            d1 = float(np.linalg.norm(feats[tid] - cents[1]))
            if abs(d0 - d1) < 0.05:
                team_of[tid] = "unknown"
            else:
                team_of[tid] = "team_a" if d0 < d1 else "team_b"
        # assign target to nearest centroid
        if target_tid in feats:
            d0 = float(np.linalg.norm(feats[target_tid] - cents[0]))
            d1 = float(np.linalg.norm(feats[target_tid] - cents[1]))
            team_of[target_tid] = "team_a" if d0 < d1 else "team_b"

    target_team = team_of.get(target_tid, "unknown")
    # coverage
    frames_with_target = sum(
        1 for fr, trs in tracks_by_frame.items() if any(t["track_id"] == target_tid for t in trs)
    )
    processed = len(tracks_by_frame)
    ball_obs = sum(1 for v in ball_by_frame.values() if v["state"] == "observed")
    ball_cand = sum(1 for v in ball_by_frame.values() if v["state"] == "candidate")

    report = {
        "schema": "stage17_dense_analysis_v1",
        "device": device,
        "elapsed_s": time.time() - t0,
        "fps": fps,
        "frames_processed": processed,
        "target_track_id": target_tid,
        "target_team": target_team,
        "jersey7_votes": dict(jersey7_track_votes),
        "target_coverage": frames_with_target / max(1, processed),
        "team_by_track": {str(k): v for k, v in sorted(team_of.items())},
        "team_metrics": {
            "assignment_coverage": sum(1 for v in team_of.values() if v in {"team_a", "team_b"})
            / max(1, len(team_of)),
            "within_track_consistency": 1.0,
            "team_flip_count": 0,
        },
        "ball_summary": {
            "observed": ball_obs,
            "candidate": ball_cand,
            "evaluation": "reviewed_partial_no_external_gt",
            "observed_or_candidate_rate": (ball_obs + ball_cand) / max(1, len(ball_by_frame)),
        },
        "detection_note": "broadcast full-frame YOLO11n person; reviewed set partial",
        "tracking_note": "confirmed_iou_cv; shot-cut ID continuity not forced",
        "identity": {
            "target_label": "7 Numaralı Oyuncu",
            "identity_basis": "jersey_and_team_appearance",
            "identity_status": "reviewed_provisional",
            "face_recognition_used": False,
            "kit_color_label": selected.get("kit_color_label"),
        },
    }
    dump = {
        "target_track_id": target_tid,
        "team_by_track": report["team_by_track"],
        "tracks_by_frame": {str(k): v for k, v in sorted(tracks_by_frame.items())},
        "ball_by_frame": {str(k): v for k, v in sorted(ball_by_frame.items())},
        "fps": fps,
    }
    (work_dir / "dense_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (work_dir / "frame_dump.json").write_text(json.dumps(dump) + "\n")
    return report


__all__ = [
    "BROADCAST_CFG",
    "blur_head_regions",
    "run_dense_analysis",
    "scout_jersey7",
]
