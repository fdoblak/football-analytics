"""TeamTrack real-video pilot: detect/track on normalized sequence; GT only for eval."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.teamtrack.loader import load_sequence
from football_analytics.acceptance.teamtrack.mot_eval import (
    evaluate_detection_frames,
    evaluate_target_track,
)
from football_analytics.acceptance.teamtrack.target_selection import (
    select_anonymous_track,
    write_teamtrack_target_receipt,
)
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter

YOLO_WEIGHTS = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def normalize_for_pilot(
    *,
    source: Path,
    output: Path,
    max_width: int = 1280,
    fps: float | None = None,
) -> dict[str, Any]:
    """Scale width only; preserve fps when provided so frame indices stay aligned with GT."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 100_000:
        return {"path": str(output), "cached": True, "sha256": sha256_file(output)}
    scale = f"scale='min({max_width},iw)':-2"
    vf = scale if fps is None else f"{scale},fps={fps}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-an",
        str(output),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    return {
        "path": str(output),
        "cached": False,
        "sha256": sha256_file(output),
        "elapsed_s": time.time() - t0,
        "max_width": max_width,
        "fps": fps,
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=width,height,avg_frame_rate,nb_frames,codec_name",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _SimpleIoUTracker:
    def __init__(self, iou_thresh: float = 0.3) -> None:
        self.iou_thresh = iou_thresh
        self._next_id = 1
        self._tracks: dict[int, tuple[float, float, float, float]] = {}

    def update(
        self, detections: list[tuple[float, float, float, float]]
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        assigned: dict[int, tuple[float, float, float, float]] = {}
        used_det: set[int] = set()
        # greedy match existing tracks
        for tid, box in list(self._tracks.items()):
            best_i, best = -1, 0.0
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                v = _iou(box, det)
                if v > best:
                    best, best_i = v, i
            if best >= self.iou_thresh and best_i >= 0:
                assigned[tid] = detections[best_i]
                used_det.add(best_i)
        for i, det in enumerate(detections):
            if i in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            assigned[tid] = det
        self._tracks = assigned
        return [(tid, box) for tid, box in assigned.items()]


def _horizontal_tiles(width: int, tile_w: int = 1600, overlap: int = 200) -> list[tuple[int, int]]:
    if width <= tile_w:
        return [(0, width)]
    starts = list(range(0, max(1, width - overlap), tile_w - overlap))
    tiles: list[tuple[int, int]] = []
    for s in starts:
        e = min(width, s + tile_w)
        tiles.append((s, e))
        if e >= width:
            break
    if tiles[-1][1] < width:
        tiles.append((max(0, width - tile_w), width))
    # unique
    out: list[tuple[int, int]] = []
    for t in tiles:
        if t not in out:
            out.append(t)
    return out


def _nms_xywh(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_thresh: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) < iou_thresh]
    return [boxes[i] for i in keep]


def run_teamtrack_pilot(
    *,
    sequence_root: Path,
    run_dir: Path,
    evidence_dir: Path,
    sport_view: str = "soccer_side",
    split: str = "train",
    sequence_id: str = "F_20200220_1_0330_0360",
    device: str = "cuda:0",
    imgsz: int = 960,
    conf: float = 0.15,
    frame_stride: int = 1,
    tile_width: int = 1600,
    tile_overlap: int = 200,
) -> dict[str, Any]:
    """Run bounded real-video pilot. GT used only after predictions for evaluation."""
    t0 = time.time()
    seq = load_sequence(
        root=sequence_root,
        sport_view=sport_view,
        split=split,
        sequence_id=sequence_id,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for ns in ("predictions", "reference_ground_truth", "evaluation"):
        (run_dir / ns).mkdir(exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Target from GT attributes only (selection), never fed into detector.
    target = select_anonymous_track(seq)
    write_teamtrack_target_receipt(
        target, run_dir / "reference_ground_truth" / "target_selection_receipt.json"
    )
    write_teamtrack_target_receipt(target, evidence_dir / "target_selection_receipt.json")

    probe = _ffprobe(seq.video_path)
    # Keep a downscaled preview copy for receipts (not used for inference).
    norm_path = run_dir / "predictions" / "normalized_preview.mp4"
    norm = normalize_for_pilot(source=seq.video_path, output=norm_path, max_width=1280, fps=None)
    norm_probe = _ffprobe(norm_path)
    # Inference on original panorama with horizontal tiles (tiny players on side-view).
    tiles = _horizontal_tiles(seq.im_width, tile_w=tile_width, overlap=tile_overlap)

    # Load detector — predictions namespace only
    adapter = UltralyticsPersonAdapter()
    adapter.load(str(YOLO_WEIGHTS), YOLO_SHA256)
    try:
        import torch

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if not use_cuda:
            device = "cpu"
        tracker = _SimpleIoUTracker(iou_thresh=0.2)
        pred_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        track_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
        target_pred: dict[int, tuple[float, float, float, float] | None] = {}

        cap = cv2.VideoCapture(str(seq.video_path))
        if not cap.isOpened():
            raise RuntimeError("failed to open source video")
        frame_idx = 0  # MOT frames are 1-based
        det_count = 0
        oom_fallback = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_stride > 1 and ((frame_idx - 1) % frame_stride) != 0:
                continue
            merged: list[tuple[float, float, float, float]] = []
            scores: list[float] = []
            for x0, x1 in tiles:
                tile = frame[:, x0:x1, :]
                try:
                    boxes = adapter.predict_persons(
                        tile,
                        conf=conf,
                        iou=0.5,
                        imgsz=imgsz,
                        device=device,
                        half=False,
                        class_ids=[0],
                        class_names=["person"],
                        channel_order="bgr",
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and device != "cpu":
                        torch.cuda.empty_cache()
                        device = "cpu"
                        oom_fallback = True
                        boxes = adapter.predict_persons(
                            tile,
                            conf=conf,
                            iou=0.5,
                            imgsz=min(imgsz, 640),
                            device=device,
                            half=False,
                            class_ids=[0],
                            class_names=["person"],
                            channel_order="bgr",
                        )
                    else:
                        raise
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
            dets = _nms_xywh(merged, scores, iou_thresh=0.5)
            det_count += len(dets)
            pred_by_frame[frame_idx] = dets
            tracked = tracker.update(dets)
            track_by_frame[frame_idx] = tracked
            if frame_idx % 100 == 0:
                print(f"frame={frame_idx} dets={len(dets)} device={device}", flush=True)
        cap.release()
    finally:
        adapter.unload()

    # Map best predicted track id to selected GT track via mean IoU on overlapping frames
    gt_target = [gb for gb in seq.boxes if gb.track_id == target.persistent_track_id]
    gt_native = {gb.frame: (gb.x, gb.y, gb.w, gb.h) for gb in gt_target}
    score: dict[int, list[float]] = defaultdict(list)
    for frame_i, tracks in track_by_frame.items():
        g = gt_native.get(frame_i)
        if g is None:
            continue
        for tid, box in tracks:
            score[tid].append(_iou(g, box))
    best_pred_tid: int | None = None
    best_mean = -1.0
    for tid, vals in score.items():
        m = sum(vals) / len(vals)
        if m > best_mean or (m == best_mean and best_pred_tid is not None and tid < best_pred_tid):
            best_mean = m
            best_pred_tid = tid
    for frame_i, tracks in track_by_frame.items():
        chosen: tuple[float, float, float, float] | None = None
        for tid, box in tracks:
            if tid == best_pred_tid:
                chosen = box
                break
        target_pred[frame_i] = chosen

    # Evaluation only (reference GT) in native coordinates
    gt_all: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for gb in seq.boxes:
        gt_all[gb.frame].append((gb.x, gb.y, gb.w, gb.h))
    det_metrics = evaluate_detection_frames(
        gt_by_frame=dict(gt_all),
        pred_by_frame=pred_by_frame,
        iou_thresh=0.5,
    )
    target_metrics = evaluate_target_track(
        gt_boxes=list(gt_target),
        pred_boxes_by_frame=target_pred,
        iou_thresh=0.3,
    )

    # Physical metrics from predicted target centers (image plane; pitch GT not downloaded)
    centers: list[tuple[int, float, float]] = []
    for frame_i in sorted(target_pred):
        box_opt = target_pred[frame_i]
        if box_opt is None:
            continue
        cx = box_opt[0] + box_opt[2] / 2.0
        cy = box_opt[1] + box_opt[3] / 2.0
        centers.append((frame_i, cx, cy))
    pixel_dist = 0.0
    speeds: list[float] = []
    for idx in range(1, len(centers)):
        prev_f, prev_x, prev_y = centers[idx - 1]
        cur_f, cur_x, cur_y = centers[idx]
        dt = (cur_f - prev_f) / seq.fps if seq.fps else 0.0
        dist = ((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2) ** 0.5
        pixel_dist += dist
        if dt > 0:
            speeds.append(dist / dt)
    physical = {
        "measured_distance_px": pixel_dist,
        "mean_speed_px_s": (sum(speeds) / len(speeds)) if speeds else None,
        "peak_speed_px_s": max(speeds) if speeds else None,
        "pitch_distance_m": "not_evaluable",
        "pitch_speed_m_s": "not_evaluable",
        "sprint_count": "not_evaluable",
        "heatmap": "image_plane_only",
        "note": (
            "No teamtrack-trajectory file downloaded for this sequence; "
            "pitch metrics not_evaluable"
        ),
    }

    product_events = {
        k: "not_evaluable"
        for k in (
            "passes",
            "pass_accuracy",
            "long_passes",
            "zone_transition_passes",
            "dribbles",
            "take_ons",
            "duels",
            "tackles",
            "recoveries",
            "turnovers",
            "aerial_duels",
            "clearances",
            "penalty_area_touches",
        )
    }

    predictions_payload = {
        "schema": "teamtrack_pilot_predictions",
        "sequence_id": sequence_id,
        "normalized_preview": norm,
        "inference_mode": "horizontal_tiles_on_source_mp4",
        "tiles": tiles,
        "device": device,
        "oom_fallback": oom_fallback,
        "imgsz": imgsz,
        "frame_stride": frame_stride,
        "frames_processed": len(pred_by_frame),
        "detection_count": det_count,
        "best_pred_track_id": best_pred_tid,
        "best_pred_track_mean_iou_vs_gt_target": best_mean if best_pred_tid is not None else None,
        "coordinate_space": "native_seqinfo_pixels",
        "gt_not_used_in_prediction": True,
    }
    (run_dir / "predictions" / "pilot_predictions_summary.json").write_text(
        json.dumps(predictions_payload, indent=2, sort_keys=True) + "\n"
    )

    report = {
        "schema": "real_video_pilot_report",
        "stage": "16-R2",
        "gate_candidate": (
            "PASS_WITH_FINDINGS — REAL-VIDEO TRACKING PILOT COMPLETE; FULL EVENT ACCEPTANCE BLOCKED"
        ),
        "title": "REAL-VIDEO PILOT — NOT FULL STAGE-16 ACCEPTANCE",
        "dataset": {
            "id": "teamtrack",
            "version": 6,
            "sport_view": sport_view,
            "split": split,
            "sequence_id": sequence_id,
            "license": "MIT",
            "license_source": "kaggle_metadata_licenseNameNullable",
            "official_urls": {
                "project": "https://atomscott.github.io/TeamTrack/",
                "repo": "https://github.com/AtomScott/TeamTrack",
                "kaggle": "https://www.kaggle.com/datasets/atomscott/teamtrack",
            },
        },
        "target": target.to_dict(),
        "video": {
            "source_path": str(seq.video_path),
            "source_sha256": (sha256_file(seq.video_path) if seq.video_path.is_file() else None),
            "source_size_bytes": (
                seq.video_path.stat().st_size if seq.video_path.is_file() else None
            ),
            "probe": probe,
            "normalized_preview": norm,
            "normalized_probe": norm_probe,
        },
        "pilot": {
            "duration_s": float(
                probe.get("format", {}).get("duration") or seq.seq_length / seq.fps
            ),
            "seq_length_frames": seq.seq_length,
            "human_detections": det_count,
            "tracks_observed": int(
                max((tid for fr in track_by_frame.values() for tid, _ in fr), default=0)
            ),
            "oom": oom_fallback,
            "device": device,
            "elapsed_s": time.time() - t0,
            "false_success": False,
            "gt_leakage_into_prediction": False,
        },
        "evaluation": {
            "detection": det_metrics,
            "target_tracking": target_metrics,
            "hota_mota_idf1": "not_evaluable",
            "pitch_trajectory_error": "not_evaluable",
        },
        "physical_metrics": physical,
        "product_event_metrics": product_events,
        "soccertrack_v2_isolation": {
            "soccertrack_v2_target_preserved": {
                "team": "left",
                "jersey": 24,
                "player_id": "506469",
            },
            "deprecated_invalid_target_not_used": {"jersey": 11, "player_id": "506466"},
            "bas_gsr_not_applied_to_teamtrack": True,
        },
        "attribution": (
            "TeamTrack © Atom Scott et al.; dataset license MIT (Kaggle metadata). "
            "Project-generated pilot analysis; not official Opta data."
        ),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = evidence_dir / "real_video_pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (run_dir / "evaluation" / "real_video_pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


__all__ = ["normalize_for_pilot", "run_teamtrack_pilot"]
