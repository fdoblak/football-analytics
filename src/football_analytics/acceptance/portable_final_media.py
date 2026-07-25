"""Rebuild portable Windows/GitHub-safe final_delivery media (Stage 16-R4-FIX2)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.namespaces import AUTHORITATIVE_SOCCERTRACK_TARGET
from football_analytics.acceptance.teamtrack.loader import load_sequence
from football_analytics.acceptance.teamtrack.mot_eval import (
    evaluate_detection_frames,
    evaluate_target_track,
)
from football_analytics.acceptance.teamtrack.pilot_runner import _iou

EXPECTED_MP4_SHA = "fd42dbe6df8b85946d6ee82fc11863ea94dd2735b3b126e9bfca3972a3149db1"
EXPECTED_MP4_SIZE = 13_086_969
EXPECTED_GT_SHA = "80fd710824a4f006528aa0bb0207ae34c1869e469373cdb276800260d96a7ff4"
DRIVE_FILE_ID = "1R3t04im2hsp52_G_JCAwB4dPbxyA062z"

COLOR_PRED = (255, 255, 0)  # cyan BGR
COLOR_GT = (0, 255, 0)
COLOR_TARGET = (0, 255, 255)
COLOR_UNMATCHED = (0, 0, 255)

OUT_W, OUT_H = 1280, 720


def _sha256(path: Path) -> str:
    return sha256_file(path)


def download_source(staging_dir: Path) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    out = staging_dir / "img1.mp4"
    partial = staging_dir / "img1.mp4.partial"
    if (
        out.is_file()
        and out.stat().st_size == EXPECTED_MP4_SIZE
        and _sha256(out) == EXPECTED_MP4_SHA
    ):
        return out
    import gdown

    if partial.exists():
        partial.unlink()
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, str(partial), quiet=False)
    if not partial.is_file():
        raise RuntimeError("NO-GO — REAL VIDEO SOURCE INTEGRITY FAILURE: download missing")
    if partial.stat().st_size != EXPECTED_MP4_SIZE or _sha256(partial) != EXPECTED_MP4_SHA:
        raise RuntimeError("NO-GO — REAL VIDEO SOURCE INTEGRITY FAILURE: size/sha mismatch")
    partial.replace(out)
    return out


def _letterbox(
    frame: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H
) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    # force even dims
    nw -= nw % 2
    nh -= nh % 2
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def _scale_box(box: tuple[float, float, float, float], scale: float, x0: int, y0: int):
    x, y, w, h = box
    return (
        int(round(x * scale)) + x0,
        int(round(y * scale)) + y0,
        int(round(w * scale)),
        int(round(h * scale)),
    )


def build_portable_proof_mp4(
    *,
    sequence_root: Path,
    predictions_json: Path,
    staging_mp4: Path,
    final_mp4: Path,
) -> dict[str, Any]:
    seq = load_sequence(
        root=sequence_root,
        sport_view="soccer_side",
        split="train",
        sequence_id="F_20200220_1_0330_0360",
    )
    dump = json.loads(predictions_json.read_text())
    pred_by_frame = {int(k): [tuple(b) for b in v] for k, v in dump["pred_by_frame"].items()}
    gt_target = {int(k): tuple(v) for k, v in dump["gt_target_by_frame"].items()}
    target_pred = {
        int(k): (tuple(v) if v is not None else None)
        for k, v in dump["target_pred_by_frame"].items()
    }

    staging_mp4.parent.mkdir(parents=True, exist_ok=True)
    if staging_mp4.exists():
        staging_mp4.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OUT_W}x{OUT_H}",
        "-r",
        str(seq.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-profile:v",
        "main",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-crf",
        "23",
        str(staging_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    cap = cv2.VideoCapture(str(seq.video_path))
    tp = fp = fn = 0
    frame_idx = 0
    selected: dict[str, dict[str, Any]] = {}
    pick = {
        "start": 1,
        "mid": max(1, seq.seq_length // 2),
        "crowd": max(1, int(seq.seq_length * 0.7)),
        "end": seq.seq_length,
    }
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            preds = pred_by_frame.get(frame_idx, [])
            gts = [gb for gb in seq.boxes if gb.frame == frame_idx]
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            for pi, pb in enumerate(preds):
                best_j, best = -1, 0.0
                for j, gb in enumerate(gts):
                    if j in matched_gt:
                        continue
                    v = _iou(pb, (gb.x, gb.y, gb.w, gb.h))
                    if v > best:
                        best, best_j = v, j
                if best >= 0.5 and best_j >= 0:
                    matched_gt.add(best_j)
                    matched_pred.add(pi)
                    tp += 1
                else:
                    fp += 1
            fn += len(gts) - len(matched_gt)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            tgt_gt = gt_target.get(frame_idx)
            tgt_pr = target_pred.get(frame_idx)
            tgt_iou = _iou(tgt_pr, tgt_gt) if (tgt_pr and tgt_gt) else 0.0
            matched_tgt = bool(tgt_pr and tgt_gt and tgt_iou >= 0.3)

            canvas, scale, x0, y0 = _letterbox(frame)
            for pi, pb in enumerate(preds):
                sx, sy, sw, sh = _scale_box(pb, scale, x0, y0)
                color = COLOR_UNMATCHED if pi not in matched_pred else COLOR_PRED
                if tgt_pr and abs(pb[0] - tgt_pr[0]) < 1e-3 and abs(pb[1] - tgt_pr[1]) < 1e-3:
                    color = COLOR_TARGET
                cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, 2)
            if tgt_gt:
                sx, sy, sw, sh = _scale_box(tgt_gt, scale, x0, y0)
                cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), COLOR_GT, 2)
                cv2.putText(
                    canvas,
                    "GT Track7",
                    (sx, max(20, sy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_GT,
                    2,
                )

            t_s = (frame_idx - 1) / seq.fps
            hud = [
                "REAL TEAMTRACK VIDEO — TRACKING PILOT",
                f"t={t_s:.2f}s frame={frame_idx}/{seq.seq_length} device=cuda:0",
                f"TP={tp} FP={fp} FN={fn}  P={precision:.3f} R={recall:.3f} F1={f1:.3f}",
                f"Track7 IoU={tgt_iou:.3f} matched={matched_tgt}",
                "cyan=pred green=GT yellow=target-pred red=unmatched",
            ]
            ytxt = 24
            for line in hud:
                cv2.putText(
                    canvas, line, (16, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )
                ytxt += 22

            proc.stdin.write(canvas.tobytes())

            for label, fi in pick.items():
                if frame_idx == fi:
                    selected[label] = {
                        "frame_index": frame_idx,
                        "t_s": t_s,
                        "target_iou": tgt_iou,
                        "matched": matched_tgt,
                        "rgb": cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
                    }
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed: {stderr[-2000:]}")

    # validate staging then atomic replace
    validate_portable_mp4(staging_mp4, expected_frames=seq.seq_length, expected_duration=30.0)
    final_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging_mp4, final_mp4)
    validate_portable_mp4(final_mp4, expected_frames=seq.seq_length, expected_duration=30.0)

    # recount metrics from dump (canonical)
    gt_all: dict[int, list[tuple[float, float, float, float]]] = {}
    for gb in seq.boxes:
        gt_all.setdefault(gb.frame, []).append((gb.x, gb.y, gb.w, gb.h))
    det = evaluate_detection_frames(gt_by_frame=gt_all, pred_by_frame=pred_by_frame, iou_thresh=0.5)
    gt_boxes = [gb for gb in seq.boxes if gb.track_id == dump["target_gt_track_id"]]
    track = evaluate_target_track(
        gt_boxes=gt_boxes, pred_boxes_by_frame=target_pred, iou_thresh=0.3
    )
    return {
        "path": str(final_mp4),
        "sha256": _sha256(final_mp4),
        "size_bytes": final_mp4.stat().st_size,
        "selected_frames": selected,
        "detection": det,
        "target_tracking": track,
        "running_end": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "fps": seq.fps,
        "seq_length": seq.seq_length,
    }


def validate_portable_mp4(
    path: Path, *, expected_frames: int | None = None, expected_duration: float | None = None
) -> dict[str, Any]:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise RuntimeError("no video stream")
    s0 = streams[0]
    if s0.get("codec_name") != "h264":
        raise RuntimeError(f"codec_name={s0.get('codec_name')}")
    if s0.get("codec_tag_string") != "avc1":
        raise RuntimeError(f"codec_tag={s0.get('codec_tag_string')}")
    if s0.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"pix_fmt={s0.get('pix_fmt')}")
    if s0.get("profile") != "Main":
        raise RuntimeError(f"profile={s0.get('profile')}")
    # atom order
    data = path.read_bytes()
    i = 0
    atoms: list[tuple[str, int, int]] = []
    while i + 8 <= len(data) and len(atoms) < 30:
        size = int.from_bytes(data[i : i + 4], "big")
        typ = data[i + 4 : i + 8].decode("latin1", errors="replace")
        if size < 8:
            break
        atoms.append((typ, i, size))
        i += size
    moov = next((a for a in atoms if a[0] == "moov"), None)
    mdat = next((a for a in atoms if a[0] == "mdat"), None)
    if not moov or not mdat or not (moov[1] < mdat[1]):
        raise RuntimeError(f"moov not before mdat: {atoms[:8]}")
    if any(a[0] in ("moof", "mfhd", "traf") for a in atoms):
        raise RuntimeError("fragmented MP4")

    # full decode
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or r.stderr.strip():
        raise RuntimeError(f"full decode failed: {r.stderr}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("opencv open failed")
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok0, f0 = cap.read()
    if not ok0:
        raise RuntimeError("frame0 failed")
    mid = max(0, n_meta // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    okm, fm = cap.read()
    if not okm:
        raise RuntimeError("middle frame failed")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_meta - 1))
    okl, fl = cap.read()
    if not okl:
        raise RuntimeError("last frame failed")
    cap.release()
    # sequential
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    if count <= 0:
        raise RuntimeError("decoded frame count 0")
    if expected_frames is not None and abs(count - expected_frames) > 2:
        raise RuntimeError(f"frame count {count} != {expected_frames}")
    dur = float(probe["format"]["duration"])
    if expected_duration is not None and abs(dur - expected_duration) > 0.5:
        raise RuntimeError(f"duration {dur}")
    return {
        "codec_name": s0["codec_name"],
        "codec_tag_string": s0["codec_tag_string"],
        "pix_fmt": s0["pix_fmt"],
        "profile": s0["profile"],
        "level": s0.get("level"),
        "width": s0["width"],
        "height": s0["height"],
        "duration": dur,
        "nb_frames_meta": n_meta,
        "sequential_frames": count,
        "moov_before_mdat": True,
        "atoms_head": atoms[:6],
        "frame0_shape": list(f0.shape),
        "middle_shape": list(fm.shape),
        "last_shape": list(fl.shape),
    }


def render_rgb_png(
    *,
    proof: dict[str, Any],
    reference: dict[str, Any],
    pilot_meta: dict[str, Any],
    output_png: Path,
) -> str:
    det = proof["detection"]
    tr = proof["target_tracking"]
    metrics = reference["metrics"]
    tgt = AUTHORITATIVE_SOCCERTRACK_TARGET
    frames = proof["selected_frames"]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0b1210")
    fig.text(
        0.02,
        0.96,
        "Football Analytics — Single Player Technical Preview",
        color="#ffe082",
        fontsize=18,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.02,
        0.925,
        (
            f"SoccerTrack v2 Match {tgt['match_id']} · Team {tgt['team_side']} / "
            f"Jersey {tgt['jersey_number']} / Player {tgt['player_id']}"
        ),
        color="#c8e6c9",
        fontsize=12,
    )
    fig.text(
        0.02,
        0.88,
        "REAL-VIDEO TRACKING VALIDATION — DIFFERENT DATASET/TARGET (TeamTrack Track 7)",
        color="#80cbc4",
        fontsize=11,
    )
    for i, key in enumerate(["start", "mid", "crowd", "end"]):
        ax = fig.add_axes((0.02 + i * 0.24, 0.52, 0.23, 0.32))
        ax.set_facecolor("#1b2a22")
        ax.set_xticks([])
        ax.set_yticks([])
        fr = frames.get(key)
        if fr:
            ax.imshow(fr["rgb"])
            ax.set_title(
                f"{key} t={fr['t_s']:.2f}s IoU={fr['target_iou']:.2f}",
                color="#e8f5e9",
                fontsize=8,
            )
    axm = fig.add_axes((0.02, 0.30, 0.40, 0.18))
    axm.set_facecolor("#1b2a22")
    axm.axis("off")
    axm.text(
        0.02,
        0.95,
        (
            "TeamTrack real-video metrics (recounted)\n"
            f"detections={det['tp'] + det['fp']}\n"
            f"TP={det['tp']} FP={det['fp']} FN={det['fn']}\n"
            f"P={det['precision']:.6f} R={det['recall']:.6f} F1={det['f1']:.6f}\n"
            f"coverage={tr['target_coverage_ratio']:.6f} mean_IoU={tr['mean_iou']:.6f}\n"
            f"device={pilot_meta.get('device')} runtime_s={pilot_meta.get('elapsed_s')}"
        ),
        va="top",
        color="#e8f5e9",
        fontsize=10,
        family="DejaVu Sans",
    )
    axr = fig.add_axes((0.46, 0.30, 0.52, 0.18))
    axr.set_facecolor("#1b2a22")
    axr.axis("off")
    lines = ["SoccerTrack v2 annotation-derived (NOT video prediction)"]
    for key in (
        "measured_distance_m",
        "mean_speed_m_s",
        "peak_speed_m_s",
        "sprint_count",
        "sprint_distance_m",
        "activity_index",
        "bas_pass_attempts",
        "bas_drive_actions",
        "bas_high_pass_attempts",
        "bas_header_actions",
        "bas_successful_tackles",
        "penalty_area_presence_points",
    ):
        m = metrics.get(key) or {}
        lines.append(f"{key}={m.get('value')} [{m.get('status')}]")
    axr.text(0.02, 0.95, "\n".join(lines[:14]), va="top", color="#e8f5e9", fontsize=8)
    ne = [k for k, m in metrics.items() if (m or {}).get("status") == "NOT_EVALUABLE"]
    fig.text(
        0.02,
        0.22,
        "N/A — NOT EVALUABLE: " + ", ".join(ne + ["video-event inference accuracy"]),
        color="#ffcc80",
        fontsize=9,
    )
    fig.text(
        0.02,
        0.10,
        (
            "SoccerTrack metrics are annotation-derived. "
            "TeamTrack panel validates real-video detection/tracking only. "
            "Datasets/player identities are not mixed. Not official Opta data."
        ),
        color="#90a4ae",
        fontsize=8,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    # RGB only (no alpha): save to buffer then convert
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    fig.savefig(tmp_path, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    from PIL import Image

    im = Image.open(tmp_path).convert("RGB")
    im.save(output_png, format="PNG", optimize=True)
    tmp_path.unlink(missing_ok=True)
    # verify
    im2 = Image.open(output_png)
    im2.verify()
    im2 = Image.open(output_png)
    im2.load()
    if im2.mode != "RGB":
        raise RuntimeError(f"png mode {im2.mode}")
    arr = cv2.imread(str(output_png), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("opencv cannot read png")
    return _sha256(output_png)


def write_open_results_html(delivery: Path, summary: dict[str, Any]) -> None:
    rv = summary["real_video_validation"]["metrics"]
    # HTML template kept compact; long lines are intentional markup.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Football Analytics — Technical Preview Results</title>
<style>
body {{
  font-family: Segoe UI, Arial, sans-serif;
  background:#0b1210; color:#e8f5e9; margin:0; padding:24px;
}}
h1 {{ color:#ffe082; margin-bottom:4px; }}
h2 {{ color:#80cbc4; }}
.card {{ background:#1b2a22; padding:16px; margin:16px 0; border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; }}
td,th {{ border:1px solid #345; padding:8px; text-align:left; }}
img {{ max-width:100%; height:auto; background:#000; }}
video {{ width:100%; max-width:1280px; background:#000; }}
a {{ color:#81d4fa; }}
.note {{ color:#ffcc80; }}
</style>
</head>
<body>
<h1>Football Analytics — Single Player Technical Preview</h1>
<p>SoccerTrack v2 Match 128057 · Team left / Jersey 24 / Player 506469</p>
<p class="note">
TeamTrack Track 7 is a different person/dataset used only for real-video tracking proof.
</p>

<div class="card">
<h2>Real-video proof (TeamTrack)</h2>
<video controls preload="metadata" src="real_video_tracking_proof.mp4">
Your browser cannot play this MP4. Open the file directly.
</video>
</div>

<div class="card">
<h2>Summary visual</h2>
<img src="single_player_analysis_summary.png" alt="Single player analysis summary"/>
</div>

<div class="card">
<h2>Key metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Evidence</th></tr>
<tr><td>Detections</td><td>{rv['detections']['value']}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
<tr><td>Precision</td><td>{rv['precision']['value']:.6f}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
<tr><td>Recall</td><td>{rv['recall']['value']:.6f}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
<tr><td>F1</td><td>{rv['f1']['value']:.6f}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
<tr><td>Target coverage</td><td>{rv['target_coverage']['value']:.6f}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
<tr><td>Mean IoU</td><td>{rv['mean_iou']['value']:.6f}</td>
<td>REAL_VIDEO_VALIDATED</td></tr>
</table>
<p>Annotation-derived SoccerTrack metrics and NOT EVALUABLE fields are in the JSON/PNG.</p>
</div>

<div class="card">
<h2>Files</h2>
<ul>
<li><a href="real_video_tracking_proof.mp4">Download MP4</a></li>
<li><a href="single_player_analysis_summary.png">Open PNG</a></li>
<li><a href="single_player_analysis_summary.json">Open JSON</a></li>
<li><a href="evidence_manifest.json">Evidence manifest</a></li>
<li><a href="checksums.sha256">Checksums</a></li>
</ul>
<p>
Not official Opta data. Video-event inference accuracy is not validated.
No internet required.
</p>
</div>
</body>
</html>
"""
    (delivery / "OPEN_RESULTS.html").write_text(html, encoding="utf-8")


__all__ = [
    "EXPECTED_MP4_SHA",
    "EXPECTED_MP4_SIZE",
    "build_portable_proof_mp4",
    "download_source",
    "render_rgb_png",
    "validate_portable_mp4",
    "write_open_results_html",
]
