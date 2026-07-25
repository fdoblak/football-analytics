"""Turkish product proof video — confirmed tracks only; no mass GT dual boxes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.final_perception_repair.pipeline import (
    COLOR_BALL_CAND_BGR,
    COLOR_BALL_OBS_BGR,
    COLOR_REF_BGR,
    COLOR_TARGET_BGR,
    COLOR_TEAM_A_BGR,
    COLOR_TEAM_B_BGR,
    COLOR_UNKNOWN_BGR,
)
from football_analytics.acceptance.portable_final_media import OUT_H, OUT_W, _letterbox, _scale_box
from football_analytics.acceptance.teamtrack.loader import load_sequence

TEAM_LABEL_TR = {
    "team_a": "Takım A",
    "team_b": "Takım B",
    "unknown": "Bilinmeyen",
    "referee_or_staff": "Hakem/Personel",
    "goalkeeper_candidate": "Kaleci Adayı",
}

BALL_STATE_TR = {
    "observed": "Top Gözlendi",
    "candidate": "Top Adayı",
    "ambiguous": "Top Belirsiz",
    "lost": "Top Kayıp",
    "not_visible": "Top Görünmüyor",
    "not_evaluable": "Top Değerlendirilemez",
}


def _team_color(team: str) -> tuple[int, int, int]:
    return {
        "team_a": COLOR_TEAM_A_BGR,
        "team_b": COLOR_TEAM_B_BGR,
        "unknown": COLOR_UNKNOWN_BGR,
        "referee_or_staff": COLOR_REF_BGR,
        "goalkeeper_candidate": COLOR_UNKNOWN_BGR,
    }.get(team, COLOR_UNKNOWN_BGR)


def build_analysis_proof_mp4(
    *,
    sequence_root: Path,
    frame_dump_json: Path,
    perception_report_json: Path,
    final_mp4: Path,
    show_target_gt_dashed: bool = True,
) -> dict[str, Any]:
    seq = load_sequence(
        root=sequence_root,
        sport_view="soccer_side",
        split="train",
        sequence_id="F_20200220_1_0330_0360",
    )
    dump = json.loads(frame_dump_json.read_text())
    report = json.loads(perception_report_json.read_text())
    team_by_track = {int(k): v for k, v in dump.get("team_by_track", {}).items()}
    tracks_by_frame = {int(k): v for k, v in dump.get("tracks_by_frame", {}).items()}
    gt_target = {int(k): tuple(v) for k, v in dump.get("gt_target_by_frame", {}).items()}
    ball_by_frame = {int(k): v for k, v in dump.get("ball_by_frame", {}).items()}
    best_tid = dump.get("best_pred_track_id")

    det = report.get("detection", {}).get("full_selected", {})
    trk = report.get("tracking", {}).get("confirmed_iou", {})
    team_m = report.get("team", {}).get("metrics", {})
    ball_s = report.get("ball", {}).get("summary", {})
    tgt = report.get("tracking", {}).get("target_tracking_eval", {})

    final_mp4.parent.mkdir(parents=True, exist_ok=True)
    staging = final_mp4.with_suffix(".staging.mp4")
    if staging.exists():
        staging.unlink()

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
        str(staging),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    cap = cv2.VideoCapture(str(seq.video_path))
    frame_idx = 0
    selected_rgb: dict[str, Any] = {}
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
            canvas, scale, x0, y0 = _letterbox(frame)

            for item in tracks_by_frame.get(frame_idx, []):
                if not item.get("confirmed"):
                    continue
                tid = int(item["track_id"])
                box = tuple(item["bbox"])
                team = team_by_track.get(tid, "unknown")
                color = _team_color(team)
                thickness = 2
                if best_tid is not None and tid == int(best_tid):
                    color = COLOR_TARGET_BGR
                    thickness = 3
                sx, sy, sw, sh = _scale_box(box, scale, x0, y0)
                cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, thickness)
                label = f"ID{tid} {TEAM_LABEL_TR.get(team, team)}"
                if best_tid is not None and tid == int(best_tid):
                    label = f"Hedef Takip ID{tid}"
                cv2.putText(
                    canvas,
                    label,
                    (sx, max(18, sy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Target GT: thin dashed outline ONLY for Track 7 (evaluation cue)
            if show_target_gt_dashed and frame_idx in gt_target:
                gx, gy, gw, gh = _scale_box(gt_target[frame_idx], scale, x0, y0)
                # dashed rectangle
                pts = [
                    (gx, gy),
                    (gx + gw, gy),
                    (gx + gw, gy + gh),
                    (gx, gy + gh),
                ]
                for i in range(4):
                    p1, p2 = pts[i], pts[(i + 1) % 4]
                    # draw segments
                    nseg = 8
                    for s in range(nseg):
                        if s % 2:
                            continue
                        t0 = s / nseg
                        t1 = (s + 1) / nseg
                        a = (int(p1[0] + (p2[0] - p1[0]) * t0), int(p1[1] + (p2[1] - p1[1]) * t0))
                        b = (int(p1[0] + (p2[0] - p1[0]) * t1), int(p1[1] + (p2[1] - p1[1]) * t1))
                        cv2.line(canvas, a, b, (0, 200, 0), 1, cv2.LINE_AA)

            ball = ball_by_frame.get(frame_idx, {})
            bstate = ball.get("state", "not_visible")
            bb = ball.get("bbox")
            if bstate in {"observed", "candidate"} and bb:
                bx, by, bw, bh = _scale_box(tuple(bb), scale, x0, y0)
                bcol = COLOR_BALL_OBS_BGR if bstate == "observed" else COLOR_BALL_CAND_BGR
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bcol, 2)
                cv2.putText(
                    canvas,
                    BALL_STATE_TR.get(bstate, bstate),
                    (bx, max(18, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    bcol,
                    1,
                    cv2.LINE_AA,
                )
            ball_hud = BALL_STATE_TR.get(bstate, bstate)
            if bstate not in {"observed", "candidate"}:
                ball_hud = "Top: güvenilir biçimde tespit edilemedi"

            t_s = (frame_idx - 1) / seq.fps
            # Main Turkish HUD
            hud = [
                "GERÇEK VİDEO SİSTEM KANITI — TeamTrack (SoccerTrack oyuncusu DEĞİL)",
                f"t={t_s:.2f}s  kare={frame_idx}/{seq.seq_length}",
                ball_hud,
            ]
            ytxt = 22
            for line in hud:
                cv2.putText(
                    canvas,
                    line,
                    (14, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ytxt += 20

            # Evaluation inset (corner)
            inset_x, inset_y, inset_w, inset_h = OUT_W - 340, 8, 328, 150
            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (inset_x, inset_y),
                (inset_x + inset_w, inset_y + inset_h),
                (20, 20, 20),
                -1,
            )
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
            inset_lines = [
                "DEĞERLENDİRME",
                (
                    f"P={det.get('precision', 0):.3f} "
                    f"R={det.get('recall', 0):.3f} "
                    f"F1={det.get('f1', 0):.3f}"
                ),
                (
                    f"ID sw={trk.get('id_switches', '?')} "
                    f"frag={trk.get('fragmentation_events', '?')}"
                ),
                (
                    f"Hedef kap={tgt.get('target_coverage_ratio', 0):.3f} "
                    f"IoU={tgt.get('mean_iou', 0):.3f}"
                ),
                f"Takım tutarl.={team_m.get('within_track_consistency', 0):.2f}",
                f"Top obs+cand={ball_s.get('observed', 0)+ball_s.get('candidate', 0)}",
            ]
            iy = inset_y + 18
            for line in inset_lines:
                cv2.putText(
                    canvas,
                    line,
                    (inset_x + 8, iy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                iy += 22

            proc.stdin.write(canvas.tobytes())
            for label, fi in pick.items():
                if frame_idx == fi:
                    selected_rgb[label] = {
                        "frame_index": frame_idx,
                        "t_s": t_s,
                        "rgb": cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
                    }
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")

    staging.replace(final_mp4)
    return {
        "path": str(final_mp4),
        "sha256": sha256_file(final_mp4),
        "size_bytes": final_mp4.stat().st_size,
        "selected_frames": {
            k: {"frame_index": v["frame_index"], "t_s": v["t_s"]} for k, v in selected_rgb.items()
        },
        "selected_rgb": selected_rgb,
        "no_mass_gt_boxes": True,
        "confirmed_tracks_only": True,
    }


__all__ = ["build_analysis_proof_mp4"]
