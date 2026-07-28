#!/usr/bin/env python3
"""Build R1-F1 blind GT review package artifacts (no detector acceptance)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from football_analytics.annotation.coordinates import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    make_source_bbox,
    validate_source_bbox_xyxy,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "artifacts/evidence/reboot_01"
OUT = EVIDENCE / "r1_f1_gt_review"
VIDEO = Path("/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4")
SELECTION = Path("/home/fdoblak/workspace/own_video_analysis/r1_blind_gt/frame_selection.json")
SMOKE = EVIDENCE / "coordinate_smoke_annotations.json"
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection")
WS_ANN = Path("/home/fdoblak/workspace/own_video_analysis/r1_blind_gt/annotations")
FPS = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def color_for(h: dict[str, Any]) -> tuple[int, int, int]:
    if h.get("ignore"):
        return (0, 220, 255)  # yellow BGR
    if h.get("uncertain"):
        return (0, 140, 255)  # orange BGR
    return (0, 220, 0)  # green accepted draft


def draw_boxes(
    frame: np.ndarray, humans: list[dict[str, Any]], *, label: bool = True
) -> np.ndarray:
    out = frame.copy()
    for h in humans:
        x1, y1, x2, y2 = map(int, [h["x1"], h["y1"], h["x2"], h["y2"]])
        c = color_for(h)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        if label:
            tag = "IGNORE" if h.get("ignore") else ("UNCERTAIN" if h.get("uncertain") else "HUMAN")
            cv2.putText(
                out, tag, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
            )
    return out


def center_turf_diagnostic(frame: np.ndarray, box: list[float]) -> bool:
    """True if bbox center looks like empty green turf (diagnostic only)."""
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    if not (0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]):
        return True
    b, g, r = [int(v) for v in frame[cy, cx]]
    return g > r + 25 and g > b + 15 and g > 80


def foreground_hint(frame: np.ndarray, box: list[float]) -> bool:
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # non-green-ish pixels ratio
    green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    return float((~green).mean()) > 0.12


def build_draft() -> dict[str, Any]:
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    by_smoke = {int(f["frame_idx"]): f for f in smoke["frames"]}
    frames_out = []
    n_pos = n_ign = n_unc = n_complete = n_incomplete = 0
    for item in sel["frames"]:
        fi = int(item["frame_idx"])
        base = {
            "frame_idx": fi,
            "t_s": float(item["t_s"]),
            "video_time_us": int(round(float(item["t_s"]) * 1_000_000)),
            "split": item["split"],
            "source_width": SOURCE_WIDTH,
            "source_height": SOURCE_HEIGHT,
            "completed": False,
            "review_status": "not_reviewed",
            "humans": [],
            "notes": "Awaiting human blind review via R1 GT review tool.",
        }
        if fi in by_smoke:
            humans = []
            for h in by_smoke[fi]["humans"]:
                box = validate_source_bbox_xyxy([h["x1"], h["y1"], h["x2"], h["y2"]])
                bb = make_source_bbox(frame_index=fi, fps=FPS, bbox_xyxy=box)
                rec = bb.to_dict()
                rec.update(
                    {
                        "ignore": bool(h.get("ignore")),
                        "uncertain": bool(h.get("uncertain", False)),
                        "difficult": bool(h.get("difficult")),
                        "on_pitch": bool(h.get("on_pitch", True)),
                        "occlusion": h.get("occlusion", "none"),
                        "truncated": bool(h.get("truncated", False)),
                        "visible_fraction": float(h.get("visible_fraction", 1.0)),
                        "size": h.get("size", "medium"),
                        "class": "human_on_pitch",
                        "review_status": "agent_blind_reviewed_draft",
                        "proposal_method": h.get("proposal_method", "visual_curation"),
                    }
                )
                humans.append(rec)
                if rec["ignore"]:
                    n_ign += 1
                elif rec["uncertain"]:
                    n_unc += 1
                else:
                    n_pos += 1
            base["humans"] = humans
            base["completed"] = False  # draft only; freeze is R1-F2
            base["review_status"] = "agent_blind_reviewed_draft"
            base["notes"] = (
                "Coordinate-smoke visually curated draft boxes only; "
                "not human_approved; not frozen GT."
            )
            n_complete += 0
            n_incomplete += 1  # still needs human approval
        else:
            n_incomplete += 1
        frames_out.append(base)
    draft = {
        "schema": "own_video_human_blind_gt_draft_v1",
        "dataset_id": "own_video_human_blind_gt_v1",
        "status": "BLIND_GT_DRAFT_REVIEW_PACKAGE",
        "provenance": "agent_blind_reviewed_draft",
        "human_approved": False,
        "gt_frozen": False,
        "acceptance_eligible": False,
        "prediction_used": False,
        "source_id": "own_video_97b298e4",
        "source_sha256": "97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160",
        "canonical_path": str(VIDEO),
        "image_size": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT},
        "fps": FPS,
        "coordinate_space": "source_xyxy_px_v1",
        "n_frames": len(frames_out),
        "counts": {
            "frames_by_split": {
                "train": sum(1 for f in frames_out if f["split"] == "train"),
                "dev": sum(1 for f in frames_out if f["split"] == "dev"),
                "holdout": sum(1 for f in frames_out if f["split"] == "holdout"),
            },
            "frames_with_draft_boxes": sum(1 for f in frames_out if f["humans"]),
            "positive_boxes": n_pos,
            "ignore_boxes": n_ign,
            "uncertain_boxes": n_unc,
            "incomplete_frames": n_incomplete,
            "human_complete_frames": n_complete,
        },
        "frames": frames_out,
        "written_at_utc": utc_now(),
    }
    return draft


def run_auto_qa(draft: dict[str, Any]) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(VIDEO))
    issues: list[dict[str, Any]] = []
    turf_hits = 0
    fg_miss = 0
    n_boxes = 0
    sizes = []
    by_frame_boxes: dict[int, list[list[float]]] = {}
    for fr in draft["frames"]:
        fi = int(fr["frame_idx"])
        humans = fr["humans"]
        if abs(fr["t_s"] - fi / FPS) > 0.05:
            issues.append({"type": "timestamp_mismatch", "frame_idx": fi})
        if len(humans) > 40:
            issues.append({"type": "implausible_box_count", "frame_idx": fi, "n": len(humans)})
        seen = set()
        boxes = []
        frame = None
        if humans:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                issues.append({"type": "frame_read_fail", "frame_idx": fi})
                continue
        for h in humans:
            n_boxes += 1
            box = [h["x1"], h["y1"], h["x2"], h["y2"]]
            key = tuple(round(v, 1) for v in box)
            if key in seen:
                issues.append({"type": "duplicate_bbox", "frame_idx": fi, "box": box})
            seen.add(key)
            try:
                validate_source_bbox_xyxy(box)
            except Exception as exc:  # noqa: BLE001
                issues.append({"type": "validate_fail", "frame_idx": fi, "error": str(exc)})
            w, hh = box[2] - box[0], box[3] - box[1]
            area = w * hh
            sizes.append(area)
            if area < 40 or area > SOURCE_WIDTH * SOURCE_HEIGHT * 0.5:
                issues.append({"type": "size_outlier", "frame_idx": fi, "area": area})
            if frame is not None:
                if center_turf_diagnostic(frame, box):
                    turf_hits += 1
                if not foreground_hint(frame, box):
                    fg_miss += 1
            boxes.append(box)
        by_frame_boxes[fi] = boxes
    # cross-frame identical box carry (simple)
    carry = 0
    prev = None
    for fi in sorted(by_frame_boxes):
        cur = by_frame_boxes[fi]
        if prev is not None and cur and prev and cur == prev:
            carry += 1
            issues.append({"type": "identical_boxes_across_frames", "frame_idx": fi})
        prev = cur
    cap.release()
    report = {
        "schema": "r1_f1_gt_draft_auto_qa_v1",
        "n_boxes": n_boxes,
        "empty_turf_center_rate": (turf_hits / n_boxes) if n_boxes else None,
        "weak_foreground_rate": (fg_miss / n_boxes) if n_boxes else None,
        "identical_carry_events": carry,
        "issue_count": len(issues),
        "issues_sample": issues[:50],
        "note": "Diagnostic only; does not replace visual review.",
        "written_at_utc": utc_now(),
    }
    return report


def contact_sheet(
    paths_or_frames: list[tuple[str, np.ndarray]],
    out: Path,
    *,
    cols: int = 3,
) -> None:
    if not paths_or_frames:
        raise SystemExit("empty contact sheet")
    thumbs = []
    for title, img in paths_or_frames:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((446, 248), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, 445, 18], fill=(0, 0, 0))
        draw.text((4, 2), title, fill=(255, 255, 0))
        thumbs.append(pil)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 446, rows * 248), (20, 20, 20))
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(th, (c * 446, r * 248))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")


def build_proof_and_sheets(draft: dict[str, Any]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    by = {int(f["frame_idx"]): f for f in draft["frames"] if f["humans"]}
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_avi = OUT / "_tmp_proof.avi"
    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    samples: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    neg: list[tuple[str, np.ndarray]] = []
    hard: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        fr = by.get(i)
        humans = fr["humans"] if fr else []
        vis = draw_boxes(frame, humans) if humans else frame
        # HUD
        ts = i / FPS
        hud = f"f={i} t={ts:.2f}s boxes={len(humans)} BLIND-GT-DRAFT (no predictions)"
        cv2.putText(
            vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        writer.write(vis)
        if fr and len(samples["all"]) < 12:
            samples["all"].append((f"f{i}", vis))
        if fr:
            split = fr["split"]
            if split == "train" and len(samples["start"]) < 6:
                samples["start"].append((f"train f{i}", vis))
            if split == "dev" and len(samples["middle"]) < 6:
                samples["middle"].append((f"dev f{i}", vis))
            if split == "holdout" and len(samples["holdout"]) < 6:
                samples["holdout"].append((f"holdout f{i}", vis))
            if any(h.get("ignore") for h in humans) and len(neg) < 6:
                neg.append((f"ignore f{i}", vis))
            if any(h.get("difficult") for h in humans) and len(hard) < 6:
                hard.append((f"difficult f{i}", vis))
            if len(hard) < 6 and any(
                (h["x2"] - h["x1"]) * (h["y2"] - h["y1"]) < 1200 for h in humans
            ):
                hard.append((f"small f{i}", vis))
    writer.release()
    cap.release()
    proof = OUT / "R1_blind_GT_draft_proof.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_avi),
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(FPS)),
        "-movflags",
        "+faststart",
        "-an",
        str(proof),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_avi.unlink(missing_ok=True)

    # ensure sheets have content even if few ignore/difficult
    if len(neg) < 3:
        # use empty-ish frames from draft without boxes as negative context examples
        cap = cv2.VideoCapture(str(VIDEO))
        for fr in draft["frames"]:
            if fr["humans"]:
                continue
            fi = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            vis = frame.copy()
            cv2.putText(
                vis,
                "NO DRAFT BOXES (pending human review)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )
            neg.append((f"pending f{fi}", vis))
            if len(neg) >= 6:
                break
        cap.release()
    if len(hard) < 3:
        hard = samples["all"][:6]

    paths = {
        "proof": str(proof),
        "start": str(OUT / "R1_GT_draft_start.png"),
        "middle": str(OUT / "R1_GT_draft_middle.png"),
        "holdout": str(OUT / "R1_GT_draft_holdout.png"),
        "negative": str(OUT / "R1_GT_negative_examples.png"),
        "difficult": str(OUT / "R1_GT_difficult_examples.png"),
        "before_after": str(EVIDENCE / "R1_coordinate_before_after.png"),
    }
    contact_sheet(samples["start"] or samples["all"][:6], Path(paths["start"]))
    contact_sheet(samples["middle"] or samples["all"][3:9], Path(paths["middle"]))
    contact_sheet(samples["holdout"] or samples["all"][-6:], Path(paths["holdout"]))
    contact_sheet(neg[:6], Path(paths["negative"]))
    contact_sheet(hard[:6], Path(paths["difficult"]))
    return paths


def write_html(
    draft: dict[str, Any], qa: dict[str, Any], paths: dict[str, str], root_cause: dict[str, Any]
) -> Path:
    html_path = OUT / "OPEN_R1_GT_DRAFT_RESULTS.html"
    counts = draft["counts"]
    rc = root_cause.get("primary_root_cause") or root_cause.get("root_cause") or {}
    if isinstance(rc, dict):
        rc_txt = rc.get("primary") or rc.get("id") or json.dumps(rc)[:200]
    else:
        rc_txt = str(rc)
    body = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1-F1 Blind GT Draft Results</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#8cf}} img{{max-width:100%;border:1px solid #333;margin:8px 0}}
.box{{background:#1b1b1b;padding:12px;margin:12px 0;border-radius:6px}}
.warn{{color:#ffd36a}}
</style></head><body>
<h1>R1-F1 Blind GT Draft Review Package</h1>
<p class="warn"><b>Bu paket model prediction içermez.</b> Kutular yalnız blind GT draft / coordinate smoke doğrulamasıdır.
human_approved / GT freeze değildir. Acceptance metriği hesaplanmamıştır.</p>
<div class="box">
<h2>Özet</h2>
<ul>
<li>Frames: {draft['n_frames']} (train {counts['frames_by_split']['train']} / dev {counts['frames_by_split']['dev']} / holdout {counts['frames_by_split']['holdout']})</li>
<li>Frames with draft boxes: {counts['frames_with_draft_boxes']}</li>
<li>Positive draft boxes: {counts['positive_boxes']}</li>
<li>Ignore: {counts['ignore_boxes']} · Uncertain: {counts['uncertain_boxes']}</li>
<li>Incomplete (needs human review): {counts['incomplete_frames']}</li>
<li>Provenance: {draft['provenance']}</li>
<li>Coordinate root cause: {rc_txt}</li>
<li>Auto QA empty-turf center rate: {qa.get('empty_turf_center_rate')}</li>
</ul>
</div>
<div class="box">
<h2>Proof video</h2>
<video controls width="960" src="R1_blind_GT_draft_proof.mp4"></video>
</div>
<div class="box"><h2>Contact sheets</h2>
<p>Start</p><img src="R1_GT_draft_start.png"/>
<p>Middle</p><img src="R1_GT_draft_middle.png"/>
<p>Holdout</p><img src="R1_GT_draft_holdout.png"/>
<p>Negative / pending</p><img src="R1_GT_negative_examples.png"/>
<p>Difficult</p><img src="R1_GT_difficult_examples.png"/>
<p>Coordinate before/after</p><img src="../R1_coordinate_before_after.png"/>
</div>
</body></html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def write_windows_package() -> None:
    WIN.mkdir(parents=True, exist_ok=True)
    readme = """R1 Human Detection — Blind GT Review Paketi
==========================================

Bu klasör prediction/YOLO göstermeden insan kutularını gözle incelemeniz içindir.

Nasıl açılır
------------
1) START_R1_GT_REVIEW.bat dosyasına çift tıklayın.
2) Tarayıcı otomatik açılır (http://127.0.0.1:8765/).
3) Kutuları çizin / taşıyın / silin; Mark frame complete ile ilerleyin.
4) Bitince BAT penceresinde bir tuşa basın; sunucu durur.

Sonuçları görmek
----------------
OPEN_R1_GT_DRAFT_RESULTS.html — proof video ve contact sheet’ler (prediction yok).

Önemli
------
- İnternet/CDN kullanılmaz; yalnız localhost.
- Secret istenmez.
- Eski geçersiz (çime düşen) kutular acceptance’a dahil değildir.
- human_approved / GT freeze R1-F2’dedir.
"""
    (WIN / "README_TR.txt").write_text(readme, encoding="utf-8")
    bat = r"""@echo off
setlocal
title R1 Blind GT Review
echo Starting local R1 blind GT review server (WSL localhost:8765)...
echo Do not close this window until you finish reviewing.
wsl -e bash -lc "cd /home/fdoblak/projects/football-analytics && /home/fdoblak/miniconda3/envs/ai-dev/bin/python scripts/r1_blind_gt_review_server.py --host 127.0.0.1 --port 8765 --blind" 
"""
    # Better BAT: start server in background via wsl, open browser, wait for key, kill
    bat = r"""@echo off
setlocal EnableExtensions
title R1 Blind GT Review
cd /d "%~dp0"
echo.
echo === R1 Blind GT Review ===
echo Local only. No internet. No secrets.
echo.

REM Start review server inside WSL on localhost:8765
wsl -e bash -lc "pkill -f 'r1_blind_gt_review_server.py' >/dev/null 2>&1 || true; cd /home/fdoblak/projects/football-analytics && nohup /home/fdoblak/miniconda3/envs/ai-dev/bin/python scripts/r1_blind_gt_review_server.py --host 127.0.0.1 --port 8765 --blind >/tmp/r1_gt_review_server.log 2>&1 & echo $!" > "%TEMP%\r1_gt_review_pid.txt"
timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:8765/"
start "" "%~dp0OPEN_R1_GT_REVIEW.html"

echo Browser opened at http://127.0.0.1:8765/
echo.
echo When finished, press any key here to stop the server...
pause >nul

wsl -e bash -lc "pkill -f 'r1_blind_gt_review_server.py' >/dev/null 2>&1 || true"
echo Server stopped.
pause
"""
    (WIN / "START_R1_GT_REVIEW.bat").write_text(bat, encoding="utf-8")
    open_html = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1 GT Review Launcher</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#111;color:#eee}
a{color:#8cf;font-size:1.2rem}</style></head><body>
<h1>R1 Blind GT Review</h1>
<p>Önce <b>START_R1_GT_REVIEW.bat</b> çalıştırın, sonra:</p>
<p><a href="http://127.0.0.1:8765/">Review aracını aç (localhost:8765)</a></p>
<p><a href="OPEN_R1_GT_DRAFT_RESULTS.html">Draft sonuçları / proof video</a></p>
<p>Blind mode: YOLO / eski GT / tracker / takım-rol / confidence gizli.</p>
</body></html>
"""
    (WIN / "OPEN_R1_GT_REVIEW.html").write_text(open_html, encoding="utf-8")


def mirror_to_windows(paths: dict[str, str], html: Path) -> dict[str, Any]:
    WIN.mkdir(parents=True, exist_ok=True)
    copied = []
    for _key, p in paths.items():
        src = Path(p)
        if not src.is_file():
            continue
        dest = WIN / src.name
        shutil.copy2(src, dest)
        copied.append(
            {"file": dest.name, "sha256": sha256_file(dest), "bytes": dest.stat().st_size}
        )
    # also copy HTML and smoke contact sheet / root cause summary
    for extra in (
        html,
        EVIDENCE / "coordinate_smoke_contact_sheet.png",
        EVIDENCE / "annotation_coordinate_root_cause.json",
        EVIDENCE / "coordinate_roundtrip.json",
        OUT / "blind_gt_draft_annotations.json",
        OUT / "gt_draft_auto_qa.json",
    ):
        if extra.is_file():
            dest = WIN / extra.name
            shutil.copy2(extra, dest)
            copied.append(
                {"file": dest.name, "sha256": sha256_file(dest), "bytes": dest.stat().st_size}
            )
    # repo-side mirrors of the three launcher files for SHA equality checks
    for name in ("START_R1_GT_REVIEW.bat", "OPEN_R1_GT_REVIEW.html", "README_TR.txt"):
        src = WIN / name
        if src.is_file():
            dest = OUT / name
            shutil.copy2(src, dest)
            copied.append(
                {"file": f"repo:{name}", "sha256": sha256_file(dest), "bytes": dest.stat().st_size}
            )
    return {"schema": "r1_f1_windows_mirror_sha_v1", "written_at_utc": utc_now(), "files": copied}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WS_ANN.mkdir(parents=True, exist_ok=True)
    # clear any leftover invalid runtime copies (workspace annotations dir already empty)
    for p in WS_ANN.glob("*.json"):
        # never silently overwrite — only remove if marked invalid or empty leftover
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            p.unlink()
            continue
        if data.get("acceptance_eligible") is False or data.get("invalid") is True:
            p.unlink()

    draft = build_draft()
    atomic_write_json(OUT / "blind_gt_draft_annotations.json", draft)
    atomic_write_json(WS_ANN / "blind_gt_draft_annotations.json", draft)
    # replace repo gt template with draft pointer note (keep old empty template superseded)
    atomic_write_json(EVIDENCE / "gt" / "blind_gt_annotations.json", draft)

    qa = run_auto_qa(draft)
    atomic_write_json(OUT / "gt_draft_auto_qa.json", qa)
    paths = build_proof_and_sheets(draft)
    root_cause = json.loads(
        (EVIDENCE / "annotation_coordinate_root_cause.json").read_text(encoding="utf-8")
    )
    html = write_html(draft, qa, paths, root_cause)
    write_windows_package()
    mirror = mirror_to_windows(paths, html)
    atomic_write_json(OUT / "WINDOWS_MIRROR_SHA.json", mirror)
    atomic_write_json(EVIDENCE / "WINDOWS_MIRROR_SHA.json", mirror)

    # checksums
    files = [
        OUT / "blind_gt_draft_annotations.json",
        OUT / "gt_draft_auto_qa.json",
        OUT / "R1_blind_GT_draft_proof.mp4",
        OUT / "R1_GT_draft_start.png",
        OUT / "R1_GT_draft_middle.png",
        OUT / "R1_GT_draft_holdout.png",
        OUT / "R1_GT_negative_examples.png",
        OUT / "R1_GT_difficult_examples.png",
        html,
        EVIDENCE / "R1_coordinate_before_after.png",
        EVIDENCE / "coordinate_smoke_contact_sheet.png",
        EVIDENCE / "annotation_coordinate_root_cause.json",
    ]
    lines = []
    for f in files:
        if f.is_file():
            lines.append(f"{sha256_file(f)}  {f.name}")
    (OUT / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gate = {
        "schema": "r1_f1_gate_status_v1",
        "gate": "PASS — BLIND GT REVIEW PACKAGE READY",
        "note": (
            "Coordinate system repaired; review tool + Windows package + draft proof ready. "
            "Human approval / GT freeze / detector acceptance are R1-F2."
        ),
        "acceptance_metrics_computed": False,
        "written_at_utc": utc_now(),
    }
    atomic_write_json(OUT / "GATE_STATUS.json", gate)
    atomic_write_json(EVIDENCE / "GATE_STATUS.json", gate)
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "draft_boxes": draft["counts"],
                "qa": {"turf": qa["empty_turf_center_rate"], "issues": qa["issue_count"]},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
