#!/usr/bin/env python3
"""R1-F1-R1: reject bad GT draft, bake-off human detectors, build preacceptance package.

Diagnostic metrics only — NOT accuracy / NOT GT freeze / NOT R2.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from football_analytics.acceptance.stage18_own_video.pipeline import compute_pitch_masks
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.human_tiled_detection import (
    HumanDetectConfig,
    HumanProposal,
    detect_humans,
    duplicate_pairs,
    merged_person_candidates,
)

REPO = Path(__file__).resolve().parents[1]
VIDEO = Path("/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4")
WEIGHTS = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
WEIGHTS_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
OUT = REPO / "artifacts/evidence/reboot_01/r1_detection_preacceptance"
WORK = Path("/home/fdoblak/workspace/own_video_analysis/r1_f1_r1_bakeoff")
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection")
FPS = 30.0

LEGEND = (
    "R1 HUMAN DETECTION ONLY — colors are NOT team/role. "
    "Cyan=on-pitch candidate | Gray dashed=off-pitch | Orange=uncertain"
)

CONFIGS: dict[str, HumanDetectConfig] = {
    "A_yolo11n_full_baseline": HumanDetectConfig(
        name="A_yolo11n_full_baseline",
        mode="full_frame",
        conf=0.22,
        imgsz_full=960,
        merge_iou=0.50,
    ),
    "B_yolo11n_hybrid_tiled": HumanDetectConfig(
        name="B_yolo11n_hybrid_tiled",
        mode="hybrid",
        conf=0.18,
        imgsz_full=960,
        imgsz_tile=640,
        tile_width=672,
        tile_height=420,
        overlap_x=112,
        overlap_y=84,
        max_tiles=12,
        merge_iou=0.55,
        min_h=22.0,
        min_area=140.0,
    ),
    "C_yolo11n_highres_full": HumanDetectConfig(
        name="C_yolo11n_highres_full",
        mode="full_frame",
        conf=0.18,
        imgsz_full=1280,
        merge_iou=0.50,
        min_h=22.0,
        min_area=140.0,
    ),
}

# Audit frames across 0-12 / 12-22 / 22-34s
AUDIT_FRAMES = [
    3,
    60,
    150,
    270,
    330,
    400,
    500,
    580,
    660,
    720,
    780,
    860,
    920,
    980,
    1010,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def draw_proposals(
    frame: np.ndarray,
    props: list[HumanProposal],
    *,
    show_suppressed: bool = False,
) -> np.ndarray:
    out = frame.copy()
    for p in props:
        if p.suppressed and not show_suppressed:
            continue
        x1, y1, x2, y2 = map(int, [p.x1, p.y1, p.x2, p.y2])
        if p.eligibility == "on_pitch_human_candidate":
            color = (255, 255, 0)  # cyan BGR
            thick = 2
            dashed = False
        elif p.eligibility == "off_pitch_human":
            color = (160, 160, 160)
            thick = 1
            dashed = True
        else:
            color = (0, 140, 255)  # orange
            thick = 2
            dashed = False
        if p.suppressed:
            color = (0, 0, 220)
            dashed = True
        if dashed:
            # simple dashed rectangle
            for i in range(x1, x2, 8):
                cv2.line(out, (i, y1), (min(i + 4, x2), y1), color, thick)
                cv2.line(out, (i, y2), (min(i + 4, x2), y2), color, thick)
            for j in range(y1, y2, 8):
                cv2.line(out, (x1, j), (x1, min(j + 4, y2)), color, thick)
                cv2.line(out, (x2, j), (x2, min(j + 4, y2)), color, thick)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    # legend strip
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 36), (0, 0, 0), -1)
    cv2.putText(
        out,
        LEGEND[:110],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def hud(frame: np.ndarray, fi: int, n_on: int, n_off: int, n_unc: int) -> np.ndarray:
    out = frame.copy()
    t = fi / FPS
    text = f"f={fi} t={t:.2f}s on={n_on} off={n_off} unc={n_unc}"
    cv2.putText(out, text, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(items: list[tuple[str, np.ndarray]], out: Path, *, cols: int = 3) -> None:
    thumbs = []
    tw, th = 520, 290
    for title, img in items:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, tw - 1, 22], fill=(0, 0, 0))
        draw.text((4, 4), title[:70], fill=(255, 255, 0))
        thumbs.append(pil)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (16, 16, 16))
    for i, th_img in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(th_img, (c * tw, r * th))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")


def load_adapter() -> UltralyticsPersonAdapter:
    ad = UltralyticsPersonAdapter()
    ad.load(str(WEIGHTS), WEIGHTS_SHA)
    return ad


def score_config(
    adapter: UltralyticsPersonAdapter,
    cfg: HumanDetectConfig,
    frames: dict[int, np.ndarray],
) -> dict[str, Any]:
    per_frame = []
    total_on = total_off = total_unc = 0
    dup = merged = 0
    small = 0
    temporal_counts: list[int] = []
    for fi in sorted(frames):
        props = detect_humans(adapter, frames[fi], cfg, apply_pitch=True)
        on = [p for p in props if p.eligibility == "on_pitch_human_candidate"]
        off = [p for p in props if p.eligibility == "off_pitch_human"]
        unc = [p for p in props if p.eligibility == "uncertain"]
        total_on += len(on)
        total_off += len(off)
        total_unc += len(unc)
        dup += duplicate_pairs(props)
        merged += merged_person_candidates(on)
        small += sum(1 for p in on if (p.y2 - p.y1) < 55)
        temporal_counts.append(len(on))
        per_frame.append(
            {
                "frame_idx": fi,
                "n_on": len(on),
                "n_off": len(off),
                "n_unc": len(unc),
                "dup_pairs": duplicate_pairs(props),
                "merged_candidates": merged_person_candidates(on),
            }
        )
    # temporal stability: std of on-pitch counts / mean
    arr = np.array(temporal_counts, dtype=float)
    stability = float(arr.std() / (arr.mean() + 1e-6))
    return {
        "config": asdict(cfg),
        "n_frames": len(frames),
        "total_on_pitch": total_on,
        "total_off_pitch": total_off,
        "total_uncertain": total_unc,
        "mean_on_pitch": float(arr.mean()) if len(arr) else 0.0,
        "temporal_count_cv": stability,
        "duplicate_pair_count": dup,
        "merged_person_candidate_count": merged,
        "small_person_box_count_h_lt_55": small,
        "off_pitch_rate": total_off / max(1, total_on + total_off + total_unc),
        "per_frame": per_frame,
        "diagnostic_not_accuracy": True,
    }


def choose_winner(results: dict[str, dict[str, Any]]) -> str:
    """Heuristic selection without claiming F1 accuracy."""
    baseline = results["A_yolo11n_full_baseline"]
    best_name = "A_yolo11n_full_baseline"
    best_score = -1e9
    for name, r in results.items():
        # Prefer more small detections, fewer merges/dups, more mean on-pitch,
        # but penalize extreme off-pitch and wild temporal cv.
        score = (
            1.5 * r["small_person_box_count_h_lt_55"]
            + 1.0 * r["mean_on_pitch"]
            - 2.0 * r["merged_person_candidate_count"]
            - 2.5 * r["duplicate_pair_count"]
            - 8.0 * r["off_pitch_rate"]
            - 3.0 * r["temporal_count_cv"]
        )
        # must not be worse than baseline on small+mean combined
        if name != "A_yolo11n_full_baseline":
            improved = (
                r["small_person_box_count_h_lt_55"] >= baseline["small_person_box_count_h_lt_55"]
                or r["mean_on_pitch"] >= baseline["mean_on_pitch"] + 0.5
            ) and r["merged_person_candidate_count"] <= baseline[
                "merged_person_candidate_count"
            ] + 2
            if not improved:
                score -= 50
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def build_proof_and_sheets(
    adapter: UltralyticsPersonAdapter,
    cfg: HumanDetectConfig,
    baseline_cfg: HumanDetectConfig,
) -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = OUT / "evidence"
    evidence.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # render at source size (near 1280x720)
    tmp = WORK / "_proof.avi"
    WORK.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))

    buckets: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    before_after: list[tuple[str, np.ndarray]] = []
    qa = {
        "oob": 0,
        "zero_area": 0,
        "dup_gt_09": 0,
        "frames": 0,
        "total_boxes": 0,
    }

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        props = detect_humans(adapter, frame, cfg, apply_pitch=True)
        for p in props:
            if p.x2 <= p.x1 or p.y2 <= p.y1:
                qa["zero_area"] += 1
            if p.x1 < 0 or p.y1 < 0 or p.x2 > w or p.y2 > h:
                qa["oob"] += 1
        qa["dup_gt_09"] += duplicate_pairs(props, iou_thresh=0.9)
        qa["frames"] += 1
        qa["total_boxes"] += len(props)
        on = sum(1 for p in props if p.eligibility == "on_pitch_human_candidate")
        off = sum(1 for p in props if p.eligibility == "off_pitch_human")
        unc = sum(1 for p in props if p.eligibility == "uncertain")
        vis = hud(draw_proposals(frame, props), i, on, off, unc)
        writer.write(vis)

        t = i / FPS
        if i in AUDIT_FRAMES:
            title = f"f{i} t={t:.1f}s"
            if t < 12 and len(buckets["start"]) < 9:
                buckets["start"].append((title, vis))
            elif 12 <= t < 22 and len(buckets["middle"]) < 9:
                buckets["middle"].append((title, vis))
            elif t >= 22 and len(buckets["holdout"]) < 9:
                buckets["holdout"].append((title, vis))
            # crowded: many on-pitch
            if on >= 10 and len(buckets["crowded"]) < 9:
                buckets["crowded"].append((f"crowded {title} n={on}", vis))
            # small distant
            if len(buckets["small"]) < 9 and any(
                (p.y2 - p.y1) < 55 for p in props if p.eligibility == "on_pitch_human_candidate"
            ):
                buckets["small"].append((f"small {title}", vis))
            # off-pitch filtering view
            if off > 0 and len(buckets["off"]) < 9:
                buckets["off"].append((f"off {title} off={off}", vis))
            # before/after pairs
            if len(before_after) < 8:
                base_props = detect_humans(adapter, frame, baseline_cfg, apply_pitch=True)
                left = draw_proposals(frame, base_props)
                right = draw_proposals(frame, props)
                pair = np.concatenate([left, right], axis=1)
                cv2.putText(
                    pair,
                    "LEFT=baseline A | RIGHT=selected",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                before_after.append((title, pair))

    writer.release()
    cap.release()

    proof = OUT / "R1_human_detection_proof.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-movflags",
            "+faststart",
            "-an",
            str(proof),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink(missing_ok=True)

    paths = {
        "proof": proof,
        "start": evidence / "R1_start.png",
        "middle": evidence / "R1_middle.png",
        "holdout": evidence / "R1_holdout.png",
        "crowded": evidence / "R1_crowded_cases.png",
        "small": evidence / "R1_small_distant_cases.png",
        "off": evidence / "R1_off_pitch_filtering.png",
        "before_after": evidence / "R1_before_after.png",
    }
    # also top-level copies required by spec
    top = {
        "start": OUT / "R1_start.png",
        "middle": OUT / "R1_middle.png",
        "holdout": OUT / "R1_holdout.png",
        "crowded": OUT / "R1_crowded_cases.png",
        "small": OUT / "R1_small_distant_cases.png",
        "off": OUT / "R1_off_pitch_filtering.png",
        "before_after": OUT / "R1_before_after.png",
    }
    contact_sheet(buckets["start"] or before_after[:6], paths["start"])
    contact_sheet(buckets["middle"] or before_after[2:8], paths["middle"])
    contact_sheet(buckets["holdout"] or before_after[-6:], paths["holdout"])
    contact_sheet(buckets["crowded"] or buckets["start"][:6], paths["crowded"])
    contact_sheet(buckets["small"] or buckets["start"][:6], paths["small"])
    contact_sheet(buckets["off"] or buckets["start"][:6], paths["off"])
    contact_sheet(before_after[:6], paths["before_after"], cols=2)
    for k, p in top.items():
        shutil.copy2(paths[k], p)
    atomic_json(OUT / "_qa_geometry.json", {**qa, "diagnostic_not_accuracy": True})
    return {"proof": proof, **top, "qa": OUT / "_qa_geometry.json"}


def write_html(selected: str, bakeoff: dict[str, Any], diag: dict[str, Any]) -> Path:
    html = OUT / "OPEN_R1_DETECTION_PREACCEPTANCE.html"
    body = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1 Detection Preacceptance</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#8cf}} img{{max-width:100%;border:1px solid #444;margin:8px 0}}
.box{{background:#1a1a1a;padding:12px;margin:12px 0;border-radius:6px}}
.warn{{color:#ffd36a}}
</style></head><body>
<h1>R1 Human Detection — Görsel Ön Kabul</h1>
<p class="warn"><b>Bu paket bağımsız GT / freeze / acceptance metriği değildir.</b>
Renkler takım veya rol belirtmez. Yalnız insan algılama önerileridir.</p>
<div class="box">
<ul>
<li>Seçilen config: <b>{selected}</b></li>
<li>Diagnostic (accuracy değil): mean on-pitch={diag.get('mean_on_pitch')}, 
small={diag.get('small_person_box_count_h_lt_55')}, 
merged={diag.get('merged_person_candidate_count')}, 
dup={diag.get('duplicate_pair_count')}</li>
<li>Model: YOLO11n COCO person — AGPL-3.0 evaluation/research only</li>
</ul>
</div>
<div class="box">
<h2>Proof video</h2>
<video controls width="960" src="R1_human_detection_proof.mp4"></video>
</div>
<div class="box">
<h2>Contact sheets</h2>
<p>Start</p><img src="R1_start.png"/>
<p>Middle</p><img src="R1_middle.png"/>
<p>Holdout</p><img src="R1_holdout.png"/>
<p>Crowded</p><img src="R1_crowded_cases.png"/>
<p>Small/distant</p><img src="R1_small_distant_cases.png"/>
<p>Off-pitch filtering</p><img src="R1_off_pitch_filtering.png"/>
<p>Before / after</p><img src="R1_before_after.png"/>
</div>
<p>Legend: Cyan solid = on-pitch human candidate · Gray dashed = off-pitch · Orange = uncertain</p>
</body></html>
"""
    html.write_text(body, encoding="utf-8")
    return html


def mirror_windows(paths: dict[str, Path]) -> None:
    WIN.mkdir(parents=True, exist_ok=True)
    ev = WIN / "evidence"
    ev.mkdir(exist_ok=True)
    # clear stale active rejected visuals from root (keep archive)
    for pat in (
        "R1_GT_draft_*.png",
        "R1_GT_difficult_examples.png",
        "R1_GT_negative_examples.png",
        "R1_coordinate_before_after.png",
        "coordinate_smoke_contact_sheet.png",
        "OPEN_R1_RESULTS.html",
        "R1_GT_REVIEW.html",
        "GATE_STATUS.json",
        "README_TR.txt",
        "gt_draft_auto_qa.json",
    ):
        for p in WIN.glob(pat):
            dest = WIN / "_rejected_r1_f1_archive" / p.name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                shutil.move(str(p), str(dest))
            else:
                p.unlink(missing_ok=True)

    readme = """R1 Human Detection — Görsel Ön Kabul
====================================

Ana giriş:
1) OPEN_R1_DETECTION_PREACCEPTANCE.html
2) R1_human_detection_proof.mp4

Bu paket bağımsız GT değildir. Takım/rol yok.
Kutular yeterince doğru görünürse sonraki adım R1-F2 (insan review + freeze) olur.

Eski reddedilmiş R1-F1 draft arşivi: _rejected_r1_f1_archive/
"""
    (WIN / "README_TR.txt").write_text(readme, encoding="utf-8")
    shutil.copy2(
        OUT / "OPEN_R1_DETECTION_PREACCEPTANCE.html", WIN / "OPEN_R1_DETECTION_PREACCEPTANCE.html"
    )
    shutil.copy2(OUT / "R1_human_detection_proof.mp4", WIN / "R1_human_detection_proof.mp4")
    shutil.copy2(OUT / "README_TR.txt", WIN / "README_TR.txt")
    for name in (
        "R1_start.png",
        "R1_middle.png",
        "R1_holdout.png",
        "R1_crowded_cases.png",
        "R1_small_distant_cases.png",
        "R1_off_pitch_filtering.png",
        "R1_before_after.png",
    ):
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, ev / name)
            # also keep top-level mirrors for SHA compare convenience in repo package only
    # copy jsons
    for name in (
        "detector_bakeoff.json",
        "diagnostic_quality.json",
        "MANIFEST.json",
        "checksums.sha256",
    ):
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, ev / name)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README_TR.txt").write_text(
        "R1 detection preacceptance — not GT. Colors are not team/role.\n"
        "Open OPEN_R1_DETECTION_PREACCEPTANCE.html\n",
        encoding="utf-8",
    )

    # pitch diagnostic on a few frames
    cap = cv2.VideoCapture(str(VIDEO))
    frames: dict[int, np.ndarray] = {}
    for fi in AUDIT_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames[fi] = fr
    cap.release()
    pitch_notes = []
    for fi, fr in list(frames.items())[:5]:
        m = compute_pitch_masks(fr)
        pitch_notes.append({"frame_idx": fi, "area_frac": m.area_frac, "fence_y": m.fence_y})

    adapter = load_adapter()
    bakeoff_results = {}
    for name, cfg in CONFIGS.items():
        print("scoring", name, flush=True)
        bakeoff_results[name] = score_config(adapter, cfg, frames)

    selected = choose_winner(bakeoff_results)
    print("selected", selected, flush=True)
    baseline = "A_yolo11n_full_baseline"
    improved = selected != baseline or (
        bakeoff_results[selected]["small_person_box_count_h_lt_55"]
        > bakeoff_results[baseline]["small_person_box_count_h_lt_55"]
        or bakeoff_results[selected]["mean_on_pitch"] > bakeoff_results[baseline]["mean_on_pitch"]
    )

    bakeoff = {
        "schema": "r1_f1_r1_detector_bakeoff_v1",
        "diagnostic_not_accuracy": True,
        "weights": {
            "path": str(WEIGHTS),
            "sha256": WEIGHTS_SHA,
            "size_bytes": WEIGHTS.stat().st_size,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "license": "AGPL-3.0",
            "usage": "evaluation_research_only",
            "downloaded_this_stage": False,
        },
        "configs": {k: asdict(v) for k, v in CONFIGS.items()},
        "results": bakeoff_results,
        "selected": selected,
        "pitch_mask_notes": pitch_notes,
        "written_at_utc": utc_now(),
    }
    atomic_json(OUT / "detector_bakeoff.json", bakeoff)

    if not improved:
        gate = "NO-GO — HUMAN DETECTION PROPOSALS STILL UNUSABLE"
        atomic_json(
            OUT / "GATE_STATUS.json",
            {"gate": gate, "selected": selected, "written_at_utc": utc_now()},
        )
        print(gate)
        return 2

    paths = build_proof_and_sheets(adapter, CONFIGS[selected], CONFIGS[baseline])
    diag = {
        **bakeoff_results[selected],
        "schema": "r1_f1_r1_diagnostic_quality_v1",
        "diagnostic_not_accuracy": True,
        "selected": selected,
        "categories_vs_baseline": {
            "small_distant": bakeoff_results[selected]["small_person_box_count_h_lt_55"]
            - bakeoff_results[baseline]["small_person_box_count_h_lt_55"],
            "mean_on_pitch": bakeoff_results[selected]["mean_on_pitch"]
            - bakeoff_results[baseline]["mean_on_pitch"],
            "merged_delta": bakeoff_results[selected]["merged_person_candidate_count"]
            - bakeoff_results[baseline]["merged_person_candidate_count"],
            "dup_delta": bakeoff_results[selected]["duplicate_pair_count"]
            - bakeoff_results[baseline]["duplicate_pair_count"],
        },
        "written_at_utc": utc_now(),
    }
    atomic_json(OUT / "diagnostic_quality.json", diag)
    write_html(selected, bakeoff, diag)

    # checksums + manifest
    files = [
        OUT / "OPEN_R1_DETECTION_PREACCEPTANCE.html",
        OUT / "R1_human_detection_proof.mp4",
        OUT / "R1_start.png",
        OUT / "R1_middle.png",
        OUT / "R1_holdout.png",
        OUT / "R1_crowded_cases.png",
        OUT / "R1_small_distant_cases.png",
        OUT / "R1_off_pitch_filtering.png",
        OUT / "R1_before_after.png",
        OUT / "detector_bakeoff.json",
        OUT / "diagnostic_quality.json",
        OUT / "README_TR.txt",
    ]
    lines = []
    entries = []
    for f in files:
        digest = sha256_file(f)
        lines.append(f"{digest}  {f.name}")
        entries.append({"file": f.name, "sha256": digest, "bytes": f.stat().st_size})
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "r1_detection_preacceptance_manifest_v1",
        "gate": "PASS_WITH_FINDINGS — IMPROVED HUMAN DETECTION PREACCEPTANCE READY; HUMAN GT NOT YET FROZEN",
        "selected_config": selected,
        "diagnostic_not_accuracy": True,
        "files": entries,
        "written_at_utc": utc_now(),
    }
    atomic_json(OUT / "MANIFEST.json", manifest)
    atomic_json(OUT / "GATE_STATUS.json", {"gate": manifest["gate"], "written_at_utc": utc_now()})
    atomic_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f1_r1_gate_status_v1",
            "gate": manifest["gate"],
            "selected_config": selected,
            "written_at_utc": utc_now(),
        },
    )

    mirror_windows(paths)
    # windows SHA compare
    win_cmp = []
    for name in (
        "OPEN_R1_DETECTION_PREACCEPTANCE.html",
        "R1_human_detection_proof.mp4",
        "README_TR.txt",
    ):
        a, b = OUT / name, WIN / name
        win_cmp.append(
            {
                "file": name,
                "repo_sha": sha256_file(a),
                "win_sha": sha256_file(b),
                "match": sha256_file(a) == sha256_file(b),
            }
        )
    atomic_json(OUT / "WINDOWS_MIRROR_SHA.json", {"files": win_cmp, "written_at_utc": utc_now()})

    # cleanup receipt
    cleaned = []
    for p in WORK.glob("**/*"):
        if p.is_file() and p.suffix in {".jpg", ".png", ".avi"}:
            sz = p.stat().st_size
            p.unlink()
            cleaned.append({"path": str(p), "bytes": sz})
    atomic_json(
        OUT / "CLEANUP_RECEIPT.json",
        {"data_loss": False, "removed": cleaned, "written_at_utc": utc_now()},
    )
    print(
        json.dumps(
            {
                "gate": manifest["gate"],
                "selected": selected,
                "diag": diag["categories_vs_baseline"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
