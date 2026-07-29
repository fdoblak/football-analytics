#!/usr/bin/env python3
"""R1-F1-R2: stronger human detector bake-off + temporal stability + candidate v2.

Produces diagnostic_not_accuracy metrics only. No GT freeze / team / acceptance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.human_temporal_stability import (
    TemporalHumanStabilizer,
    TemporalProposal,
    compute_temporal_diagnostics,
)
from football_analytics.perception.human_tiled_detection import (
    HumanDetectConfig,
    detect_humans,
    duplicate_pairs,
    merged_person_candidates,
)

REPO = Path(__file__).resolve().parents[1]
VIDEO = Path("/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4")
ARCH = Path("/home/fdoblak/football_data/model_archive")
YOLO_N = ARCH / "yolo11n.pt"
YOLO_M = ARCH / "yolo11m.pt"
SHA_N = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SHA_M = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
OUT = REPO / "artifacts/evidence/reboot_01/r1_detection_candidate_v2"
WORK = Path("/home/fdoblak/workspace/own_video_analysis/r1_f1_r2")
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection")
FPS = 30.0

LEGEND = "R1 — YALNIZ INSAN TESPITI | RENKLER TAKIM VEYA ROL BELIRTMEZ | Cyan=observed Orange=carried Gray=off/unknown"

# Fixed audit set (30)
AUDIT_FRAMES = [
    3,
    30,
    60,
    90,
    120,
    150,
    180,
    210,
    240,
    270,
    300,
    330,
    360,
    400,
    450,
    500,
    540,
    580,
    620,
    660,
    700,
    740,
    780,
    820,
    860,
    900,
    940,
    980,
    1000,
    1010,
]

CONFIGS: dict[str, tuple[Path, str, HumanDetectConfig, bool]] = {
    # name -> (weights, sha, cfg, use_temporal)
    "A_yolo11n_full": (
        YOLO_N,
        SHA_N,
        HumanDetectConfig(
            name="A_yolo11n_full",
            mode="full_frame",
            conf=0.22,
            imgsz_full=960,
            merge_iou=0.50,
            half=True,
        ),
        False,
    ),
    "B_yolo11n_hybrid": (
        YOLO_N,
        SHA_N,
        HumanDetectConfig(
            name="B_yolo11n_hybrid",
            mode="hybrid",
            conf=0.18,
            imgsz_full=960,
            imgsz_tile=640,
            merge_iou=0.55,
            half=True,
        ),
        False,
    ),
    "C_yolo11m_hybrid": (
        YOLO_M,
        SHA_M,
        HumanDetectConfig(
            name="C_yolo11m_hybrid",
            mode="hybrid",
            conf=0.15,
            imgsz_full=1280,
            imgsz_tile=640,
            tile_width=672,
            tile_height=420,
            overlap_x=112,
            overlap_y=84,
            max_tiles=12,
            merge_iou=0.55,
            min_h=20.0,
            min_area=120.0,
            half=True,
        ),
        False,
    ),
    "D_yolo11m_hybrid_temporal": (
        YOLO_M,
        SHA_M,
        HumanDetectConfig(
            name="D_yolo11m_hybrid_temporal",
            mode="hybrid",
            conf=0.15,
            imgsz_full=1280,
            imgsz_tile=640,
            tile_width=672,
            tile_height=420,
            overlap_x=112,
            overlap_y=84,
            max_tiles=12,
            merge_iou=0.55,
            min_h=20.0,
            min_area=120.0,
            half=True,
        ),
        True,
    ),
}


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


def load_adapter(weights: Path, sha: str) -> UltralyticsPersonAdapter:
    ad = UltralyticsPersonAdapter()
    ad.load(str(weights), sha)
    return ad


def run_frame(
    adapter: UltralyticsPersonAdapter,
    frame: np.ndarray,
    cfg: HumanDetectConfig,
    stabilizer: TemporalHumanStabilizer | None,
) -> list[TemporalProposal]:
    props = detect_humans(adapter, frame, cfg, apply_pitch=True)
    if stabilizer is None:
        return [
            TemporalProposal(
                x1=p.x1,
                y1=p.y1,
                x2=p.x2,
                y2=p.y2,
                score=p.score,
                eligibility=p.eligibility,
                temporal_status="observed",
                source=p.source,
            )
            for p in props
        ]
    return stabilizer.update(frame, props)


def draw_temporal(frame: np.ndarray, props: list[TemporalProposal]) -> np.ndarray:
    out = frame.copy()
    for p in props:
        x1, y1, x2, y2 = map(int, p.as_xyxy())
        if p.temporal_status == "carried":
            color = (0, 140, 255)
            dashed = True
        elif p.eligibility in {"off_pitch_human", "unknown"}:
            color = (160, 160, 160)
            dashed = True
        else:
            color = (255, 255, 0)
            dashed = False
        if dashed:
            for i in range(x1, x2, 8):
                cv2.line(out, (i, y1), (min(i + 4, x2), y1), color, 2)
                cv2.line(out, (i, y2), (min(i + 4, x2), y2), color, 2)
            for j in range(y1, y2, 8):
                cv2.line(out, (x1, j), (x1, min(j + 4, y2)), color, 2)
                cv2.line(out, (x2, j), (x2, min(j + 4, y2)), color, 2)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 40), (0, 0, 0), -1)
    cv2.putText(
        out, LEGEND[:120], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA
    )
    return out


def contact_sheet(items: list[tuple[str, np.ndarray]], out: Path, *, cols: int = 2) -> None:
    tw, th = 640, 356
    thumbs = []
    for title, img in items:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, tw - 1, 24], fill=(0, 0, 0))
        draw.text((4, 4), title[:80], fill=(255, 255, 0))
        thumbs.append(pil)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (16, 16, 16))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * tw, r * th))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")


def evaluate_config(
    name: str,
    weights: Path,
    sha: str,
    cfg: HumanDetectConfig,
    use_temporal: bool,
    frames: dict[int, np.ndarray],
    ordered_ids: list[int],
) -> dict[str, Any]:
    adapter = load_adapter(weights, sha)
    stab = TemporalHumanStabilizer() if use_temporal else None
    seq: list[list[TemporalProposal]] = []
    small = merged = dup = 0
    n_obs = n_car = n_off = 0
    for fi in ordered_ids:
        props = run_frame(adapter, frames[fi], cfg, stab)
        seq.append(props)
        obs = [p for p in props if p.temporal_status == "observed"]
        # rebuild HumanProposal-like for merged/dup on observed only
        from football_analytics.perception.human_tiled_detection import HumanProposal as HP

        hp = [
            HP(
                x1=p.x1,
                y1=p.y1,
                x2=p.x2,
                y2=p.y2,
                score=p.score,
                eligibility=p.eligibility,  # type: ignore[arg-type]
                source=p.source,
            )
            for p in obs
        ]
        small += sum(
            1 for p in obs if (p.y2 - p.y1) < 55 and p.eligibility == "on_pitch_human_candidate"
        )
        merged += merged_person_candidates(hp)
        dup += duplicate_pairs(hp)
        n_obs += len(obs)
        n_car += sum(1 for p in props if p.temporal_status == "carried")
        n_off += sum(1 for p in props if p.eligibility in {"off_pitch_human", "unknown"})
    temporal = compute_temporal_diagnostics(seq)
    return {
        "name": name,
        "config": asdict(cfg),
        "use_temporal": use_temporal,
        "weights": str(weights),
        "n_frames": len(ordered_ids),
        "mean_observed": n_obs / max(1, len(ordered_ids)),
        "mean_carried": n_car / max(1, len(ordered_ids)),
        "small_person_count": small,
        "merged_person_candidate_count": merged,
        "duplicate_pair_count": dup,
        "off_or_unknown_count": n_off,
        "temporal": temporal,
        "diagnostic_not_accuracy": True,
    }


def choose_winner(results: dict[str, dict[str, Any]]) -> str | None:
    b = results["B_yolo11n_hybrid"]
    candidates = []
    for name in ("D_yolo11m_hybrid_temporal", "C_yolo11m_hybrid"):
        if name not in results:
            continue
        r = results[name]
        # Prefer effective gap (carry-aware) and jitter; require small not worse
        flicker_metric = r["temporal"].get(
            "effective_one_frame_gap", r["temporal"]["one_frame_disappearance"]
        )
        b_flicker = b["temporal"]["one_frame_disappearance"]
        improved_flicker = flicker_metric <= b_flicker * 0.85
        improved_jitter = r["temporal"]["center_jitter"] <= b["temporal"]["center_jitter"] * 0.95
        small_ok = r["small_person_count"] >= b["small_person_count"] * 0.9
        dup_ok = r["duplicate_pair_count"] <= b["duplicate_pair_count"] + 2
        merged_ok = r["merged_person_candidate_count"] <= b["merged_person_candidate_count"] + 2
        if small_ok and dup_ok and merged_ok and (improved_flicker or improved_jitter):
            score = (
                -4.0 * flicker_metric
                - 3.0 * r["temporal"]["center_jitter"]
                + 1.0 * r["small_person_count"]
                - 3.0 * r["merged_person_candidate_count"]
            )
            candidates.append((score, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def write_nogo(reason: str, bakeoff: dict[str, Any]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # remove any media if present
    for p in OUT.glob("*"):
        if p.suffix.lower() in {".mp4", ".png", ".html"}:
            p.unlink()
    atomic_json(
        OUT / "diagnostic_not_accuracy.json",
        {"gate": "NO-GO", "reason": reason, "bakeoff": bakeoff, "written_at_utc": utc_now()},
    )
    atomic_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "gate": "NO-GO — STRONGER HUMAN DETECTION STILL VISUALLY INSUFFICIENT",
            "reason": reason,
            "written_at_utc": utc_now(),
        },
    )
    WIN.mkdir(parents=True, exist_ok=True)
    for p in WIN.glob("*"):
        if p.is_file():
            p.unlink()
    (WIN / "NO_GO_STATUS.txt").write_text(
        "NO-GO — STRONGER HUMAN DETECTION STILL VISUALLY INSUFFICIENT\n" + reason + "\n",
        encoding="utf-8",
    )
    print("NO-GO", reason)
    return 2


def build_package(selected: str, bakeoff: dict[str, Any], audit: dict[str, Any]) -> None:
    if OUT.exists():
        for p in OUT.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".mp4", ".png", ".html", ".avi"}:
                p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    weights, sha, cfg, use_temporal = CONFIGS[selected]
    base_w, base_sha, base_cfg, _ = CONFIGS["B_yolo11n_hybrid"]
    adapter = load_adapter(weights, sha)
    base_ad = load_adapter(base_w, base_sha)
    stab = TemporalHumanStabilizer() if use_temporal else None

    # proof video
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = WORK / "_proof.avi"
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    before_after: list[tuple[str, np.ndarray]] = []
    difficult: list[tuple[str, np.ndarray]] = []
    off_pitch: list[tuple[str, np.ndarray]] = []

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        props = run_frame(adapter, frame, cfg, stab)
        vis = draw_temporal(frame, props)
        n_obs = sum(1 for p in props if p.temporal_status == "observed")
        n_car = sum(1 for p in props if p.temporal_status == "carried")
        cv2.putText(
            vis,
            f"f={i} t={i/FPS:.2f}s obs={n_obs} carry={n_car}",
            (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        writer.write(vis)
        if i in {90, 270, 450, 660, 780, 920}:
            base_props = run_frame(base_ad, frame, base_cfg, None)
            left = draw_temporal(frame, base_props)
            right = vis.copy()
            cv2.putText(
                left,
                "LEFT=B yolo11n hybrid",
                (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                right,
                f"RIGHT={selected}",
                (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            before_after.append((f"f{i}", np.concatenate([left, right], axis=1)))
        if i in {60, 330, 500, 660, 860, 1010}:
            difficult.append((f"diff f{i}", vis))
        if (
            i in AUDIT_FRAMES
            and len(off_pitch) < 6
            and any(p.eligibility in {"off_pitch_human", "unknown"} for p in props)
        ):
            off_pitch.append((f"off/unk f{i}", vis))
    writer.release()
    cap.release()

    proof = OUT / "R1_human_detection_candidate_v2.mp4"
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

    contact_sheet(before_after[:6], OUT / "R1_before_after_v2.png", cols=2)
    contact_sheet(difficult[:6], OUT / "R1_difficult_cases_v2.png", cols=3)
    contact_sheet(off_pitch[:6] or difficult[:6], OUT / "R1_off_pitch_v2.png", cols=3)

    atomic_json(
        OUT / "diagnostic_not_accuracy.json",
        {
            "schema": "r1_f1_r2_diagnostic_v1",
            "diagnostic_not_accuracy": True,
            "selected": selected,
            "bakeoff": bakeoff,
            "pixel_audit": audit,
            "written_at_utc": utc_now(),
        },
    )
    atomic_json(
        OUT / "model_provenance.json",
        {
            "id": "ultralytics_yolo11m_coco_person",
            "path": str(YOLO_M),
            "sha256": SHA_M,
            "size_bytes": YOLO_M.stat().st_size,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            "license": "AGPL-3.0",
            "usage": "evaluation_only",
            "downloaded_this_stage": True,
            "written_at_utc": utc_now(),
        },
    )

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>R1 Candidate v2</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#111;color:#eee}}
img{{max-width:100%;border:1px solid #444}} .warn{{color:#ffd36a}}</style></head><body>
<h1>R1 Human Detection Candidate v2</h1>
<p class="warn"><b>Bağımsız GT değildir.</b> Renkler takım/rol belirtmez. Cyan=observed, Orange dashed=carried(≤2), Gray=off/unknown.</p>
<p>Seçilen: {selected}</p>
<video controls width="960" src="R1_human_detection_candidate_v2.mp4"></video>
<p>Before/after</p><img src="R1_before_after_v2.png"/>
<p>Difficult</p><img src="R1_difficult_cases_v2.png"/>
<p>Off-pitch / unknown</p><img src="R1_off_pitch_v2.png"/>
</body></html>"""
    (OUT / "OPEN_R1_CANDIDATE_V2.html").write_text(html, encoding="utf-8")
    (OUT / "README_TR.txt").write_text(
        "R1 Candidate v2 — yalnız insan tespiti görsel ön kabul adayı.\n"
        "GT freeze / takım / acceptance yok.\n"
        "OPEN_R1_CANDIDATE_V2.html ve R1_human_detection_candidate_v2.mp4 açın.\n",
        encoding="utf-8",
    )

    files = [
        "R1_human_detection_candidate_v2.mp4",
        "R1_before_after_v2.png",
        "R1_difficult_cases_v2.png",
        "R1_off_pitch_v2.png",
        "OPEN_R1_CANDIDATE_V2.html",
        "diagnostic_not_accuracy.json",
        "model_provenance.json",
        "README_TR.txt",
    ]
    lines = []
    entries = []
    for name in files:
        p = OUT / name
        d = sha256_file(p)
        lines.append(f"{d}  {name}")
        entries.append({"file": name, "sha256": d, "bytes": p.stat().st_size})
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gate = "PASS_WITH_FINDINGS — STRONGER HUMAN DETECTION CANDIDATE READY; TEAM AND GT NOT YET VALIDATED"
    atomic_json(
        OUT / "MANIFEST.json",
        {
            "schema": "r1_detection_candidate_v2_manifest_v1",
            "gate": gate,
            "selected": selected,
            "diagnostic_not_accuracy": True,
            "files": entries,
            "written_at_utc": utc_now(),
        },
    )
    atomic_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "gate": gate,
            "selected": selected,
            "written_at_utc": utc_now(),
        },
    )

    # Windows mirror — only successful package
    if WIN.exists():
        for p in WIN.rglob("*"):
            if p.is_file():
                p.unlink()
    WIN.mkdir(parents=True, exist_ok=True)
    for name in [
        "OPEN_R1_CANDIDATE_V2.html",
        "R1_human_detection_candidate_v2.mp4",
        "README_TR.txt",
        "R1_before_after_v2.png",
        "R1_difficult_cases_v2.png",
        "R1_off_pitch_v2.png",
    ]:
        shutil.copy2(OUT / name, WIN / name)
    print("package ready", gate)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not YOLO_M.is_file():
        return write_nogo("yolo11m weights missing after download attempt", {})

    # provenance always
    provenance = {
        "path": str(YOLO_M),
        "sha256": sha256_file(YOLO_M),
        "size_bytes": YOLO_M.stat().st_size,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "license": "AGPL-3.0",
        "usage": "evaluation_only",
        "expected_sha256": SHA_M,
        "sha_match": sha256_file(YOLO_M) == SHA_M,
    }
    if not provenance["sha_match"]:
        return write_nogo("yolo11m sha mismatch", {"provenance": provenance})

    print("cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print(
            "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3), flush=True
        )

    # load audit frames once
    cap = cv2.VideoCapture(str(VIDEO))
    frames: dict[int, np.ndarray] = {}
    for fi in AUDIT_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames[fi] = fr
    cap.release()
    ordered = sorted(frames)

    results = {}
    for name, (weights, sha, cfg, use_temp) in CONFIGS.items():
        print("eval", name, flush=True)
        results[name] = evaluate_config(name, weights, sha, cfg, use_temp, frames, ordered)
        t = results[name]["temporal"]
        print(
            name,
            "flicker1",
            t["one_frame_disappearance"],
            "eff",
            t.get("effective_one_frame_gap"),
            "jitter",
            round(t["center_jitter"], 3),
            "small",
            results[name]["small_person_count"],
            "merged",
            results[name]["merged_person_candidate_count"],
            "dup",
            results[name]["duplicate_pair_count"],
            flush=True,
        )

    bakeoff = {
        "schema": "r1_f1_r2_bakeoff_v1",
        "diagnostic_not_accuracy": True,
        "model_provenance": provenance,
        "results": results,
        "written_at_utc": utc_now(),
    }
    selected = choose_winner(results)
    bakeoff["selected"] = selected
    print("selected", selected, flush=True)

    if selected is None:
        return write_nogo("no stronger candidate beat B on flicker/small/jitter jointly", bakeoff)

    # Pixel-grounded audit on selected (agent diagnostic)
    weights, sha, cfg, use_temp = CONFIGS[selected]
    adapter = load_adapter(weights, sha)
    stab = TemporalHumanStabilizer() if use_temp else None
    audit_rows = []
    # save a few overlays for Read inspection
    audit_dir = WORK / "audit_overlays"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for fi in ordered:
        props = run_frame(adapter, frames[fi], cfg, stab)
        vis = draw_temporal(frames[fi], props)
        if fi in {3, 60, 330, 500, 660, 780, 920, 1010}:
            cv2.imwrite(str(audit_dir / f"audit_f{fi:04d}.jpg"), vis)
        obs = [p for p in props if p.temporal_status == "observed"]
        audit_rows.append(
            {
                "frame_idx": fi,
                "detected_observed": len(obs),
                "temporally_carried": sum(1 for p in props if p.temporal_status == "carried"),
                "off_or_unknown": sum(
                    1 for p in props if p.eligibility in {"off_pitch_human", "unknown"}
                ),
                "merged_person_box": merged_person_candidates(
                    [
                        __import__(
                            "football_analytics.perception.human_tiled_detection",
                            fromlist=["HumanProposal"],
                        ).HumanProposal(
                            x1=p.x1,
                            y1=p.y1,
                            x2=p.x2,
                            y2=p.y2,
                            score=p.score,
                            eligibility=p.eligibility,  # type: ignore[arg-type]
                            source=p.source,
                        )
                        for p in obs
                    ]
                ),
                "duplicate_box": duplicate_pairs(
                    [
                        __import__(
                            "football_analytics.perception.human_tiled_detection",
                            fromlist=["HumanProposal"],
                        ).HumanProposal(
                            x1=p.x1,
                            y1=p.y1,
                            x2=p.x2,
                            y2=p.y2,
                            score=p.score,
                            eligibility=p.eligibility,  # type: ignore[arg-type]
                            source=p.source,
                        )
                        for p in obs
                    ]
                ),
            }
        )

    audit = {
        "schema": "AGENT_PIXEL_GROUNDED_DIAGNOSTIC — NOT INDEPENDENT GT",
        "n_frames": len(audit_rows),
        "rows": audit_rows,
        "sum_merged": sum(r["merged_person_box"] for r in audit_rows),
        "sum_duplicate": sum(r["duplicate_box"] for r in audit_rows),
        "mean_observed": float(np.mean([r["detected_observed"] for r in audit_rows])),
        "diagnostic_not_accuracy": True,
        "written_at_utc": utc_now(),
    }

    # hard fail if too many merges/dups on audit
    if audit["sum_merged"] > 8 or audit["sum_duplicate"] > 5:
        return write_nogo(
            f"pixel audit merges/dups too high merged={audit['sum_merged']} dup={audit['sum_duplicate']}",
            {**bakeoff, "audit": audit},
        )

    # compare vs B using effective gap when temporal
    b = results["B_yolo11n_hybrid"]
    s = results[selected]
    s_flicker = s["temporal"].get(
        "effective_one_frame_gap", s["temporal"]["one_frame_disappearance"]
    )
    b_flicker = b["temporal"]["one_frame_disappearance"]
    clearly_better = s_flicker < b_flicker * 0.85 or (
        s["temporal"]["center_jitter"] < b["temporal"]["center_jitter"] * 0.92
        and s["small_person_count"] >= b["small_person_count"] * 0.95
    )
    if not clearly_better:
        return write_nogo(
            "selected not clearly better than B on effective-flicker/small/jitter",
            {**bakeoff, "audit": audit},
        )

    build_package(selected, bakeoff, audit)
    # cleanup workspace media except small audit jpgs used for inspection — remove after packaging
    for p in WORK.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".avi", ".mp4"}:
            p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
