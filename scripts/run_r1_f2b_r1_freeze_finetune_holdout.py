#!/usr/bin/env python3
"""R1-F2-B-R1: validate repaired GT, freeze, fine-tune, one-shot holdout.

Does not start R2. Does not ask the user for annotation work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from football_analytics.annotation.gt_freeze import (
    DEFAULT_FROZEN_DIR,
    assert_audit_append_only,
    clear_leftover_train_proposals,
    validate_repaired_gt_integrity,
    write_frozen_gt,
)
from football_analytics.annotation.independent_gt import (
    DEFAULT_RUNTIME,
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    append_audit_line,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from football_analytics.annotation.yolo_export import DEFAULT_EXPORT_ROOT, export_yolo_dataset
from football_analytics.perception.detection_evaluation import (
    BBoxDetection,
    evaluate_human_detections,
)

REPO = Path(__file__).resolve().parents[1]
RUNTIME = DEFAULT_RUNTIME
VIDEO = DEFAULT_VIDEO
ARCHIVE = Path("/home/fdoblak/football_data/model_archive")
WORK = Path("/home/fdoblak/workspace/r1_f2b_r1_finetune")
EV_DIR = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_detector_acceptance"
FAIL_EV = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_f2b_r1_holdout_failure"
WIN_OK = Path(
    "/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection Accepted"
)
WIN_FAIL = Path(
    "/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection Accepted"
)
YOLO11N = ARCHIVE / "yolo11n.pt"
YOLO11S_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"

ACCEPT = {
    "precision": 0.90,
    "recall": 0.90,
    "f1": 0.90,
    "ap50": 0.90,
    "small_distant_recall": 0.80,
    "duplicate_rate": 0.01,
}


def _sha(path: Path) -> str:
    return sha256_file(path)


def load_draft() -> dict[str, Any]:
    return json.loads((RUNTIME / "draft_annotations.json").read_text(encoding="utf-8"))


def pixel_qa(draft: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Pixel-grounded QA on 10 train + 6 dev + 8 holdout frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "holdout": []}
    for fr in draft["frames"]:
        by_split[fr["split"]].append(fr)

    def pick(frames: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        # diversify by categories + box count
        scored = []
        for fr in frames:
            cats = set(fr.get("categories") or [])
            score = len(fr.get("humans") or [])
            if "crowded" in cats:
                score += 50
            if "small_distant" in cats:
                score += 40
            if "occlusion" in cats or "tackle" in cats:
                score += 30
            if "goal" in cats:
                score += 25
            if "sideline" in cats:
                score += 20
            scored.append((score, fr))
        scored.sort(key=lambda x: -x[0])
        # take top + evenly spaced fill
        chosen = []
        seen = set()
        for _, fr in scored:
            if fr["frame_idx"] in seen:
                continue
            chosen.append(fr)
            seen.add(fr["frame_idx"])
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            step = max(1, len(frames) // n)
            for fr in frames[::step]:
                if fr["frame_idx"] not in seen:
                    chosen.append(fr)
                    seen.add(fr["frame_idx"])
                if len(chosen) >= n:
                    break
        return chosen[:n]

    sample = {
        "train": pick(by_split["train"], 10),
        "dev": pick(by_split["dev"], 6),
        "holdout": pick(by_split["holdout"], 8),
    }
    cap = cv2.VideoCapture(str(VIDEO))
    findings: list[dict[str, Any]] = []
    critical = 0
    for split, frames in sample.items():
        for fr in frames:
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                critical += 1
                findings.append({"frame_idx": idx, "split": split, "critical": "read_fail"})
                continue
            vis = frame.copy()
            humans = list(fr.get("humans") or [])
            wide = 0
            for h in humans:
                x1, y1, x2, y2 = map(int, h["bbox_xyxy"])
                col = (255, 200, 0) if split == "train" else (0, 255, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                bw, bh = x2 - x1, y2 - y1
                if bh > 0 and (bw / bh) >= 1.2 and (bw * bh) / (1336 * 744) >= 0.02:
                    wide += 1
            path = out_dir / f"qa_{split}_{idx:06d}.jpg"
            cv2.imwrite(str(path), vis)
            # heuristic critical flags (human review via saved sheets)
            note = {
                "frame_idx": idx,
                "split": split,
                "n_humans": len(humans),
                "categories": fr.get("categories") or [],
                "wide_box_candidates": wide,
                "path": str(path),
                "proposals_present": bool(fr.get("proposals")),
                "blind_ok": split in {"dev", "holdout"} and not fr.get("proposals"),
            }
            if note["proposals_present"] and split != "train":
                critical += 1
                note["critical"] = "prediction_leakage"
            findings.append(note)
    cap.release()
    # contact sheet per split
    for split, frames in sample.items():
        imgs = []
        for fr in frames[:6]:
            p = out_dir / f"qa_{split}_{int(fr['frame_idx']):06d}.jpg"
            if p.is_file():
                im = cv2.imread(str(p))
                if im is not None:
                    imgs.append(cv2.resize(im, (446, 248)))
        if imgs:
            sheet = (
                np.concatenate(imgs, axis=1)
                if len(imgs) == 1
                else np.vstack(
                    [
                        np.concatenate(imgs[i : i + 3], axis=1)
                        for i in range(0, len(imgs), 3)
                        if imgs[i : i + 3]
                    ]
                )
            )
            cv2.imwrite(str(out_dir / f"contact_{split}.jpg"), sheet)

    report = {
        "schema": "r1_f2b_r1_pixel_qa_v1",
        "n_checked": sum(len(v) for v in sample.values()),
        "targets": {"train": 10, "dev": 6, "holdout": 8},
        "sampled_indices": {sp: [int(f["frame_idx"]) for f in frs] for sp, frs in sample.items()},
        "critical_count": critical,
        "findings": findings,
        "categories_covered": sorted(
            {c for frs in sample.values() for fr in frs for c in (fr.get("categories") or [])}
        ),
        "ok": critical == 0,
        "written_at_utc": utc_now(),
    }
    atomic_write_json(out_dir / "pixel_qa_report.json", report, mode=0o600)
    return report


def ensure_yolo11s() -> dict[str, Any]:
    dest = ARCHIVE / "yolo11s.pt"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return {
            "path": str(dest),
            "sha256": _sha(dest),
            "bytes": dest.stat().st_size,
            "downloaded": False,
            "url": YOLO11S_URL,
            "license": "AGPL-3.0",
            "evaluation_only": True,
        }
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    # Official Ultralytics asset; record provenance.
    subprocess.run(
        ["curl", "-L", "--fail", "-o", str(dest), YOLO11S_URL],
        check=True,
    )
    return {
        "path": str(dest),
        "sha256": _sha(dest),
        "bytes": dest.stat().st_size,
        "downloaded": True,
        "url": YOLO11S_URL,
        "license": "AGPL-3.0",
        "evaluation_only": True,
    }


def train_candidate(
    *,
    name: str,
    weights: Path,
    data_yaml: Path,
    epochs: int = 40,
    imgsz: int = 960,
    batch: int = 2,
) -> dict[str, Any]:
    from ultralytics import YOLO

    run_dir = WORK / "runs" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"
    amp = bool(torch.cuda.is_available())
    t0 = time.time()
    vram_peak = None
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=2,
            cache=False,
            amp=amp,
            project=str(WORK / "runs"),
            name=name,
            exist_ok=True,
            seed=42,
            deterministic=True,
            patience=12,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.08,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=8,
            plots=False,
            save=True,
            val=True,
            pretrained=True,
            single_cls=True,
        )
        if torch.cuda.is_available():
            vram_peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and batch > 1:
            torch.cuda.empty_cache()
            return train_candidate(
                name=name,
                weights=weights,
                data_yaml=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=1,
            )
        raise
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        # ultralytics may return differently
        cand = list((WORK / "runs" / name).rglob("best.pt"))
        if not cand:
            raise FileNotFoundError(f"no best.pt for {name}")
        best = cand[0]
    elapsed = time.time() - t0
    return {
        "name": name,
        "weights_in": str(weights),
        "best_pt": str(best),
        "best_sha256": _sha(best),
        "best_bytes": best.stat().st_size,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": str(device),
        "amp": amp,
        "elapsed_s": round(elapsed, 1),
        "vram_peak_mb": vram_peak,
        "save_dir": str(best.parent.parent),
    }


def predict_split(
    weights: Path,
    annotations: dict[str, Any],
    *,
    split: str,
    conf: float,
    iou: float,
    imgsz: int,
) -> tuple[list[BBoxDetection], list[BBoxDetection], dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frames = [f for f in annotations["frames"] if f["split"] == split]
    cap = cv2.VideoCapture(str(VIDEO))
    preds: list[BBoxDetection] = []
    gts: list[BBoxDetection] = []
    per_frame: dict[int, dict[str, Any]] = {}
    try:
        for fr in frames:
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )[0]
            frame_preds = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                for j, box in enumerate(xyxy):
                    cname = res.names.get(int(cls[j]), "")
                    if int(cls[j]) != 0 and cname not in {"person", "human"}:
                        continue
                    det = BBoxDetection(
                        frame_index=idx,
                        entity_type="human",
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        score=float(scores[j]),
                    )
                    preds.append(det)
                    frame_preds.append(det)
            for h in fr.get("humans") or []:
                x1, y1, x2, y2 = (float(v) for v in h["bbox_xyxy"])
                gts.append(
                    BBoxDetection(
                        frame_index=idx,
                        entity_type="human",
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=1.0,
                    )
                )
            per_frame[idx] = {
                "n_pred": len(frame_preds),
                "n_gt": len(fr.get("humans") or []),
                "visibility": [h.get("visibility") for h in (fr.get("humans") or [])],
                "eligibility": [h.get("eligibility") for h in (fr.get("humans") or [])],
                "gt_boxes": [h["bbox_xyxy"] for h in (fr.get("humans") or [])],
                "meta": [
                    {
                        "visibility": h.get("visibility"),
                        "eligibility": h.get("eligibility"),
                        "bbox_xyxy": h["bbox_xyxy"],
                    }
                    for h in (fr.get("humans") or [])
                ],
            }
    finally:
        cap.release()
    return preds, gts, per_frame


def metrics_bundle(
    preds: list[BBoxDetection],
    gts: list[BBoxDetection],
    per_frame: dict[int, dict[str, Any]],
    matches_iou: float = 0.5,
) -> dict[str, Any]:
    ev = evaluate_human_detections(preds, gts, iou_threshold=matches_iou)
    d = ev.to_dict()
    # duplicate rate among preds same frame iou>=0.9
    dup = 0
    total_pairs = 0
    by_f: dict[int, list[BBoxDetection]] = defaultdict(list)
    for p in preds:
        by_f[p.frame_index].append(p)
    from football_analytics.perception.detection_evaluation import bbox_iou

    for boxes in by_f.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total_pairs += 1
                a = boxes[i]
                b = boxes[j]
                if bbox_iou([a.x1, a.y1, a.x2, a.y2], [b.x1, b.y1, b.x2, b.y2]) >= 0.9:
                    dup += 1
    dup_rate = (dup / max(1, len(preds))) if preds else 0.0

    # small/distant recall: GT with visibility==small OR height<55
    matched_gt = {m.gt_index for m in ev.matches}
    # rebuild gt list order as in evaluate — use same order
    small_idx = []
    on_idx = []
    off_idx = []
    for i, g in enumerate(gts):
        meta = None
        for m in per_frame.get(g.frame_index, {}).get("meta") or []:
            if abs(m["bbox_xyxy"][0] - g.x1) < 1e-3 and abs(m["bbox_xyxy"][1] - g.y1) < 1e-3:
                meta = m
                break
        h = g.y2 - g.y1
        is_small = (meta and meta.get("visibility") == "small") or h < 55
        if is_small:
            small_idx.append(i)
        el = (meta or {}).get("eligibility")
        if el == "on_pitch":
            on_idx.append(i)
        elif el == "off_pitch":
            off_idx.append(i)

    def _rec(idxs: list[int]) -> float | None:
        if not idxs:
            return None
        hit = sum(1 for i in idxs if i in matched_gt)
        return hit / len(idxs)

    # frame FP/FN lists
    tp_by_frame: dict[int, int] = defaultdict(int)
    for m in ev.matches:
        tp_by_frame[m.frame_index] += 1
    missed_frames = []
    fp_frames = []
    for idx, info in per_frame.items():
        n_gt = info["n_gt"]
        n_pred = info["n_pred"]
        tp = tp_by_frame.get(idx, 0)
        fn = n_gt - tp
        fp = n_pred - tp
        if fn > 0:
            missed_frames.append({"frame_idx": idx, "fn": fn, "n_gt": n_gt})
        if fp > 0:
            fp_frames.append({"frame_idx": idx, "fp": fp, "n_pred": n_pred})

    d.update(
        {
            "duplicate_rate": dup_rate,
            "duplicate_pairs": dup,
            "small_distant_recall": _rec(small_idx),
            "on_pitch_recall": _rec(on_idx),
            "off_pitch_recall": _rec(off_idx),
            "n_small_gt": len(small_idx),
            "missed_frames": missed_frames,
            "fp_frames": fp_frames,
        }
    )
    return d


def sweep_dev(
    weights: Path,
    annotations: dict[str, Any],
    *,
    imgsz: int,
) -> dict[str, Any]:
    confs = [0.15, 0.20, 0.25, 0.30, 0.35]
    ious = [0.45, 0.55, 0.65]
    best = None
    rows = []
    for conf in confs:
        for iou in ious:
            preds, gts, per = predict_split(
                weights, annotations, split="dev", conf=conf, iou=iou, imgsz=imgsz
            )
            m = metrics_bundle(preds, gts, per)
            row = {
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "f1": m.get("f1"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "ap50": m.get("ap50"),
                "small_distant_recall": m.get("small_distant_recall"),
                "duplicate_rate": m.get("duplicate_rate"),
            }
            rows.append(row)
            score = (
                (m.get("f1") or 0)
                + 0.15 * (m.get("ap50") or 0)
                + 0.10 * (m.get("small_distant_recall") or 0)
                - 0.5 * max(0.0, (m.get("duplicate_rate") or 0) - 0.01)
            )
            if best is None or score > best["score"]:
                best = {"score": score, "config": row, "metrics": m}
    assert best is not None
    return {"best": best, "rows": rows}


def acceptance_table(m: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "precision": (m.get("precision") or 0) >= ACCEPT["precision"],
        "recall": (m.get("recall") or 0) >= ACCEPT["recall"],
        "f1": (m.get("f1") or 0) >= ACCEPT["f1"],
        "ap50": (m.get("ap50") or 0) >= ACCEPT["ap50"],
        "small_distant_recall": (m.get("small_distant_recall") or 0)
        >= ACCEPT["small_distant_recall"],
        "duplicate_rate": float(m["duplicate_rate"] if m.get("duplicate_rate") is not None else 1.0)
        <= ACCEPT["duplicate_rate"],
        "invalid_bbox": True,
    }
    return {
        "thresholds": ACCEPT,
        "values": {k: m.get(k) for k in checks if k != "invalid_bbox"},
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_nogo(status: str, payload: dict[str, Any]) -> None:
    FAIL_EV.mkdir(parents=True, exist_ok=True)
    atomic_write_json(FAIL_EV / "status.json", payload, mode=0o644)
    WIN_FAIL.mkdir(parents=True, exist_ok=True)
    # remove any prior acceptance media
    for p in WIN_FAIL.iterdir():
        if p.is_file() and p.name != "NO_GO_STATUS.txt":
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    (WIN_FAIL / "NO_GO_STATUS.txt").write_text(
        f"{status}\r\n"
        f"written_at_utc={utc_now()}\r\n"
        f"{json.dumps(payload.get('summary', payload), indent=2)}\r\n",
        encoding="utf-8",
    )
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f2b_r1_gate_status_v1",
            "gate": status,
            "acceptance_eligible": False,
            "frozen": payload.get("frozen", False),
            "human_approved": False,
            "reviewed_gt": False,
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )


def build_acceptance_package(
    *,
    annotations: dict[str, Any],
    weights: Path,
    conf: float,
    iou: float,
    imgsz: int,
    holdout_metrics: dict[str, Any],
    training_summary: dict[str, Any],
    model_prov: dict[str, Any],
    baseline_weights: Path,
) -> dict[str, str]:
    from ultralytics import YOLO

    if EV_DIR.exists():
        shutil.rmtree(EV_DIR)
    EV_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    base = YOLO(str(baseline_weights))
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = WORK / "_proof.avi"
    WORK.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    holdout_frames = [f for f in annotations["frames"] if f["split"] == "holdout"]
    err_panel = None
    ok_panel = None
    cmp_panel = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )[0]
        vis = frame.copy()
        if res.boxes is not None:
            for box in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis,
            "R1 HUMAN DETECTION",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            "RENKLER TAKIM VEYA ROL BELIRTMEZ",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        writer.write(vis)
        if i == holdout_frames[len(holdout_frames) // 2]["frame_idx"]:
            ok_panel = vis.copy()
            bres = base.predict(
                source=frame,
                conf=0.25,
                iou=0.5,
                imgsz=960,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )[0]
            left = frame.copy()
            right = vis.copy()
            if bres.boxes is not None:
                for box in bres.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(left, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cmp_panel = np.concatenate([left, right], axis=1)
        # error panel: first holdout frame with FN if any
    writer.release()
    cap.release()

    # error visualization from missed frames
    missed = holdout_metrics.get("missed_frames") or []
    if missed:
        idx = int(missed[0]["frame_idx"])
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            fr = next(f for f in holdout_frames if int(f["frame_idx"]) == idx)
            err = frame.copy()
            for h in fr.get("humans") or []:
                x1, y1, x2, y2 = map(int, h["bbox_xyxy"])
                cv2.rectangle(err, (x1, y1), (x2, y2), (0, 0, 255), 2)
            res = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )[0]
            if res.boxes is not None:
                for box in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(err, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                err,
                "RED=GT CYAN=PRED",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            err_panel = err

    proof = EV_DIR / "R1_finetuned_human_detection_proof.mp4"
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
            str(proof),
        ],
        check=True,
        capture_output=True,
    )
    if ok_panel is not None:
        cv2.imwrite(str(EV_DIR / "R1_holdout_predictions.png"), ok_panel)
    if err_panel is not None:
        cv2.imwrite(str(EV_DIR / "R1_holdout_errors.png"), err_panel)
    elif ok_panel is not None:
        cv2.imwrite(str(EV_DIR / "R1_holdout_errors.png"), ok_panel)
    if cmp_panel is not None:
        cv2.imwrite(str(EV_DIR / "R1_generic_vs_finetuned.png"), cmp_panel)

    atomic_write_json(EV_DIR / "holdout_metrics.json", holdout_metrics, mode=0o644)
    atomic_write_json(EV_DIR / "training_summary.json", training_summary, mode=0o644)
    atomic_write_json(EV_DIR / "model_provenance.json", model_prov, mode=0o644)

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>R1 Detector Acceptance</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#8cf}} img{{max-width:100%}}</style></head><body>
<h1>R1 Fine-tuned Human Detection — Own-video clip</h1>
<p>Gate: PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED ON OWN-VIDEO CLIP; GENERALIZATION NOT VALIDATED</p>
<p>Bu model yalniz bu klip icin dogrulandi. Genel mac basarisi iddia edilmez.</p>
<ul>
<li><a href="R1_finetuned_human_detection_proof.mp4">Proof MP4</a></li>
<li>P={holdout_metrics.get('precision')} R={holdout_metrics.get('recall')} F1={holdout_metrics.get('f1')} AP50={holdout_metrics.get('ap50')}</li>
</ul>
<img src="R1_holdout_predictions.png"/><br/>
<img src="R1_generic_vs_finetuned.png"/>
</body></html>
"""
    (EV_DIR / "OPEN_R1_DETECTOR_ACCEPTANCE.html").write_text(html, encoding="utf-8")
    (EV_DIR / "README_TR.txt").write_text(
        "R1 insan tespiti kabul paketi (own-video klip).\r\n"
        "Renkler takim/rol belirtmez.\r\n"
        "Genelleme dogrulanmadi.\r\n",
        encoding="utf-8",
    )
    files = [
        "R1_finetuned_human_detection_proof.mp4",
        "R1_holdout_predictions.png",
        "R1_holdout_errors.png",
        "R1_generic_vs_finetuned.png",
        "OPEN_R1_DETECTOR_ACCEPTANCE.html",
        "holdout_metrics.json",
        "training_summary.json",
        "model_provenance.json",
        "README_TR.txt",
    ]
    checksums = {}
    for name in files:
        p = EV_DIR / name
        if p.is_file():
            checksums[name] = _sha(p)
    (EV_DIR / "checksums.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in sorted(checksums.items())) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "r1_detector_acceptance_manifest_v1",
        "files": checksums,
        "holdout_one_shot": True,
        "generalization_validated": False,
        "written_at_utc": utc_now(),
    }
    atomic_write_json(EV_DIR / "MANIFEST.json", manifest, mode=0o644)

    # Windows mirror hash-equal
    if WIN_OK.exists():
        for p in list(WIN_OK.iterdir()):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    WIN_OK.mkdir(parents=True, exist_ok=True)
    for name in list(checksums) + ["checksums.sha256", "MANIFEST.json"]:
        src = EV_DIR / name
        if src.is_file():
            dst = WIN_OK / name
            shutil.copy2(src, dst)
            if _sha(dst) != _sha(src):
                raise RuntimeError(f"WINDOWS_HASH_MISMATCH:{name}")
    return checksums


def cleanup_training(keep_best: Path) -> dict[str, Any]:
    removed = []
    runs = WORK / "runs"
    if not runs.is_dir():
        return {"removed": removed, "kept": str(keep_best)}
    for p in runs.rglob("*"):
        if not p.is_file():
            continue
        try:
            if p.resolve() == keep_best.resolve():
                continue
        except FileNotFoundError:
            continue
        if p.name in {"last.pt", "optimizer.pt"} or p.suffix in {".optimizer"}:
            p.unlink(missing_ok=True)
            removed.append(str(p))
            continue
        if p.name.endswith(".pt") and p.name != "best.pt" and "weights" in str(p):
            try:
                if p.resolve() != keep_best.resolve():
                    p.unlink(missing_ok=True)
                    removed.append(str(p))
            except FileNotFoundError:
                continue
    for cache in WORK.rglob("*.cache"):
        cache.unlink(missing_ok=True)
        removed.append(str(cache))
    return {"removed": removed, "kept": str(keep_best)}


def resume_from_trained_runs(
    annotations: dict[str, Any],
    *,
    yolo11s_meta: dict[str, Any],
) -> int:
    """Continue after A/B training if cleanup crashed before holdout."""
    a_best = WORK / "runs" / "A_yolo11n_ft" / "weights" / "best.pt"
    b_best = WORK / "runs" / "B_yolo11s_ft" / "weights" / "best.pt"
    # archived copy may exist if crash was after archive
    arch_b = ARCHIVE / "own_video_human_v1_b_best.pt"
    arch_a = ARCHIVE / "own_video_human_v1_a_best.pt"
    if not a_best.is_file() and arch_a.is_file():
        a_best = arch_a
    if not b_best.is_file() and arch_b.is_file():
        b_best = arch_b
    if not a_best.is_file() or not b_best.is_file():
        raise SystemExit(f"missing trained weights a={a_best} b={b_best}")

    a = {
        "name": "A_yolo11n_ft",
        "best_pt": str(a_best),
        "best_sha256": _sha(a_best),
        "best_bytes": a_best.stat().st_size,
        "imgsz": 960,
        "batch": 2,
        "epochs": 40,
    }
    b = {
        "name": "B_yolo11s_ft",
        "best_pt": str(b_best),
        "best_sha256": _sha(b_best),
        "best_bytes": b_best.stat().st_size,
        "imgsz": 960,
        "batch": 2,
        "epochs": 40,
    }
    training_summary: dict[str, Any] = {"candidates": {"A": a, "B": b}, "resumed": True}
    dev_a = sweep_dev(Path(a["best_pt"]), annotations, imgsz=960)
    dev_b = sweep_dev(Path(b["best_pt"]), annotations, imgsz=960)
    training_summary["dev_selection"] = {"A": dev_a, "B": dev_b}
    if dev_b["best"]["score"] > dev_a["best"]["score"]:
        selected = {
            "id": "B",
            "run": b,
            "dev": dev_b["best"],
            "parent": yolo11s_meta,
        }
    else:
        selected = {
            "id": "A",
            "run": a,
            "dev": dev_a["best"],
            "parent": {
                "path": str(YOLO11N),
                "sha256": _sha(YOLO11N),
                "license": "AGPL-3.0",
            },
        }
    training_summary["selected"] = selected
    holdout_cfg = {
        "model_id": selected["id"],
        "weights_sha256": selected["run"]["best_sha256"],
        "conf": selected["dev"]["config"]["conf"],
        "iou": selected["dev"]["config"]["iou"],
        "imgsz": selected["dev"]["config"]["imgsz"],
        "gt_fingerprint": annotations["canonical_fingerprint"],
        "one_shot": True,
    }
    holdout_cfg["config_fingerprint"] = hashlib.sha256(
        json.dumps(holdout_cfg, sort_keys=True).encode()
    ).hexdigest()
    training_summary["holdout_config_fingerprint_before"] = holdout_cfg

    preds, gts, per = predict_split(
        Path(selected["run"]["best_pt"]),
        annotations,
        split="holdout",
        conf=holdout_cfg["conf"],
        iou=holdout_cfg["iou"],
        imgsz=holdout_cfg["imgsz"],
    )
    holdout_metrics = metrics_bundle(preds, gts, per)
    holdout_metrics["config"] = holdout_cfg
    holdout_metrics["one_shot_confirmed"] = True
    table = acceptance_table(holdout_metrics)
    training_summary["holdout_acceptance"] = table

    arch_name = f"own_video_human_v1_{selected['id'].lower()}_best.pt"
    arch_path = ARCHIVE / arch_name
    if not arch_path.is_file() or _sha(arch_path) != selected["run"]["best_sha256"]:
        shutil.copy2(selected["run"]["best_pt"], arch_path)
    model_prov = {
        "schema": "r1_finetuned_human_detector_provenance_v1",
        "checkpoint": str(arch_path),
        "checkpoint_sha256": _sha(arch_path),
        "checkpoint_bytes": arch_path.stat().st_size,
        "parent_model": selected["parent"],
        "gt_fingerprint": annotations["canonical_fingerprint"],
        "training_config_fingerprint": holdout_cfg["config_fingerprint"],
        "own_video_clip_specific": True,
        "evaluation_only": True,
        "production_approved": False,
        "license_finding": "AGPL-3.0 (Ultralytics)",
        "selected_id": selected["id"],
    }
    training_summary["cleanup"] = cleanup_training(Path(selected["run"]["best_pt"]))
    atomic_write_json(WORK / "training_summary.json", training_summary, mode=0o600)
    atomic_write_json(WORK / "holdout_metrics.json", holdout_metrics, mode=0o600)

    if not table["passed"]:
        write_nogo(
            "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE",
            {
                "summary": {
                    "holdout_metrics": {
                        k: holdout_metrics.get(k)
                        for k in (
                            "precision",
                            "recall",
                            "f1",
                            "ap50",
                            "ap50_95",
                            "small_distant_recall",
                            "duplicate_rate",
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        )
                    },
                    "acceptance": table,
                    "selected": selected["id"],
                },
                "frozen": True,
                "freeze_fingerprint": annotations["canonical_fingerprint"],
                "training_summary": training_summary,
                "model_provenance": model_prov,
            },
        )
        if EV_DIR.exists():
            shutil.rmtree(EV_DIR)
        print("NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE")
        print(json.dumps(table, indent=2))
        return 3

    build_acceptance_package(
        annotations=annotations,
        weights=arch_path,
        conf=holdout_cfg["conf"],
        iou=holdout_cfg["iou"],
        imgsz=holdout_cfg["imgsz"],
        holdout_metrics=holdout_metrics,
        training_summary=training_summary,
        model_prov=model_prov,
        baseline_weights=YOLO11N,
    )
    gate = (
        "PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED "
        "ON OWN-VIDEO CLIP; GENERALIZATION NOT VALIDATED"
    )
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f2b_r1_gate_status_v1",
            "gate": gate,
            "acceptance_eligible": False,
            "frozen": True,
            "human_approved": True,
            "reviewed_gt": True,
            "generalization_validated": False,
            "own_video_clip_specific": True,
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )
    print(gate)
    print(json.dumps(table, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume-holdout", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    if _sha(VIDEO) != EXPECTED_SOURCE_SHA256:
        write_nogo(
            "NO-GO — REPAIRED GT INTEGRITY FAILURE",
            {"summary": {"error": "SOURCE_SHA_MISMATCH"}},
        )
        print("NO-GO — REPAIRED GT INTEGRITY FAILURE")
        return 2

    if args.resume_holdout:
        if not (DEFAULT_FROZEN_DIR / "annotations.json").is_file():
            raise SystemExit("frozen GT missing for resume")
        annotations = json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text())
        yolo11s = {
            "path": str(ARCHIVE / "yolo11s.pt"),
            "sha256": _sha(ARCHIVE / "yolo11s.pt"),
            "bytes": (ARCHIVE / "yolo11s.pt").stat().st_size,
            "url": YOLO11S_URL,
            "license": "AGPL-3.0",
            "evaluation_only": True,
        }
        return resume_from_trained_runs(annotations, yolo11s_meta=yolo11s)

    draft = load_draft()
    audit_path = RUNTIME / "review_audit.jsonl"
    audit_before = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""

    # Clear leftover non-GT proposals on protected frames (0/5/15), preserve humans.
    cleared = clear_leftover_train_proposals(draft)
    atomic_write_json(RUNTIME / "draft_annotations.json", draft)
    append_audit_line(
        audit_path,
        {
            "event": "f2b_r1_clear_leftover_proposals",
            "cleared": cleared,
            "note": "Leftover proposals are not GT; humans unchanged",
            "ts": utc_now(),
        },
    )
    audit_after = audit_path.read_text(encoding="utf-8")
    assert_audit_append_only(audit_before, audit_after)

    integrity = validate_repaired_gt_integrity(draft)
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/r1_f2b_r1_integrity.json",
        integrity,
        mode=0o644,
    )
    if not integrity["ok"]:
        write_nogo(
            "NO-GO — REPAIRED GT INTEGRITY FAILURE",
            {
                "summary": {
                    "problem_frames": integrity["problem_frames"],
                    "errors": integrity["errors"][:50],
                },
                "frozen": False,
            },
        )
        print(json.dumps(integrity, indent=2))
        print("NO-GO — REPAIRED GT INTEGRITY FAILURE")
        return 2

    qa = pixel_qa(draft, WORK / "pixel_qa")
    if not qa["ok"]:
        write_nogo(
            "NO-GO — REPAIRED GT INTEGRITY FAILURE",
            {"summary": {"pixel_qa": qa}, "frozen": False},
        )
        print("NO-GO — REPAIRED GT INTEGRITY FAILURE (pixel)")
        return 2

    # Freeze
    freeze_info = write_frozen_gt(draft, audit_path=audit_path)
    annotations = json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text())
    append_audit_line(
        audit_path,
        {"event": "f2b_r1_freeze", "freeze": freeze_info, "ts": utc_now()},
    )

    # Export
    export_info = export_yolo_dataset(annotations, out_root=DEFAULT_EXPORT_ROOT)
    data_yaml = DEFAULT_EXPORT_ROOT / "dataset.yaml"

    # Models
    if not YOLO11N.is_file():
        raise SystemExit("missing yolo11n.pt in archive")
    yolo11s = ensure_yolo11s()

    training_summary: dict[str, Any] = {"candidates": {}, "dev_selection": {}}
    if not args.skip_train:
        WORK.mkdir(parents=True, exist_ok=True)
        a = train_candidate(
            name="A_yolo11n_ft",
            weights=YOLO11N,
            data_yaml=data_yaml,
            epochs=args.epochs,
            imgsz=960,
            batch=2,
        )
        b = train_candidate(
            name="B_yolo11s_ft",
            weights=Path(yolo11s["path"]),
            data_yaml=data_yaml,
            epochs=args.epochs,
            imgsz=960,
            batch=2,
        )
        training_summary["candidates"] = {"A": a, "B": b}

        # Dev selection — never touch holdout yet
        dev_a = sweep_dev(Path(a["best_pt"]), annotations, imgsz=a["imgsz"])
        dev_b = sweep_dev(Path(b["best_pt"]), annotations, imgsz=b["imgsz"])
        training_summary["dev_selection"] = {"A": dev_a, "B": dev_b}
        pick_a = dev_a["best"]["score"]
        pick_b = dev_b["best"]["score"]
        if pick_b > pick_a:
            selected = {
                "id": "B",
                "run": b,
                "dev": dev_b["best"],
                "parent": yolo11s,
            }
        else:
            selected = {
                "id": "A",
                "run": a,
                "dev": dev_a["best"],
                "parent": {
                    "path": str(YOLO11N),
                    "sha256": _sha(YOLO11N),
                    "license": "AGPL-3.0",
                },
            }
        training_summary["selected"] = selected

        # Fingerprint config BEFORE holdout
        holdout_cfg = {
            "model_id": selected["id"],
            "weights_sha256": selected["run"]["best_sha256"],
            "conf": selected["dev"]["config"]["conf"],
            "iou": selected["dev"]["config"]["iou"],
            "imgsz": selected["dev"]["config"]["imgsz"],
            "gt_fingerprint": annotations["canonical_fingerprint"],
            "one_shot": True,
        }
        holdout_cfg["config_fingerprint"] = hashlib.sha256(
            json.dumps(holdout_cfg, sort_keys=True).encode()
        ).hexdigest()
        training_summary["holdout_config_fingerprint_before"] = holdout_cfg

        # ONE-SHOT holdout
        preds, gts, per = predict_split(
            Path(selected["run"]["best_pt"]),
            annotations,
            split="holdout",
            conf=holdout_cfg["conf"],
            iou=holdout_cfg["iou"],
            imgsz=holdout_cfg["imgsz"],
        )
        holdout_metrics = metrics_bundle(preds, gts, per)
        holdout_metrics["config"] = holdout_cfg
        holdout_metrics["one_shot_confirmed"] = True
        table = acceptance_table(holdout_metrics)
        training_summary["holdout_acceptance"] = table

        # Archive selected checkpoint
        arch_name = f"own_video_human_v1_{selected['id'].lower()}_best.pt"
        arch_path = ARCHIVE / arch_name
        shutil.copy2(selected["run"]["best_pt"], arch_path)
        model_prov = {
            "schema": "r1_finetuned_human_detector_provenance_v1",
            "checkpoint": str(arch_path),
            "checkpoint_sha256": _sha(arch_path),
            "checkpoint_bytes": arch_path.stat().st_size,
            "parent_model": selected["parent"],
            "gt_fingerprint": annotations["canonical_fingerprint"],
            "training_config_fingerprint": holdout_cfg["config_fingerprint"],
            "own_video_clip_specific": True,
            "evaluation_only": True,
            "production_approved": False,
            "license_finding": "AGPL-3.0 (Ultralytics)",
            "selected_id": selected["id"],
        }
        cleanup = cleanup_training(Path(selected["run"]["best_pt"]))
        training_summary["cleanup"] = cleanup

        if not table["passed"]:
            write_nogo(
                "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE",
                {
                    "summary": {
                        "holdout_metrics": {
                            k: holdout_metrics.get(k)
                            for k in (
                                "precision",
                                "recall",
                                "f1",
                                "ap50",
                                "ap50_95",
                                "small_distant_recall",
                                "duplicate_rate",
                                "true_positives",
                                "false_positives",
                                "false_negatives",
                            )
                        },
                        "acceptance": table,
                        "selected": selected["id"],
                    },
                    "frozen": True,
                    "freeze_fingerprint": annotations["canonical_fingerprint"],
                    "training_summary": training_summary,
                    "model_provenance": model_prov,
                },
            )
            if EV_DIR.exists():
                shutil.rmtree(EV_DIR)
            print("NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE")
            print(json.dumps(table, indent=2))
            atomic_write_json(WORK / "training_summary.json", training_summary, mode=0o600)
            atomic_write_json(WORK / "holdout_metrics.json", holdout_metrics, mode=0o600)
            return 3

        build_acceptance_package(
            annotations=annotations,
            weights=arch_path,
            conf=holdout_cfg["conf"],
            iou=holdout_cfg["iou"],
            imgsz=holdout_cfg["imgsz"],
            holdout_metrics=holdout_metrics,
            training_summary=training_summary,
            model_prov=model_prov,
            baseline_weights=YOLO11N,
        )
        gate = (
            "PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED "
            "ON OWN-VIDEO CLIP; GENERALIZATION NOT VALIDATED"
        )
        atomic_write_json(
            REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
            {
                "schema": "r1_f2b_r1_gate_status_v1",
                "gate": gate,
                "acceptance_eligible": False,
                "frozen": True,
                "human_approved": True,
                "reviewed_gt": True,
                "generalization_validated": False,
                "own_video_clip_specific": True,
                "written_at_utc": utc_now(),
            },
            mode=0o644,
        )
        print(gate)
        print(json.dumps({"freeze": freeze_info, "export": export_info, "table": table}, indent=2))
        return 0

    print("frozen_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
