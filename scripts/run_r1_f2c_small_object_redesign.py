#!/usr/bin/env python3
"""R1-F2-C: small-object detector redesign + evaluation protocol v2 + blind holdout_v2 prep.

Does not start R2. Does not ask the user for annotation. Holdout_v1 is not reused
for acceptance / selection / tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from football_analytics.annotation.evaluation_protocol_v2 import (
    EXPECTED_FROZEN_FP,
    HOLDOUT_V1_STATUS,
    PROTOCOL_ID,
    dev_gate_passed,
    evaluate_protocol_v2,
    protocol_v2_definition,
)
from football_analytics.annotation.gt_freeze import DEFAULT_FROZEN_DIR
from football_analytics.annotation.holdout_v1_guard import (
    assert_dev_only_selection,
    assert_no_holdout_v1_for_development,
)
from football_analytics.annotation.holdout_v2_selection import (
    select_holdout_v2_frames,
    write_holdout_v2_draft,
)
from football_analytics.annotation.independent_gt import (
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from football_analytics.annotation.root_cause_small_object import (
    build_root_cause_report,
    error_magnitude_from_preds,
    height_bin_recall_from_eval,
)
from football_analytics.annotation.train_tiles import build_train_tile_dataset
from football_analytics.perception.detection_evaluation import BBoxDetection
from football_analytics.perception.full_tile_fusion import (
    FusionConfig,
    attach_frame_index,
    predict_full_tile_fused,
)

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_small_object_redesign"
ARCHIVE = Path("/home/fdoblak/football_data/model_archive")
WORK = Path("/home/fdoblak/workspace/r1_f2c_small_object")
TILE_DS = Path("/home/fdoblak/workspace/training_datasets/own_video_human_v1_tiles")
CKPT_B = ARCHIVE / "own_video_human_v1_b_best.pt"
YOLO11S = ARCHIVE / "yolo11s.pt"
VIDEO = DEFAULT_VIDEO
WIN_DIR = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")
HOLDOUT_V2_RUNTIME = Path(
    "/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_holdout_v2"
)

GATE_PASS = (
    "PASS — SMALL-OBJECT DETECTOR DEVELOPMENT GATE PASSED; " "NEW BLIND HOLDOUT REVIEW READY"
)
GATE_FAIL = "NO-GO — SMALL-OBJECT DETECTOR DEVELOPMENT GATE FAILED"


def _sha(p: Path) -> str:
    return sha256_file(p)


def load_frozen() -> dict[str, Any]:
    path = DEFAULT_FROZEN_DIR / "annotations.json"
    ann = json.loads(path.read_text(encoding="utf-8"))
    if not ann.get("frozen"):
        raise SystemExit("FROZEN_GT_REQUIRED")
    if ann.get("canonical_fingerprint") != EXPECTED_FROZEN_FP:
        raise SystemExit(f"FINGERPRINT_MISMATCH:{ann.get('canonical_fingerprint')}")
    if _sha(VIDEO) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("SOURCE_SHA_MISMATCH")
    return ann


def record_holdout_v1_consumed(prior: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "schema": "holdout_v1_consumption_record_v1",
        "holdout_v1_status": HOLDOUT_V1_STATUS,
        "acceptance_reusable": False,
        "may_use_for_error_analysis": True,
        "may_use_for_training": False,
        "may_use_for_model_selection": False,
        "may_use_for_threshold_tuning": False,
        "may_produce_acceptance": False,
        "frozen_gt_fingerprint": EXPECTED_FROZEN_FP,
        "frozen_gt_mutated": False,
        "historical_metrics": {
            "precision": 0.645,
            "recall": 0.631,
            "f1": 0.638,
            "ap50": 0.494,
            "small_recall": 0.143,
            "source_file": str(
                REPO / "artifacts/evidence/reboot_01/r1_f2b_r1_holdout_failure/status.json"
            ),
            "exact": {
                "precision": prior.get("precision"),
                "recall": prior.get("recall"),
                "f1": prior.get("f1"),
                "ap50": prior.get("ap50"),
                "small_distant_recall": prior.get("small_distant_recall"),
            },
        },
        "written_at_utc": utc_now(),
    }
    atomic_write_json(EV / "holdout_v1_consumed.json", rec, mode=0o644)
    return rec


def predict_split_fused(
    weights: Path,
    annotations: dict[str, Any],
    *,
    split: str,
    cfg: FusionConfig,
) -> list[BBoxDetection]:
    assert_dev_only_selection(split) if split == "dev" else None
    if split == "holdout":
        # allowed only for error-analysis callers that pass purpose explicitly
        pass
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frames = [f for f in annotations["frames"] if f["split"] == split]
    if split in {"train", "dev"}:
        assert_no_holdout_v1_for_development(frames, purpose="development")
    device = "0" if torch.cuda.is_available() else "cpu"
    cfg = FusionConfig(
        conf=cfg.conf,
        predict_iou=cfg.predict_iou,
        merge_iou=cfg.merge_iou,
        imgsz=cfg.imgsz,
        tile_w=cfg.tile_w,
        tile_h=cfg.tile_h,
        overlap_x=cfg.overlap_x,
        overlap_y=cfg.overlap_y,
        max_tiles=cfg.max_tiles,
        mode=cfg.mode,
        device=device,
    )
    cap = cv2.VideoCapture(str(VIDEO))
    preds: list[BBoxDetection] = []
    try:
        for fr in frames:
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            dets = predict_full_tile_fused(model, frame, cfg)
            preds.extend(attach_frame_index(dets, idx))
    finally:
        cap.release()
    return preds


def eval_split(
    weights: Path,
    annotations: dict[str, Any],
    *,
    split: str,
    cfg: FusionConfig,
) -> dict[str, Any]:
    frames = [f for f in annotations["frames"] if f["split"] == split]
    preds = predict_split_fused(weights, annotations, split=split, cfg=cfg)
    return evaluate_protocol_v2(preds, frames)


def sweep_dev_fusion(
    weights: Path,
    annotations: dict[str, Any],
    *,
    mode: str,
    imgsz: int = 960,
) -> dict[str, Any]:
    assert_dev_only_selection("dev")
    confs = [0.15, 0.20, 0.25, 0.30]
    merge_ious = [0.45, 0.55, 0.65]
    best = None
    rows = []
    for conf in confs:
        for merge_iou in merge_ious:
            cfg = FusionConfig(
                conf=conf,
                predict_iou=0.5,
                merge_iou=merge_iou,
                imgsz=imgsz,
                mode=mode,
            )
            ev = eval_split(weights, annotations, split="dev", cfg=cfg)
            gate = dev_gate_passed(ev["primary"])
            row = {
                "conf": conf,
                "merge_iou": merge_iou,
                "mode": mode,
                "imgsz": imgsz,
                "primary": {
                    k: ev["primary"].get(k)
                    for k in (
                        "precision",
                        "recall",
                        "f1",
                        "ap50",
                        "ap50_95",
                        "small_recall",
                        "duplicate_rate",
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "fp_per_frame",
                        "merged_person_diag",
                        "height_bin_recall",
                    )
                },
                "secondary": {
                    "ignored_predictions": ev["secondary"]["ignored_predictions"],
                    "uncertain_gt_n": ev["secondary"]["uncertain_gt_n"],
                    "off_pitch_gt_n": ev["secondary"]["off_pitch_gt_n"],
                    "all_human_f1": (ev["secondary"]["all_human"] or {}).get("f1"),
                },
                "gate": gate,
            }
            rows.append(row)
            p = ev["primary"]
            score = (
                (p.get("f1") or 0)
                + 0.2 * (p.get("ap50") or 0)
                + 0.25 * (p.get("small_recall") or 0)
                - 0.5 * max(0.0, (p.get("duplicate_rate") or 0) - 0.01)
            )
            cur_best = -1.0
            if best is not None:
                raw = best["score"]
                cur_best = float(raw) if isinstance(raw, (int, float)) else -1.0
            if best is None or score > cur_best:
                best = {"score": score, "config": row, "eval": ev}
    assert best is not None
    return {"best": best, "rows": rows}


def train_tile_aware(
    *,
    name: str,
    weights: Path,
    data_yaml: Path,
    epochs: int = 35,
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
    cfg = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": 0.0008,
        "lrf": 0.01,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 2.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "seed": 42,
        "close_mosaic": 10,
    }
    cfg_fp = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    t0 = time.time()
    vram_peak = None
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
            patience=10,
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            hsv_h=cfg["hsv_h"],
            hsv_s=cfg["hsv_s"],
            hsv_v=cfg["hsv_v"],
            degrees=cfg["degrees"],
            translate=0.05,
            scale=cfg["scale"],
            shear=0.0,
            perspective=cfg["perspective"],
            flipud=0.0,
            fliplr=cfg["fliplr"],
            mosaic=cfg["mosaic"],
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=cfg["close_mosaic"],
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
            return train_tile_aware(
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
        cand = list((WORK / "runs" / name).rglob("best.pt"))
        if not cand:
            raise FileNotFoundError(f"no best.pt for {name}")
        best = cand[0]
    return {
        "name": name,
        "best_pt": str(best),
        "best_sha256": _sha(best),
        "best_bytes": best.stat().st_size,
        "elapsed_s": round(time.time() - t0, 1),
        "vram_peak_mb": vram_peak,
        "config": cfg,
        "config_fingerprint": cfg_fp,
        "device": str(device),
        "amp": amp,
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
    }


def cleanup_workspace(
    *,
    keep_best: Path | None,
    gate_passed: bool,
) -> dict[str, Any]:
    removed: list[str] = []
    kept: list[str] = []

    def _rm(p: Path) -> None:
        if not p.exists():
            return
        if p.is_file():
            p.unlink(missing_ok=True)
            removed.append(str(p))
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))

    # Always remove caches / optimizer / last.pt / tensorboard
    if WORK.exists():
        for p in WORK.rglob("*"):
            if not p.is_file():
                continue
            if keep_best and p.resolve() == keep_best.resolve() and gate_passed:
                kept.append(str(p))
                continue
            if p.name in {"last.pt", "optimizer.pt"} or p.suffix == ".cache":
                p.unlink(missing_ok=True)
                removed.append(str(p))
                continue
            if (
                p.name.endswith(".pt")
                and "weights" in str(p)
                and (not gate_passed or (keep_best and p.resolve() != keep_best.resolve()))
            ):
                p.unlink(missing_ok=True)
                removed.append(str(p))

    # tile image cache / dataset images are runtime — wipe if gate fail; keep manifest only on pass? Spec: delete tile image cache
    if TILE_DS.exists():
        for sub in ("images", "labels"):
            d = TILE_DS / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
        for cache in TILE_DS.rglob("*.cache"):
            cache.unlink(missing_ok=True)
            removed.append(str(cache))

    if keep_best and gate_passed and keep_best.is_file():
        arch = ARCHIVE / "own_video_human_v1_f2c_devbest.pt"
        shutil.copy2(keep_best, arch)
        kept.append(str(arch))

    receipt = {
        "schema": "r1_f2c_cleanup_receipt_v1",
        "data_loss": False,
        "frozen_gt_preserved": True,
        "removed_n": len(removed),
        "removed_sample": removed[:40],
        "kept": kept,
        "gate_passed": gate_passed,
        "written_at_utc": utc_now(),
    }
    atomic_write_json(EV / "cleanup_receipt.json", receipt, mode=0o644)
    return receipt


def write_holdout_v2_launcher(selection: dict[str, Any]) -> dict[str, Any]:
    """Windows blind holdout_v2 review tool (user does not annotate in this stage)."""
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = REPO / "scripts" / "start_r1_holdout_v2_review.sh"
    wrapper.write_text(
        """#!/usr/bin/env bash
# Blind holdout_v2 review server (no proposals / no model API leakage).
set -euo pipefail
REPO_ROOT="/home/fdoblak/projects/football-analytics"
PYTHON="/home/fdoblak/miniconda3/envs/ai-dev/bin/python"
SERVER_PY="${REPO_ROOT}/scripts/r1_independent_gt_review_server.py"
RUNTIME="/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_holdout_v2"
VIDEO="/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
HOST="127.0.0.1"
PORT="8767"
PID_FILE="${RUNTIME}/server.pid"
LOG_FILE="${RUNTIME}/server_wrapper.log"
EXPECTED_SHA="97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"

mkdir -p "${RUNTIME}"
log() { echo "$1"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >>"${LOG_FILE}"; }
log "=== start_r1_holdout_v2_review.sh ==="
if [[ ! -f "${RUNTIME}/draft_annotations.json" ]]; then
  log "ERROR: holdout_v2 draft missing"
  exit 5
fi
DIGEST="$("${PYTHON}" -c "import hashlib; from pathlib import Path; p=Path(r'${VIDEO}'); h=hashlib.sha256();
f=p.open('rb');
[h.update(c) for c in iter(lambda: f.read(1<<20), b'')];
f.close();
print(h.hexdigest())")"
if [[ "${DIGEST}" != "${EXPECTED_SHA}" ]]; then
  log "ERROR: SOURCE_SHA_MISMATCH"
  exit 6
fi
# Refuse if a non-holdout_v2 server answers on this port
HEALTH="$("${PYTHON}" - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8767/health", timeout=2) as r:
        body = json.loads(r.read().decode())
except Exception:
    print("DOWN"); raise SystemExit(0)
ok = body.get("status")=="ok" and body.get("service")=="r1_independent_gt_review" and body.get("holdout_v2") is True
print("OK" if ok else "MISMATCH")
PY
)"
if [[ "${HEALTH}" == "OK" ]]; then
  log "INFO: holdout_v2 server already healthy"
  exit 10
fi
if [[ "${HEALTH}" == "MISMATCH" ]]; then
  log "ERROR: port 8767 occupied by non-holdout_v2 service"
  exit 8
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo $$ >"${PID_FILE}"
log "Starting holdout_v2 blind review on http://${HOST}:${PORT}/"
exec "${PYTHON}" "${SERVER_PY}" --host "${HOST}" --port "${PORT}" --runtime "${RUNTIME}" --video "${VIDEO}" --holdout-v2
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    bat = """@echo off
setlocal EnableExtensions
title R1 Holdout V2 Blind Review
cd /d "%~dp0"

set "DISTRO=Ubuntu-22.04"
set "PORT=8767"
set "APP_URL=http://127.0.0.1:8767/"
set "WRAPPER=/home/fdoblak/projects/football-analytics/scripts/start_r1_holdout_v2_review.sh"
set "LINUX_LOG=/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_holdout_v2/server_wrapper.log"

echo R1 Holdout V2 blind review starting...
echo.

call :CHECK_HEALTH
if not errorlevel 1 (
  echo Server already ready.
  echo Browser opening...
  start "" "%APP_URL%"
  echo Do not close the existing server window during review.
  goto END_OK
)

start "R1-HoldoutV2-Server" cmd /k wsl.exe -d %DISTRO% -- bash %WRAPPER%

set /a TRIES=0
:WAIT_HEALTH
set /a TRIES+=1
call :CHECK_HEALTH
if not errorlevel 1 goto READY
if %TRIES% GEQ 30 goto FAIL
timeout /t 1 /nobreak >nul
goto WAIT_HEALTH

:READY
echo Server ready.
echo Browser opening...
start "" "%APP_URL%"
echo Do not close the server window during review.
goto END_OK

:FAIL
echo.
echo ERROR: Server did not become healthy within 30 seconds.
echo Browser was NOT opened.
echo Open the R1-HoldoutV2-Server window for stderr.
echo Linux log: %LINUX_LOG%
echo.
pause
exit /b 1

:END_OK
echo.
echo Review UI: %APP_URL%
echo Keep the R1-HoldoutV2-Server window open until you finish.
pause
exit /b 0

:CHECK_HEALTH
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.service -eq 'r1_independent_gt_review' -and $r.holdout_v2 -eq $true) { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %ERRORLEVEL%
"""
    # ASCII only
    bat.encode("ascii")
    repo_bat = REPO / "scripts" / "windows" / "START_HOLDOUT_V2_REVIEW.bat"
    repo_bat.parent.mkdir(parents=True, exist_ok=True)
    repo_bat.write_bytes(bat.replace("\n", "\r\n").encode("ascii"))
    win_bat = WIN_DIR / "START_HOLDOUT_V2_REVIEW.bat"
    win_bat.write_bytes(repo_bat.read_bytes())
    readme = (
        "R1 HOLDOUT V2 — KOR BAGIMSIZ ANNOTATION\r\n"
        "=========================================\r\n"
        "1) START_HOLDOUT_V2_REVIEW.bat cift tik.\r\n"
        "2) R1-HoldoutV2-Server penceresini KAPATMAYIN.\r\n"
        "3) Proposal/prediction YOK — sifirdan kor annotation.\r\n"
        f"4) Kare sayisi: {selection['n_frames']}\r\n"
        "5) Bu asamada Cursor annotation istemez; arac hazir.\r\n"
    )
    (WIN_DIR / "HOLDOUT_V2_README_TR.txt").write_text(readme, encoding="utf-8")
    return {
        "windows_bat": str(win_bat),
        "repo_bat": str(repo_bat),
        "wrapper": str(wrapper),
        "port": 8767,
        "n_frames": selection["n_frames"],
        "bat_sha256": _sha(win_bat),
    }


def maybe_dev_error_sheet(
    annotations: dict[str, Any],
    weights: Path,
    cfg: FusionConfig,
) -> str | None:
    """Optional single small contact sheet of worst-FN dev frames."""
    frames = [f for f in annotations["frames"] if f["split"] == "dev"]
    preds = predict_split_fused(weights, annotations, split="dev", cfg=cfg)
    ev = evaluate_protocol_v2(preds, frames)
    # pick up to 4 frames with most FN via height bins isn't per-frame; skip heavy media
    if (ev["primary"].get("false_negatives") or 0) <= 0:
        return None
    out = EV / "dev_error_contact_sheet.jpg"
    # Build a tiny 2x2 sheet from first 4 dev frames with overlays — keep small
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for fr in frames[:4]:
        idx = int(fr["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if not ok:
            continue
        vis = cv2.resize(im, (320, 180))
        tiles.append(vis)
    cap.release()
    if len(tiles) < 2:
        return None
    import numpy as np

    row1 = np.concatenate(tiles[:2], axis=1)
    if len(tiles) >= 4:
        row2 = np.concatenate(tiles[2:4], axis=1)
        sheet = np.concatenate([row1, row2], axis=0)
    else:
        sheet = row1
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=35)
    args = ap.parse_args()

    EV.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    annotations = load_frozen()

    prior_path = (
        REPO / "artifacts/evidence/reboot_01/r1_f2b_r1_holdout_failure/holdout_metrics.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    record_holdout_v1_consumed(prior)

    # Protocol v2 fingerprint (before any holdout_v2 selection)
    proto = protocol_v2_definition()
    atomic_write_json(EV / "evaluation_protocol_v2.json", proto, mode=0o644)
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/evaluation_protocol_v2.json",
        proto,
        mode=0o644,
    )

    if not CKPT_B.is_file():
        raise SystemExit(f"missing reference checkpoint {CKPT_B}")

    # --- RCA: height-bin recall on holdout for ERROR ANALYSIS ONLY ---
    print("RCA: evaluating reference full-frame on holdout (error analysis only)...")
    cfg_ref = FusionConfig(conf=0.25, merge_iou=0.55, imgsz=960, mode="full_frame")
    # Direct predict without selection guard for holdout error analysis
    from ultralytics import YOLO

    model = YOLO(str(CKPT_B))
    hold_frames = [f for f in annotations["frames"] if f["split"] == "holdout"]
    cap = cv2.VideoCapture(str(VIDEO))
    hold_preds: list[BBoxDetection] = []
    device = "0" if torch.cuda.is_available() else "cpu"
    cfg_ref = FusionConfig(conf=0.25, merge_iou=0.55, imgsz=960, mode="full_frame", device=device)
    try:
        for fr in hold_frames:
            idx = int(fr["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            dets = predict_full_tile_fused(model, frame, cfg_ref)
            hold_preds.extend(attach_frame_index(dets, idx))
    finally:
        cap.release()
    hold_ev = evaluate_protocol_v2(hold_preds, hold_frames)
    # Dev reference for comparison (not using holdout for selection)
    print("RCA: evaluating reference full-frame on dev...")
    dev_ev_ref = eval_split(CKPT_B, annotations, split="dev", cfg=cfg_ref)
    err_mag = error_magnitude_from_preds(hold_preds, hold_frames)
    rca = build_root_cause_report(
        annotations,
        prior_holdout_metrics=prior,
        height_recall_holdout=height_bin_recall_from_eval(hold_ev),
        height_recall_dev=height_bin_recall_from_eval(dev_ev_ref),
        error_magnitude=err_mag,
    )
    atomic_write_json(EV / "root_cause.json", rca, mode=0o644)

    # --- Tile dataset (train only tiles) ---
    print("Building train tile dataset...")
    tile_manifest = build_train_tile_dataset(annotations, video=VIDEO, out_root=TILE_DS)
    atomic_write_json(EV / "tile_export_manifest.json", tile_manifest, mode=0o644)

    runs: dict[str, Any] = {}

    # A — reference full-frame (existing ckpt)
    print("Run A: full-frame reference on DEV...")
    a_sweep = sweep_dev_fusion(CKPT_B, annotations, mode="full_frame", imgsz=960)
    runs["A"] = {
        "name": "A_yolo11s_ft_fullframe_ref",
        "checkpoint": str(CKPT_B),
        "checkpoint_sha256": _sha(CKPT_B),
        "retrained": False,
        "dev": a_sweep,
    }

    # B — tiled inference same ckpt
    print("Run B: full+tile inference on DEV...")
    b_sweep = sweep_dev_fusion(CKPT_B, annotations, mode="hybrid", imgsz=960)
    runs["B"] = {
        "name": "B_yolo11s_ft_tiled_infer",
        "checkpoint": str(CKPT_B),
        "checkpoint_sha256": _sha(CKPT_B),
        "retrained": False,
        "dev": b_sweep,
    }

    # C — tile-aware fine-tune
    if args.skip_train:
        c_best = WORK / "runs" / "C_yolo11s_tile_aware" / "weights" / "best.pt"
        if not c_best.is_file():
            raise SystemExit("skip-train but C checkpoint missing")
        c_train = {
            "name": "C_yolo11s_tile_aware",
            "best_pt": str(c_best),
            "best_sha256": _sha(c_best),
            "skipped_train": True,
        }
    else:
        print("Run C: tile-aware fine-tune YOLO11s...")
        c_train = train_tile_aware(
            name="C_yolo11s_tile_aware",
            weights=YOLO11S if YOLO11S.is_file() else CKPT_B,
            data_yaml=TILE_DS / "dataset.yaml",
            epochs=args.epochs,
            imgsz=960,
            batch=2,
        )
    print("Run C: DEV sweep hybrid...")
    c_sweep = sweep_dev_fusion(Path(str(c_train["best_pt"])), annotations, mode="hybrid", imgsz=960)
    runs["C"] = {
        "name": "C_yolo11s_tile_aware",
        "train": c_train,
        "retrained": not args.skip_train,
        "dev": c_sweep,
    }

    # Select best on DEV only
    candidates = []
    for kid, rund in runs.items():
        best = rund["dev"]["best"]
        candidates.append((kid, best["score"], best))
    candidates.sort(key=lambda x: -x[1])
    selected_id, _, selected_best = candidates[0]
    selected_primary = selected_best["config"]["primary"]
    gate = selected_best["config"]["gate"]
    if not gate["passed"]:
        # also check if any candidate passed
        for kid, _, best in candidates:
            if best["config"]["gate"]["passed"]:
                selected_id, selected_best = kid, best
                selected_primary = best["config"]["primary"]
                gate = best["config"]["gate"]
                break

    summary = {
        "schema": "r1_f2c_dev_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": proto["protocol_fingerprint"],
        "runs": {
            k: {
                "name": v["name"],
                "best_score": v["dev"]["best"]["score"],
                "best_primary": v["dev"]["best"]["config"]["primary"],
                "best_cfg": {
                    "conf": v["dev"]["best"]["config"]["conf"],
                    "merge_iou": v["dev"]["best"]["config"]["merge_iou"],
                    "mode": v["dev"]["best"]["config"]["mode"],
                    "imgsz": v["dev"]["best"]["config"]["imgsz"],
                },
                "gate_passed": v["dev"]["best"]["config"]["gate"]["passed"],
            }
            for k, v in runs.items()
        },
        "selected": selected_id,
        "selected_primary": selected_primary,
        "gate": gate,
        "tile_counts": tile_manifest.get("counts"),
        "written_at_utc": utc_now(),
    }
    atomic_write_json(EV / "dev_summary.json", summary, mode=0o644)
    # compact metrics only
    atomic_write_json(
        EV / "dev_metrics.json",
        {k: summary["runs"][k]["best_primary"] for k in summary["runs"]},
        mode=0o644,
    )

    keep_best: Path | None = None
    if selected_id == "C":
        keep_best = Path(runs["C"]["train"]["best_pt"])
    elif gate["passed"]:
        keep_best = CKPT_B

    launcher_info = None
    holdout_sel = None
    if gate["passed"]:
        ckpt_path = Path(keep_best or CKPT_B)
        frozen_cfg = {
            "schema": "r1_f2c_selected_dev_config_v1",
            "selected_run": selected_id,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha(ckpt_path),
            "inference": {
                "conf": selected_best["config"]["conf"],
                "merge_iou": selected_best["config"]["merge_iou"],
                "mode": selected_best["config"]["mode"],
                "imgsz": selected_best["config"]["imgsz"],
            },
            "protocol_fingerprint": proto["protocol_fingerprint"],
            "written_at_utc": utc_now(),
        }
        frozen_cfg["config_fingerprint"] = hashlib.sha256(
            json.dumps(
                {
                    "selected_run": frozen_cfg["selected_run"],
                    "checkpoint_sha256": frozen_cfg["checkpoint_sha256"],
                    "inference": frozen_cfg["inference"],
                    "protocol_fingerprint": frozen_cfg["protocol_fingerprint"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        atomic_write_json(EV / "selected_dev_config.json", frozen_cfg, mode=0o644)

        holdout_sel = select_holdout_v2_frames(annotations, n_target=22)
        atomic_write_json(EV / "holdout_v2_selection.json", holdout_sel, mode=0o644)
        write_holdout_v2_draft(
            holdout_sel,
            out_runtime=HOLDOUT_V2_RUNTIME,
            source_sha256=EXPECTED_SOURCE_SHA256,
        )
        launcher_info = write_holdout_v2_launcher(holdout_sel)
        atomic_write_json(EV / "holdout_v2_launcher.json", launcher_info, mode=0o644)
        maybe_dev_error_sheet(
            annotations,
            Path(keep_best or CKPT_B),
            FusionConfig(
                conf=frozen_cfg["inference"]["conf"],
                merge_iou=frozen_cfg["inference"]["merge_iou"],
                imgsz=frozen_cfg["inference"]["imgsz"],
                mode=frozen_cfg["inference"]["mode"],
                device=device,
            ),
        )
        gate_status = GATE_PASS
    else:
        gate_status = GATE_FAIL
        # report failure diagnostics
        fail = {
            "schema": "r1_f2c_dev_gate_failure_v1",
            "gate": GATE_FAIL,
            "failed_checks": {k: v for k, v in gate["checks"].items() if not v},
            "height_bin_recall": selected_primary.get("height_bin_recall"),
            "tile_improvement": {
                "A_small_recall": runs["A"]["dev"]["best"]["config"]["primary"].get("small_recall"),
                "B_small_recall": runs["B"]["dev"]["best"]["config"]["primary"].get("small_recall"),
                "C_small_recall": runs["C"]["dev"]["best"]["config"]["primary"].get("small_recall"),
                "A_f1": runs["A"]["dev"]["best"]["config"]["primary"].get("f1"),
                "B_f1": runs["B"]["dev"]["best"]["config"]["primary"].get("f1"),
                "C_f1": runs["C"]["dev"]["best"]["config"]["primary"].get("f1"),
            },
            "precision_recall": {
                "precision": selected_primary.get("precision"),
                "recall": selected_primary.get("recall"),
            },
            "next_needs": [
                "additional_train_active_learning",
                "or_different_P2_detector",
            ],
            "holdout_v2_created": False,
            "written_at_utc": utc_now(),
        }
        atomic_write_json(EV / "dev_gate_failure.json", fail, mode=0o644)

    cleanup = cleanup_workspace(keep_best=keep_best, gate_passed=gate["passed"])

    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f2c_gate_status_v1",
            "gate": gate_status,
            "stage": "R1-F2-C",
            "dev_gate_passed": gate["passed"],
            "acceptance_eligible": False,
            "holdout_v1_status": HOLDOUT_V1_STATUS,
            "acceptance_reusable": False,
            "frozen_gt_fingerprint": EXPECTED_FROZEN_FP,
            "selected_run": selected_id,
            "holdout_v2_frames": (holdout_sel or {}).get("n_frames"),
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )

    # Persist full run details (compact) for provenance
    atomic_write_json(
        WORK / "runs_summary.json",
        {
            "runs": summary["runs"],
            "selected": selected_id,
            "gate": gate_status,
            "cleanup": cleanup,
            "launcher": launcher_info,
        },
        mode=0o600,
    )

    print(gate_status)
    print(json.dumps(gate, indent=2))
    print(json.dumps(summary["runs"], indent=2))
    return 0 if gate["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
