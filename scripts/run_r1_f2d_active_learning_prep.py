#!/usr/bin/env python3
"""R1-F2-D: Protocol v3, blind holdout_v2, active-learning package, review launcher.

Does not train. Does not start R2. Does not freeze new annotations.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from football_analytics.annotation.active_learning_selection import select_active_learning_frames
from football_analytics.annotation.evaluation_protocol_v3 import (
    EXPECTED_FROZEN_FP,
    HOLDOUT_V1_STATUS,
    PROTOCOL_ID,
    protocol_v3_definition,
)
from football_analytics.annotation.gt_freeze import DEFAULT_FROZEN_DIR
from football_analytics.annotation.holdout_v2_blind import select_blind_holdout_v2
from football_analytics.annotation.independent_gt import (
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.human_tiled_detection import HumanDetectConfig, detect_humans

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_f2d_active_learning"
ARCHIVE = Path("/home/fdoblak/football_data/model_archive")
CKPT_B = ARCHIVE / "own_video_human_v1_b_best.pt"
YOLO11N = ARCHIVE / "yolo11n.pt"
YOLO11N_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
VIDEO = DEFAULT_VIDEO
RUNTIME = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning")
WIN_DIR = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")
WORK_F2C = Path("/home/fdoblak/workspace/r1_f2c_small_object")
TILE_DS = Path("/home/fdoblak/workspace/training_datasets/own_video_human_v1_tiles")

GATE_PASS = "PASS — ACTIVE LEARNING AND NEW BLIND HOLDOUT REVIEW READY"
GATE_FAIL = "NO-GO — ACTIVE LEARNING REVIEW PACKAGE FAILURE"


def load_frozen() -> dict[str, Any]:
    ann = json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text(encoding="utf-8"))
    if not ann.get("frozen"):
        raise SystemExit("FROZEN_REQUIRED")
    if ann.get("canonical_fingerprint") != EXPECTED_FROZEN_FP:
        raise SystemExit("FINGERPRINT_MISMATCH")
    if sha256_file(VIDEO) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("SOURCE_SHA_MISMATCH")
    return ann


def generate_al_proposals(
    video: Path,
    al_frames: list[dict[str, Any]],
    holdout_set: set[int],
) -> dict[int, list[dict[str, Any]]]:
    """Proposals for AL section only — never holdout_v2."""
    if not YOLO11N.is_file() or sha256_file(YOLO11N) != YOLO11N_SHA:
        raise SystemExit("yolo11n baseline missing/mismatch")
    adapter = UltralyticsPersonAdapter()
    adapter.load(str(YOLO11N), YOLO11N_SHA)
    cfg = HumanDetectConfig(
        name="al_proposal_yolo11n_hybrid",
        mode="hybrid",
        conf=0.18,
        imgsz_full=960,
        imgsz_tile=640,
        merge_iou=0.55,
        half=True,
        device="auto",
    )
    cap = cv2.VideoCapture(str(video))
    out: dict[int, list[dict[str, Any]]] = {}
    want = {int(f["frame_idx"]) for f in al_frames}
    if want & holdout_set:
        raise SystemExit("AL_PROPOSAL_HOLDOUT_INTERSECTION")
    try:
        for idx in sorted(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            props = detect_humans(adapter, frame, cfg)
            boxes = []
            for j, p in enumerate(props):
                if p.score < 0.22:
                    continue
                if p.eligibility == "off_pitch_human":
                    elig = "off_pitch"
                elif p.eligibility == "on_pitch_human_candidate":
                    elig = "on_pitch"
                else:
                    elig = "uncertain"
                boxes.append(
                    {
                        "proposal_id": f"al_{idx}_{j}",
                        "bbox_xyxy": [p.x1, p.y1, p.x2, p.y2],
                        "score": float(p.score),
                        "eligibility_hint": elig,
                        "origin": "proposal_unreviewed",
                        "class_name": "human",
                        "note": "PROPOSAL_ONLY_NOT_GT",
                    }
                )
            boxes.sort(key=lambda b: -b["score"])
            out[idx] = boxes[:24]
    finally:
        cap.release()
    return out


def write_dual_draft(
    *,
    al_sel: dict[str, Any],
    hold_sel: dict[str, Any],
    proposals: dict[int, list[dict[str, Any]]],
    source_sha: str,
) -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    # Section A first (AL), then B (blind holdout)
    for fr in al_sel["frames"]:
        idx = int(fr["frame_idx"])
        frames.append(
            {
                "frame_idx": idx,
                "t_s": float(fr["t_s"]),
                "split": "train",
                "section": "active_learning",
                "categories": list(fr.get("planned_categories") or []),
                "selection_reasons": list(fr.get("selection_reasons") or []),
                "completed": False,
                "humans": [],
                "proposals": list(proposals.get(idx) or []),
                "rejected_proposals": [],
                "no_human_confirmed": False,
                "provenance": {
                    "origin": "active_learning_v1",
                    "selection_fingerprint": al_sel["selection_fingerprint"],
                },
            }
        )
    for fr in hold_sel["frames"]:
        idx = int(fr["frame_idx"])
        frames.append(
            {
                "frame_idx": idx,
                "t_s": float(fr["t_s"]),
                "split": "holdout",
                "section": "holdout_v2",
                "categories": list(fr.get("planned_categories") or []),
                "completed": False,
                "humans": [],
                "proposals": [],
                "rejected_proposals": [],
                "no_human_confirmed": False,
                "provenance": {
                    "origin": "holdout_v2_blind",
                    "selection_fingerprint": hold_sel["selection_fingerprint"],
                },
            }
        )
    draft = {
        "schema": "independent_gt_draft_v1",
        "dataset_id": "own_video_97b298e4_active_learning_holdout_v2",
        "source_id": "own_video_97b298e4",
        "source_sha256": source_sha,
        "active_learning": True,
        "holdout_v2": True,
        "blind_holdout_section": True,
        "frozen": False,
        "acceptance_eligible": False,
        "frames": frames,
        "al_selection_fingerprint": al_sel["selection_fingerprint"],
        "holdout_v2_selection_fingerprint": hold_sel["selection_fingerprint"],
        "protocol_id": PROTOCOL_ID,
    }
    path = RUNTIME / "draft_annotations.json"
    atomic_write_json(path, draft, mode=0o600)
    n_al = len(al_sel["frames"])
    n_ho = len(hold_sel["frames"])
    atomic_write_json(
        RUNTIME / "progress.json",
        {
            "schema": "active_learning_progress_v1",
            "n_frames": len(frames),
            "n_complete": 0,
            "active_learning": {"n": n_al, "complete": 0},
            "holdout_v2": {"n": n_ho, "complete": 0},
            "by_split": {
                "train": {"n": n_al, "complete": 0},
                "dev": {"n": 0, "complete": 0},
                "holdout": {"n": n_ho, "complete": 0},
            },
            "updated_at_utc": utc_now(),
        },
        mode=0o600,
    )
    atomic_write_json(
        RUNTIME / "session_state.json",
        {
            "schema": "independent_gt_session_state_v1",
            "index": 0,
            "active_learning": True,
            "video": str(VIDEO),
            "source_sha256": source_sha,
            "runtime": str(RUNTIME),
            "updated_at_utc": utc_now(),
        },
        mode=0o600,
    )
    atomic_write_json(RUNTIME / "al_selection.json", al_sel, mode=0o600)
    atomic_write_json(RUNTIME / "holdout_v2_selection.json", hold_sel, mode=0o600)
    return path


def write_detector_plan() -> dict[str, Any]:
    plan = {
        "schema": "r1_f2d_next_detector_plan_v1",
        "status": "PLAN_ONLY_NO_TRAINING",
        "max_candidates": 3,
        "candidates": [
            {
                "id": "D1",
                "name": "YOLO11s_fullframe_hires",
                "arch": "YOLO11s",
                "inference": "full_frame",
                "imgsz": [960, 1280],
                "notes": "Recover precision; higher res for small players without tile FP flood",
            },
            {
                "id": "D2",
                "name": "YOLO11s_full_tile_precision_fusion",
                "arch": "YOLO11s",
                "inference": "hybrid_precision_preserving",
                "fusion_precision_strategy": {
                    "tile_only_requires_fullframe_support": True,
                    "temporal_persistence": True,
                    "pitch_eligibility_gate": True,
                    "size_aware_confidence": True,
                    "cross_tile_agreement": True,
                    "goal": "prevent F2-C precision collapse from tile-only FP",
                },
            },
            {
                "id": "D3",
                "name": "P2_small_object_if_locally_verified",
                "arch": "conditional",
                "status": "PENDING_LOCAL_VERIFY",
                "notes": (
                    "Only if local Ultralytics install exposes a verified P2/small-object "
                    "YAML; do not invent YOLO11-P2 YAML. Prefer official package YAMLs."
                ),
                "verified_local_p2_yaml": False,
            },
        ],
        "holdout_v2_tuning_forbidden": True,
        "holdout_v1_acceptance_forbidden": True,
        "written_at_utc": utc_now(),
    }
    # Verify local P2 yaml without fabricating YOLO11-specific configs.
    try:
        import ultralytics

        root = Path(ultralytics.__file__).resolve().parent
        p2_hits = sorted(str(p) for p in root.rglob("*p2*.yaml"))[:8]
        plan["candidates"][2]["local_p2_yaml_sample"] = p2_hits
        plan["candidates"][2]["verified_local_p2_yaml"] = bool(p2_hits)
        plan["candidates"][2]["status"] = (
            "LOCALLY_VERIFIED_P2_YAML_AVAILABLE" if p2_hits else "NO_LOCAL_P2_YAML"
        )
        plan["candidates"][2]["notes"] = (
            "Local Ultralytics ships P2 YAMLs (e.g. yolov8-p2 / yolo26-p2). "
            "Next round may try one only after confirming trainability; "
            "do not invent a YOLO11-P2 YAML."
            if p2_hits
            else "No local P2 YAML found; skip D3 rather than fabricating."
        )
    except Exception as exc:  # noqa: BLE001
        plan["candidates"][2]["verify_error"] = str(exc)
        plan["candidates"][2]["status"] = "VERIFY_FAILED"
    atomic_write_json(EV / "next_detector_plan.json", plan, mode=0o644)
    return plan


def write_launcher() -> dict[str, Any]:
    wrapper = REPO / "scripts" / "start_r1_active_learning_review.sh"
    wrapper.write_text(
        """#!/usr/bin/env bash
# Active-learning + blind holdout_v2 dual review (foreground WSL helper).
set -euo pipefail
REPO_ROOT="/home/fdoblak/projects/football-analytics"
PYTHON="/home/fdoblak/miniconda3/envs/ai-dev/bin/python"
SERVER_PY="${REPO_ROOT}/scripts/r1_independent_gt_review_server.py"
RUNTIME="/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning"
VIDEO="/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
HOST="127.0.0.1"
PORT="8768"
PID_FILE="${RUNTIME}/server.pid"
LOG_FILE="${RUNTIME}/server_wrapper.log"
EXPECTED_SHA="97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"

mkdir -p "${RUNTIME}"
log() { echo "$1"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >>"${LOG_FILE}"; }
log "=== start_r1_active_learning_review.sh ==="
if [[ ! -f "${RUNTIME}/draft_annotations.json" ]]; then
  log "ERROR: active learning draft missing"
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
HEALTH="$("${PYTHON}" - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8768/health", timeout=2) as r:
        body = json.loads(r.read().decode())
except Exception:
    print("DOWN"); raise SystemExit(0)
ok = (
    body.get("status")=="ok"
    and body.get("service")=="r1_independent_gt_review"
    and body.get("active_learning") is True
)
print("OK" if ok else "MISMATCH")
PY
)"
if [[ "${HEALTH}" == "OK" ]]; then
  log "INFO: active learning server already healthy"
  exit 10
fi
if [[ "${HEALTH}" == "MISMATCH" ]]; then
  log "ERROR: port 8768 occupied by non-AL service"
  exit 8
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo $$ >"${PID_FILE}"
log "Starting AL+holdout_v2 review on http://${HOST}:${PORT}/"
exec "${PYTHON}" "${SERVER_PY}" --host "${HOST}" --port "${PORT}" --runtime "${RUNTIME}" --video "${VIDEO}" --active-learning
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    bat = """@echo off
setlocal EnableExtensions
title R1 Active Learning Review
cd /d "%~dp0"

set "DISTRO=Ubuntu-22.04"
set "PORT=8768"
set "APP_URL=http://127.0.0.1:8768/"
set "WRAPPER=/home/fdoblak/projects/football-analytics/scripts/start_r1_active_learning_review.sh"
set "LINUX_LOG=/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning/server_wrapper.log"

echo R1 Active Learning + Blind Holdout review starting...
echo.

call :CHECK_HEALTH
if not errorlevel 1 (
  echo Server already ready.
  echo Browser opening...
  start "" "%APP_URL%"
  echo Do not close the existing server window during review.
  goto END_OK
)

start "R1-AL-Server" cmd /k wsl.exe -d %DISTRO% -- bash %WRAPPER%

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
echo Open the R1-AL-Server window for stderr.
echo Linux log: %LINUX_LOG%
echo.
pause
exit /b 1

:END_OK
echo.
echo Review UI: %APP_URL%
echo Keep the R1-AL-Server window open until you finish.
pause
exit /b 0

:CHECK_HEALTH
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8768/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.service -eq 'r1_independent_gt_review' -and $r.active_learning -eq $true) { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %ERRORLEVEL%
"""
    bat.encode("ascii")
    repo_bat = REPO / "scripts" / "windows" / "START_ACTIVE_LEARNING_REVIEW.bat"
    repo_bat.parent.mkdir(parents=True, exist_ok=True)
    repo_bat.write_bytes(bat.replace("\n", "\r\n").encode("ascii"))
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    win_bat = WIN_DIR / "START_ACTIVE_LEARNING_REVIEW.bat"
    win_bat.write_bytes(repo_bat.read_bytes())
    readme = (
        "R1 ACTIVE LEARNING + BLIND HOLDOUT V2\r\n"
        "=====================================\r\n"
        "1) START_ACTIVE_LEARNING_REVIEW.bat cift tik.\r\n"
        "2) R1-AL-Server penceresini KAPATMAYIN.\r\n"
        "3) Bolum A: Active learning — turuncu proposal (GT degil).\r\n"
        "4) Bolum B: Blind holdout — proposal/prediction YOK.\r\n"
        "5) Progress: active learning X/N ve blind holdout X/30.\r\n"
        "6) Freeze / egitim / R2 YOK. Bitince Cursor'a haber verin.\r\n"
    )
    (WIN_DIR / "ACTIVE_LEARNING_README_TR.txt").write_text(readme, encoding="utf-8")
    return {
        "windows_bat": str(win_bat),
        "repo_bat": str(repo_bat),
        "wrapper": str(wrapper),
        "port": 8768,
        "bat_sha256": sha256_file(win_bat),
    }


def cleanup() -> dict[str, Any]:
    removed: list[dict[str, Any]] = []
    total = 0

    def _rm(p: Path, reason: str) -> None:
        nonlocal total
        if not p.exists():
            return
        if p.is_file():
            sz = p.stat().st_size
            p.unlink(missing_ok=True)
            removed.append({"path": str(p), "bytes": sz, "reason": reason})
            total += sz
        elif p.is_dir():
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
            removed.append({"path": str(p), "bytes": sz, "reason": reason})
            total += sz

    # F2-C runtime leftovers
    if WORK_F2C.exists():
        for p in WORK_F2C.rglob("last.pt"):
            _rm(p, "rejected_last_pt")
        for p in WORK_F2C.rglob("*.cache"):
            _rm(p, "training_cache")
        for p in WORK_F2C.rglob("best.pt"):
            # rejected C checkpoint — do not delete yolo11s archive
            if "C_yolo11s" in str(p) or "tile_aware" in str(p):
                _rm(p, "rejected_f2c_checkpoint")
    for sub in ("images", "labels"):
        _rm(TILE_DS / sub, "temp_tile_images_labels")
    # empty dirs
    for d in (WORK_F2C / "runs", TILE_DS):
        if d.is_dir() and not any(d.rglob("*")):
            _rm(d, "empty_dir")

    # never touch canonical/archive baselines
    receipt = {
        "schema": "r1_f2d_cleanup_receipt_v1",
        "data_loss": False,
        "preserved": [
            str(VIDEO),
            str(DEFAULT_FROZEN_DIR),
            str(ARCHIVE / "yolo11n.pt"),
            str(ARCHIVE / "yolo11s.pt"),
            str(CKPT_B),
            str(REPO / "model_registry.yaml"),
            str(REPO / "artifacts/evidence/reboot_01/r1_small_object_redesign"),
        ],
        "removed": removed,
        "byte_total": total,
        "recoverability": "runtime regenerable; Git history intact",
        "written_at_utc": utc_now(),
    }
    atomic_write_json(EV / "cleanup_receipt.json", receipt, mode=0o644)
    return receipt


def validate_package(
    al_sel: dict[str, Any],
    hold_sel: dict[str, Any],
    launcher: dict[str, Any],
) -> dict[str, Any]:
    errs: list[str] = []
    al_i = set(al_sel["frame_indices"])
    ho_i = set(hold_sel["frame_indices"])
    old = {
        int(f["frame_idx"])
        for f in json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text())["frames"]
    }
    if al_i & ho_i:
        errs.append("AL_HOLDOUT_OVERLAP")
    if al_i & old or ho_i & old:
        errs.append("OVERLAP_OLD_80")
    if hold_sel["n_frames"] != 30:
        errs.append(f"HOLDOUT_N:{hold_sel['n_frames']}")
    if al_sel["n_frames"] > 100 or al_sel["n_frames"] < 40:
        errs.append(f"AL_N:{al_sel['n_frames']}")
    draft = json.loads((RUNTIME / "draft_annotations.json").read_text())
    for fr in draft["frames"]:
        if fr.get("section") == "holdout_v2" and fr.get("proposals"):
            errs.append("HOLDOUT_PROPOSAL_LEAK")
        if fr.get("section") == "holdout_v2" and fr.get("split") != "holdout":
            errs.append("HOLDOUT_SPLIT")
    if not Path(launcher["windows_bat"]).is_file():
        errs.append("BAT_MISSING")
    fp = json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text())["canonical_fingerprint"]
    if fp != EXPECTED_FROZEN_FP:
        errs.append("FROZEN_FP_CHANGED")
    return {"ok": not errs, "errors": errs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-al-infer", action="store_true")
    ap.add_argument("--al-n", type=int, default=100)
    args = ap.parse_args()

    EV.mkdir(parents=True, exist_ok=True)
    annotations = load_frozen()
    old_80 = {int(f["frame_idx"]) for f in annotations["frames"]}

    # 1) Protocol v3 BEFORE inference
    proto = protocol_v3_definition()
    atomic_write_json(EV / "evaluation_protocol_v3.json", proto, mode=0o644)
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/evaluation_protocol_v3.json",
        proto,
        mode=0o644,
    )
    atomic_write_json(
        EV / "holdout_v1_lineage.json",
        {
            "holdout_v1_status": HOLDOUT_V1_STATUS,
            "acceptance_reusable": False,
            "role": "development_error_analysis_only",
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )

    # 2) Blind holdout_v2 lock (no model)
    print("Selecting blind holdout_v2 (30 frames, no inference)...")
    hold_sel = select_blind_holdout_v2(annotations, n_target=30)
    atomic_write_json(EV / "holdout_v2_selection.json", hold_sel, mode=0o644)
    holdout_set = set(hold_sel["frame_indices"])

    # 3) Active learning (dev signals only; never holdout_v2)
    if not CKPT_B.is_file():
        raise SystemExit(f"missing development checkpoint {CKPT_B}")
    print("Selecting active-learning frames with development checkpoint...")
    if args.skip_al_infer and (EV / "active_learning_selection.json").is_file():
        al_sel = json.loads((EV / "active_learning_selection.json").read_text())
    else:
        al_sel = select_active_learning_frames(
            video=VIDEO,
            weights=CKPT_B,
            old_80=old_80,
            holdout_v2=holdout_set,
            n_target=min(100, args.al_n),
        )
    atomic_write_json(EV / "active_learning_selection.json", al_sel, mode=0o644)

    print("Generating AL proposals (train section only)...")
    proposals = generate_al_proposals(VIDEO, al_sel["frames"], holdout_set)
    write_dual_draft(
        al_sel=al_sel,
        hold_sel=hold_sel,
        proposals=proposals,
        source_sha=EXPECTED_SOURCE_SHA256,
    )

    plan = write_detector_plan()
    launcher = write_launcher()
    atomic_write_json(EV / "launcher.json", launcher, mode=0o644)
    cleanup_receipt = cleanup()

    checks = validate_package(al_sel, hold_sel, launcher)
    atomic_write_json(EV / "package_validation.json", checks, mode=0o644)

    gate = GATE_PASS if checks["ok"] else GATE_FAIL
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f2d_gate_status_v1",
            "gate": gate,
            "stage": "R1-F2-D",
            "acceptance_eligible": False,
            "detector_trained": False,
            "holdout_v1_status": HOLDOUT_V1_STATUS,
            "acceptance_reusable": False,
            "frozen_gt_fingerprint": EXPECTED_FROZEN_FP,
            "protocol_v3_fingerprint": proto["protocol_fingerprint"],
            "holdout_v2_n": hold_sel["n_frames"],
            "active_learning_n": al_sel["n_frames"],
            "launcher": launcher.get("windows_bat"),
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )
    summary = {
        "gate": gate,
        "protocol_fingerprint": proto["protocol_fingerprint"],
        "holdout_v2": {
            "n": hold_sel["n_frames"],
            "fingerprint": hold_sel["selection_fingerprint"],
        },
        "active_learning": {
            "n": al_sel["n_frames"],
            "counts": al_sel.get("counts"),
            "fingerprint": al_sel["selection_fingerprint"],
        },
        "launcher": launcher,
        "cleanup_bytes": cleanup_receipt.get("byte_total"),
        "validation": checks,
        "detector_plan_ids": [c["id"] for c in plan["candidates"]],
    }
    atomic_write_json(EV / "summary.json", summary, mode=0o644)
    print(gate)
    print(json.dumps(summary, indent=2))
    return 0 if checks["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
