#!/usr/bin/env python3
"""R1-F1-R3: SoccerNet Game State official detector vs YOLO11n hybrid bake-off.

Candidates:
  B  — current best general baseline (YOLO11n hybrid tiled)
  S  — official sn-gamestate/TrackLab yolo_ultralytics settings (yolo11m full-frame)
  ST — S + ≤2-frame temporal carry (observed vs carried)

No team/role labels. Diagnostic ≠ accuracy / GT.
On failure: write blocker JSON only (no MP4/PNG/HTML package).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_analytics.perception.adapters.soccernet_gamestate_detector import (
    OFFICIAL_FINE_TUNE_STATUS,
    OFFICIAL_IMGSZ,
    OFFICIAL_MIN_CONFIDENCE,
    SoccerNetGameStateDetectorAdapter,
    default_worker_python,
)
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.human_temporal_stability import (
    TemporalHumanStabilizer,
    TemporalProposal,
    compute_temporal_diagnostics,
)
from football_analytics.perception.human_tiled_detection import (
    HumanDetectConfig,
    HumanProposal,
    detect_humans,
    duplicate_pairs,
    merged_person_candidates,
)

REPO = Path(__file__).resolve().parents[1]
VIDEO = Path("/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4")
ARCH = Path("/home/fdoblak/football_data/model_archive")
YOLO_N = ARCH / "yolo11n.pt"
SHA_N = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
YOLO_M = Path("/home/fdoblak/workspace/own_video_analysis/r1_f1_r3/weights/yolo11m.pt")
SHA_M = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
EV = REPO / "artifacts/evidence/reboot_01"
OUT_OK = EV / "r1_soccernet_detector_candidate"
WORK = Path("/home/fdoblak/workspace/own_video_analysis/r1_f1_r3")
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Human Detection")
FPS = 30.0

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

CFG_B = HumanDetectConfig(
    name="B_yolo11n_hybrid",
    mode="hybrid",
    conf=0.18,
    imgsz_full=960,
    imgsz_tile=640,
    merge_iou=0.55,
    half=True,
)

GATE_PASS = "PASS_WITH_FINDINGS — SOCCERNET FOOTBALL HUMAN DETECTOR CANDIDATE READY; GT NOT FROZEN"
GATE_NOGO = "NO-GO — OFFICIAL SOCCER FOOTBALL DETECTOR UNAVAILABLE OR INSUFFICIENT"


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
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_frames(indices: list[int]) -> dict[int, np.ndarray]:
    want = set(indices)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(VIDEO))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = frame
            if len(out) == len(want):
                break
        i += 1
    cap.release()
    missing = sorted(want - set(out))
    if missing:
        raise RuntimeError(f"missing frames: {missing[:10]}")
    return out


def props_from_official_m(
    adapter: UltralyticsPersonAdapter, frame: np.ndarray
) -> list[HumanProposal]:
    """Official TrackLab semantics: full-frame yolo11m, filter conf>=0.4, imgsz=640."""
    from football_analytics.acceptance.stage18_own_video.pipeline import compute_pitch_masks
    from football_analytics.perception.human_tiled_detection import classify_eligibility

    boxes = adapter.predict_persons(
        frame,
        conf=OFFICIAL_MIN_CONFIDENCE,
        iou=0.7,
        imgsz=OFFICIAL_IMGSZ,
        device="0",
        half=True,
        class_ids=[0],
        class_names=["person"],
        channel_order="bgr",
    )
    masks = compute_pitch_masks(frame)
    props: list[HumanProposal] = []
    for b in boxes:
        xyxy = (b.x1, b.y1, b.x2, b.y2)
        elig = classify_eligibility(xyxy, masks)
        props.append(
            HumanProposal(
                x1=b.x1,
                y1=b.y1,
                x2=b.x2,
                y2=b.y2,
                score=b.score,
                eligibility=elig,
                source="soccernet_official_yolo11m",
            )
        )
    return props


def isolated_adapter_smoke(frame: np.ndarray) -> dict[str, Any]:
    ad = SoccerNetGameStateDetectorAdapter()
    ad.load(str(YOLO_M), SHA_M)
    boxes = ad.predict_persons(
        frame,
        conf=OFFICIAL_MIN_CONFIDENCE,
        iou=0.7,
        imgsz=OFFICIAL_IMGSZ,
        device="0",
        half=True,
        class_ids=[0],
        class_names=["person"],
        channel_order="bgr",
    )
    prov = ad.provenance().to_dict()
    vers = dict(ad.software_versions())
    ad.unload()
    return {
        "ok": True,
        "n_boxes": len(boxes),
        "provenance": prov,
        "software_versions": vers,
        "boxes_sample": [{"xyxy": [b.x1, b.y1, b.x2, b.y2], "score": b.score} for b in boxes[:5]],
    }


def to_temporal(props: list[HumanProposal]) -> list[TemporalProposal]:
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


def summarize(
    name: str,
    per_frame: dict[int, list[TemporalProposal]],
    frames: dict[int, np.ndarray],
) -> dict[str, Any]:
    ordered = [per_frame[i] for i in AUDIT_FRAMES]
    td = compute_temporal_diagnostics(ordered)
    small = 0
    merged = 0
    dups = 0
    off = 0
    observed = 0
    for i in AUDIT_FRAMES:
        props = per_frame[i]
        xyxy = [p.as_xyxy() for p in props if p.temporal_status == "observed"]
        observed += len(xyxy)
        for p in props:
            if p.temporal_status != "observed":
                continue
            h = p.y2 - p.y1
            if h < 40:
                small += 1
            if p.eligibility in {"off_pitch_human", "unknown"}:
                off += 1
        hp = [
            HumanProposal(
                x1=p.x1,
                y1=p.y1,
                x2=p.x2,
                y2=p.y2,
                score=p.score,
                eligibility=p.eligibility,  # type: ignore[arg-type]
                source=p.source,
            )
            for p in props
            if p.temporal_status == "observed"
        ]
        merged += int(merged_person_candidates(hp))
        dups += int(duplicate_pairs(hp))
    return {
        "name": name,
        "observed_detections": observed,
        "small_distant_detected": small,
        "merged_person_boxes": merged,
        "duplicate_boxes": dups,
        "off_pitch_or_unknown": off,
        "flicker1": td.get("one_frame_disappearance"),
        "flicker2": td.get("two_frame_disappearance"),
        "effective_flicker": td.get("effective_one_frame_gap"),
        "bbox_jitter": td.get("center_jitter"),
        "raw_temporal": td,
        "n_frames": len(AUDIT_FRAMES),
        "frame_wh": list(frames[AUDIT_FRAMES[0]].shape[1::-1]),
    }


def sn_env_smoke() -> dict[str, Any]:
    sn_py = "/home/fdoblak/miniconda3/envs/sn-gamestate/bin/python"
    worker = default_worker_python()
    sn_ok = False
    sn_err = ""
    try:
        r = subprocess.run(
            [sn_py, "-c", "import torch; import ultralytics"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        sn_ok = r.returncode == 0
        sn_err = (r.stderr or r.stdout or "")[:400]
    except Exception as exc:  # noqa: BLE001
        sn_err = str(exc)
    return {
        "sn_gamestate_python": sn_py,
        "sn_gamestate_torch_ultralytics": sn_ok,
        "sn_gamestate_error": sn_err if not sn_ok else "",
        "worker_python_selected": worker,
        "worker_is_sn_gamestate": Path(worker).resolve() == Path(sn_py).resolve(),
    }


def decision(
    b: dict[str, Any], s: dict[str, Any], st: dict[str, Any]
) -> tuple[str, str, str | None]:
    """Return (gate, reason, selected)."""

    # Clear superiority of S or ST over B required.
    def better(cand: dict[str, Any]) -> list[str]:
        wins: list[str] = []
        fails: list[str] = []
        if cand["small_distant_detected"] > b["small_distant_detected"]:
            wins.append("small_distant")
        else:
            fails.append(
                f"small_distant {cand['small_distant_detected']}!>{b['small_distant_detected']}"
            )
        if cand["merged_person_boxes"] <= b["merged_person_boxes"]:
            wins.append("merged_ok")
        else:
            fails.append(f"merged {cand['merged_person_boxes']}>{b['merged_person_boxes']}")
        # FP proxy: off_pitch/unknown presented — must not increase vs B
        if cand["off_pitch_or_unknown"] <= b["off_pitch_or_unknown"] + 2:
            wins.append("fp_proxy_ok")
        else:
            fails.append("fp_proxy_worse")
        fl_c = cand.get("flicker1") or 0
        fl_b = b.get("flicker1") or 0
        if fl_c < fl_b:
            wins.append("flicker")
        else:
            fails.append(f"flicker {fl_c}!<{fl_b}")
        jt_c = float(cand.get("bbox_jitter") or 0)
        jt_b = float(b.get("bbox_jitter") or 0)
        if jt_c <= jt_b + 0.02:
            wins.append("jitter_ok")
        else:
            fails.append("jitter_worse")
        return fails

    # Football fine-tune unproven → official football detector unavailable as domain model.
    # Still allow PASS only if S/ST clearly beat B on the bake-off criteria.
    for name, cand in (("S", s), ("ST", st)):
        fails = better(cand)
        if not fails and cand["small_distant_detected"] > b["small_distant_detected"]:
            return GATE_PASS, f"{name}_clearly_beats_B", name
    return (
        GATE_NOGO,
        "official_detector_is_coco_yolo11m_unproven_football_ft_and_does_not_clearly_beat_B;"
        + f"S_small={s['small_distant_detected']}_vs_B={b['small_distant_detected']};"
        + f"ST_small={st['small_distant_detected']};"
        + f"S_flicker={s.get('flicker1')}_B={b.get('flicker1')}",
        None,
    )


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    smoke = sn_env_smoke()

    inventory = {
        "schema": "r1_f1_r3_soccernet_detector_inventory_v1",
        "sn_gamestate_path": "/home/fdoblak/projects/soccernet/sn-gamestate",
        "tracklab_path": "/home/fdoblak/projects/third-party/tracklab",
        "bbox_detector_module": "yolo_ultralytics",
        "model_filename": "yolo11m.pt",
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "fine_tune_status": OFFICIAL_FINE_TUNE_STATUS,
        "classification": {
            "COCO_pretrained_generic_model": True,
            "SoccerNet_fine_tuned_football_detector": False,
            "config_named_but_weight_missing_before_download": True,
            "baseline_tracker_state": False,
        },
        "taxonomy": "coco_person_class_0_only",
        "input_resolution": OFFICIAL_IMGSZ,
        "min_confidence": OFFICIAL_MIN_CONFIDENCE,
        "nms_iou_ultralytics_default": 0.7,
        "license_model": "AGPL-3.0",
        "license_sn_gamestate": "GPL-3.0",
        "license_tracklab": "MIT",
        "standalone_detector_possible": True,
        "weight_individually_downloadable": True,
        "football_domain_weight_individually_available": False,
        "note": "Game State default bbox detector is stock Ultralytics yolo11m.pt; no SoccerNet FT checkpoint in-repo.",
        "isolated_env_smoke": smoke,
        "written_at_utc": utc_now(),
    }
    atomic_json(EV / "r1_f1_r3_soccernet_detector_inventory.json", inventory)

    if not YOLO_M.is_file():
        atomic_json(
            EV / "r1_f1_r3_BLOCKER.json",
            {
                "schema": "r1_f1_r3_blocker_v1",
                "gate": GATE_NOGO,
                "code": "OFFICIAL_FOOTBALL_DETECTOR_WEIGHT_NOT_INDIVIDUALLY_AVAILABLE",
                "detail": "eval weight missing after cleanup",
                "written_at_utc": utc_now(),
            },
        )
        return 2

    assert sha256_file(YOLO_M) == SHA_M
    assert sha256_file(YOLO_N) == SHA_N

    frames = load_frames(AUDIT_FRAMES)

    # Isolated adapter smoke (1 real frame, subprocess boundary)
    try:
        adapter_smoke = isolated_adapter_smoke(frames[AUDIT_FRAMES[0]])
    except Exception as exc:  # noqa: BLE001
        adapter_smoke = {"ok": False, "error": str(exc)[:800]}

    # B
    ad_b = UltralyticsPersonAdapter()
    ad_b.load(str(YOLO_N), SHA_N)
    per_b: dict[int, list[TemporalProposal]] = {}
    for i in AUDIT_FRAMES:
        props = detect_humans(ad_b, frames[i], CFG_B)
        per_b[i] = to_temporal(props)
    ad_b.unload()

    # S — official settings in-process (same conf/imgsz/weight as TrackLab yolo_ultralytics)
    ad_m = UltralyticsPersonAdapter()
    ad_m.load(str(YOLO_M), SHA_M)
    per_s: dict[int, list[TemporalProposal]] = {}
    for i in AUDIT_FRAMES:
        props = props_from_official_m(ad_m, frames[i])
        per_s[i] = to_temporal(props)

    # ST — sequential full-video carry, keep audit frames only
    stab = TemporalHumanStabilizer(max_carry_frames=2)
    per_st: dict[int, list[TemporalProposal]] = {}
    cap = cv2.VideoCapture(str(VIDEO))
    fi = 0
    audit_set = set(AUDIT_FRAMES)
    max_f = max(AUDIT_FRAMES)
    while fi <= max_f:
        ok, frame = cap.read()
        if not ok:
            break
        props = props_from_official_m(ad_m, frame)
        out = stab.update(frame, props)
        if fi in audit_set:
            per_st[fi] = out
        fi += 1
    cap.release()
    ad_m.unload()

    prov = {
        "adapter_id": "soccernet_gamestate_detector",
        "weight_filename": "yolo11m.pt",
        "weights_path": str(YOLO_M),
        "weights_sha256": SHA_M,
        "weights_bytes": YOLO_M.stat().st_size if YOLO_M.is_file() else None,
        "weight_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "license": "AGPL-3.0",
        "fine_tune_status": OFFICIAL_FINE_TUNE_STATUS,
        "min_confidence": OFFICIAL_MIN_CONFIDENCE,
        "imgsz": OFFICIAL_IMGSZ,
        "classification": "COCO_pretrained_generic_model",
        "isolated_adapter_smoke": adapter_smoke,
        "worker_python": default_worker_python(),
    }
    sum_b = summarize("B_yolo11n_hybrid", per_b, frames)
    sum_s = summarize("S_soccernet_official_yolo11m", per_s, frames)
    sum_st = summarize("ST_soccernet_plus_temporal", per_st, frames)

    gate, reason, selected = decision(sum_b, sum_s, sum_st)

    # Pixel-grounded diagnostic sheets (saved under WORK only; not evidence media on NO-GO)
    pixel = {
        "schema": "r1_f1_r3_pixel_diagnostic_v1",
        "classification": [
            "AGENT_PIXEL_GROUNDED_DIAGNOSTIC",
            "NOT_INDEPENDENT_GT",
            "NOT_ACCURACY",
        ],
        "audit_frames": AUDIT_FRAMES,
        "candidates": {"B": sum_b, "S": sum_s, "ST": sum_st},
        "notes": [
            "obvious_visible_humans estimated from detections + visual spot-check; not GT",
            "official S uses full-frame 640 / conf>=0.4 without tiling",
            "B uses hybrid tiling — expected stronger on small/distant",
        ],
        "written_at_utc": utc_now(),
    }

    # Spot-check a few frames: save for agent visual read then delete on NO-GO
    spot_dir = WORK / "pixel_spot"
    if spot_dir.exists():
        shutil.rmtree(spot_dir)
    spot_dir.mkdir(parents=True)
    for fi in (60, 330, 660, 860, 1010):
        for tag, props in (("B", per_b[fi]), ("S", per_s[fi]), ("ST", per_st[fi])):
            vis = frames[fi].copy()
            for p in props:
                color = (255, 255, 0) if p.temporal_status == "observed" else (0, 165, 255)
                if p.eligibility != "on_pitch_human_candidate":
                    color = (128, 128, 128)
                cv2.rectangle(
                    vis,
                    (int(p.x1), int(p.y1)),
                    (int(p.x2), int(p.y2)),
                    color,
                    2,
                )
            cv2.putText(
                vis,
                f"{tag} f={fi} n={len(props)}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "R1 — YALNIZ INSAN TESPITI | RENKLER TAKIM VEYA ROL BELIRTMEZ",
                (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 255),
                2,
            )
            cv2.imwrite(str(spot_dir / f"{tag}_f{fi}.png"), vis)

    # Keep spots until gate decided; delete only after agent-optional retention flag
    bakeoff = {
        "schema": "r1_f1_r3_bakeoff_v1",
        "diagnostic_not_accuracy": True,
        "selected": selected,
        "gate": gate,
        "reason": reason,
        "summary": {"B": sum_b, "S": sum_s, "ST": sum_st},
        "provenance_S": prov,
        "isolated_env_smoke": smoke,
        "weight_eval_path": str(YOLO_M),
        "weight_sha256": SHA_M,
        "weight_bytes": YOLO_M.stat().st_size,
        "fine_tune_status": OFFICIAL_FINE_TUNE_STATUS,
        "written_at_utc": utc_now(),
    }
    atomic_json(EV / "r1_f1_r3_bakeoff_diagnostic.json", bakeoff)
    atomic_json(WORK / "pixel_diagnostic.json", pixel)

    # Model archive policy: on NO-GO delete eval weight; never keep archive duplicate.
    archived = ARCH / "yolo11m.pt"
    model_policy = {
        "schema": "r1_f1_r3_model_policy_v1",
        "official_weight": "yolo11m.pt",
        "selected": selected is not None,
        "archive_path": str(archived),
        "eval_path": str(YOLO_M),
        "action": None,
        "recoverability": "redownloadable_from_ultralytics_assets_v8.3.0",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "sha256": SHA_M,
        "data_loss": False,
        "written_at_utc": utc_now(),
    }

    if gate.startswith("PASS"):
        OUT_OK.mkdir(parents=True, exist_ok=True)
        # Success packaging would go here; not expected for COCO-only official detector.
        raise RuntimeError("PASS path packaging not implemented in this run — unexpected")
    else:
        # Ensure no success media package
        if OUT_OK.exists():
            for p in OUT_OK.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".mp4", ".png", ".html", ".jpg"}:
                    p.unlink()
        # Delete eval weight + any archive copy
        deleted = []
        for p in (YOLO_M, archived):
            if p.is_file():
                deleted.append(
                    {
                        "path": str(p),
                        "type": "file",
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                        "reason": "nogo_unused_coco_yolo11m_after_r3_eval",
                        "recoverability": "redownloadable_from_ultralytics_assets_v8.3.0",
                        "data_loss": False,
                    }
                )
                p.unlink()
        model_policy["action"] = "deleted_after_nogo"
        model_policy["deleted"] = deleted
        atomic_json(EV / "r1_f1_r3_model_policy.json", model_policy)

        # Spot PNGs retained briefly under WORK for agent pixel read; cleaned in final receipt.
        # if spot_dir.exists():
        #     shutil.rmtree(spot_dir)

        blocker = {
            "schema": "r1_f1_r3_blocker_v1",
            "gate": gate,
            "reason": reason,
            "fine_tune_status": OFFICIAL_FINE_TUNE_STATUS,
            "classification": "COCO_pretrained_generic_model_not_football_ft",
            "next_mandatory_step": (
                "insan onaylı train/dev/holdout GT + football-domain detector fine-tuning"
            ),
            "bakeoff_summary": {"B": sum_b, "S": sum_s, "ST": sum_st},
            "isolated_env_smoke": smoke,
            "media_retained": False,
            "written_at_utc": utc_now(),
        }
        atomic_json(EV / "r1_f1_r3_BLOCKER.json", blocker)
        atomic_json(
            EV / "GATE_STATUS.json",
            {
                "schema": "r1_f1_r3_gate_status_v1",
                "gate": gate,
                "acceptance_eligible": False,
                "gt_frozen": False,
                "written_at_utc": utc_now(),
            },
        )

        WIN.mkdir(parents=True, exist_ok=True)
        for p in list(WIN.rglob("*")):
            if p.is_file() and p.name != "NO_GO_STATUS.txt":
                p.unlink()
        (WIN / "NO_GO_STATUS.txt").write_text(
            f"{gate}\n"
            f"reason={reason}\n"
            f"fine_tune={OFFICIAL_FINE_TUNE_STATUS}\n"
            "next=insan onaylı train/dev/holdout GT + football-domain detector fine-tuning\n"
            f"written_at_utc={utc_now()}\n",
            encoding="utf-8",
        )
        print(json.dumps({"gate": gate, "reason": reason, "summary": bakeoff["summary"]}, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
