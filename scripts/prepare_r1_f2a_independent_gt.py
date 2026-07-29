#!/usr/bin/env python3
"""Prepare R1-F2-A independent GT runtime workspace (Git-external).

- Selects ~80 stratified frames (train/dev/holdout time-isolated)
- Generates YOLO11n-hybrid proposals for TRAIN only (never GT)
- Writes selected_frames.json, draft_annotations.json, progress, session_state
- Does NOT freeze; does NOT copy source video
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import cv2

from football_analytics.annotation.frame_selection import build_independent_gt_selection
from football_analytics.annotation.independent_gt import (
    DEFAULT_RUNTIME,
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    atomic_write_json,
    empty_draft,
    sha256_file,
    utc_now,
)
from football_analytics.perception.adapters.ultralytics_person import UltralyticsPersonAdapter
from football_analytics.perception.human_tiled_detection import HumanDetectConfig, detect_humans

ARCH = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
SHA_N = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
REPO = Path(__file__).resolve().parents[1]


def _chmod_tree(root: Path, mode: int = 0o700) -> None:
    with contextlib.suppress(OSError):
        root.chmod(mode)
    for p in root.rglob("*"):
        with contextlib.suppress(OSError):
            p.chmod(0o600 if p.is_file() else 0o700)


def generate_train_proposals(
    video: Path,
    train_frames: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    if not ARCH.is_file():
        raise SystemExit(f"missing baseline weights: {ARCH}")
    if sha256_file(ARCH) != SHA_N:
        raise SystemExit("yolo11n sha mismatch")
    adapter = UltralyticsPersonAdapter()
    adapter.load(str(ARCH), SHA_N)
    cfg = HumanDetectConfig(
        name="train_proposal_yolo11n_hybrid",
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
    want = {int(f["frame_idx"]) for f in train_frames}
    i = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
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
                        "proposal_id": f"p_{i}_{j}",
                        "bbox_xyxy": [p.x1, p.y1, p.x2, p.y2],
                        "score": float(p.score),
                        "eligibility_hint": elig,
                        "origin": "proposal_unreviewed",
                        "class_name": "human",
                        "note": "PROPOSAL_ONLY_NOT_GT",
                    }
                )
            # Cap extreme proposal density for UI usability (still not GT).
            boxes.sort(key=lambda b: -b["score"])
            boxes = boxes[:24]
            out[i] = boxes
            want.remove(i)
        i += 1
    cap.release()
    adapter.unload()
    return out


def write_windows_package(win_dir: Path, *, port: int = 8766) -> None:
    del port  # port is fixed at 8766 in the canonical ASCII BAT template
    import importlib.util
    import sys

    mod_path = Path(__file__).resolve().parent / "r1_gt_windows_package.py"
    spec = importlib.util.spec_from_file_location("r1_gt_windows_package", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r1_gt_windows_package"] = mod
    spec.loader.exec_module(mod)
    mod.write_windows_package(win_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--skip-proposals", action="store_true")
    ap.add_argument(
        "--windows-dir",
        type=Path,
        default=Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT"),
    )
    args = ap.parse_args()

    video = args.video
    digest = sha256_file(video)
    if digest != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"SOURCE_SHA_MISMATCH {digest}")
    if oct(video.stat().st_mode)[-3:] != "444":
        # warn only — mode may display differently
        pass

    runtime: Path = args.runtime
    runtime.mkdir(parents=True, exist_ok=True)
    cache = runtime / "frame_cache"
    cache.mkdir(exist_ok=True)

    selection = build_independent_gt_selection()
    # Repo-side manifest (no images) for git
    repo_manifest = (
        REPO / "artifacts" / "evidence" / "reboot_01" / "independent_gt" / "selected_frames.json"
    )
    atomic_write_json(repo_manifest, selection, mode=0o644)

    sel_path = runtime / "selected_frames.json"
    atomic_write_json(sel_path, selection)

    proposals: dict[int, list[dict[str, Any]]] = {}
    if not args.skip_proposals:
        train_frames = [f for f in selection["frames"] if f["split"] == "train"]
        proposals = generate_train_proposals(video, train_frames)

    frames_for_draft = []
    for fr in selection["frames"]:
        row = dict(fr)
        if fr["split"] == "train":
            row["proposals"] = proposals.get(int(fr["frame_idx"]), [])
        else:
            row["proposals"] = []
        frames_for_draft.append(row)

    audit_path = runtime / "review_audit.jsonl"
    if not audit_path.is_file():
        audit_path.write_text("", encoding="utf-8")
        with contextlib.suppress(OSError):
            audit_path.chmod(0o600)

    draft_path = runtime / "draft_annotations.json"
    if draft_path.is_file():
        # Resume: keep existing draft; refresh proposals only for empty train frames
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        by_idx = {int(f["frame_idx"]): f for f in draft.get("frames", [])}
        for fr in frames_for_draft:
            idx = int(fr["frame_idx"])
            if idx not in by_idx:
                continue
            if fr["split"] == "train" and not by_idx[idx].get("humans"):
                by_idx[idx]["proposals"] = fr["proposals"]
        draft["updated_at_utc"] = utc_now()
        atomic_write_json(draft_path, draft)
    else:
        draft = empty_draft(
            video=video,
            source_sha256=digest,
            frames=frames_for_draft,
            audit_log_path=str(audit_path),
        )
        atomic_write_json(draft_path, draft)

    progress = {
        "schema": "independent_gt_progress_v1",
        "n_frames": selection["counts"]["total"],
        "n_complete": 0,
        "by_split": {
            "train": {"n": selection["counts"]["train"], "complete": 0},
            "dev": {"n": selection["counts"]["dev"], "complete": 0},
            "holdout": {"n": selection["counts"]["holdout"], "complete": 0},
        },
        "updated_at_utc": utc_now(),
    }
    # recompute if resuming
    d = json.loads(draft_path.read_text(encoding="utf-8"))
    for fr in d["frames"]:
        if fr.get("completed"):
            progress["n_complete"] += 1
            progress["by_split"][fr["split"]]["complete"] += 1
    atomic_write_json(runtime / "progress.json", progress)

    session = {
        "schema": "independent_gt_session_state_v1",
        "index": 0,
        "video": str(video),
        "source_sha256": digest,
        "runtime": str(runtime),
        "frame_cache_dir": str(cache),
        "frame_cache_cleanup_after_review": True,
        "bind": "127.0.0.1",
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(runtime / "session_state.json", session)
    _chmod_tree(runtime, 0o700)

    write_windows_package(args.windows_dir)

    gate = {
        "schema": "r1_f2a_gate_status_v1",
        "gate": "PASS — INDEPENDENT HUMAN GT REVIEW TOOL READY",
        "acceptance_eligible": False,
        "human_approved": False,
        "reviewed_gt": False,
        "frozen": False,
        "runtime": str(runtime),
        "windows_dir": str(args.windows_dir),
        "n_frames": selection["counts"],
        "written_at_utc": utc_now(),
    }
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        gate,
        mode=0o644,
    )
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/independent_gt/PREPARE_RECEIPT.json",
        {
            "schema": "r1_f2a_prepare_receipt_v1",
            "source_sha256": digest,
            "runtime": str(runtime),
            "n_train_proposals_frames": len(proposals),
            "n_proposal_boxes": sum(len(v) for v in proposals.values()),
            "video_copied": False,
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )
    print(json.dumps(gate, indent=2))
    print("uid", os.getuid(), "runtime", runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
