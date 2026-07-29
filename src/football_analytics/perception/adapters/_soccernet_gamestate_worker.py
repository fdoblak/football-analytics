#!/usr/bin/env python3
"""Subprocess worker for SoccerNet Game State official YOLO person detector.

Runs in the isolated worker Python (prefer sn-gamestate env when Torch exists).
Does not import football_analytics. Outputs source-space xyxy JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_imports() -> int:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401

    print(json.dumps({"ok": True}))
    return 0


def versions() -> int:
    import torch
    import ultralytics

    out = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "ultralytics": ultralytics.__version__,
    }
    print(json.dumps(out))
    return 0


def infer(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    digest = _sha256(weights)
    if digest.lower() != args.expected_sha256.lower():
        print(f"SHA_MISMATCH got={digest}", file=sys.stderr)
        return 2

    if args.frame_format == "bgr_png":
        import cv2

        frame = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if frame is None:
            print("FRAME_READ_FAILED", file=sys.stderr)
            return 3
    elif args.frame_format == "bgr_npy":
        frame = np.load(args.frame)
    else:
        print(f"BAD_FRAME_FORMAT:{args.frame_format}", file=sys.stderr)
        return 4

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))
    # TrackLab calls model(images) without overriding conf/iou; defaults apply,
    # then TrackLab filters bbox.conf >= min_confidence (0.4).
    results = model.predict(
        source=frame,
        conf=0.25,
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        device=device,
        half=bool(int(args.half)) and device != "cpu",
        classes=[0],
        verbose=False,
    )
    boxes = []
    min_conf = float(args.min_confidence)
    if results:
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy()
            h, w = frame.shape[:2]
            for (x1, y1, x2, y2), sc, cl in zip(xyxy, confs, clss, strict=True):
                if int(cl) != 0:
                    continue
                if float(sc) < min_conf:
                    continue
                x1 = float(max(0.0, min(w, x1)))
                y1 = float(max(0.0, min(h, y1)))
                x2 = float(max(0.0, min(w, x2)))
                y2 = float(max(0.0, min(h, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(sc),
                        "class_id": 0,
                        "class_name": "person",
                    }
                )

    payload = {
        "boxes": boxes,
        "provenance": {
            "weight": str(weights),
            "sha256": digest,
            "min_confidence": min_conf,
            "imgsz": int(args.imgsz),
            "iou": float(args.iou),
            "device": device,
            "fine_tune_status": "UNPROVEN_COCO_PRETRAINED_GENERIC",
            "source_space": "xyxy",
            "image_wh": [int(frame.shape[1]), int(frame.shape[0])],
        },
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-imports", action="store_true")
    p.add_argument("--versions", action="store_true")
    p.add_argument("--weights")
    p.add_argument("--expected-sha256")
    p.add_argument("--frame")
    p.add_argument("--frame-format", default="bgr_png")
    p.add_argument("--out")
    p.add_argument("--min-confidence", default="0.4")
    p.add_argument("--imgsz", default="640")
    p.add_argument("--iou", default="0.7")
    p.add_argument("--device", default="auto")
    p.add_argument("--half", default="0")
    args = p.parse_args()
    if args.probe_imports:
        return probe_imports()
    if args.versions:
        return versions()
    if not args.weights or not args.expected_sha256 or not args.frame or not args.out:
        print("MISSING_ARGS", file=sys.stderr)
        return 1
    return infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
