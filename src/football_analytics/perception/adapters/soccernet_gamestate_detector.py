"""Isolated SoccerNet Game State bbox-detector adapter (R1-F1-R3).

Wraps the *official TrackLab / sn-gamestate* person detector settings behind a
subprocess boundary so TrackLab/mmcv are never imported into the host process.

Official config (TrackLab ``yolo_ultralytics``):
  - weights filename: ``yolo11m.pt``
  - taxonomy: COCO person class 0 only (remapped category_id=1 in TrackLab)
  - post-filter ``min_confidence: 0.4``
  - Ultralytics default ``imgsz=640``, predict conf default 0.25 then filtered
  - football / SoccerNet fine-tune: **UNPROVEN** (stock Ultralytics COCO weights)

Import does **not** load weights or import Ultralytics/Torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_analytics.perception.adapters.base import PersonDetectorAdapter, RawPersonBox

ADAPTER_ID = "soccernet_gamestate_detector"
ADAPTER_VERSION = "1.0.0"

# Official TrackLab defaults (read-only inventory; do not invent football FT).
OFFICIAL_WEIGHT_FILENAME = "yolo11m.pt"
OFFICIAL_MIN_CONFIDENCE = 0.4
OFFICIAL_IMGSZ = 640
OFFICIAL_PREDICT_IOU = 0.7
OFFICIAL_PERSON_CLASS_ID = 0
OFFICIAL_FINE_TUNE_STATUS = "UNPROVEN_COCO_PRETRAINED_GENERIC"
OFFICIAL_CONFIG_PATH = (
    "/home/fdoblak/projects/third-party/tracklab/"
    "tracklab/configs/modules/bbox_detector/yolo_ultralytics.yaml"
)
OFFICIAL_PIPELINE_CONFIG = (
    "/home/fdoblak/projects/soccernet/sn-gamestate/" "sn_gamestate/configs/soccernet.yaml"
)
DEFAULT_WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
DEFAULT_LICENSE = "AGPL-3.0"


class SoccerNetGameStateDetectorError(ValueError):
    """Isolated SoccerNet Game State detector adapter failure."""


@dataclass(frozen=True)
class SoccerNetDetectorProvenance:
    adapter_id: str
    adapter_version: str
    weight_filename: str
    weights_path: str | None
    weights_sha256: str | None
    weights_bytes: int | None
    weight_url: str
    license: str
    fine_tune_status: str
    taxonomy: str
    min_confidence: float
    imgsz: int
    predict_iou: float
    official_config: str
    pipeline_config: str
    worker_python: str
    isolated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "weight_filename": self.weight_filename,
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "weight_url": self.weight_url,
            "license": self.license,
            "fine_tune_status": self.fine_tune_status,
            "taxonomy": self.taxonomy,
            "min_confidence": self.min_confidence,
            "imgsz": self.imgsz,
            "predict_iou": self.predict_iou,
            "official_config": self.official_config,
            "pipeline_config": self.pipeline_config,
            "worker_python": self.worker_python,
            "isolated": self.isolated,
            "classification": "COCO_pretrained_generic_model",
            "not": [
                "SoccerNet_fine_tuned_football_detector",
                "baseline_tracker_state",
            ],
        }


def default_worker_python() -> str:
    """Prefer preserved sn-gamestate env; fall back to FA_SOCCERNET_WORKER_PYTHON / ai-dev."""
    env_override = os.environ.get("FA_SOCCERNET_WORKER_PYTHON")
    if env_override:
        return env_override
    sn = Path("/home/fdoblak/miniconda3/envs/sn-gamestate/bin/python")
    ai = Path("/home/fdoblak/miniconda3/envs/ai-dev/bin/python")
    # Prefer sn-gamestate only when Torch is importable there.
    if sn.is_file():
        try:
            r = subprocess.run(
                [str(sn), "-c", "import torch, ultralytics"],
                check=False,
                capture_output=True,
                timeout=30,
            )
            if r.returncode == 0:
                return str(sn)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if ai.is_file():
        return str(ai)
    return "python3"


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "_soccernet_gamestate_worker.py"


class SoccerNetGameStateDetectorAdapter(PersonDetectorAdapter):
    """Subprocess-isolated official Game State person detector settings."""

    def __init__(self, *, worker_python: str | None = None) -> None:
        self._worker_python = worker_python or default_worker_python()
        self._weights_path: Path | None = None
        self._weights_sha256: str | None = None
        self._weights_bytes: int | None = None
        self._loaded = False

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return ADAPTER_VERSION

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, weights_path: str, expected_sha256: str) -> None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise SoccerNetGameStateDetectorError(f"WEIGHTS_MISSING:{path}")
        if path.name != OFFICIAL_WEIGHT_FILENAME:
            raise SoccerNetGameStateDetectorError(
                f"WEIGHT_FILENAME_NOT_OFFICIAL:{path.name}!={OFFICIAL_WEIGHT_FILENAME}"
            )
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise SoccerNetGameStateDetectorError(
                f"WEIGHT_SHA256_MISMATCH:got={digest}:expected={expected_sha256}"
            )
        # Smoke that worker can import without loading into this process.
        probe = subprocess.run(
            [
                self._worker_python,
                str(_worker_script_path()),
                "--probe-imports",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if probe.returncode != 0:
            raise SoccerNetGameStateDetectorError(
                "WORKER_IMPORT_FAILED:" + (probe.stderr or probe.stdout or "")[:800]
            )
        self._weights_path = path
        self._weights_sha256 = digest
        self._weights_bytes = path.stat().st_size
        self._loaded = True

    def unload(self) -> None:
        self._weights_path = None
        self._weights_sha256 = None
        self._weights_bytes = None
        self._loaded = False

    def provenance(self) -> SoccerNetDetectorProvenance:
        return SoccerNetDetectorProvenance(
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
            weight_filename=OFFICIAL_WEIGHT_FILENAME,
            weights_path=str(self._weights_path) if self._weights_path else None,
            weights_sha256=self._weights_sha256,
            weights_bytes=self._weights_bytes,
            weight_url=DEFAULT_WEIGHT_URL,
            license=DEFAULT_LICENSE,
            fine_tune_status=OFFICIAL_FINE_TUNE_STATUS,
            taxonomy="coco_person_class_0_only",
            min_confidence=OFFICIAL_MIN_CONFIDENCE,
            imgsz=OFFICIAL_IMGSZ,
            predict_iou=OFFICIAL_PREDICT_IOU,
            official_config=OFFICIAL_CONFIG_PATH,
            pipeline_config=OFFICIAL_PIPELINE_CONFIG,
            worker_python=self._worker_python,
            isolated=True,
        )

    def software_versions(self) -> Mapping[str, str]:
        if not self._loaded:
            return {"worker_python": self._worker_python}
        r = subprocess.run(
            [self._worker_python, str(_worker_script_path()), "--versions"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            return {"worker_python": self._worker_python, "error": "versions_failed"}
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"worker_python": self._worker_python, "raw": r.stdout[:200]}

    def predict_persons(
        self,
        image_bgr_or_rgb: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        half: bool,
        class_ids: Sequence[int],
        class_names: Sequence[str],
        channel_order: str = "bgr",
    ) -> list[RawPersonBox]:
        if not self._loaded or self._weights_path is None or self._weights_sha256 is None:
            raise SoccerNetGameStateDetectorError("MODEL_NOT_LOADED")
        # Official settings win; callers may pass matching values.
        min_conf = max(float(conf), OFFICIAL_MIN_CONFIDENCE)
        use_imgsz = int(imgsz) if imgsz > 0 else OFFICIAL_IMGSZ
        use_iou = float(iou) if iou > 0 else OFFICIAL_PREDICT_IOU

        arr = np.asarray(image_bgr_or_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise SoccerNetGameStateDetectorError("IMAGE_SHAPE_INVALID")
        if channel_order.lower() == "rgb":
            arr = arr[:, :, ::-1].copy()
        elif channel_order.lower() != "bgr":
            raise SoccerNetGameStateDetectorError(f"CHANNEL_ORDER_UNSUPPORTED:{channel_order}")

        with tempfile.TemporaryDirectory(prefix="sn_gs_det_") as td:
            td_path = Path(td)
            frame_path = td_path / "frame.png"
            out_path = td_path / "out.json"
            # Encode via OpenCV in worker would require host cv2; use numpy PNG via PIL-free path.
            # Prefer cv2 if available in host (ai-dev has it); else raw .npy.
            try:
                import cv2

                ok = cv2.imwrite(str(frame_path), arr)
                if not ok:
                    raise SoccerNetGameStateDetectorError("FRAME_WRITE_FAILED")
                frame_arg = str(frame_path)
                frame_fmt = "bgr_png"
            except Exception:
                npy_path = td_path / "frame.npy"
                np.save(str(npy_path), arr)
                frame_arg = str(npy_path)
                frame_fmt = "bgr_npy"

            cmd = [
                self._worker_python,
                str(_worker_script_path()),
                "--weights",
                str(self._weights_path),
                "--expected-sha256",
                self._weights_sha256,
                "--frame",
                frame_arg,
                "--frame-format",
                frame_fmt,
                "--out",
                str(out_path),
                "--min-confidence",
                str(min_conf),
                "--imgsz",
                str(use_imgsz),
                "--iou",
                str(use_iou),
                "--device",
                str(device),
                "--half",
                "1" if half else "0",
            ]
            # class filter stays person-only regardless of caller names.
            _ = class_ids, class_names
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                raise SoccerNetGameStateDetectorError(
                    "WORKER_INFER_FAILED:" + (proc.stderr or proc.stdout or "")[:1200]
                )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            boxes: list[RawPersonBox] = []
            for row in payload.get("boxes", []):
                boxes.append(
                    RawPersonBox(
                        x1=float(row["x1"]),
                        y1=float(row["y1"]),
                        x2=float(row["x2"]),
                        y2=float(row["y2"]),
                        score=float(row["score"]),
                        class_id=int(row.get("class_id", OFFICIAL_PERSON_CLASS_ID)),
                        class_name=str(row.get("class_name", "person")),
                    )
                )
            return boxes


def get_soccernet_gamestate_detector_adapter(
    *, worker_python: str | None = None
) -> SoccerNetGameStateDetectorAdapter:
    return SoccerNetGameStateDetectorAdapter(worker_python=worker_python)


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "OFFICIAL_FINE_TUNE_STATUS",
    "OFFICIAL_IMGSZ",
    "OFFICIAL_MIN_CONFIDENCE",
    "OFFICIAL_WEIGHT_FILENAME",
    "SoccerNetDetectorProvenance",
    "SoccerNetGameStateDetectorAdapter",
    "SoccerNetGameStateDetectorError",
    "default_worker_python",
    "get_soccernet_gamestate_detector_adapter",
]
