"""Tests for SoccerNet Game State isolated detector adapter (R1-F1-R3)."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from football_analytics.perception.adapters.soccernet_gamestate_detector import (
    OFFICIAL_FINE_TUNE_STATUS,
    OFFICIAL_MIN_CONFIDENCE,
    OFFICIAL_WEIGHT_FILENAME,
    SoccerNetGameStateDetectorAdapter,
    SoccerNetGameStateDetectorError,
)


class SoccerNetGameStateAdapterTests(unittest.TestCase):
    def test_import_does_not_load_model(self) -> None:
        mod = importlib.import_module(
            "football_analytics.perception.adapters.soccernet_gamestate_detector"
        )
        ad = mod.SoccerNetGameStateDetectorAdapter(worker_python="/bin/false")
        self.assertFalse(ad.is_loaded())
        self.assertEqual(ad.provenance().fine_tune_status, OFFICIAL_FINE_TUNE_STATUS)
        self.assertEqual(ad.provenance().min_confidence, OFFICIAL_MIN_CONFIDENCE)

    def test_rejects_non_official_filename(self) -> None:
        ad = SoccerNetGameStateDetectorAdapter(worker_python="/bin/false")
        with self.assertRaises(SoccerNetGameStateDetectorError):
            ad.load("/tmp/yolo11n.pt", "0" * 64)

    def test_predict_requires_load(self) -> None:
        ad = SoccerNetGameStateDetectorAdapter(worker_python="/bin/false")
        with self.assertRaises(SoccerNetGameStateDetectorError):
            ad.predict_persons(
                np.zeros((64, 64, 3), dtype=np.uint8),
                conf=0.4,
                iou=0.7,
                imgsz=640,
                device="cpu",
                half=False,
                class_ids=[0],
                class_names=["person"],
            )

    def test_xyxy_passthrough_from_worker_json(self) -> None:
        ad = SoccerNetGameStateDetectorAdapter(worker_python="python")
        ad._loaded = True  # noqa: SLF001
        ad._weights_path = Path("/tmp/yolo11m.pt")  # noqa: SLF001
        ad._weights_sha256 = "abc"  # noqa: SLF001

        payload = {
            "boxes": [
                {
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 30.0,
                    "y2": 40.0,
                    "score": 0.9,
                    "class_id": 0,
                    "class_name": "person",
                }
            ]
        }

        class _Proc:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(__import__("json").dumps(payload), encoding="utf-8")
            return _Proc()

        with mock.patch("subprocess.run", side_effect=fake_run):
            boxes = ad.predict_persons(
                np.zeros((100, 100, 3), dtype=np.uint8),
                conf=0.4,
                iou=0.7,
                imgsz=640,
                device="cpu",
                half=False,
                class_ids=[0],
                class_names=["person"],
            )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(
            (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2), (10.0, 20.0, 30.0, 40.0)
        )
        self.assertEqual(OFFICIAL_WEIGHT_FILENAME, "yolo11m.pt")


if __name__ == "__main__":
    unittest.main()
