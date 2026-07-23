"""Stage 5B human detection unit tests (mocks preferred; optional model smoke)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root
from football_analytics.perception.adapters.base import RawPersonBox
from football_analytics.perception.detection_evaluation import (
    NOT_EVALUATED,
    bbox_iou,
    evaluate_from_rows,
    evaluate_human_detections,
    greedy_match_iou,
    parse_detection_boxes,
)
from football_analytics.perception.human_detection import (
    filter_raw_person_boxes,
    resolve_source_bbox,
)
from football_analytics.perception.human_detector_config import (
    default_human_detector_config_path,
    human_detector_config_fingerprint,
    load_human_detector_config,
)
from football_analytics.perception.human_fixtures import (
    RUNTIME_ROOT,
    make_analysis_window_row,
    make_frame_rows,
    write_tiny_mp4,
)
from football_analytics.perception.taxonomy import load_detection_taxonomy, map_model_class
from football_analytics.perception.transforms import (
    build_preprocessing_transform,
    clip_bbox_xyxy,
    inverse_bbox,
)

REPO = default_project_root()
PY = sys.executable
MODEL = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
MODEL_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def _unfreeze(value: Any) -> Any:
    from types import MappingProxyType

    if isinstance(value, (MappingProxyType, dict)):
        return {k: _unfreeze(v) for k, v in dict(value).items()}
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    if isinstance(value, list):
        return [_unfreeze(v) for v in value]
    return value


class LazyImportTests(unittest.TestCase):
    def test_perception_import_does_not_load_ultralytics(self) -> None:
        before = {k for k in sys.modules if "ultralytics" in k.lower()}
        import importlib

        importlib.reload(__import__("football_analytics.perception", fromlist=["*"]))
        after = {k for k in sys.modules if "ultralytics" in k.lower()}
        self.assertEqual(after - before, set())


class ConfigAndMappingTests(unittest.TestCase):
    def test_config_fingerprint_stable(self) -> None:
        cfg = load_human_detector_config(default_human_detector_config_path())
        a = human_detector_config_fingerprint(cfg)
        b = human_detector_config_fingerprint(cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_person_maps_human_unknown_never_player(self) -> None:
        tax = load_detection_taxonomy()
        m = map_model_class(0, "person", taxonomy=tax)
        self.assertEqual(m.entity_type.value, "human")
        self.assertEqual(m.role_label.value, "unknown")
        self.assertNotEqual(m.role_label.value, "player")

    def test_non_person_filtered(self) -> None:
        tax = load_detection_taxonomy()
        out = filter_raw_person_boxes(
            [
                RawPersonBox(0, 0, 30, 60, 0.9, 0, "person"),
                RawPersonBox(0, 0, 10, 10, 0.99, 32, "sports_ball"),
                RawPersonBox(0, 0, 20, 40, 0.05, 0, "person"),
            ],
            confidence_threshold=0.25,
            minimum_bbox_area=64.0,
            maximum_aspect_ratio=8.0,
            frame_width=200,
            frame_height=200,
            model_input_size=640,
            taxonomy=tax,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].role_label.value, "unknown")
        self.assertEqual(out[0].entity_type.value, "human")


class TransformFilterTests(unittest.TestCase):
    def test_inverse_letterbox_and_clip(self) -> None:
        tf = build_preprocessing_transform(
            source_width=100, source_height=50, model_input_width=640, model_input_height=640
        )
        src = inverse_bbox((100.0, 100.0, 200.0, 200.0), tf)
        clipped, was = clip_bbox_xyxy(src, frame_width=100, frame_height=50)
        self.assertTrue(was or clipped[0] >= 0)

    def test_invalid_bbox_rejected(self) -> None:
        box, flags, err = resolve_source_bbox(
            RawPersonBox(10, 10, 5, 5, 0.9, 0, "person"),
            frame_width=100,
            frame_height=100,
            boxes_in_source_space=True,
            model_input_size=640,
        )
        self.assertEqual(err, "INVALID_BBOX")


class EvaluationTests(unittest.TestCase):
    def test_iou_matching_duplicates_and_empty(self) -> None:
        preds = parse_detection_boxes(
            [
                {
                    "frame_index": 0,
                    "entity_type": "human",
                    "bbox": [0, 0, 10, 10],
                    "confidence": 0.9,
                },
                {
                    "frame_index": 0,
                    "entity_type": "human",
                    "bbox": [0.5, 0.5, 10.5, 10.5],
                    "confidence": 0.8,
                },
            ]
        )
        gts = parse_detection_boxes(
            [
                {
                    "frame_index": 0,
                    "entity_type": "human",
                    "bbox": [0, 0, 10, 10],
                    "is_reviewed_ground_truth": True,
                }
            ]
        )
        matches = greedy_match_iou(preds, gts, iou_threshold=0.5)
        self.assertEqual(len(matches), 1)
        metrics = evaluate_human_detections(preds, gts, iou_threshold=0.5)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertGreater(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 0.99)

        empty = evaluate_human_detections([], [], iou_threshold=0.5)
        self.assertEqual(empty.true_positives, 0)
        self.assertIsNone(empty.precision)

    def test_no_reviewed_gt_not_evaluated(self) -> None:
        preds = [
            {"frame_index": 0, "entity_type": "human", "bbox": [0, 0, 1, 1], "confidence": 1.0}
        ]
        m = evaluate_from_rows(preds, None)
        self.assertEqual(m.status, NOT_EVALUATED)
        m2 = evaluate_from_rows(
            preds,
            [
                {
                    "frame_index": 0,
                    "entity_type": "human",
                    "bbox": [0, 0, 1, 1],
                    "review_status": "unreviewed",
                }
            ],
        )
        self.assertEqual(m2.status, NOT_EVALUATED)


class AdapterGateTests(unittest.TestCase):
    def test_missing_and_hash_mismatch(self) -> None:
        from football_analytics.perception.adapters.ultralytics_person import (
            UltralyticsPersonAdapter,
            UltralyticsPersonAdapterError,
        )

        adapter = UltralyticsPersonAdapter()
        with self.assertRaises(UltralyticsPersonAdapterError):
            adapter.load("/tmp/does_not_exist_yolo11n.pt", MODEL_SHA)
        if MODEL.is_file():
            with self.assertRaises(UltralyticsPersonAdapterError):
                adapter.load(str(MODEL), "0" * 64)
            with self.assertRaises(UltralyticsPersonAdapterError):
                adapter.load("https://example.com/yolo11n.pt", MODEL_SHA)


class ServiceRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        self.config = _unfreeze(load_human_detector_config(default_human_detector_config_path()))
        self.config["maximum_frames_per_run"] = 4
        self.config["input_size"] = 320

    def _write_inputs(self, session: Path, *, tracking: str = "eligible", n: int = 4):
        video = session / "tiny.mp4"
        write_tiny_mp4(video, n_frames=n)
        rid = generate_run_id()
        frames = make_frame_rows(rid, "vid_t", n)
        windows = [make_analysis_window_row(rid, "vid_t", n_frames=n, tracking=tracking)]
        frames_path = session / "frames.parquet"
        windows_path = session / "windows.parquet"
        write_contract_parquet(
            pa.Table.from_pylist(frames, schema=compile_arrow_schema(get_contract("frames", 1))),
            frames_path,
            get_contract("frames", 1),
            contain_root=RUNTIME_ROOT,
        )
        write_contract_parquet(
            pa.Table.from_pylist(
                windows, schema=compile_arrow_schema(get_contract("analysis_windows", 1))
            ),
            windows_path,
            get_contract("analysis_windows", 1),
            contain_root=RUNTIME_ROOT,
        )
        return video, frames_path, windows_path, rid

    def test_eligible_skipped_with_mock_adapter(self) -> None:
        from football_analytics.perception.detection_service import run_human_detection

        if not MODEL.is_file():
            self.skipTest("model weights absent")
        session = Path(tempfile.mkdtemp(prefix="human_svc_", dir=str(RUNTIME_ROOT)))
        try:
            video, frames_path, windows_path, rid = self._write_inputs(
                session, tracking="ineligible", n=3
            )
            fake = mock.MagicMock()
            fake.is_loaded.return_value = True
            fake.adapter_id = "ultralytics_person"
            fake.adapter_version = "1.0.0"
            fake.software_versions.return_value = {}
            fake.predict_persons.return_value = []
            out = session / "out"
            out.mkdir()
            res = run_human_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_t",
                project_root=REPO,
                adapter=fake,
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertEqual(res.ball_detection_count, 0)
            receipt = json.loads(Path(str(res.receipt_json)).read_text(encoding="utf-8"))
            self.assertEqual(receipt["ball_detection_count"], 0)
            self.assertEqual(fake.predict_persons.call_count, 0)
            self.assertGreater(receipt["skipped_frame_count"], 0)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_atomic_no_overwrite(self) -> None:
        from football_analytics.perception.detection_service import run_human_detection

        if not MODEL.is_file():
            self.skipTest("model weights absent")
        session = Path(tempfile.mkdtemp(prefix="human_ow_", dir=str(RUNTIME_ROOT)))
        try:
            video, frames_path, windows_path, rid = self._write_inputs(session, n=2)
            out = session / "out"
            out.mkdir()
            res1 = run_human_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_t",
                project_root=REPO,
            )
            self.assertTrue(res1.accepted, res1.error_code)
            res2 = run_human_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_t",
                project_root=REPO,
            )
            self.assertFalse(res2.accepted)
            self.assertEqual(res2.error_code, "OVERWRITE_FORBIDDEN")
        finally:
            shutil.rmtree(session, ignore_errors=True)


@unittest.skipUnless(MODEL.is_file(), "yolo11n.pt not present")
class OptionalModelSmokeTests(unittest.TestCase):
    def test_cpu_adapter_smoke(self) -> None:
        import numpy as np

        from football_analytics.perception.adapters.ultralytics_person import (
            UltralyticsPersonAdapter,
        )

        adapter = UltralyticsPersonAdapter()
        adapter.load(str(MODEL), MODEL_SHA)
        try:
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            boxes = adapter.predict_persons(
                img,
                conf=0.25,
                iou=0.5,
                imgsz=320,
                device="cpu",
                half=False,
                class_ids=[0],
                class_names=["person"],
            )
            self.assertIsInstance(boxes, list)
        finally:
            adapter.unload()

    def test_gpu_conditional(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import numpy as np

        from football_analytics.perception.adapters.ultralytics_person import (
            UltralyticsPersonAdapter,
        )

        adapter = UltralyticsPersonAdapter()
        adapter.load(str(MODEL), MODEL_SHA)
        try:
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            boxes = adapter.predict_persons(
                img,
                conf=0.25,
                iou=0.5,
                imgsz=320,
                device="cuda:0",
                half=True,
                class_ids=[0],
                class_names=["person"],
            )
            self.assertIsInstance(boxes, list)
        finally:
            adapter.unload()


class CliAndGitTests(unittest.TestCase):
    def test_cli_help(self) -> None:
        import subprocess

        proc = subprocess.run(
            [PY, "-m", "football_analytics", "perception", "humans", "--help"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("detect", proc.stdout.lower() + proc.stderr.lower())

    def test_no_model_video_tracked(self) -> None:
        import subprocess

        proc = subprocess.run(
            ["git", "ls-files", "*.pt", "*.mp4"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
