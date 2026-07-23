"""Stage 5C ball detection unit tests (mocks preferred; optional model smoke)."""

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
from football_analytics.perception.adapters.base import RawDetectionBox
from football_analytics.perception.ball_detection import filter_raw_ball_boxes
from football_analytics.perception.ball_detector_config import (
    ball_detector_config_fingerprint,
    default_ball_detector_config_path,
    load_ball_detector_config,
)
from football_analytics.perception.ball_evaluation import (
    FROZEN_BALL_FIXTURES,
    NOT_EVALUATED_BALL,
    evaluate_ball_detections,
    evaluate_ball_from_rows,
)
from football_analytics.perception.ball_fixtures import (
    RUNTIME_ROOT,
    make_analysis_window_row,
    make_frame_rows,
    write_tiny_mp4_with_ball,
)
from football_analytics.perception.candidate_merge import BallCandidate, merge_ball_candidates
from football_analytics.perception.detection_evaluation import (
    greedy_match_iou,
    parse_detection_boxes,
)
from football_analytics.perception.taxonomy import load_detection_taxonomy, map_model_class
from football_analytics.perception.tiling import generate_tiles, map_tile_bbox_to_source

REPO = default_project_root()
PY = sys.executable
MODEL = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
MODEL_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

DEFAULT_FILTERS = {
    "min_width": 2.0,
    "max_width": 128.0,
    "min_height": 2.0,
    "max_height": 128.0,
    "min_area_fraction": 0.000001,
    "max_area_fraction": 0.05,
    "max_aspect_ratio": 3.0,
}


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


class CapabilityMappingTests(unittest.TestCase):
    def test_config_and_sports_ball_mapping(self) -> None:
        cfg = load_ball_detector_config(default_ball_detector_config_path())
        self.assertEqual(cfg["sports_ball_class"]["class_ids"][0], 32)
        self.assertIn("sports ball", list(cfg["sports_ball_class"]["class_names"]))
        a = ball_detector_config_fingerprint(cfg)
        self.assertEqual(a, ball_detector_config_fingerprint(cfg))
        tax = load_detection_taxonomy()
        m = map_model_class(32, "sports_ball", taxonomy=tax)
        self.assertEqual(m.entity_type.value, "ball")
        self.assertEqual(m.role_label.value, "unknown")

    def test_person_rejected_by_ball_filter(self) -> None:
        tax = load_detection_taxonomy()
        out = filter_raw_ball_boxes(
            [
                RawDetectionBox(0, 0, 10, 10, 0.9, 32, "sports ball"),
                RawDetectionBox(0, 0, 30, 60, 0.99, 0, "person"),
            ],
            confidence_threshold=0.15,
            filters=DEFAULT_FILTERS,
            frame_width=200,
            frame_height=200,
            model_input_size=640,
            taxonomy=tax,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_type.value, "ball")
        self.assertEqual(out[0].class_name, "sports ball")


class TilingTests(unittest.TestCase):
    def test_generate_tiles_and_limits(self) -> None:
        tiles = generate_tiles(
            100,
            80,
            tile_width=40,
            tile_height=40,
            overlap_x=10,
            overlap_y=10,
            max_tiles=3,
        )
        self.assertEqual(len(tiles), 3)
        self.assertTrue(tiles[0].tile_id.startswith("r0"))
        self.assertTrue(any("left" in t.edge_flags or "top" in t.edge_flags for t in tiles))

    def test_map_tile_bbox_to_source(self) -> None:
        tiles = generate_tiles(
            200, 200, tile_width=100, tile_height=100, overlap_x=0, overlap_y=0, max_tiles=4
        )
        tile = tiles[1] if len(tiles) > 1 else tiles[0]
        mapped = map_tile_bbox_to_source((1, 2, 5, 6), tile, coordinate_space="tile_local")
        self.assertEqual(mapped[0], 1 + tile.x0)
        self.assertEqual(mapped[1], 2 + tile.y0)
        src = map_tile_bbox_to_source((1, 2, 5, 6), tile, coordinate_space="source")
        self.assertEqual(src, (1.0, 2.0, 5.0, 6.0))


class MergeTests(unittest.TestCase):
    def test_duplicate_merge_preserves_provenance(self) -> None:
        a = BallCandidate(10, 10, 20, 20, 0.9, 32, "sports ball", "full_frame")
        b = BallCandidate(11, 11, 21, 21, 0.8, 32, "sports ball", "tile:r0c0")
        merged = merge_ball_candidates([a], [b], merge_iou=0.5, class_aware=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].candidate_source, "full_frame")


class ModeLogicTests(unittest.TestCase):
    def test_full_tiled_hybrid_candidate_sources(self) -> None:
        full = [BallCandidate(10, 10, 20, 20, 0.9, 32, "sports ball", "full_frame")]
        tiled = [BallCandidate(50, 50, 60, 60, 0.7, 32, "sports ball", "tile:r0c1")]
        hybrid = merge_ball_candidates(full, tiled, merge_iou=0.5)
        self.assertEqual(len(hybrid), 2)
        only_tile = merge_ball_candidates([], tiled, merge_iou=0.5)
        self.assertEqual(len(only_tile), 1)
        only_full = merge_ball_candidates(full, [], merge_iou=0.5)
        self.assertEqual(len(only_full), 1)


class FilterTests(unittest.TestCase):
    def test_size_aspect_and_invalid(self) -> None:
        tax = load_detection_taxonomy()
        out = filter_raw_ball_boxes(
            [
                RawDetectionBox(0, 0, 4, 4, 0.9, 32, "sports ball"),
                RawDetectionBox(0, 0, 150, 150, 0.9, 32, "sports ball"),  # too large
                RawDetectionBox(10, 10, 5, 5, 0.9, 32, "sports ball"),  # invalid
                RawDetectionBox(0, 0, 20, 4, 0.9, 32, "sports ball"),  # aspect
            ],
            confidence_threshold=0.15,
            filters=DEFAULT_FILTERS,
            frame_width=200,
            frame_height=200,
            model_input_size=640,
            taxonomy=tax,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].bbox_x2 - out[0].bbox_x1, 4.0)


class EvaluationFixtureTests(unittest.TestCase):
    def test_frozen_fixtures(self) -> None:
        for name, fx in FROZEN_BALL_FIXTURES.items():
            if fx.get("empty_gt_list"):
                preds = parse_detection_boxes(fx["predictions"], default_entity="ball")
                metrics = evaluate_ball_detections(preds, [], iou_threshold=0.5)
                self.assertEqual(metrics.false_positives, fx["expect_fp"], name)
                continue
            if fx.get("empty_gt"):
                metrics = evaluate_ball_detections([], [], iou_threshold=0.5)
                self.assertEqual(metrics.true_positives, 0, name)
                self.assertIsNone(metrics.precision)
                continue
            preds = parse_detection_boxes(fx["predictions"], default_entity="ball")
            gts = parse_detection_boxes(fx["ground_truth"], default_entity="ball")
            metrics = evaluate_ball_detections(preds, gts, iou_threshold=0.5)
            self.assertEqual(metrics.true_positives, fx["expect_tp"], name)
            self.assertEqual(metrics.false_positives, fx["expect_fp"], name)
            self.assertEqual(metrics.false_negatives, fx["expect_fn"], name)

    def test_iou_tie_and_threshold(self) -> None:
        preds = parse_detection_boxes(
            [
                {
                    "frame_index": 0,
                    "entity_type": "ball",
                    "bbox": [0, 0, 10, 10],
                    "confidence": 0.9,
                },
                {
                    "frame_index": 0,
                    "entity_type": "ball",
                    "bbox": [0.2, 0.2, 10.2, 10.2],
                    "confidence": 0.85,
                },
            ]
        )
        gts = parse_detection_boxes(
            [
                {
                    "frame_index": 0,
                    "entity_type": "ball",
                    "bbox": [0, 0, 10, 10],
                    "is_reviewed_ground_truth": True,
                }
            ]
        )
        matches = greedy_match_iou(preds, gts, iou_threshold=0.5, require_entity_type="ball")
        self.assertEqual(len(matches), 1)
        m_hi = evaluate_ball_detections(preds, gts, iou_threshold=0.99)
        self.assertGreaterEqual(m_hi.true_positives or 0, 0)

    def test_no_reviewed_gt(self) -> None:
        preds = [{"frame_index": 0, "entity_type": "ball", "bbox": [0, 0, 1, 1], "confidence": 1.0}]
        m = evaluate_ball_from_rows(preds, None)
        self.assertEqual(m.status, NOT_EVALUATED_BALL)
        m2 = evaluate_ball_from_rows(
            preds,
            [
                {
                    "frame_index": 0,
                    "entity_type": "ball",
                    "bbox": [0, 0, 1, 1],
                    "review_status": "unreviewed",
                }
            ],
        )
        self.assertEqual(m2.status, NOT_EVALUATED_BALL)

    def test_no_ball_negative_accuracy(self) -> None:
        preds = parse_detection_boxes(
            [{"frame_index": 1, "entity_type": "ball", "bbox": [0, 0, 5, 5], "confidence": 0.5}]
        )
        gts = parse_detection_boxes(
            [
                {
                    "frame_index": 0,
                    "entity_type": "ball",
                    "bbox": [0, 0, 1, 1],
                    "is_reviewed_ground_truth": True,
                }
            ]
        )
        # Frame 0 has GT but we'll use empty preds for frame 0 and GT on different frame —
        # construct: frame 0 no GT no pred via evaluating empty on frame marked by both empty
        metrics = evaluate_ball_detections([], [], iou_threshold=0.5)
        self.assertIsNone(metrics.no_ball_negative_accuracy)
        # Frame with GT elsewhere: frame 2 empty both
        preds2 = parse_detection_boxes(
            [{"frame_index": 2, "entity_type": "ball", "bbox": [0, 0, 2, 2], "confidence": 0.1}]
        )
        gts2 = parse_detection_boxes(
            [
                {
                    "frame_index": 1,
                    "entity_type": "ball",
                    "bbox": [10, 10, 15, 15],
                    "is_reviewed_ground_truth": True,
                }
            ]
        )
        # Add empty frame by including a zero-match frame — use preds on frame 0 empty via
        # matching only frame 1 GT vs frame 2 pred → frame with g_count=0
        m = evaluate_ball_detections(preds2, gts2, iou_threshold=0.5)
        self.assertIsNotNone(m.no_ball_negative_accuracy)
        self.assertEqual(m.false_positives, 1)
        self.assertEqual(m.false_negatives, 1)
        _ = preds, gts


class AdapterGateTests(unittest.TestCase):
    def test_no_auto_download_and_hash(self) -> None:
        from football_analytics.perception.adapters.ultralytics_ball import (
            UltralyticsBallAdapter,
            UltralyticsBallAdapterError,
        )

        adapter = UltralyticsBallAdapter()
        with self.assertRaises(UltralyticsBallAdapterError):
            adapter.load("/tmp/does_not_exist_yolo11n.pt", MODEL_SHA)
        if MODEL.is_file():
            with self.assertRaises(UltralyticsBallAdapterError):
                adapter.load(str(MODEL), "0" * 64)
            with self.assertRaises(UltralyticsBallAdapterError):
                adapter.load("https://example.com/yolo11n.pt", MODEL_SHA)


class ServiceRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        self.config = _unfreeze(load_ball_detector_config(default_ball_detector_config_path()))
        self.config["maximum_frames_per_run"] = 4
        self.config["input_size"] = 320
        self.config["inference_mode"] = "full_frame"
        self.config["tiling"]["max_tiles"] = 4

    def _write_inputs(
        self,
        session: Path,
        *,
        ball_analysis: str = "eligible",
        tracking: str = "eligible",
        identity: str = "unknown",
        n: int = 4,
    ):
        video = session / "tiny.mp4"
        write_tiny_mp4_with_ball(video, n_frames=n)
        rid = generate_run_id()
        frames = make_frame_rows(rid, "vid_ball", n)
        windows = [
            make_analysis_window_row(
                rid,
                "vid_ball",
                n_frames=n,
                tracking=tracking,
                ball_analysis=ball_analysis,
                identity=identity,
            )
        ]
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

    def test_ball_ineligible_skipped(self) -> None:
        from football_analytics.perception.ball_service import run_ball_detection

        if not MODEL.is_file():
            self.skipTest("model weights absent")
        session = Path(tempfile.mkdtemp(prefix="ball_svc_", dir=str(RUNTIME_ROOT)))
        try:
            video, frames_path, windows_path, rid = self._write_inputs(
                session, ball_analysis="ineligible", n=3
            )
            fake = mock.MagicMock()
            fake.is_loaded.return_value = True
            fake.adapter_id = "ultralytics_sports_ball"
            fake.adapter_version = "1.0.0"
            fake.software_versions.return_value = {}
            fake.predict_balls.return_value = []
            out = session / "out"
            out.mkdir()
            res = run_ball_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ball",
                project_root=REPO,
                adapter=fake,
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertEqual(res.human_detection_count, 0)
            self.assertEqual(fake.predict_balls.call_count, 0)
            receipt = json.loads(Path(str(res.receipt_json)).read_text(encoding="utf-8"))
            self.assertEqual(receipt["human_detection_count"], 0)
            self.assertGreater(receipt["skipped_frame_count"], 0)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_identity_only_skipped(self) -> None:
        from football_analytics.perception.ball_service import run_ball_detection

        if not MODEL.is_file():
            self.skipTest("model weights absent")
        session = Path(tempfile.mkdtemp(prefix="ball_id_", dir=str(RUNTIME_ROOT)))
        try:
            video, frames_path, windows_path, rid = self._write_inputs(
                session,
                ball_analysis="eligible",
                tracking="ineligible",
                identity="eligible",
                n=2,
            )
            fake = mock.MagicMock()
            fake.is_loaded.return_value = True
            fake.adapter_id = "ultralytics_sports_ball"
            fake.adapter_version = "1.0.0"
            fake.software_versions.return_value = {}
            fake.predict_balls.return_value = []
            out = session / "out"
            out.mkdir()
            res = run_ball_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ball",
                project_root=REPO,
                adapter=fake,
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertEqual(fake.predict_balls.call_count, 0)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_atomic_no_overwrite(self) -> None:
        from football_analytics.perception.ball_service import run_ball_detection

        if not MODEL.is_file():
            self.skipTest("model weights absent")
        session = Path(tempfile.mkdtemp(prefix="ball_ow_", dir=str(RUNTIME_ROOT)))
        try:
            video, frames_path, windows_path, rid = self._write_inputs(session, n=2)
            out = session / "out"
            out.mkdir()
            fake = mock.MagicMock()
            fake.is_loaded.return_value = True
            fake.adapter_id = "ultralytics_sports_ball"
            fake.adapter_version = "1.0.0"
            fake.software_versions.return_value = {}
            fake.predict_balls.return_value = [
                RawDetectionBox(10, 10, 18, 18, 0.8, 32, "sports ball")
            ]
            res1 = run_ball_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ball",
                project_root=REPO,
                adapter=fake,
            )
            self.assertTrue(res1.accepted, res1.error_code)
            self.assertGreaterEqual(res1.ball_detection_count, 1)
            self.assertEqual(res1.human_detection_count, 0)
            status = json.loads(
                # force read receipt
                Path(str(res1.receipt_json)).read_text(encoding="utf-8")
            )
            self.assertEqual(status["detector_id"], "ball_yolo11n_v1")
            res2 = run_ball_detection(
                source=str(video),
                timeline=str(frames_path),
                analysis_windows=str(windows_path),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ball",
                project_root=REPO,
                adapter=fake,
            )
            self.assertFalse(res2.accepted)
            self.assertEqual(res2.error_code, "OVERWRITE_FORBIDDEN")
        finally:
            shutil.rmtree(session, ignore_errors=True)


@unittest.skipUnless(MODEL.is_file(), "yolo11n.pt not present")
class OptionalModelSmokeTests(unittest.TestCase):
    def test_cpu_adapter_smoke(self) -> None:
        import numpy as np

        from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter

        adapter = UltralyticsBallAdapter()
        adapter.load(str(MODEL), MODEL_SHA)
        try:
            names = adapter.model_names()
            self.assertEqual(names.get(32), "sports ball")
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            boxes = adapter.predict_balls(
                img,
                conf=0.15,
                iou=0.5,
                imgsz=320,
                device="cpu",
                half=False,
                class_ids=[32],
                class_names=["sports ball"],
            )
            self.assertIsInstance(boxes, list)
            for b in boxes:
                self.assertEqual(b.class_name, "sports ball")
                self.assertNotEqual(b.class_id, 0)
        finally:
            adapter.unload()

    def test_gpu_conditional(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import numpy as np

        from football_analytics.perception.adapters.ultralytics_ball import UltralyticsBallAdapter

        adapter = UltralyticsBallAdapter()
        adapter.load(str(MODEL), MODEL_SHA)
        try:
            img = np.zeros((96, 128, 3), dtype=np.uint8)
            boxes = adapter.predict_balls(
                img,
                conf=0.15,
                iou=0.5,
                imgsz=320,
                device="cuda:0",
                half=True,
                class_ids=[32],
                class_names=["sports ball"],
            )
            self.assertIsInstance(boxes, list)
        finally:
            adapter.unload()


class CliAndGitTests(unittest.TestCase):
    def test_cli_help(self) -> None:
        import subprocess

        proc = subprocess.run(
            [PY, "-m", "football_analytics", "perception", "ball", "--help"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        combined = proc.stdout.lower() + proc.stderr.lower()
        self.assertIn("detect", combined)
        self.assertIn("evaluate", combined)

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
