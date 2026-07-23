"""BBox / preprocessing transform tests."""

from __future__ import annotations

import math
import unittest

from football_analytics.perception.contracts import (
    load_perception_json_schema,
    validate_against_json_schema,
)
from football_analytics.perception.transforms import (
    TransformError,
    build_preprocessing_transform,
    clip_bbox_xyxy,
    forward_bbox,
    inverse_bbox,
    roundtrip_bbox,
    stretch_params,
    validate_bbox_xyxy,
)
from football_analytics.perception.types import ResizeMode


class DetectionTransformTests(unittest.TestCase):
    def test_01_letterbox_roundtrip(self) -> None:
        t = build_preprocessing_transform(
            source_width=1280,
            source_height=720,
            model_input_width=640,
            model_input_height=640,
            resize_mode=ResizeMode.LETTERBOX,
        )
        bbox = (100.0, 50.0, 200.0, 300.0)
        back = roundtrip_bbox(bbox, t)
        for a, b in zip(bbox, back, strict=True):
            self.assertLessEqual(abs(a - b), t.roundtrip_tolerance_px)
        validate_against_json_schema(
            t.to_dict(), load_perception_json_schema("preprocessing_transform")
        )

    def test_02_stretch_inverse(self) -> None:
        t = build_preprocessing_transform(
            source_width=100,
            source_height=50,
            model_input_width=200,
            model_input_height=200,
            resize_mode="stretch",
        )
        params = stretch_params(100, 50, 200, 200)
        self.assertEqual(t.scale_x, params["scale_x"])
        bbox = (10.0, 5.0, 40.0, 25.0)
        fwd = forward_bbox(bbox, t)
        inv = inverse_bbox(fwd, t)
        for a, b in zip(bbox, inv, strict=True):
            self.assertAlmostEqual(a, b, places=5)

    def test_03_reject_nan_inf_zero_area(self) -> None:
        with self.assertRaises(TransformError):
            validate_bbox_xyxy((float("nan"), 0.0, 1.0, 1.0))
        with self.assertRaises(TransformError):
            validate_bbox_xyxy((0.0, 0.0, float("inf"), 1.0))
        with self.assertRaises(TransformError):
            validate_bbox_xyxy((10.0, 10.0, 10.0, 20.0))
        with self.assertRaises(TransformError):
            validate_bbox_xyxy((10.0, 20.0, 5.0, 30.0))

    def test_04_clip_and_bounds(self) -> None:
        clipped, was = clip_bbox_xyxy((-5.0, -2.0, 50.0, 60.0), frame_width=40, frame_height=50)
        self.assertTrue(was)
        self.assertEqual(clipped, (0.0, 0.0, 40.0, 50.0))
        with self.assertRaises(TransformError):
            validate_bbox_xyxy((0.0, 0.0, 50.0, 10.0), frame_width=40, frame_height=20)

    def test_05_small_ball_bbox(self) -> None:
        t = build_preprocessing_transform(
            source_width=1280,
            source_height=720,
            model_input_width=640,
            model_input_height=640,
        )
        ball = (640.0, 360.0, 646.0, 366.0)
        validate_bbox_xyxy(ball, frame_width=1280, frame_height=720)
        back = roundtrip_bbox(ball, t)
        self.assertTrue(all(math.isfinite(v) for v in back))
        self.assertLess(back[2] - back[0], 20)

    def test_06_fingerprint_changes_with_mode(self) -> None:
        a = build_preprocessing_transform(
            source_width=1280,
            source_height=720,
            model_input_width=640,
            model_input_height=640,
            resize_mode="letterbox",
        )
        b = build_preprocessing_transform(
            source_width=1280,
            source_height=720,
            model_input_width=640,
            model_input_height=640,
            resize_mode="stretch",
        )
        self.assertNotEqual(a.transform_fingerprint, b.transform_fingerprint)


if __name__ == "__main__":
    unittest.main()
