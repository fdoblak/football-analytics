"""R1-F1 canonical coordinate round-trip and hard-fail tests."""

from __future__ import annotations

import unittest

from football_analytics.annotation.coordinates import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    CoordinateError,
    canvas_bbox_to_source,
    make_letterbox_transform,
    make_source_bbox,
    make_stretch_transform,
    reject_xywh,
    roundtrip_error_px,
    source_bbox_to_canvas,
    source_point_to_canvas,
    validate_source_bbox_xyxy,
)


class CoordinateRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = make_letterbox_transform(
            SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_WIDTH, SOURCE_HEIGHT
        )
        self.letterbox = make_letterbox_transform(SOURCE_WIDTH, SOURCE_HEIGHT, 1280, 720)
        self.small_canvas = make_letterbox_transform(SOURCE_WIDTH, SOURCE_HEIGHT, 800, 450)
        self.stretch = make_stretch_transform(SOURCE_WIDTH, SOURCE_HEIGHT, 1000, 500)

    def _assert_rt(self, box: list[float], transform) -> None:
        err = roundtrip_error_px(box, transform)
        self.assertLessEqual(err, 1.0, msg=f"box={box} err={err} fp={transform.fingerprint}")

    def test_01_source_corners(self) -> None:
        corners = [
            [0.0, 0.0, 1.0, 1.0],
            [SOURCE_WIDTH - 1.0, 0.0, SOURCE_WIDTH, 1.0],
            [0.0, SOURCE_HEIGHT - 1.0, 1.0, SOURCE_HEIGHT],
            [SOURCE_WIDTH - 1.0, SOURCE_HEIGHT - 1.0, SOURCE_WIDTH, SOURCE_HEIGHT],
        ]
        for box in corners:
            validate_source_bbox_xyxy(box)
            for t in (self.identity, self.letterbox, self.small_canvas, self.stretch):
                self._assert_rt(box, t)

    def test_02_center_point_roundtrip(self) -> None:
        cx, cy = SOURCE_WIDTH / 2.0, SOURCE_HEIGHT / 2.0
        for t in (self.identity, self.letterbox, self.small_canvas):
            dx, dy = source_point_to_canvas(cx, cy, t)
            back = canvas_bbox_to_source([dx, dy, dx + 1, dy + 1], t)
            self.assertLessEqual(abs(back[0] - cx), 1.0)
            self.assertLessEqual(abs(back[1] - cy), 1.0)

    def test_03_full_frame_bbox(self) -> None:
        box = [0.0, 0.0, float(SOURCE_WIDTH), float(SOURCE_HEIGHT)]
        for t in (self.identity, self.letterbox, self.small_canvas, self.stretch):
            self._assert_rt(box, t)

    def test_04_small_player_bbox(self) -> None:
        box = [640.0, 300.0, 668.0, 360.0]
        for t in (self.identity, self.letterbox, self.small_canvas):
            self._assert_rt(box, t)

    def test_05_bottom_right_edge_bbox(self) -> None:
        box = [1280.0, 680.0, 1336.0, 744.0]
        for t in (self.identity, self.letterbox, self.small_canvas):
            self._assert_rt(box, t)

    def test_06_letterboxed_view(self) -> None:
        box = [100.0, 120.0, 160.0, 240.0]
        canvas = source_bbox_to_canvas(box, self.letterbox)
        # content should be centered: pad >= 0
        self.assertGreaterEqual(self.letterbox.pad_x, 0.0)
        self.assertGreaterEqual(self.letterbox.pad_y, 0.0)
        back = canvas_bbox_to_source(canvas, self.letterbox)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(box, back, strict=True)), 1.0)

    def test_07_different_browser_canvas_sizes(self) -> None:
        box = [220.0, 180.0, 280.0, 320.0]
        for cw, ch in ((1920, 1080), (1024, 768), (640, 360), (1336, 744)):
            t = make_letterbox_transform(SOURCE_WIDTH, SOURCE_HEIGHT, cw, ch)
            self._assert_rt(box, t)

    def test_08_canvas_source_canvas_roundtrip(self) -> None:
        box = [400.0, 250.0, 460.0, 390.0]
        c1 = source_bbox_to_canvas(box, self.small_canvas)
        s = canvas_bbox_to_source(c1, self.small_canvas)
        c2 = source_bbox_to_canvas(s, self.small_canvas)
        err = max(abs(a - b) for a, b in zip(c1, c2, strict=True))
        self.assertLessEqual(err, 1.0)

    def test_09_xywh_hard_fail(self) -> None:
        with self.assertRaises(CoordinateError):
            reject_xywh([10, 20, 30, 40], format_tag="xywh")
        # Ambiguous small x2/y2 looking like width/height
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([100.0, 100.0, 40.0, 80.0])

    def test_10_out_of_bounds_hard_fail(self) -> None:
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([-1.0, 0.0, 10.0, 10.0])
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([0.0, 0.0, 1400.0, 10.0])

    def test_11_zero_area_hard_fail(self) -> None:
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([10.0, 10.0, 10.0, 20.0])

    def test_12_nan_inf_hard_fail(self) -> None:
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([float("nan"), 0.0, 1.0, 1.0])
        with self.assertRaises(CoordinateError):
            validate_source_bbox_xyxy([0.0, 0.0, float("inf"), 1.0])

    def test_13_make_source_bbox_fields(self) -> None:
        bb = make_source_bbox(
            frame_index=660,
            fps=30.0,
            bbox_xyxy=[10.0, 20.0, 50.0, 100.0],
            transform=self.letterbox,
        )
        self.assertEqual(bb.frame_index, 660)
        self.assertEqual(bb.video_time_us, 22_000_000)
        self.assertEqual(bb.source_width, SOURCE_WIDTH)
        self.assertEqual(bb.coordinate_space, "source_xyxy_px_v1")
        self.assertEqual(bb.transform_fingerprint, self.letterbox.fingerprint)


if __name__ == "__main__":
    unittest.main()
