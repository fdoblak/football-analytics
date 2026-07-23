"""Unit tests for camera sampling."""

from __future__ import annotations

import unittest

from football_analytics.broadcast.camera_config import (
    default_camera_config_path,
    load_camera_view_config,
)
from football_analytics.broadcast.camera_sampling import plan_sample_points
from football_analytics.broadcast.shot_features import build_cfr_timeline


class CameraSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_camera_view_config(default_camera_config_path())

    def test_equal_time_deterministic(self) -> None:
        tl = build_cfr_timeline(frame_count=50, fps_num=25, fps_den=1)
        a = plan_sample_points(tl, start_time_us=0, end_time_us=2_000_000, config=self.config)
        b = plan_sample_points(tl, start_time_us=0, end_time_us=2_000_000, config=self.config)
        self.assertEqual(a, b)
        self.assertGreaterEqual(len(a), int(self.config["sampling"]["min_samples"]))
        self.assertLessEqual(len(a), int(self.config["sampling"]["max_samples"]))
        times = [p.time_us for p in a]
        self.assertEqual(times, sorted(times))
        # Within shot
        for p in a:
            self.assertGreaterEqual(p.time_us, 0)
            self.assertLess(p.time_us, 2_000_000)

    def test_edge_margin_avoids_ends(self) -> None:
        tl = build_cfr_timeline(frame_count=100, fps_num=25, fps_den=1)
        pts = plan_sample_points(tl, start_time_us=0, end_time_us=4_000_000, config=self.config)
        edge = float(self.config["sampling"]["edge_exclude_fraction"])
        margin = int(round(4_000_000 * edge))
        for p in pts:
            self.assertGreaterEqual(p.time_us, margin)
            self.assertLess(p.time_us, 4_000_000 - margin)


if __name__ == "__main__":
    unittest.main()
