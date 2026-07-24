#!/usr/bin/env python3
"""Stage 9D heatmap / zones / activity baseline tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.physical.activity import classify_speed_mps, compute_activity_distribution
from football_analytics.physical.heatmap import (
    compute_time_weighted_heatmap,
    smooth_conserve_mass,
)
from football_analytics.physical.spatial_config import (
    load_spatial_baseline_config,
    spatial_baseline_config_fingerprint,
)
from football_analytics.physical.spatial_evaluation import NOT_EVALUATED_SPATIAL
from football_analytics.physical.spatial_fixtures import (
    penalty_presence_points,
    speed_class_ladder,
    stationary_zone_dwell,
)
from football_analytics.physical.spatial_service import compute_spatial_metrics
from football_analytics.physical.zone_occupancy import compute_zone_occupancy


class HeatmapActivityBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_spatial_baseline_config()
        self.fp = spatial_baseline_config_fingerprint(self.cfg)
        self.tmp = Path(tempfile.mkdtemp(prefix="spatial9d_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_config(self) -> None:
        self.assertEqual(self.cfg["stage"], "9D")
        self.assertEqual(self.cfg["attack_direction"], "unknown")
        self.assertFalse(self.cfg["output_policy"]["write_visuals_to_git"])
        self.assertTrue(self.cfg["coverage"]["missing_coverage_is_not_inactive"])

    def test_02_time_weighted_heatmap(self) -> None:
        pts = stationary_zone_dwell(self.fp)
        hm = compute_time_weighted_heatmap(pts, config=self.cfg)
        self.assertEqual(hm.status, "computed")
        self.assertGreater(hm.total_dwell_seconds, 0.0)
        self.assertAlmostEqual(hm.mass_before_smooth, hm.mass_after_smooth, places=6)

    def test_03_smoothing_mass(self) -> None:
        g = [[1.0, 0.0], [0.0, 2.0]]
        out = smooth_conserve_mass(g, sigma_cells=1.0, radius_cells=1)
        self.assertAlmostEqual(sum(sum(r) for r in g), sum(sum(r) for r in out), places=9)

    def test_04_zones_and_penalty_semantics(self) -> None:
        z = compute_zone_occupancy(penalty_presence_points(self.fp), config=self.cfg)
        self.assertEqual(z["attack_direction"], "unknown")
        pens = [r for r in z["zones"] if r["zone_id"] == "goal_a_penalty"]
        self.assertTrue(pens)
        self.assertTrue(pens[0]["not_touch_or_possession"])

    def test_05_activity_classes(self) -> None:
        classes = self.cfg["activity"]["classes"]
        self.assertEqual(classify_speed_mps(8.0, classes=classes), "sprinting")
        self.assertEqual(classify_speed_mps(0.2, classes=classes), "stationary")
        act = compute_activity_distribution(speed_class_ladder(self.fp), config=self.cfg)
        self.assertEqual(act["status"], "computed")
        self.assertFalse(act["missing_coverage_counted_as_inactive"])

    def test_06_service_receipt(self) -> None:
        out = self.tmp / "run"
        res = compute_spatial_metrics(
            primary_points=stationary_zone_dwell(self.fp),
            output_dir=out,
            config=self.cfg,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.summary["evaluation_status"], NOT_EVALUATED_SPATIAL)
        self.assertFalse(res.summary["visuals_committed_to_git"])
        self.assertTrue(Path(str(res.receipt_json)).is_file())
        self.assertTrue(Path(str(res.heatmap_json)).is_file())


if __name__ == "__main__":
    unittest.main()
