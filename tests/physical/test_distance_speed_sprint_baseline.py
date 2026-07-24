#!/usr/bin/env python3
"""Stage 9C distance / speed / sprint baseline tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.physical.distance import compute_segment_distance
from football_analytics.physical.motion_config import (
    load_motion_baseline_config,
    motion_baseline_config_fingerprint,
)
from football_analytics.physical.motion_evaluation import NOT_EVALUATED_MOTION
from football_analytics.physical.motion_fixtures import (
    constant_speed_points,
    hard_gap_two_segments,
    single_sprint_points,
)
from football_analytics.physical.motion_service import compute_physical_motion
from football_analytics.physical.speed import compute_segment_speeds, mps_to_kmh
from football_analytics.physical.sprint import (
    count_evaluable_sprints,
    extract_sprint_bouts_for_segment,
)


class DistanceSpeedSprintBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_motion_baseline_config()
        self.fp = motion_baseline_config_fingerprint(self.cfg)
        self.tmp = Path(tempfile.mkdtemp(prefix="motion9c_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_config_metadata(self) -> None:
        self.assertEqual(self.cfg["stage"], "9C")
        self.assertEqual(self.cfg["metric_origin"], "project_generated")
        self.assertEqual(self.cfg["definition_style"], "opta_style_metric_definition")
        self.assertEqual(self.cfg["primary_sample_layer"], "filtered")
        self.assertTrue(self.cfg["sprint"]["not_official_opta"])
        self.assertFalse(self.cfg["coverage"]["extrapolate_uncovered_time"])

    def test_02_constant_speed_distance(self) -> None:
        pts = constant_speed_points(self.fp, speed_mps=5.0, n=8)
        d = compute_segment_distance(
            pts, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=self.cfg
        )
        self.assertEqual(d.status, "computed")
        self.assertAlmostEqual(float(d.distance_m or -1), 3.5, places=6)

    def test_03_speed_from_video_time(self) -> None:
        pts = constant_speed_points(self.fp, speed_mps=5.0, n=8)
        s = compute_segment_speeds(
            pts, trajectory_segment_id="traj_seg_01", sample_layer="filtered", config=self.cfg
        )
        self.assertEqual(s.status, "computed")
        self.assertAlmostEqual(float(s.robust_mean_mps or -1), 5.0, places=2)
        self.assertAlmostEqual(mps_to_kmh(5.0), 18.0, places=6)

    def test_04_hard_gap_no_bridge(self) -> None:
        gap = hard_gap_two_segments(self.fp)
        out = self.tmp / "gap"
        res = compute_physical_motion(
            primary_points=gap["traj_seg_a"] + gap["traj_seg_b"],
            output_dir=out,
            config=self.cfg,
        )
        self.assertTrue(res.accepted)
        da = compute_segment_distance(
            gap["traj_seg_a"],
            trajectory_segment_id="traj_seg_a",
            sample_layer="filtered",
            config=self.cfg,
        )
        db = compute_segment_distance(
            gap["traj_seg_b"],
            trajectory_segment_id="traj_seg_b",
            sample_layer="filtered",
            config=self.cfg,
        )
        self.assertAlmostEqual(
            float(res.summary.get("measured_distance_m") or -1),
            float(da.distance_m or 0) + float(db.distance_m or 0),
            places=5,
        )

    def test_05_sprint_and_receipt(self) -> None:
        out = self.tmp / "sprint"
        pts = single_sprint_points(self.fp)
        res = compute_physical_motion(primary_points=pts, output_dir=out, config=self.cfg)
        self.assertTrue(res.accepted)
        self.assertGreaterEqual(int(res.summary.get("sprint_count") or 0), 1)
        self.assertEqual(res.summary.get("evaluation_status"), NOT_EVALUATED_MOTION)
        self.assertTrue(Path(str(res.receipt_json)).is_file())
        bouts = extract_sprint_bouts_for_segment(
            pts,
            trajectory_segment_id="traj_seg_sprint1",
            sample_layer="filtered",
            config=self.cfg,
            config_fingerprint=self.fp,
        )
        self.assertEqual(count_evaluable_sprints(bouts)["sprint_count"], 1)
        self.assertEqual(bouts[0].metric_origin, "project_generated")

    def test_06_deterministic_fingerprint(self) -> None:
        a = motion_baseline_config_fingerprint(load_motion_baseline_config())
        b = motion_baseline_config_fingerprint(load_motion_baseline_config())
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
