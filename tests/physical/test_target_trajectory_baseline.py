#!/usr/bin/env python3
"""Stage 9B target trajectory preparation tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.evidence.collector import is_safe_evidence_file
from football_analytics.physical.trajectory_config import (
    load_trajectory_baseline_config,
    trajectory_baseline_config_fingerprint,
)
from football_analytics.physical.trajectory_evaluation import NOT_EVALUATED_TRAJECTORY
from football_analytics.physical.trajectory_fixtures import (
    continuous_movement_bundle,
    hard_gap_bundle,
    jump_spike_bundle,
    revoked_identity_bundle,
)
from football_analytics.physical.trajectory_service import prepare_target_trajectory


class TargetTrajectoryBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_trajectory_baseline_config()
        self.fp = trajectory_baseline_config_fingerprint(self.cfg)
        self.tmp = Path(tempfile.mkdtemp(prefix="traj9b_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_config_forbids_customer_metrics(self) -> None:
        self.assertTrue(self.cfg["customer_metrics_forbidden"]["distance"])
        self.assertTrue(self.cfg["quality_filter"]["enabled"])
        self.assertTrue(self.cfg["resample"]["enabled"])
        self.assertEqual(self.cfg["attack_direction"], "unknown")

    def test_02_continuous_prepare(self) -> None:
        out = self.tmp / "cont"
        res = prepare_target_trajectory(
            candidates=continuous_movement_bundle(self.fp), output_dir=out, config=self.cfg
        )
        self.assertTrue(res.accepted)
        self.assertGreaterEqual(res.summary["raw_count"], 2)
        self.assertGreaterEqual(res.summary["resampled_count"], 2)
        self.assertEqual(res.summary["evaluation_status"], NOT_EVALUATED_TRAJECTORY)
        self.assertFalse(res.summary["customer_metrics_computed"])
        self.assertTrue(Path(str(res.receipt_json)).is_file())

    def test_03_jump_rejected(self) -> None:
        out = self.tmp / "jump"
        res = prepare_target_trajectory(
            candidates=jump_spike_bundle(self.fp), output_dir=out, config=self.cfg
        )
        self.assertTrue(res.accepted)
        self.assertGreaterEqual(res.summary["rejected_count"], 1)

    def test_04_hard_gap_no_bridge(self) -> None:
        out = self.tmp / "gap"
        res = prepare_target_trajectory(
            candidates=hard_gap_bundle(self.fp), output_dir=out, config=self.cfg
        )
        self.assertTrue(res.accepted)
        self.assertGreaterEqual(res.summary["gap_count"], 1)
        self.assertGreaterEqual(res.summary["segment_count"], 2)

    def test_05_revoked_excluded(self) -> None:
        out = self.tmp / "rev"
        res = prepare_target_trajectory(
            candidates=revoked_identity_bundle(self.fp), output_dir=out, config=self.cfg
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.summary["raw_count"], 0)

    def test_06_deterministic_fingerprint(self) -> None:
        a = trajectory_baseline_config_fingerprint(load_trajectory_baseline_config())
        b = trajectory_baseline_config_fingerprint(load_trajectory_baseline_config())
        self.assertEqual(a, b)

    def test_07_evidence_safety_helper(self) -> None:
        p = self.tmp / "ok.json"
        p.write_text('{"a":1}\n', encoding="utf-8")
        ok, _ = is_safe_evidence_file(p)
        self.assertTrue(ok)
        bad = self.tmp / "x.mp4"
        bad.write_bytes(b"notavideo")
        ok2, reason = is_safe_evidence_file(bad)
        self.assertFalse(ok2)
        self.assertEqual(reason, "unsafe_suffix")


if __name__ == "__main__":
    unittest.main()
