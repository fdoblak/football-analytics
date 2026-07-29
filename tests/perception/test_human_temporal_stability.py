"""Tests for temporal human stabilizer (R1-F1-R2)."""

from __future__ import annotations

import unittest

import numpy as np

from football_analytics.perception.human_temporal_stability import (
    TemporalHumanStabilizer,
    compute_temporal_diagnostics,
)
from football_analytics.perception.human_tiled_detection import HumanProposal


class TemporalStabilityTests(unittest.TestCase):
    def test_max_carry_two_frames(self) -> None:
        stab = TemporalHumanStabilizer(max_carry_frames=2, match_iou=0.3)
        frame = np.zeros((744, 1336, 3), dtype=np.uint8)
        frame[:] = (40, 120, 40)
        p = HumanProposal(100, 100, 140, 200, 0.9, "on_pitch_human_candidate", "full")
        o0 = stab.update(frame, [p])
        self.assertEqual(o0[0].temporal_status, "observed")
        o1 = stab.update(frame, [])
        self.assertEqual(len(o1), 1)
        self.assertEqual(o1[0].temporal_status, "carried")
        o2 = stab.update(frame, [])
        self.assertEqual(len(o2), 1)
        self.assertEqual(o2[0].temporal_status, "carried")
        o3 = stab.update(frame, [])
        self.assertEqual(len(o3), 0)

    def test_reject_carry_gt_two(self) -> None:
        with self.assertRaises(ValueError):
            TemporalHumanStabilizer(max_carry_frames=3)

    def test_diagnostics_flag(self) -> None:
        d = compute_temporal_diagnostics([])
        self.assertTrue(d["diagnostic_not_accuracy"])


if __name__ == "__main__":
    unittest.main()
