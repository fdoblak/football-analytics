"""Unit tests for camera evaluation metrics."""

from __future__ import annotations

import unittest

from football_analytics.broadcast.camera_evaluation import (
    evaluate_axis,
    evaluate_camera_predictions,
    unsafe_playable_fp_rate,
)


class CameraEvaluationTests(unittest.TestCase):
    def test_perfect_axis(self) -> None:
        y = ["main_broadcast", "graphics", "player_isolation"]
        m = evaluate_axis(y, y, axis="view_family", labels=y + ["unknown"])
        self.assertEqual(m.status, "ok")
        self.assertEqual(m.macro_f1, 1.0)
        self.assertEqual(m.abstention_rate, 0.0)

    def test_null_when_empty(self) -> None:
        m = evaluate_axis([], [], axis="view_family")
        self.assertEqual(m.status, "not_evaluable")
        self.assertIsNone(m.macro_f1)

    def test_unsafe_playable_fp(self) -> None:
        rate = unsafe_playable_fp_rate(
            ["non_playable", "playable", "non_playable"],
            ["playable", "playable", "non_playable"],
        )
        self.assertEqual(rate, 0.5)
        self.assertIsNone(unsafe_playable_fp_rate(["playable"], ["playable"]))

    def test_pairing_and_ood(self) -> None:
        preds = [
            {
                "video_id": "v1",
                "shot_id": "s1",
                "view_family": "main_broadcast",
                "framing_scale": "wide",
                "camera_motion": "static",
                "graphics_status": "none",
                "playability": "playable",
            },
            {
                "video_id": "v2",
                "shot_id": "s2",
                "fixture_id": "ood_crowd",
                "view_family": "unknown",
                "framing_scale": "unknown",
                "camera_motion": "unknown",
                "graphics_status": "unknown",
                "playability": "uncertain",
            },
        ]
        gts = [
            {
                "video_id": "v1",
                "shot_id": "s1",
                "view_family": "main_broadcast",
                "framing_scale": "wide",
                "camera_motion": "static",
                "graphics_status": "none",
                "playability": "playable",
            },
            {
                "video_id": "v2",
                "shot_id": "s2",
                "fixture_id": "ood_crowd",
                "is_ood": True,
                "view_family": "unknown",
                "framing_scale": "unknown",
                "camera_motion": "unknown",
                "graphics_status": "unknown",
                "playability": "uncertain",
            },
        ]
        report = evaluate_camera_predictions(preds, gts, ood_fixture_ids=["ood_crowd"])
        self.assertEqual(report.n_pairs, 2)
        self.assertEqual(report.ood_abstention_rate, 1.0)
        self.assertEqual(report.unsafe_playable_false_positive_rate, None)


if __name__ == "__main__":
    unittest.main()
