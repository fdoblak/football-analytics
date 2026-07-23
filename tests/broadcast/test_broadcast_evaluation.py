"""Evaluation-focused tests (re-export module coverage)."""

from __future__ import annotations

import unittest

from football_analytics.broadcast.broadcast_evaluation import (
    evaluate_broadcast_windows,
    passes_safety_gates,
)


class BroadcastEvaluationTests(unittest.TestCase):
    def test_01_manual_review_recall(self) -> None:
        gt = [
            {
                "analysis_window_id": "aw_0000",
                "start_time_us": 0,
                "end_time_us": 10,
                "playability": "uncertain",
                "replay_status": "unknown",
                "tracking_eligibility": "unknown",
                "calibration_eligibility": "unknown",
                "identity_eligibility": "unknown",
                "ball_analysis_eligibility": "unknown",
                "live_event_eligibility": "unknown",
                "physical_metric_eligibility": "unknown",
                "decision_codes": ["UNKNOWN_VIEW_REVIEW_REQUIRED"],
                "manual_review_required": True,
            }
        ]
        pred = [{**gt[0]}]
        report = evaluate_broadcast_windows(pred, gt, repeat_predictions=pred)
        self.assertEqual(report.manual_review_recall, 1.0)
        ok, fails = passes_safety_gates(report)
        self.assertTrue(ok, fails)

    def test_02_overlap_detected(self) -> None:
        rows = [
            {
                "analysis_window_id": "aw_0000",
                "start_time_us": 0,
                "end_time_us": 100,
                "playability": "playable",
                "replay_status": "live",
                "tracking_eligibility": "eligible",
                "calibration_eligibility": "eligible",
                "identity_eligibility": "unknown",
                "ball_analysis_eligibility": "eligible",
                "live_event_eligibility": "eligible",
                "physical_metric_eligibility": "eligible",
                "decision_codes": ["PLAYABLE_WIDE_VIEW"],
                "manual_review_required": False,
            },
            {
                "analysis_window_id": "aw_0001",
                "start_time_us": 50,
                "end_time_us": 150,
                "playability": "playable",
                "replay_status": "live",
                "tracking_eligibility": "eligible",
                "calibration_eligibility": "eligible",
                "identity_eligibility": "unknown",
                "ball_analysis_eligibility": "eligible",
                "live_event_eligibility": "eligible",
                "physical_metric_eligibility": "eligible",
                "decision_codes": ["PLAYABLE_WIDE_VIEW"],
                "manual_review_required": False,
            },
        ]
        report = evaluate_broadcast_windows(rows, rows)
        self.assertGreater(report.overlap_rate or 0, 0)


if __name__ == "__main__":
    unittest.main()
