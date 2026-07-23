"""Unit tests for shot boundary evaluation matching."""

from __future__ import annotations

import unittest

from football_analytics.broadcast.shot_evaluation import (
    BoundaryEvent,
    evaluate_boundaries,
    greedy_match,
)


class ShotEvaluationTests(unittest.TestCase):
    def test_perfect_match(self) -> None:
        preds = [BoundaryEvent(100000, "hard_cut"), BoundaryEvent(500000, "fade")]
        gts = [BoundaryEvent(100000, "hard_cut"), BoundaryEvent(500000, "fade")]
        m = evaluate_boundaries(preds, gts, tolerance_us=200000, duration_us=1_000_000)
        self.assertEqual(m.true_positives, 2)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.false_negatives, 0)
        self.assertEqual(m.f1, 1.0)

    def test_tolerance_edge(self) -> None:
        preds = [BoundaryEvent(300000, "hard_cut")]
        gts = [BoundaryEvent(100000, "hard_cut")]
        m = evaluate_boundaries(preds, gts, tolerance_us=200000)
        self.assertEqual(m.true_positives, 1)
        m2 = evaluate_boundaries(preds, gts, tolerance_us=199999)
        self.assertEqual(m2.true_positives, 0)
        self.assertEqual(m2.false_positives, 1)
        self.assertEqual(m2.false_negatives, 1)

    def test_no_double_counting(self) -> None:
        preds = [BoundaryEvent(100000), BoundaryEvent(120000)]
        gts = [BoundaryEvent(110000)]
        matches = greedy_match(preds, gts, tolerance_us=200000)
        self.assertEqual(len(matches), 1)
        m = evaluate_boundaries(preds, gts, tolerance_us=200000)
        self.assertEqual(m.true_positives, 1)
        self.assertEqual(m.false_positives, 1)
        self.assertEqual(m.false_negatives, 0)

    def test_null_metrics_when_empty(self) -> None:
        m = evaluate_boundaries([], [], tolerance_us=200000)
        self.assertIsNone(m.precision)
        self.assertIsNone(m.recall)
        self.assertIsNone(m.f1)

    def test_shot_count_error(self) -> None:
        preds = [BoundaryEvent(100000), BoundaryEvent(200000)]
        gts = [BoundaryEvent(100000)]
        m = evaluate_boundaries(preds, gts, tolerance_us=50000)
        self.assertEqual(m.over_segmentation, 1)
        self.assertEqual(m.under_segmentation, 0)
        self.assertEqual(m.shot_count_error, 1)

    def test_deterministic_order(self) -> None:
        preds = [BoundaryEvent(250000), BoundaryEvent(100000)]
        gts = [BoundaryEvent(100000), BoundaryEvent(250000)]
        a = greedy_match(preds, gts, tolerance_us=10000)
        b = greedy_match(list(reversed(preds)), list(reversed(gts)), tolerance_us=10000)
        self.assertEqual(
            [(m.pred_index, m.gt_index, m.abs_error_us) for m in a],
            [(m.pred_index, m.gt_index, m.abs_error_us) for m in b],
        )


if __name__ == "__main__":
    unittest.main()
