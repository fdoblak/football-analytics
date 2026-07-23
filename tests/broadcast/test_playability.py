"""Unit tests for Stage 4D playability routing."""

from __future__ import annotations

import unittest
from pathlib import Path

from football_analytics.broadcast.playability import (
    build_review_queue,
    load_routing_policy,
    route_fused_window,
    routing_policy_fingerprint,
)
from football_analytics.broadcast.segment_fusion import FusedWindow
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.registry import default_project_root


def _fw(**kwargs):
    base = dict(
        run_id=generate_run_id(),
        video_id="v1",
        start_time_us=0,
        end_time_us=1000,
        start_frame_index=None,
        end_frame_index_exclusive=None,
        shot_id="s1",
        camera_segment_ids=("c1",),
        view_family="main_broadcast",
        framing_scale="wide",
        replay_status="live",
        graphics_status="none",
        playability="playable",
        coverage=1.0,
        confidence=0.9,
        timeline_mapping_quality="exact_identity",
        is_gap=False,
        is_conflict=False,
        source_refs=("c1", "s1"),
    )
    base.update(kwargs)
    return FusedWindow(**base)


class PlayabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_routing_policy(
            default_project_root() / "configs/broadcast/broadcast_routing_policy.yaml"
        )

    def test_01_policy_fingerprint_stable(self) -> None:
        a = routing_policy_fingerprint(self.policy)
        b = routing_policy_fingerprint(self.policy)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_02_wide_playable(self) -> None:
        row = route_fused_window(_fw(), self.policy, analysis_window_id="aw_0000")
        self.assertEqual(row["tracking_eligibility"], "eligible")
        self.assertEqual(row["calibration_eligibility"], "eligible")
        self.assertIn("PLAYABLE_WIDE_VIEW", row["decision_codes"])

    def test_03_closeup_identity_only(self) -> None:
        row = route_fused_window(
            _fw(view_family="player_isolation", framing_scale="close_up"),
            self.policy,
            analysis_window_id="aw_0001",
        )
        self.assertEqual(row["identity_eligibility"], "eligible")
        self.assertEqual(row["calibration_eligibility"], "ineligible")
        self.assertEqual(row["physical_metric_eligibility"], "ineligible")

    def test_04_graphics_block(self) -> None:
        row = route_fused_window(
            _fw(
                view_family="graphics",
                graphics_status="full_screen",
                playability="non_playable",
                framing_scale="unknown",
            ),
            self.policy,
            analysis_window_id="aw_0002",
        )
        for axis in (
            "tracking_eligibility",
            "calibration_eligibility",
            "live_event_eligibility",
            "physical_metric_eligibility",
        ):
            self.assertEqual(row[axis], "ineligible")

    def test_05_replay_unknown_blocks_live(self) -> None:
        row = route_fused_window(
            _fw(replay_status="unknown"),
            self.policy,
            analysis_window_id="aw_0003",
        )
        self.assertNotEqual(row["live_event_eligibility"], "eligible")
        self.assertTrue(row["manual_review_required"])
        self.assertIn("REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING", row["decision_codes"])

    def test_06_replay_confirmed(self) -> None:
        row = route_fused_window(
            _fw(replay_status="replay", playability="partially_playable"),
            self.policy,
            analysis_window_id="aw_0004",
        )
        self.assertEqual(row["live_event_eligibility"], "ineligible")
        self.assertNotEqual(row["physical_metric_eligibility"], "eligible")

    def test_07_mapping_unsafe(self) -> None:
        row = route_fused_window(
            _fw(timeline_mapping_quality="uncertain"),
            self.policy,
            analysis_window_id="aw_0005",
        )
        self.assertNotEqual(row["physical_metric_eligibility"], "eligible")
        self.assertIn("TIMELINE_MAPPING_INSUFFICIENT", row["decision_codes"])

    def test_08_gap_review(self) -> None:
        row = route_fused_window(
            _fw(is_gap=True, camera_segment_ids=(), view_family="unknown", playability="uncertain"),
            self.policy,
            analysis_window_id="aw_0006",
        )
        self.assertTrue(row["manual_review_required"])
        self.assertIn("CAMERA_GAP", row["decision_codes"])

    def test_09_review_queue_dedupe(self) -> None:
        rows = [
            route_fused_window(
                _fw(is_gap=True, camera_segment_ids=(), playability="uncertain"),
                self.policy,
                analysis_window_id="aw_x",
            ),
            route_fused_window(
                _fw(is_gap=True, camera_segment_ids=(), playability="uncertain"),
                self.policy,
                analysis_window_id="aw_x",
            ),
        ]
        q = build_review_queue(rows, policy_version="1")
        self.assertEqual(len(q["items"]), 1)
        self.assertEqual(q["items"][0]["status"], "pending")

    def test_10_policy_path_exists(self) -> None:
        p = default_project_root() / "configs/broadcast/broadcast_routing_policy.yaml"
        self.assertTrue(Path(p).is_file())


if __name__ == "__main__":
    unittest.main()
