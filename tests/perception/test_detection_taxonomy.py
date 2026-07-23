"""Taxonomy + policy loader / mapping tests."""

from __future__ import annotations

import unittest

from football_analytics.perception.policy import (
    load_detection_policy,
    policy_fingerprint,
    resolve_frame_routing,
)
from football_analytics.perception.taxonomy import (
    TaxonomyError,
    load_detection_taxonomy,
    map_model_class,
    taxonomy_fingerprint,
)
from football_analytics.perception.types import EntityType, RoleLabel


class DetectionTaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tax = load_detection_taxonomy()
        self.pol = load_detection_policy()

    def test_01_fingerprint_stable(self) -> None:
        self.assertEqual(taxonomy_fingerprint(self.tax), taxonomy_fingerprint(self.tax))
        self.assertEqual(policy_fingerprint(self.pol), policy_fingerprint(self.pol))
        self.assertEqual(len(taxonomy_fingerprint(self.tax)), 64)

    def test_02_person_never_player(self) -> None:
        for name in ("person", "Person", "human", "pedestrian"):
            got = map_model_class(0, name, taxonomy=self.tax)
            self.assertEqual(got.entity_type, EntityType.HUMAN)
            self.assertEqual(got.role_label, RoleLabel.UNKNOWN)
            self.assertTrue(got.mapped)

    def test_03_explicit_player_and_ball(self) -> None:
        player = map_model_class(1, "player", taxonomy=self.tax)
        self.assertEqual(player.role_label, RoleLabel.PLAYER)
        ball = map_model_class(32, "sports_ball", taxonomy=self.tax)
        self.assertEqual(ball.entity_type, EntityType.BALL)
        self.assertEqual(ball.role_label, RoleLabel.UNKNOWN)

    def test_04_unmapped_rejects(self) -> None:
        got = map_model_class(99, "spaceship", taxonomy=self.tax)
        self.assertTrue(got.rejected)
        self.assertFalse(got.mapped)

    def test_05_auto_player_flag_false(self) -> None:
        self.assertFalse(self.tax["auto_player_from_person"])

    def test_06_routing_playable_vs_graphics(self) -> None:
        playable = {
            "playability": "playable",
            "graphics_status": "none",
            "tracking_eligibility": "eligible",
            "identity_eligibility": "conditionally_eligible",
            "ball_analysis_eligibility": "eligible",
        }
        r = resolve_frame_routing(playable, policy=self.pol)
        self.assertTrue(r["process_human"])
        self.assertTrue(r["process_ball"])
        graphics = {
            "playability": "non_playable",
            "graphics_status": "full_screen",
            "tracking_eligibility": "ineligible",
            "identity_eligibility": "ineligible",
            "ball_analysis_eligibility": "ineligible",
        }
        g = resolve_frame_routing(graphics, policy=self.pol)
        self.assertFalse(g["process_human"])
        self.assertEqual(g["processing_status"], "not_eligible")

    def test_07_identity_only_closeup_allows_human_skips_ball(self) -> None:
        window = {
            "playability": "partially_playable",
            "graphics_status": "none",
            "tracking_eligibility": "ineligible",
            "identity_eligibility": "eligible",
            "ball_analysis_eligibility": "ineligible",
        }
        r = resolve_frame_routing(window, policy=self.pol)
        self.assertTrue(r["process_human"])
        self.assertFalse(r["process_ball"])
        self.assertTrue(r.get("identity_only"))

    def test_08_live_event_unknown_does_not_block(self) -> None:
        window = {
            "playability": "playable",
            "graphics_status": "none",
            "tracking_eligibility": "eligible",
            "identity_eligibility": "unknown",
            "ball_analysis_eligibility": "eligible",
            "live_event_eligibility": "unknown",
        }
        r = resolve_frame_routing(window, policy=self.pol)
        self.assertTrue(r["process_human"])
        self.assertFalse(self.pol["routing"]["live_event_unknown_blocks_visual_detection"])

    def test_09_empty_class_name_rejected(self) -> None:
        with self.assertRaises(TaxonomyError):
            map_model_class(0, "  ", taxonomy=self.tax)


if __name__ == "__main__":
    unittest.main()
