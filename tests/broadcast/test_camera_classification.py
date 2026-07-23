"""Unit / integration tests for camera classification rules."""

from __future__ import annotations

import json
import unittest

from football_analytics.broadcast.camera_classification import (
    AxisDecision,
    SampleDecision,
    aggregate_shot_classification,
    classify_sample,
)
from football_analytics.broadcast.camera_config import (
    camera_config_fingerprint,
    default_camera_config_path,
    load_camera_view_config,
)
from football_analytics.broadcast.camera_features import CameraSampleFeatures
from football_analytics.broadcast.types import CameraPosition, ReplayStatus


def _feat(**kwargs: float | int) -> CameraSampleFeatures:
    base = dict(
        frame_index=0,
        time_us=0,
        pitch_green_fraction=0.0,
        pitch_center_fraction=0.0,
        pitch_spatial_spread=0.0,
        hist_entropy=4.0,
        edge_density=0.05,
        texture_entropy=4.0,
        center_periphery_ratio=0.5,
        overlay_high_contrast_fraction=0.0,
        skin_like_fraction=0.0,
        mean_luma=0.4,
        frame_diff_mean=0.0,
        flow_mag_mean=0.0,
        flow_mag_std=0.0,
        flow_horizontal_ratio=0.5,
        flow_radial_consistency=0.0,
    )
    base.update(kwargs)
    return CameraSampleFeatures(**base)  # type: ignore[arg-type]


class CameraClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_camera_view_config(default_camera_config_path())
        cls.fp = camera_config_fingerprint(cls.config)

    def test_wide_main_broadcast(self) -> None:
        d = classify_sample(
            _feat(
                pitch_green_fraction=0.7,
                pitch_spatial_spread=0.35,
                overlay_high_contrast_fraction=0.02,
            ),
            self.config,
        )
        self.assertEqual(d.view_family.label, "main_broadcast")
        self.assertEqual(d.framing_scale.label, "wide")
        self.assertEqual(d.graphics_status.label, "none")
        self.assertEqual(d.playability.label, "playable")

    def test_player_isolation(self) -> None:
        d = classify_sample(
            _feat(
                pitch_green_fraction=0.02,
                skin_like_fraction=0.12,
                overlay_high_contrast_fraction=0.02,
            ),
            self.config,
        )
        self.assertEqual(d.view_family.label, "player_isolation")
        self.assertEqual(d.framing_scale.label, "close_up")
        self.assertEqual(d.playability.label, "partially_playable")

    def test_fullscreen_graphics(self) -> None:
        d = classify_sample(
            _feat(
                pitch_green_fraction=0.01,
                overlay_high_contrast_fraction=0.7,
                hist_entropy=3.0,
                edge_density=0.3,
            ),
            self.config,
        )
        self.assertEqual(d.view_family.label, "graphics")
        self.assertEqual(d.graphics_status.label, "full_screen")
        self.assertEqual(d.playability.label, "non_playable")
        self.assertEqual(d.framing_scale.label, "unknown")

    def test_ood_abstains(self) -> None:
        d = classify_sample(
            _feat(
                pitch_green_fraction=0.02,
                hist_entropy=7.0,
                edge_density=0.25,
                skin_like_fraction=0.0,
                overlay_high_contrast_fraction=0.05,
            ),
            self.config,
        )
        self.assertTrue(d.ood_like)
        self.assertEqual(d.view_family.label, "unknown")
        self.assertEqual(d.playability.label, "uncertain")

    def test_aggregate_always_unknown_axes_and_null_confidence(self) -> None:
        decisions = [
            SampleDecision(
                frame_index=i,
                time_us=i * 40000,
                view_family=AxisDecision("main_broadcast", 0.8, False, {"main_broadcast": 0.8}),
                framing_scale=AxisDecision("wide", 0.8, False, {"wide": 0.8}),
                camera_motion=AxisDecision("static", 0.8, False, {"static": 0.8}),
                graphics_status=AxisDecision("none", 0.8, False, {"none": 0.8}),
                playability=AxisDecision("playable", 0.8, False, {"playable": 0.8}),
                ood_like=False,
            )
            for i in range(3)
        ]
        from football_analytics.core.run_id import generate_run_id

        seg = aggregate_shot_classification(
            decisions,
            run_id=generate_run_id(),
            video_id="vid_test",
            shot_id="shot_1",
            camera_segment_id="cam_0000",
            start_time_us=0,
            end_time_us=400000,
            start_frame_index=0,
            end_frame_index_exclusive=10,
            config=self.config,
            config_fingerprint=self.fp,
        )
        self.assertIsNone(seg.confidence)
        self.assertEqual(seg.camera_position, CameraPosition.UNKNOWN)
        self.assertEqual(seg.replay_status, ReplayStatus.UNKNOWN)
        prov = json.loads(str(seg.provenance_json))
        self.assertIn("heuristic_score", prov)
        self.assertEqual(prov["limitation"], "one_camera_view_segment_per_shot")


if __name__ == "__main__":
    unittest.main()
