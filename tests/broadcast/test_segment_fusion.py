"""Unit tests for Stage 4D segment fusion."""

from __future__ import annotations

import unittest

from football_analytics.broadcast.segment_fusion import (
    FusionError,
    fuse_shot_camera_intervals,
)
from football_analytics.core.run_id import generate_run_id


def _shot(run_id: str, shot_id: str, start: int, end: int, *, mapping: str = "exact_identity"):
    return {
        "run_id": run_id,
        "video_id": "v1",
        "shot_id": shot_id,
        "start_time_us": start,
        "end_time_us": end,
        "timeline_mapping_quality": mapping,
        "segment_status": "active",
        "duration_us": end - start,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "start_boundary_id": None,
        "end_boundary_id": None,
        "frame_count": None,
        "provenance_json": None,
        "contract_version": 1,
    }


def _cam(
    run_id: str,
    cam_id: str,
    shot_id: str,
    start: int,
    end: int,
    *,
    view: str = "main_broadcast",
    framing: str = "wide",
    playability: str = "playable",
    coverage: float = 1.0,
):
    return {
        "run_id": run_id,
        "video_id": "v1",
        "camera_segment_id": cam_id,
        "shot_id": shot_id,
        "start_time_us": start,
        "end_time_us": end,
        "view_family": view,
        "framing_scale": framing,
        "replay_status": "live",
        "graphics_status": "none",
        "playability": playability,
        "coverage": coverage,
        "confidence": 0.8,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "camera_position": "unknown",
        "camera_motion": "static",
        "calibration_suitability": "suitable",
        "tracking_suitability": "suitable",
        "target_identity_suitability": "unknown",
        "classification_source": "manual",
        "review_status": "accepted",
        "evidence_refs": [cam_id],
        "provenance_json": None,
        "contract_version": 1,
    }


class SegmentFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()

    def test_01_split_multi_camera(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 1000)]
        cams = [
            _cam(self.run_id, "c1", "s1", 0, 400),
            _cam(self.run_id, "c2", "s1", 400, 1000, view="player_isolation", framing="close_up"),
        ]
        fused = fuse_shot_camera_intervals(shots, cams)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].end_time_us, 400)
        self.assertEqual(fused[1].start_time_us, 400)
        self.assertEqual(fused[1].view_family, "player_isolation")

    def test_02_gap_not_filled(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 1000)]
        cams = [_cam(self.run_id, "c1", "s1", 0, 400)]
        fused = fuse_shot_camera_intervals(shots, cams)
        gaps = [w for w in fused if w.is_gap]
        self.assertTrue(gaps)
        self.assertEqual(gaps[0].start_time_us, 400)
        self.assertEqual(gaps[0].playability, "uncertain")

    def test_03_conflict(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 1000)]
        cams = [
            _cam(self.run_id, "c1", "s1", 0, 1000),
            _cam(self.run_id, "c2", "s1", 0, 1000, view="graphics", framing="unknown"),
        ]
        fused = fuse_shot_camera_intervals(shots, cams)
        self.assertEqual(len(fused), 1)
        self.assertTrue(fused[0].is_conflict)

    def test_04_camera_outside_shot_errors(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 500)]
        cams = [_cam(self.run_id, "c1", "s1", 0, 800)]
        with self.assertRaises(FusionError):
            fuse_shot_camera_intervals(shots, cams)

    def test_05_adjacent_merge(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 1000)]
        cams = [
            _cam(self.run_id, "c1", "s1", 0, 500),
            _cam(self.run_id, "c1b", "s1", 500, 1000),
        ]
        # Different camera ids but same decision fields → no merge (ids differ)
        fused = fuse_shot_camera_intervals(shots, cams, merge_identical_adjacent=True)
        self.assertEqual(len(fused), 2)
        # Same camera id spanning via two identical atomic windows after split+merge
        cams2 = [_cam(self.run_id, "c1", "s1", 0, 1000)]
        fused2 = fuse_shot_camera_intervals(shots, cams2)
        self.assertEqual(len(fused2), 1)
        self.assertEqual(fused2[0].start_time_us, 0)
        self.assertEqual(fused2[0].end_time_us, 1000)

    def test_06_mapping_carried(self) -> None:
        shots = [_shot(self.run_id, "s1", 0, 100, mapping="uncertain")]
        cams = [_cam(self.run_id, "c1", "s1", 0, 100)]
        fused = fuse_shot_camera_intervals(shots, cams)
        self.assertEqual(fused[0].timeline_mapping_quality, "uncertain")


if __name__ == "__main__":
    unittest.main()
