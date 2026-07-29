"""Unit tests for human tiled/hybrid detection helpers (no GPU required for pure logic)."""

from __future__ import annotations

import unittest

from football_analytics.perception.adapters.base import RawPersonBox
from football_analytics.perception.candidate_merge import BallCandidate, class_aware_nms
from football_analytics.perception.human_tiled_detection import (
    HumanDetectConfig,
    HumanProposal,
    duplicate_pairs,
    geometry_ok,
    merged_person_candidates,
    raw_to_candidates,
)
from football_analytics.perception.tiling import generate_tiles, map_tile_bbox_to_source


class HumanTiledDetectionTests(unittest.TestCase):
    def test_tile_roundtrip_offset(self) -> None:
        tiles = generate_tiles(
            1336,
            744,
            tile_width=672,
            tile_height=420,
            overlap_x=112,
            overlap_y=84,
            max_tiles=12,
        )
        self.assertGreaterEqual(len(tiles), 4)
        t = tiles[1]
        local = (10.0, 20.0, 40.0, 80.0)
        mapped = map_tile_bbox_to_source(local, t, coordinate_space="tile_local")
        self.assertEqual(mapped[0], local[0] + t.x0)
        self.assertEqual(mapped[1], local[1] + t.y0)

    def test_geometry_filters(self) -> None:
        cfg = HumanDetectConfig(name="t")
        self.assertTrue(geometry_ok((100, 100, 130, 180), cfg, 744))
        self.assertFalse(geometry_ok((100, 100, 105, 110), cfg, 744))  # tiny
        self.assertFalse(geometry_ok((10, 10, 400, 40), cfg, 744))  # flat

    def test_nms_dedup(self) -> None:
        cands = [
            BallCandidate(10, 10, 40, 80, 0.9, 0, "person", "full_frame"),
            BallCandidate(12, 12, 42, 82, 0.8, 0, "person", "tile:r0c0"),
        ]
        kept = class_aware_nms(cands, merge_iou=0.5)
        self.assertEqual(len(kept), 1)

    def test_raw_to_candidates_clip(self) -> None:
        cfg = HumanDetectConfig(name="t")
        raw = [
            RawPersonBox(-5, 10, 40, 100, 0.7, 0, "person"),
            RawPersonBox(1300, 700, 1400, 800, 0.7, 0, "person"),
        ]
        out = raw_to_candidates(raw, source="full_frame", cfg=cfg, frame_w=1336, frame_h=744)
        self.assertGreaterEqual(len(out), 1)
        for c in out:
            self.assertGreaterEqual(c.x1, 0)
            self.assertLessEqual(c.x2, 1336)

    def test_duplicate_and_merged_diagnostics(self) -> None:
        props = [
            HumanProposal(10, 10, 40, 80, 0.9, "on_pitch_human_candidate", "a"),
            HumanProposal(11, 11, 41, 81, 0.8, "on_pitch_human_candidate", "b"),
            HumanProposal(200, 200, 320, 280, 0.7, "on_pitch_human_candidate", "c"),
        ]
        self.assertGreaterEqual(duplicate_pairs(props, iou_thresh=0.9), 1)
        self.assertGreaterEqual(merged_person_candidates(props), 1)


if __name__ == "__main__":
    unittest.main()
