"""Tests for R1-F2-C small-object redesign (protocol v2, tiles, fusion, guards)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from football_analytics.annotation.evaluation_protocol_v2 import (
    EXPECTED_FROZEN_FP,
    HOLDOUT_V1_STATUS,
    LabeledBox,
    boxes_from_frozen_frame,
    dev_gate_passed,
    evaluate_protocol_v2,
    filter_predictions_with_ignore,
    height_bin,
    protocol_v2_definition,
)
from football_analytics.annotation.holdout_v1_guard import (
    assert_no_holdout_v1_for_development,
)
from football_analytics.annotation.holdout_v2_selection import select_holdout_v2_frames
from football_analytics.annotation.independent_gt import IndependentGTError
from football_analytics.annotation.train_tiles import (
    DEFAULT_TILE_H,
    DEFAULT_TILE_W,
    clip_box_to_tile,
    roundtrip_ok,
)
from football_analytics.perception.candidate_merge import BallCandidate, class_aware_nms
from football_analytics.perception.detection_evaluation import BBoxDetection
from football_analytics.perception.full_tile_fusion import FusionConfig
from football_analytics.perception.tiling import generate_tiles, map_tile_bbox_to_source

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "annotations" / "own_video_97b298e4" / "human_detection_v1"
EV = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_small_object_redesign"


class ProtocolV2Tests(unittest.TestCase):
    def test_protocol_fingerprint_stable_keys(self) -> None:
        a = protocol_v2_definition()
        b = protocol_v2_definition()
        self.assertEqual(a["protocol_id"], "own_video_human_eval_protocol_v2")
        self.assertEqual(a["protocol_fingerprint"], b["protocol_fingerprint"])
        self.assertEqual(a["holdout_v1"]["status"], HOLDOUT_V1_STATUS)
        self.assertFalse(a["holdout_v1"]["acceptance_reusable"])
        self.assertTrue(a["defined_before_holdout_v2"])

    def test_ignore_region_not_fp(self) -> None:
        ignore = [
            LabeledBox(
                frame_index=1,
                xyxy=(10, 10, 50, 80),
                role="player",
                eligibility="uncertain",
                visibility="clear",
                team_appearance="unknown",
            )
        ]
        preds = [
            BBoxDetection(1, "human", 12, 12, 48, 78, 0.9),
            BBoxDetection(1, "human", 200, 200, 240, 300, 0.8),
        ]
        kept, ignored = filter_predictions_with_ignore(preds, ignore, iou_thresh=0.5)
        self.assertEqual(len(ignored), 1)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].x1, 200)

    def test_height_bins(self) -> None:
        self.assertEqual(height_bin(10), "h_lt_16")
        self.assertEqual(height_bin(20), "h_16_24")
        self.assertEqual(height_bin(30), "h_24_40")
        self.assertEqual(height_bin(50), "h_40_64")
        self.assertEqual(height_bin(80), "h_ge_64")

    def test_dev_gate_thresholds(self) -> None:
        ok = {
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "ap50": 0.9,
            "small_recall": 0.8,
            "duplicate_rate": 0.0,
        }
        self.assertTrue(dev_gate_passed(ok)["passed"])
        bad = dict(ok)
        bad["small_recall"] = 0.5
        self.assertFalse(dev_gate_passed(bad)["passed"])

    def test_evaluate_protocol_v2_primary_scope(self) -> None:
        frames = [
            {
                "frame_idx": 0,
                "humans": [
                    {
                        "bbox_xyxy": [100, 100, 140, 200],
                        "role": "player",
                        "eligibility": "on_pitch",
                        "visibility": "clear",
                        "team_appearance": "team_a",
                    },
                    {
                        "bbox_xyxy": [10, 10, 40, 50],
                        "role": "player",
                        "eligibility": "uncertain",
                        "visibility": "clear",
                        "team_appearance": "unknown",
                    },
                ],
            }
        ]
        preds = [
            BBoxDetection(0, "human", 100, 100, 140, 200, 0.95),
            BBoxDetection(0, "human", 12, 12, 38, 48, 0.9),  # overlaps ignore → not FP
        ]
        ev = evaluate_protocol_v2(preds, frames)
        self.assertEqual(ev["n_primary_gt"], 1)
        self.assertEqual(ev["primary"]["true_positives"], 1)
        self.assertEqual(ev["secondary"]["ignored_predictions"], 1)


class TileGeometryTests(unittest.TestCase):
    def test_roundtrip_and_clip(self) -> None:
        tiles = generate_tiles(
            1336,
            744,
            tile_width=DEFAULT_TILE_W,
            tile_height=DEFAULT_TILE_H,
            overlap_x=140,
            overlap_y=102,
            max_tiles=24,
        )
        self.assertGreaterEqual(len(tiles), 4)
        box = (100.0, 100.0, 140.0, 220.0)
        tile = tiles[0]
        self.assertTrue(roundtrip_ok(box, tile))
        local = clip_box_to_tile(box, tile)
        self.assertIsNotNone(local)
        assert local is not None
        back = map_tile_bbox_to_source(local, tile, coordinate_space="tile_local")
        self.assertAlmostEqual(back[0], max(box[0], tile.x0), places=3)

    def test_min_visible_fraction_ignores_sliver(self) -> None:
        tiles = generate_tiles(
            1336,
            744,
            tile_width=DEFAULT_TILE_W,
            tile_height=DEFAULT_TILE_H,
            overlap_x=140,
            overlap_y=102,
            max_tiles=24,
        )
        tile = tiles[0]
        # almost entirely outside tile
        box = (tile.x1 - 2.0, float(tile.y0), tile.x1 + 80.0, float(tile.y0) + 100)
        self.assertIsNone(clip_box_to_tile(box, tile))


class FusionAndGuardTests(unittest.TestCase):
    def test_class_aware_nms_dedup(self) -> None:
        cands = [
            BallCandidate(10, 10, 50, 80, 0.9, 0, "human", "full_frame"),
            BallCandidate(12, 12, 52, 82, 0.8, 0, "human", "tile:r0c0"),
        ]
        merged = class_aware_nms(cands, merge_iou=0.5)
        self.assertEqual(len(merged), 1)

    def test_holdout_v1_access_denied(self) -> None:
        with self.assertRaises(IndependentGTError):
            assert_no_holdout_v1_for_development(split="holdout", purpose="training")
        with self.assertRaises(IndependentGTError):
            assert_no_holdout_v1_for_development(
                [{"split": "holdout", "frame_idx": 1}],
                purpose="model_selection",
            )

    def test_fusion_config_defaults(self) -> None:
        cfg = FusionConfig()
        self.assertEqual(cfg.mode, "hybrid")
        self.assertGreaterEqual(cfg.tile_w, 640)
        self.assertLessEqual(cfg.tile_w, 768)


class HoldoutV2SelectionTests(unittest.TestCase):
    def test_selection_isolated_from_old_80(self) -> None:
        ann = json.loads((FROZEN / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(ann["canonical_fingerprint"], EXPECTED_FROZEN_FP)
        sel = select_holdout_v2_frames(ann, n_target=22)
        used = {int(f["frame_idx"]) for f in ann["frames"]}
        self.assertGreaterEqual(sel["n_frames"], 20)
        self.assertLessEqual(sel["n_frames"], 25)
        self.assertTrue(set(sel["frame_indices"]).isdisjoint(used))
        self.assertTrue(sel["rules"]["blind"])
        self.assertTrue(sel["rules"]["no_proposals"])

    def test_frozen_boxes_from_frame(self) -> None:
        ann = json.loads((FROZEN / "annotations.json").read_text(encoding="utf-8"))
        fr = next(f for f in ann["frames"] if f["split"] == "train")
        boxes = boxes_from_frozen_frame(fr)
        self.assertGreater(len(boxes), 0)


class EvidenceIfPresentTests(unittest.TestCase):
    def test_root_cause_schema_when_written(self) -> None:
        path = EV / "root_cause.json"
        if not path.is_file():
            self.skipTest("root_cause not written yet")
        rca = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(rca["holdout_v1_status"], "CONSUMED_FAILED_EVALUATION")
        self.assertFalse(rca["acceptance_reusable"])
        self.assertIn("hypotheses", rca)


if __name__ == "__main__":
    unittest.main()
