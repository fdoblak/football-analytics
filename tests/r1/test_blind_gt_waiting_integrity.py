"""R1-F1 blind GT review package integrity (not frozen / not acceptance)."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
GT = EV / "gt"
PKG = EV / "r1_f1_gt_review"


class R1BlindGtReviewPackageTests(unittest.TestCase):
    def test_gate_package_ready(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["gate"], "PASS — BLIND GT REVIEW PACKAGE READY")
        self.assertFalse(gate.get("acceptance_metrics_computed", True))

    def test_draft_not_frozen_or_approved(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(ann["status"], "BLIND_GT_DRAFT_REVIEW_PACKAGE")
        self.assertEqual(ann["provenance"], "agent_blind_reviewed_draft")
        self.assertFalse(ann["human_approved"])
        self.assertFalse(ann["gt_frozen"])
        self.assertFalse(ann["acceptance_eligible"])
        self.assertFalse(ann["prediction_used"])
        self.assertEqual(ann["n_frames"], 150)
        self.assertEqual(ann["counts"]["frames_by_split"]["train"], 54)
        self.assertEqual(ann["counts"]["frames_by_split"]["dev"], 43)
        self.assertEqual(ann["counts"]["frames_by_split"]["holdout"], 53)
        self.assertGreater(ann["counts"]["positive_boxes"], 0)
        self.assertEqual(ann["counts"]["human_complete_frames"], 0)

    def test_draft_boxes_have_canonical_coords(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        n_with = 0
        for fr in ann["frames"]:
            self.assertIn(fr["split"], {"train", "dev", "holdout"})
            self.assertEqual(fr["source_width"], 1336)
            self.assertEqual(fr["source_height"], 744)
            for h in fr["humans"]:
                n_with += 1
                self.assertEqual(h["coordinate_space"], "source_xyxy_px_v1")
                self.assertLess(h["x1"], h["x2"])
                self.assertLess(h["y1"], h["y2"])
                self.assertGreaterEqual(h["x1"], 0)
                self.assertGreaterEqual(h["y1"], 0)
                self.assertLessEqual(h["x2"], 1336)
                self.assertLessEqual(h["y2"], 744)
                self.assertEqual(h["review_status"], "agent_blind_reviewed_draft")
        expected = (
            ann["counts"]["positive_boxes"]
            + ann["counts"]["ignore_boxes"]
            + ann["counts"]["uncertain_boxes"]
        )
        self.assertEqual(n_with, expected)

    def test_splits_cover_time_ranges(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        for fr in ann["frames"]:
            t = fr["t_s"]
            if fr["split"] == "train":
                self.assertLess(t, 12.0)
            elif fr["split"] == "dev":
                self.assertGreaterEqual(t, 12.0)
                self.assertLess(t, 22.0)
            else:
                self.assertGreaterEqual(t, 22.0)

    def test_no_freeze_fingerprint_yet(self) -> None:
        text = (GT / "blind_gt_checksums.sha256").read_text(encoding="utf-8")
        self.assertIn("NOT_FROZEN", text)
        self.assertFalse((GT / "GT_FREEZE.json").exists())

    def test_checksums_match_files(self) -> None:
        for line in (GT / "blind_gt_checksums.sha256").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            digest, name = line.split()
            path = GT / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_acceptance_metrics_blocked(self) -> None:
        blocked = json.loads(
            (EV / "evaluation" / "WAITING_NO_ACCEPTANCE_METRICS.json").read_text(encoding="utf-8")
        )
        self.assertFalse(blocked["acceptance_metrics_computed"])

    def test_prediction_namespace_separate_from_gt(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        blob = json.dumps(ann).lower()
        self.assertNotIn("yolo11n", blob)
        self.assertNotIn("ultralytics", blob)

    def test_invalid_attempt_rejected(self) -> None:
        rej = json.loads(
            (EV / "gt_invalid_attempt" / "REJECTION_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertFalse(rej["acceptance_eligible"])
        self.assertGreaterEqual(rej["bbox_count"], 600)

    def test_package_outputs_exist(self) -> None:
        for name in (
            "R1_blind_GT_draft_proof.mp4",
            "R1_GT_draft_start.png",
            "R1_GT_draft_middle.png",
            "R1_GT_draft_holdout.png",
            "R1_GT_negative_examples.png",
            "R1_GT_difficult_examples.png",
            "OPEN_R1_GT_DRAFT_RESULTS.html",
            "START_R1_GT_REVIEW.bat",
            "README_TR.txt",
        ):
            self.assertTrue((PKG / name).is_file(), msg=name)


if __name__ == "__main__":
    unittest.main()
