"""R1 blind GT integrity / leakage / freeze guards (WAITING state)."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
GT = EV / "gt"


class R1BlindGtWaitingTests(unittest.TestCase):
    def test_gate_is_waiting(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertIn("WAITING", gate["gate"])

    def test_annotations_not_frozen_with_positives(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(ann["status"], "WAITING_VISUAL_APPROVAL")
        self.assertEqual(ann["counts"]["positive_boxes"], 0)
        self.assertGreaterEqual(ann["n_frames"], 120)
        for fr in ann["frames"]:
            self.assertEqual(fr["humans"], [])
            self.assertIn(fr["split"], {"train", "dev", "holdout"})
            self.assertEqual(fr["review_status"], "not_reviewed")

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
        # predictions must not be embedded into GT humans
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        blob = json.dumps(ann)
        self.assertNotIn("yolo11n", blob.lower())
        self.assertNotIn("bbox_xyxy", blob)  # empty humans; no leaked preds


if __name__ == "__main__":
    unittest.main()
