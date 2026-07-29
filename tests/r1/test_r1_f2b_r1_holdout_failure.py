"""R1-F2-B-R1 holdout failure integrity (GT frozen, detector rejected)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
FAIL = EV / "r1_f2b_r1_holdout_failure"
FROZEN = REPO / "annotations" / "own_video_97b298e4" / "human_detection_v1"


class R1F2BR1HoldoutFailureTests(unittest.TestCase):
    def test_gate_nogo_holdout(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["gate"], "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE")
        self.assertTrue(gate["frozen"])
        self.assertFalse(gate["acceptance_eligible"])

    def test_frozen_gt_present(self) -> None:
        self.assertTrue((FROZEN / "annotations.json").is_file())
        ann = json.loads((FROZEN / "annotations.json").read_text(encoding="utf-8"))
        self.assertTrue(ann["frozen"])
        self.assertEqual(len(ann["frames"]), 80)

    def test_holdout_metrics_below_threshold(self) -> None:
        m = json.loads((FAIL / "holdout_metrics.json").read_text(encoding="utf-8"))
        self.assertLess(m["precision"], 0.90)
        self.assertLess(m["recall"], 0.90)
        self.assertTrue(m.get("one_shot_confirmed") or m.get("config", {}).get("one_shot"))

    def test_no_acceptance_media(self) -> None:
        acc = EV / "r1_detector_acceptance"
        if acc.is_dir():
            for pat in ("*.mp4", "*.png", "*.html"):
                self.assertEqual(list(acc.glob(pat)), [])


if __name__ == "__main__":
    unittest.main()
