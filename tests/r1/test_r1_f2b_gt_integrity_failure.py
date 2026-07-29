"""R1-F2-B integrity failure path: do not freeze poisoned train GT."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
FAIL = EV / "r1_f2b_gt_integrity_failure"


class R1F2BIntegrityFailureTests(unittest.TestCase):
    def test_gate_is_integrity_nogo(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["gate"], "NO-GO — REVIEWED GT INTEGRITY FAILURE")
        self.assertFalse(gate["frozen"])
        self.assertFalse(gate["acceptance_eligible"])

    def test_integrity_report_lists_failed_train_frames(self) -> None:
        rep = json.loads((FAIL / "integrity_report.json").read_text(encoding="utf-8"))
        self.assertFalse(rep["frozen"])
        self.assertGreaterEqual(rep["critical_failures"][0]["n_frames"], 30)
        failed = json.loads((FAIL / "failed_train_empty_frames.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["n"], len(failed["frame_indices"]))
        self.assertEqual(sorted(failed["train_labeled_frame_indices"]), [0, 5, 15])

    def test_no_frozen_annotations_dir(self) -> None:
        frozen = REPO / "annotations" / "own_video_97b298e4" / "human_detection_v1"
        self.assertFalse(frozen.exists())

    def test_no_acceptance_media_package(self) -> None:
        acc = EV / "r1_detector_acceptance"
        if acc.is_dir():
            for pat in ("*.mp4", "*.png", "*.html"):
                self.assertEqual(list(acc.glob(pat)), [])


if __name__ == "__main__":
    unittest.main()
