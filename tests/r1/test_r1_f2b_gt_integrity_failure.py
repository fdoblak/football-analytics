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
        # Historical F2B gate is retained in stage evidence; current GATE_STATUS may
        # have advanced to train-repair readiness (FIX2) without freezing.
        hist = FAIL / "integrity_report.json"
        self.assertTrue(hist.is_file())
        rep = json.loads(hist.read_text(encoding="utf-8"))
        self.assertFalse(rep["frozen"])
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertIn(
            gate["gate"],
            {
                "NO-GO — REVIEWED GT INTEGRITY FAILURE",
                "PASS — TRAIN ANNOTATION REPAIR READY",
                "NO-GO — TRAIN ANNOTATION REPAIR TOOL FAILURE",
                "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE",
                "NO-GO — REPAIRED GT INTEGRITY FAILURE",
                (
                    "PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED "
                    "ON OWN-VIDEO CLIP; GENERALIZATION NOT VALIDATED"
                ),
                "NO-GO — SMALL-OBJECT DETECTOR DEVELOPMENT GATE FAILED",
                (
                    "PASS — SMALL-OBJECT DETECTOR DEVELOPMENT GATE PASSED; "
                    "NEW BLIND HOLDOUT REVIEW READY"
                ),
            },
        )
        self.assertFalse(gate["acceptance_eligible"])
        if gate["gate"] == "NO-GO — REVIEWED GT INTEGRITY FAILURE":
            self.assertFalse(gate["frozen"])
        if gate["gate"] == "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE":
            self.assertTrue(gate.get("frozen", False))
        if "SMALL-OBJECT DETECTOR DEVELOPMENT GATE" in gate["gate"]:
            self.assertEqual(
                gate.get("frozen_gt_fingerprint"),
                "4e4e46d9edabd98aad53ea2538a2a67cd5cfeb6e0444abf7254b12f01ca4f9f1",
            )

    def test_integrity_report_lists_failed_train_frames(self) -> None:
        rep = json.loads((FAIL / "integrity_report.json").read_text(encoding="utf-8"))
        self.assertFalse(rep["frozen"])
        self.assertGreaterEqual(rep["critical_failures"][0]["n_frames"], 30)
        failed = json.loads((FAIL / "failed_train_empty_frames.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["n"], len(failed["frame_indices"]))
        self.assertEqual(sorted(failed["train_labeled_frame_indices"]), [0, 5, 15])

    def test_no_frozen_annotations_dir_unless_r1_accepted(self) -> None:
        frozen = REPO / "annotations" / "own_video_97b298e4" / "human_detection_v1"
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        g = gate["gate"]
        if g.startswith("PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED") or (
            g.startswith("NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE") and gate.get("frozen")
        ):
            self.assertTrue(frozen.is_dir())
            return
        if "SMALL-OBJECT DETECTOR DEVELOPMENT GATE" in g:
            self.assertTrue(frozen.is_dir())
            return
        if g == "PASS — TRAIN ANNOTATION REPAIR READY":
            # repair ready before freeze
            return
        # Historical integrity-failure era: freeze must not exist
        if g == "NO-GO — REVIEWED GT INTEGRITY FAILURE":
            self.assertFalse(frozen.exists())

    def test_no_acceptance_media_package(self) -> None:
        acc = EV / "r1_detector_acceptance"
        if acc.is_dir():
            for pat in ("*.mp4", "*.png", "*.html"):
                self.assertEqual(list(acc.glob(pat)), [])


if __name__ == "__main__":
    unittest.main()
