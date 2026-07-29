"""R1-F1-R1: rejected agent draft + detection preacceptance integrity."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
GT = EV / "gt"
REJECTED = EV / "r1_f1_gt_review_REJECTED"
PRE = EV / "r1_detection_preacceptance"


class R1F1R1RejectionAndPreacceptanceTests(unittest.TestCase):
    def test_old_draft_rejected(self) -> None:
        rej = json.loads((REJECTED / "REJECTION_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertIn("REJECTED", rej["status"])
        self.assertFalse(rej["acceptance_eligible"])
        self.assertFalse(rej["gt_freeze_eligible"])
        status = json.loads(
            (EV / "r1_f1_gt_review" / "REJECTED_STATUS.json").read_text(encoding="utf-8")
        )
        self.assertTrue(status["do_not_use_for_review"])

    def test_gt_annotations_marked_rejected(self) -> None:
        ann = json.loads((GT / "blind_gt_annotations.json").read_text(encoding="utf-8"))
        self.assertIn("REJECTED", ann["status"])
        self.assertFalse(ann["acceptance_eligible"])
        self.assertFalse(ann["human_approved"])
        self.assertFalse(ann["gt_frozen"])

    def test_no_freeze_fingerprint(self) -> None:
        text = (GT / "blind_gt_checksums.sha256").read_text(encoding="utf-8")
        self.assertIn("NOT_FROZEN", text)
        self.assertFalse((GT / "GT_FREEZE.json").exists())

    def test_acceptance_metrics_still_blocked(self) -> None:
        blocked = json.loads(
            (EV / "evaluation" / "WAITING_NO_ACCEPTANCE_METRICS.json").read_text(encoding="utf-8")
        )
        self.assertFalse(blocked["acceptance_metrics_computed"])

    def test_preacceptance_package_if_present(self) -> None:
        if not (PRE / "MANIFEST.json").is_file():
            self.skipTest("preacceptance package not built yet")
        man = json.loads((PRE / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertIn("PREACCEPTANCE", man["gate"])
        self.assertTrue(man["diagnostic_not_accuracy"])
        for name in (
            "OPEN_R1_DETECTION_PREACCEPTANCE.html",
            "R1_human_detection_proof.mp4",
            "R1_start.png",
            "R1_middle.png",
            "R1_holdout.png",
            "R1_crowded_cases.png",
            "R1_small_distant_cases.png",
            "R1_off_pitch_filtering.png",
            "R1_before_after.png",
            "detector_bakeoff.json",
            "diagnostic_quality.json",
            "checksums.sha256",
            "README_TR.txt",
        ):
            self.assertTrue((PRE / name).is_file(), msg=name)
        for line in (PRE / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, fname = line.split()
            self.assertEqual(hashlib.sha256((PRE / fname).read_bytes()).hexdigest(), digest)

    def test_gate_status_file(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertTrue(
            "PREACCEPTANCE" in gate["gate"]
            or "UNUSABLE" in gate["gate"]
            or "REJECTED" in gate.get("gate", "")
            or gate["gate"].startswith("PASS_WITH_FINDINGS")
            or gate["gate"].startswith("NO-GO")
        )


if __name__ == "__main__":
    unittest.main()
