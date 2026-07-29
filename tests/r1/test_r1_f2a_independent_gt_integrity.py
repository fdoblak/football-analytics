"""R1-F2-A integrity: independent GT tool ready; not frozen / not acceptance."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"
RUNTIME = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")


class R1F2AIntegrityTests(unittest.TestCase):
    def test_gate_tool_ready(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertTrue(
            gate["gate"]
            in {
                "PASS — INDEPENDENT HUMAN GT REVIEW TOOL READY",
                "PASS — WINDOWS GT REVIEW LAUNCHER VERIFIED",
                "PASS — TRAIN ANNOTATION REPAIR READY",
                "NO-GO — REVIEWED GT INTEGRITY FAILURE",
                "NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE",
                "NO-GO — REPAIRED GT INTEGRITY FAILURE",
                "NO-GO — TRAIN ANNOTATION REPAIR TOOL FAILURE",
                (
                    "PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED "
                    "ON OWN-VIDEO CLIP; GENERALIZATION NOT VALIDATED"
                ),
                "NO-GO — SMALL-OBJECT DETECTOR DEVELOPMENT GATE FAILED",
                (
                    "PASS — SMALL-OBJECT DETECTOR DEVELOPMENT GATE PASSED; "
                    "NEW BLIND HOLDOUT REVIEW READY"
                ),
            }
        )
        if gate["gate"].startswith("PASS —"):
            self.assertFalse(gate["acceptance_eligible"])
            if "human_approved" in gate:
                self.assertFalse(gate["human_approved"])
            if "reviewed_gt" in gate:
                self.assertFalse(gate["reviewed_gt"])
        if gate["gate"].startswith("NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE"):
            self.assertFalse(gate["acceptance_eligible"])
        if "SMALL-OBJECT DETECTOR DEVELOPMENT GATE" in gate["gate"]:
            self.assertFalse(gate["acceptance_eligible"])
            self.assertEqual(gate.get("stage"), "R1-F2-C")
            self.assertEqual(
                gate.get("frozen_gt_fingerprint"),
                "4e4e46d9edabd98aad53ea2538a2a67cd5cfeb6e0444abf7254b12f01ca4f9f1",
            )
        if gate["gate"].startswith("PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR"):
            self.assertFalse(gate["acceptance_eligible"])
            self.assertTrue(gate.get("frozen", False))

    def test_selection_manifest(self) -> None:
        sel = json.loads(
            (EV / "independent_gt" / "selected_frames.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sel["counts"]["total"], 80)
        self.assertEqual(sel["counts"]["train"], 40)
        self.assertEqual(sel["counts"]["dev"], 20)
        self.assertEqual(sel["counts"]["holdout"], 20)

    def test_runtime_draft_not_approved(self) -> None:
        self.assertTrue((RUNTIME / "draft_annotations.json").is_file())
        draft = json.loads((RUNTIME / "draft_annotations.json").read_text(encoding="utf-8"))
        self.assertFalse(draft["human_approved"])
        self.assertFalse(draft["reviewed_gt"])
        self.assertFalse(draft["frozen"])
        for fr in draft["frames"]:
            if fr["split"] in {"dev", "holdout"}:
                self.assertEqual(fr.get("proposals") or [], [])

    def test_windows_package_files_only(self) -> None:
        win = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")
        names = {p.name for p in win.iterdir() if p.is_file()}
        self.assertEqual(
            names,
            {
                "START_GT_REVIEW.bat",
                "START_TRAIN_REPAIR.bat",
                "README_TR.txt",
                "REVIEW_PROGRESS.html",
                "NO_MODEL_RESULT_YET.txt",
            },
        )


if __name__ == "__main__":
    unittest.main()
