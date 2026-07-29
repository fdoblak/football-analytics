"""R1-F1-R3 integrity: SoccerNet detector evaluation blocker / no false success media."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"


class R1F1R3IntegrityTests(unittest.TestCase):
    def test_inventory_classifies_coco_not_football_ft(self) -> None:
        inv = json.loads(
            (EV / "r1_f1_r3_soccernet_detector_inventory.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inv["model_filename"], "yolo11m.pt")
        self.assertFalse(inv["classification"]["SoccerNet_fine_tuned_football_detector"])
        self.assertTrue(inv["classification"]["COCO_pretrained_generic_model"])
        self.assertIn("UNPROVEN", inv["fine_tune_status"])

    def test_gate_is_nogo_or_pass_with_findings(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        g = gate["gate"]
        self.assertTrue(
            g.startswith("NO-GO — OFFICIAL SOCCER FOOTBALL DETECTOR")
            or g.startswith("PASS_WITH_FINDINGS — SOCCERNET FOOTBALL HUMAN DETECTOR")
            or g.startswith("PASS — INDEPENDENT HUMAN GT REVIEW TOOL READY")
            or g.startswith("PASS — WINDOWS GT REVIEW LAUNCHER VERIFIED")
            or g.startswith("NO-GO — REVIEWED GT INTEGRITY FAILURE")
            or g.startswith("NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE")
            or g.startswith("PASS_WITH_FINDINGS — INDEPENDENT HUMAN DETECTOR ACCEPTED")
        )

    def test_nogo_has_no_candidate_media(self) -> None:
        # R3 blocker artifact remains the source of truth for SoccerNet NO-GO media policy.
        blocker = json.loads((EV / "r1_f1_r3_BLOCKER.json").read_text(encoding="utf-8"))
        self.assertTrue(blocker["gate"].startswith("NO-GO — OFFICIAL SOCCER FOOTBALL DETECTOR"))
        self.assertFalse(blocker.get("media_retained", True))
        self.assertIn("football-domain detector fine-tuning", blocker["next_mandatory_step"])
        cand = EV / "r1_soccernet_detector_candidate"
        if cand.is_dir():
            for pat in ("*.mp4", "*.png", "*.html"):
                self.assertEqual(list(cand.glob(pat)), [])

    def test_yolo11m_not_in_archive_on_nogo(self) -> None:
        blocker = json.loads((EV / "r1_f1_r3_BLOCKER.json").read_text(encoding="utf-8"))
        if not blocker["gate"].startswith("NO-GO"):
            self.skipTest("selected model may remain archived")
        arch = Path("/home/fdoblak/football_data/model_archive/yolo11m.pt")
        self.assertFalse(arch.is_file())


if __name__ == "__main__":
    unittest.main()
