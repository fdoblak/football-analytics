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
        )

    def test_nogo_has_no_candidate_media(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        if not gate["gate"].startswith("NO-GO"):
            self.skipTest("PASS path retains media")
        cand = EV / "r1_soccernet_detector_candidate"
        if cand.is_dir():
            for pat in ("*.mp4", "*.png", "*.html"):
                self.assertEqual(list(cand.glob(pat)), [])
        blocker = json.loads((EV / "r1_f1_r3_BLOCKER.json").read_text(encoding="utf-8"))
        self.assertFalse(blocker.get("media_retained", True))
        self.assertIn("football-domain detector fine-tuning", blocker["next_mandatory_step"])

    def test_yolo11m_not_in_archive_on_nogo(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        if not gate["gate"].startswith("NO-GO"):
            self.skipTest("selected model may remain archived")
        arch = Path("/home/fdoblak/football_data/model_archive/yolo11m.pt")
        self.assertFalse(arch.is_file())


if __name__ == "__main__":
    unittest.main()
