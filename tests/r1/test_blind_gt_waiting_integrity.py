"""R1-F1-R2 integrity: prior packages rejected; stronger candidate NO-GO or v2."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01"


class R1F1R2IntegrityTests(unittest.TestCase):
    def test_prior_preacceptance_rejected(self) -> None:
        rej = json.loads((EV / "r1_f1_r2_REJECTION.json").read_text(encoding="utf-8"))
        self.assertIn("REJECTED_BY_USER", rej["status"])
        self.assertFalse(rej["acceptance_eligible"])
        st = json.loads(
            (EV / "r1_detection_preacceptance" / "REJECTED_STATUS.json").read_text(encoding="utf-8")
        )
        self.assertTrue(st["do_not_use_as_preacceptance"])

    def test_no_active_preacceptance_media(self) -> None:
        pre = EV / "r1_detection_preacceptance"
        for pat in ("*.mp4", "*.png", "*.html"):
            self.assertEqual(list(pre.glob(pat)), [])

    def test_gate_file(self) -> None:
        gate = json.loads((EV / "GATE_STATUS.json").read_text(encoding="utf-8"))
        self.assertTrue(
            gate["gate"].startswith("NO-GO")
            or "STRONGER HUMAN DETECTION CANDIDATE READY" in gate["gate"]
            or gate["gate"].startswith("PASS — INDEPENDENT HUMAN GT REVIEW TOOL READY")
            or gate["gate"].startswith("PASS — WINDOWS GT REVIEW LAUNCHER VERIFIED")
        )

    def test_cleanup_receipt(self) -> None:
        rec = json.loads((EV / "r1_f1_r2_CLEANUP_RECEIPT.json").read_text(encoding="utf-8"))
        self.assertFalse(rec["data_loss"])
        self.assertGreaterEqual(rec["n_git_deleted"], 1)


if __name__ == "__main__":
    unittest.main()
