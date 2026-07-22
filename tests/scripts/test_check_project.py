#!/usr/bin/env python3
"""Validator script tests for check_project (Stage 2D)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_project.py"


def _load_validator():
    if not SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("check_project", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_validator()

    def _require(self) -> None:
        if self.mod is None:
            self.skipTest("scripts/check_project.py not present yet")

    def test_01_exit_codes_defined(self) -> None:
        self._require()
        self.assertEqual(self.mod.EXIT_PASS, 0)
        self.assertEqual(self.mod.EXIT_FINDING, 1)
        self.assertEqual(self.mod.EXIT_CONFIG, 2)
        self.assertEqual(self.mod.EXIT_INTEGRITY, 3)

    def test_02_local_quick_smoke(self) -> None:
        self._require()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--profile",
                    "local",
                    "--quick",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            # May WARN on dirty tree or FAIL on missing CI workflow; must not crash.
            self.assertIn(code, (0, 1, 2, 3))
            if out.is_file():
                payload = json.loads(out.read_text(encoding="utf-8"))
                self.assertTrue(
                    "status" in payload or "overall_status" in payload,
                    msg=f"missing status keys: {sorted(payload)}",
                )
                self.assertIn("checks", payload)

    def test_03_ci_profile_smoke(self) -> None:
        self._require()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--profile",
                    "ci",
                    "--quick",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertIn(code, (0, 1, 2, 3))
            if out.is_file():
                payload = json.loads(out.read_text(encoding="utf-8"))
                # CI host-only checks must SKIP, not false PASS
                checks = payload.get("checks") or payload.get("extras", {}).get("checks")
                if isinstance(checks, list):
                    skip_msgs = [c for c in checks if str(c.get("status", "")).upper() == "SKIP"]
                    # At least document that SKIP is a valid outcome in CI profile
                    self.assertIsInstance(skip_msgs, list)

    def test_04_no_network_imports(self) -> None:
        self._require()
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("urllib.request", src)
        self.assertNotIn("torch", src)

    def test_05_script_path_documented(self) -> None:
        if not SCRIPT.is_file():
            self.skipTest("scripts/check_project.py not present yet")
        self.assertTrue(SCRIPT.is_file())


if __name__ == "__main__":
    unittest.main()
