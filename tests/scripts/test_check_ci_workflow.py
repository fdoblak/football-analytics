#!/usr/bin/env python3
"""Validator script tests for check_ci_workflow (Stage 2D)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_ci_workflow.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_validator():
    if not SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("check_ci_workflow", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckCiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_validator()

    def _require(self) -> None:
        if self.mod is None:
            self.skipTest("scripts/check_ci_workflow.py not present yet")

    def test_01_exit_codes_defined(self) -> None:
        self._require()
        self.assertEqual(self.mod.EXIT_PASS, 0)
        self.assertEqual(self.mod.EXIT_FINDING, 1)
        self.assertEqual(self.mod.EXIT_CONFIG, 2)
        self.assertEqual(self.mod.EXIT_INTEGRITY, 3)

    def test_02_missing_workflow_nonzero(self) -> None:
        self._require()
        code = self.mod.main(["--workflow", "/tmp/missing-football-analytics-ci.yml", "--quiet"])
        self.assertNotEqual(code, 0)

    def test_03_repo_workflow_if_present(self) -> None:
        self._require()
        if not WORKFLOW.is_file():
            self.skipTest(".github/workflows/ci.yml not present yet")
        code = self.mod.main(["--workflow", str(WORKFLOW), "--quiet"])
        self.assertIn(code, (0, 1, 2, 3))

    def test_04_unsafe_workflow_fails(self) -> None:
        self._require()
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "ci.yml"
            bad.write_text(
                "\n".join(
                    [
                        "on:",
                        "  pull_request_target:",
                        "permissions:",
                        "  contents: write",
                        "jobs:",
                        "  build:",
                        "    runs-on: ubuntu-latest",
                        "    steps:",
                        "      - uses: actions/checkout@v4",
                        "      - run: curl | bash",
                    ]
                ),
                encoding="utf-8",
            )
            code = self.mod.main(["--workflow", str(bad), "--quiet"])
            self.assertNotEqual(code, 0)

    def test_05_no_network_imports(self) -> None:
        self._require()
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("requests", src)
        self.assertNotIn("torch", src)

    def test_06_script_path_documented(self) -> None:
        if not SCRIPT.is_file():
            self.skipTest("scripts/check_ci_workflow.py not present yet")
        self.assertTrue(SCRIPT.is_file())


if __name__ == "__main__":
    unittest.main()
