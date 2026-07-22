#!/usr/bin/env python3
"""Validator script tests (Stage 2B)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_validator():
    path = REPO_ROOT / "scripts" / "check_runtime_foundation.py"
    spec = importlib.util.spec_from_file_location("check_runtime_foundation", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckRuntimeFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_validator()

    def test_01_defaults_pass(self) -> None:
        code = self.mod.main(
            ["--config", str(REPO_ROOT / "configs/project/defaults.yaml"), "--quiet"]
        )
        self.assertEqual(code, 0)

    def test_02_synthetic_run_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--config",
                    str(REPO_ROOT / "configs/project/defaults.yaml"),
                    "--synthetic-run",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            self.assertTrue(payload["extras"].get("synthetic_cleaned"))

    def test_03_missing_config_nonzero(self) -> None:
        code = self.mod.main(["--config", "/tmp/missing-football-analytics-config.yaml", "--quiet"])
        self.assertNotEqual(code, 0)

    def test_04_exit_codes_defined(self) -> None:
        self.assertEqual(self.mod.EXIT_PASS, 0)
        self.assertEqual(self.mod.EXIT_FINDING, 1)
        self.assertEqual(self.mod.EXIT_CONFIG, 2)
        self.assertEqual(self.mod.EXIT_INTEGRITY, 3)


if __name__ == "__main__":
    unittest.main()
