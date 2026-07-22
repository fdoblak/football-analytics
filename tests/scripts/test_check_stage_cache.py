#!/usr/bin/env python3
"""Validator script tests for check_stage_cache (Stage 2D)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_stage_cache.py"


def _load_validator():
    if not SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("check_stage_cache", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckStageCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_validator()

    def _require(self):
        if self.mod is None:
            self.skipTest("scripts/check_stage_cache.py not present yet")

    def test_01_exit_codes_defined(self) -> None:
        self._require()
        self.assertEqual(self.mod.EXIT_PASS, 0)
        self.assertEqual(self.mod.EXIT_FINDING, 1)
        self.assertEqual(self.mod.EXIT_CONFIG, 2)
        self.assertEqual(self.mod.EXIT_INTEGRITY, 3)

    def test_02_policy_load_pass(self) -> None:
        self._require()
        code = self.mod.main(
            [
                "--config",
                str(REPO_ROOT / "configs/system/cache_policy.yaml"),
                "--paths-config",
                str(REPO_ROOT / "configs/system/paths.yaml"),
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)

    def test_03_synthetic_smoke(self) -> None:
        self._require()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--config",
                    str(REPO_ROOT / "configs/system/cache_policy.yaml"),
                    "--paths-config",
                    str(REPO_ROOT / "configs/system/paths.yaml"),
                    "--synthetic",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())

    def test_04_missing_config_nonzero(self) -> None:
        self._require()
        code = self.mod.main(
            [
                "--config",
                "/tmp/missing-football-analytics-cache-policy.yaml",
                "--quiet",
            ]
        )
        self.assertNotEqual(code, 0)

    def test_05_no_network_imports(self) -> None:
        self._require()
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("requests", src)
        self.assertNotIn("urllib.request", src)
        self.assertNotIn("torch", src)

    def test_06_script_exists_or_skipped(self) -> None:
        # Documents expected landing path for Stage 2D validators.
        if not SCRIPT.is_file():
            self.skipTest("scripts/check_stage_cache.py not present yet")
        self.assertTrue(SCRIPT.is_file())


if __name__ == "__main__":
    unittest.main()
