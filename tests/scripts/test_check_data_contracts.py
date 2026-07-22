#!/usr/bin/env python3
"""Validator script tests (Stage 2C data contracts)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_validator():
    path = REPO_ROOT / "scripts" / "check_data_contracts.py"
    spec = importlib.util.spec_from_file_location("check_data_contracts", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckDataContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_validator()

    def test_01_registry_pass(self) -> None:
        code = self.mod.main(
            ["--registry", str(REPO_ROOT / "configs/data/schema_registry.yaml"), "--quiet"]
        )
        self.assertEqual(code, 0)

    def test_02_synthetic_and_migration_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--registry",
                    str(REPO_ROOT / "configs/data/schema_registry.yaml"),
                    "--synthetic-roundtrip",
                    "--migration-smoke",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            self.assertTrue(payload["extras"].get("synthetic_roundtrip"))
            self.assertTrue(payload["extras"].get("migration_smoke"))
            self.assertTrue(payload["extras"].get("fixture_cleaned"))
            fixture = payload["extras"].get("fixture_root")
            if fixture:
                self.assertFalse(Path(fixture).exists())

    def test_03_missing_registry_nonzero(self) -> None:
        code = self.mod.main(
            ["--registry", "/tmp/missing-football-analytics-registry.yaml", "--quiet"]
        )
        self.assertNotEqual(code, 0)

    def test_04_exit_codes_defined(self) -> None:
        self.assertEqual(self.mod.EXIT_PASS, 0)
        self.assertEqual(self.mod.EXIT_FINDING, 1)
        self.assertEqual(self.mod.EXIT_CONFIG, 2)
        self.assertEqual(self.mod.EXIT_INTEGRITY, 3)

    def test_05_selected_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = self.mod.main(
                [
                    "--registry",
                    str(REPO_ROOT / "configs/data/schema_registry.yaml"),
                    "--contract",
                    "events",
                    "--version",
                    "1",
                    "--json-out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["extras"]["selected"]["contract"], "events")
            self.assertEqual(len(payload["extras"]["selected"]["fingerprint"]), 64)

    def test_06_no_network_imports_in_script(self) -> None:
        src = (REPO_ROOT / "scripts" / "check_data_contracts.py").read_text(encoding="utf-8")
        self.assertNotIn("requests", src)
        self.assertNotIn("urllib.request", src)
        self.assertNotIn("torch", src)


if __name__ == "__main__":
    unittest.main()
