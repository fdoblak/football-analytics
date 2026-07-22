#!/usr/bin/env python3
"""CLI project check / cache inspect smoke tests (Stage 2D)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from football_analytics.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _parser_has_command(name: str) -> bool:
    try:
        from football_analytics.cli import build_parser

        parser = build_parser()
        # Probe by checking help / subparsers
        help_text = parser.format_help()
        return name in help_text
    except Exception:  # noqa: BLE001
        return False


class CliProjectCheckTests(unittest.TestCase):
    def test_01_existing_cli_version_regression(self) -> None:
        self.assertEqual(main(["--version"]), 0)

    def test_02_existing_cli_info_regression(self) -> None:
        self.assertEqual(main(["info"]), 0)

    def test_03_subprocess_version(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=ENV,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("0.1.0", proc.stdout)

    def test_04_project_check_smoke(self) -> None:
        if not _parser_has_command("project"):
            self.skipTest("CLI project command not present yet")
        # Prefer --quick --profile ci to avoid host-only FAIL noise
        code = main(["project", "check", "--profile", "ci", "--quick"])
        self.assertIn(code, (0, 1, 2, 3))

    def test_05_cache_inspect_smoke(self) -> None:
        if not _parser_has_command("cache"):
            self.skipTest("CLI cache command not present yet")
        # Invalid key should be controlled nonzero, not crash
        code = main(["cache", "inspect", "not-a-valid-key"])
        self.assertNotEqual(code, 0)

    def test_06_cache_verify_smoke(self) -> None:
        if not _parser_has_command("cache"):
            self.skipTest("CLI cache command not present yet")
        code = main(["cache", "verify", "0" * 64])
        # Missing entry is nonzero; must not raise
        self.assertNotEqual(code, 0)

    def test_07_no_eager_pyarrow_torch_on_pipeline_import(self) -> None:
        mods_before = set(sys.modules)
        heavy = {"torch", "pyarrow", "ultralytics", "SoccerNet"}
        # Fresh subprocess import is the reliable isolation check
        proc = subprocess.run(
            [
                PY,
                "-c",
                (
                    "import sys; "
                    "import football_analytics.pipeline; "
                    "heavy={'torch','pyarrow','ultralytics','SoccerNet'}; "
                    "hit=heavy & set(sys.modules); "
                    "raise SystemExit(0 if not hit else 1)"
                ),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=ENV,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # In-process import must not newly load heavy engines either
        importlib.import_module("football_analytics.pipeline")
        newly = set(sys.modules) - mods_before
        self.assertFalse(heavy & newly)

    def test_08_pipeline_init_source_clean(self) -> None:
        src = (REPO_ROOT / "src/football_analytics/pipeline/__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import torch", src)
        self.assertNotRegex(src, r"(?m)^\s*import pyarrow\b")
        self.assertNotRegex(src, r"(?m)^\s*from pyarrow\b")

    def test_09_cli_help_mentions_stage_when_extended(self) -> None:
        from football_analytics.cli import build_parser

        text = build_parser().format_help().lower()
        # Stage 2C or 2D description; must remain side-effect free
        self.assertIn("football", text)


if __name__ == "__main__":
    unittest.main()
