#!/usr/bin/env python3
"""CLI foundation command tests (Stage 2B)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from football_analytics.cli import main
from football_analytics.core.config import default_defaults_path
from football_analytics.core.run_id import validate_run_id

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


class CliRuntimeFoundationTests(unittest.TestCase):
    def test_01_existing_version(self) -> None:
        self.assertEqual(main(["--version"]), 0)

    def test_02_existing_info(self) -> None:
        self.assertEqual(main(["info"]), 0)

    def test_03_run_id(self) -> None:
        # Capture via subprocess for stdout
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "run-id"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0)
        validate_run_id(proc.stdout.strip())

    def test_04_config_validate(self) -> None:
        code = main(["config", "validate", "--config", str(default_defaults_path())])
        self.assertEqual(code, 0)

    def test_05_config_fingerprint_json(self) -> None:
        proc = subprocess.run(
            [
                PY,
                "-m",
                "football_analytics",
                "config",
                "fingerprint",
                "--config",
                str(default_defaults_path()),
                "--json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["digest"]), 64)

    def test_06_environment_show_json(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "environment", "show", "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(
            payload["gpu_validation"]["classification"], "AGENT_CONTEXT_GPU_UNVERIFIABLE"
        )
        blob = proc.stdout.lower()
        self.assertNotIn("ghp_", blob)
        self.assertNotIn("begin private key", blob)

    def test_07_invalid_config_nonzero(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("pipeline:\n  x: 1\n", encoding="utf-8")
            code = main(["config", "validate", "--config", str(bad)])
            self.assertNotEqual(code, 0)

    def test_08_no_torch_in_cli_source(self) -> None:
        src = (REPO_ROOT / "src/football_analytics/cli.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", src)
        self.assertNotIn("torch.cuda", src)


if __name__ == "__main__":
    unittest.main()
