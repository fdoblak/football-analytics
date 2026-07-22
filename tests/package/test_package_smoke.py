#!/usr/bin/env python3
"""Stage 2A package / metadata / CLI smoke tests."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


class PackageSmokeTests(unittest.TestCase):
    def test_01_package_import(self) -> None:
        import football_analytics

        self.assertTrue(hasattr(football_analytics, "__version__"))

    def test_02_exact_version(self) -> None:
        import football_analytics

        self.assertEqual(football_analytics.__version__, "0.1.0.dev0")

    def test_03_import_side_effect_free(self) -> None:
        # Re-import should not pull heavy engines
        mods_before = set(sys.modules)
        importlib.reload(importlib.import_module("football_analytics"))
        heavy = {"torch", "ultralytics", "SoccerNet", "cv2", "pandas"}
        newly = set(sys.modules) - mods_before
        self.assertFalse(heavy & newly)

    def test_04_module_version(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout.strip(), "0.1.0.dev0")

    def test_05_console_script_version(self) -> None:
        script = Path(sys.executable).parent / "football-analytics"
        if not script.is_file():
            self.skipTest("console script not installed yet")
        proc = subprocess.run(
            [str(script), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout.strip(), "0.1.0.dev0")

    def test_06_info(self) -> None:
        from football_analytics.cli import main

        code = main(["info"])
        self.assertEqual(code, 0)

    def test_07_unknown_command_nonzero(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "not-a-command"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_08_cli_no_secret_dump(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "info"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        blob = (proc.stdout + proc.stderr).lower()
        for bad in ("password=", "token=", "ghp_", "begin private key", "api_key="):
            self.assertNotIn(bad, blob)

    def test_09_missing_config_controlled(self) -> None:
        from football_analytics.cli import _read_active_backend

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "paths.yaml"
            val = _read_active_backend(missing)
            self.assertIn("missing", val)

    def test_10_package_metadata(self) -> None:
        try:
            import importlib.metadata as md
        except ImportError:  # pragma: no cover
            self.skipTest("importlib.metadata missing")
        try:
            dist = md.metadata("football-analytics")
        except md.PackageNotFoundError:
            self.skipTest("editable package not installed")
        self.assertEqual(dist["Name"], "football-analytics")
        self.assertEqual(dist["Version"], "0.1.0.dev0")

    def test_11_pyproject_parse(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "football-analytics"', text)
        self.assertIn('version = "0.1.0.dev0"', text)
        self.assertIn("football-analytics =", text)

    def test_12_requirements_no_file_urls(self) -> None:
        for rel in (
            "requirements/base.txt",
            "requirements/dev.txt",
            "requirements/constraints-ai-dev.txt",
        ):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("file://", text)

    def test_13_requirements_no_conflict_markers(self) -> None:
        base = (REPO_ROOT / "requirements/base.txt").read_text(encoding="utf-8")
        self.assertIn("PyYAML", base)
        self.assertIn("pyarrow", base)
        self.assertNotIn("torch", base.lower())

    def test_14_protected_constraints_present(self) -> None:
        text = (REPO_ROOT / "requirements/constraints-ai-dev.txt").read_text(encoding="utf-8")
        for pin in (
            "torch==2.11.0+cu128",
            "numpy==2.2.6",
            "opencv-python==5.0.0.93",
            "ultralytics==8.4.91",
            "SoccerNet==0.1.62",
        ):
            self.assertIn(pin, text)

    def test_15_environment_name_ai_dev(self) -> None:
        text = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
        self.assertIn("name: ai-dev", text)

    def test_16_readme_documents_existing_commands(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("football-analytics --version", readme)
        self.assertIn("football-analytics info", readme)
        self.assertNotIn("football-analytics detect", readme)
        self.assertNotIn("football-analytics track", readme)

    def test_17_import_does_not_load_heavy_engines(self) -> None:
        # Ensure package import path modules don't import torch at load
        cli_src = (REPO_ROOT / "src/football_analytics/cli.py").read_text(encoding="utf-8")
        init_src = (REPO_ROOT / "src/football_analytics/__init__.py").read_text(encoding="utf-8")
        for src in (cli_src, init_src):
            self.assertNotRegex(src, r"(?m)^\s*import torch\b")
            self.assertNotRegex(src, r"(?m)^\s*from torch")
            self.assertNotIn("ultralytics", src)
            self.assertNotIn("SoccerNet", src)

    def test_18_cli_does_not_start_gpu(self) -> None:
        cli_src = (REPO_ROOT / "src/football_analytics/cli.py").read_text(encoding="utf-8")
        self.assertNotIn("cuda", cli_src.lower().split("gpu_classification")[0])
        # info may mention classification string but must not call torch.cuda
        self.assertNotIn("torch.cuda", cli_src)

    def test_19_license_proprietary(self) -> None:
        lic = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("All rights reserved", lic)
        self.assertIn("Furkan Doblak", lic)
        self.assertIn("Permission is NOT granted", lic)

    def test_20_gitignore_env_and_artifacts(self) -> None:
        gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pat in (".env", "dist/", "*.pth", "*.mp4", "*.egg-info/"):
            self.assertIn(pat, gi)
        self.assertIn("!.env.example", gi)

    def test_21_github_workflow_security_rules(self) -> None:
        doc = (REPO_ROOT / "docs/development/git_github_workflow.md").read_text(encoding="utf-8")
        self.assertIn("git add .", doc)
        self.assertIn("Force push", doc)
        self.assertIn("private", doc.lower())
        self.assertIn("credential", doc.lower())

    def test_22_remote_url_no_credential_policy(self) -> None:
        doc = (REPO_ROOT / "docs/development/git_github_workflow.md").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"must not embed credentials|credential", doc, re.I))


if __name__ == "__main__":
    unittest.main()
