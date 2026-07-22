#!/usr/bin/env python3
"""Run context initialization tests (Stage 2B)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.config import ConfigError, default_defaults_path
from football_analytics.core.run_context import RunContextError, initialize_run_context
from football_analytics.core.run_id import generate_run_id


class RunContextTests(unittest.TestCase):
    def test_01_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init = initialize_run_context(
                runs_root=Path(tmp),
                defaults_path=default_defaults_path(),
                repo_root=Path(__file__).resolve().parents[2],
            )
            self.assertTrue(init.resolved_config_path.is_file())
            self.assertTrue(init.environment_path.is_file())
            self.assertTrue(init.run_context_path.is_file())
            self.assertTrue(init.log_path.is_file())
            ctx = json.loads(init.run_context_path.read_text(encoding="utf-8"))
            self.assertEqual(ctx["status"], "initialized")
            self.assertEqual(ctx["schema_version"], 1)

    def test_02_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rid = generate_run_id()
            initialize_run_context(
                runs_root=Path(tmp),
                run_id=rid,
                defaults_path=default_defaults_path(),
            )
            with self.assertRaises(RunContextError):
                initialize_run_context(
                    runs_root=Path(tmp),
                    run_id=rid,
                    defaults_path=default_defaults_path(),
                )

    def test_03_partial_failure_cleanup(self) -> None:
        # Invalid override causes failure before durable success; dir cleaned
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ConfigError, RunContextError)):
                initialize_run_context(
                    runs_root=Path(tmp),
                    defaults_path=default_defaults_path(),
                    overrides={"logging": {"level": "NOPE"}},
                )
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_04_hashes_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init = initialize_run_context(
                runs_root=Path(tmp),
                defaults_path=default_defaults_path(),
            )
            self.assertEqual(len(init.config_fingerprint["digest"]), 64)
            env = json.loads(init.environment_path.read_text(encoding="utf-8"))
            self.assertEqual(env["config_fingerprint"]["digest"], init.config_fingerprint["digest"])

    def test_05_distinct_from_run_manifest_status_enum(self) -> None:
        # run_context uses initialized; archive run_manifest uses pending/running/...
        schema = Path(__file__).resolve().parents[2] / "schemas" / "run_context.schema.json"
        text = schema.read_text(encoding="utf-8")
        self.assertIn('"initialized"', text)
        self.assertNotIn('"archived"', text)


if __name__ == "__main__":
    unittest.main()
