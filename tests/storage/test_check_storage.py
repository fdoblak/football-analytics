#!/usr/bin/env python3
"""Unit tests for scripts/check_storage.py (stdlib unittest, no pytest)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path


def load_check_storage():
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "check_storage.py"
    module_name = "check_storage_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses look up the module in sys.modules during class creation
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CS = load_check_storage()


REQUIRED_DIRS = [
    "videos/raw_matches",
    "videos/test_clips",
    "datasets",
    "results",
    "rendered_outputs",
    "reports",
    "model_archive",
    "experiments_archive",
    "backups",
]


def make_storage_tree(root: Path) -> None:
    for rel in REQUIRED_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


def write_config(path: Path, active_root: Path, **overrides) -> None:
    storage = {
        "active_backend": "wsl_local",
        "active_root": str(active_root),
        "planned_archive_root": "/mnt/d/football_data",
        "planned_archive_status": "unverified",
        "ssd_root": str(active_root),
        "raw_matches": str(active_root / "videos" / "raw_matches"),
        "test_clips": str(active_root / "videos" / "test_clips"),
        "datasets": str(active_root / "datasets"),
        "results": str(active_root / "results"),
        "rendered_outputs": str(active_root / "rendered_outputs"),
        "reports": str(active_root / "reports"),
        "model_archive": str(active_root / "model_archive"),
        "experiments_archive": str(active_root / "experiments_archive"),
        "backups": str(active_root / "backups"),
    }
    storage.update(overrides.get("storage_overrides", {}))
    validation = {
        "minimum_free_bytes": 1024,
        "warning_free_bytes": 2048,
        "require_absolute_paths": True,
        "require_paths_under_active_root": True,
        "reject_symlink_escape": True,
    }
    validation.update(overrides.get("validation_overrides", {}))
    text = (
        "schema_version: 1\n"
        "storage:\n"
        + "".join(f"  {k}: {json.dumps(v)}\n" for k, v in storage.items())
        + "storage_validation:\n"
        + "".join(f"  {k}: {json.dumps(v)}\n" for k, v in validation.items())
    )
    # For integer thresholds yaml needs unquoted ints - rewrite validation ints plainly
    lines = ["schema_version: 1", "storage:"]
    for k, v in storage.items():
        if isinstance(v, str):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("storage_validation:")
    for k, v in validation.items():
        if isinstance(v, bool):
            lines.append(f"  {k}: {'true' if v else 'false'}")
        else:
            lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CheckStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="fa_storage_test_")
        self.tmp = Path(self._tmpdir.name)
        self.storage = self.tmp / "football_data"
        make_storage_tree(self.storage)
        self.config = self.tmp / "paths.yaml"
        write_config(self.config, self.storage)
        self._before = set()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_01_valid_tree_pass(self) -> None:
        result = CS.run_validation(self.config)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.errors)

    def test_02_relative_active_root_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            storage_overrides={"active_root": "relative/football_data"},
        )
        result = CS.run_validation(self.config)
        self.assertNotEqual(result.exit_code, 0)
        self.assertTrue(any("absolute" in e.lower() for e in result.errors))

    def test_03_root_slash_rejected(self) -> None:
        write_config(self.config, self.storage, storage_overrides={"active_root": "/"})
        # Also force leaf paths under / would be invalid separately; root alone fails first
        result = CS.run_validation(self.config)
        self.assertTrue(any("forbidden" in e.lower() or "absolute" in e.lower() for e in result.errors) or result.errors)
        self.assertNotEqual(result.exit_code, 0)

    def test_04_broad_home_root_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            storage_overrides={"active_root": "/home/fdoblak"},
        )
        result = CS.run_validation(self.config)
        self.assertTrue(any("forbidden" in e.lower() for e in result.errors))

    def test_05_missing_required_path_rejected(self) -> None:
        text = self.config.read_text(encoding="utf-8").replace(
            f"  backups: {self.storage / 'backups'}\n", ""
        )
        self.config.write_text(text, encoding="utf-8")
        result = CS.run_validation(self.config)
        self.assertTrue(any("backups" in e for e in result.errors))
        self.assertEqual(result.exit_code, CS.EXIT_CONFIG)

    def test_06_path_outside_active_root_rejected(self) -> None:
        outside = self.tmp / "outside"
        outside.mkdir()
        write_config(
            self.config,
            self.storage,
            storage_overrides={"datasets": str(outside)},
        )
        result = CS.run_validation(self.config)
        self.assertTrue(any("datasets" in e and "escape" in e.lower() for e in result.errors))

    def test_07_dotdot_escape_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            storage_overrides={
                "datasets": str(self.storage / "datasets" / ".." / ".." / "nope")
            },
        )
        result = CS.run_validation(self.config)
        self.assertTrue(result.errors)

    def test_08_symlink_escape_rejected(self) -> None:
        outside = self.tmp / "escape_target"
        outside.mkdir()
        link = self.storage / "datasets_link"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(outside)
        write_config(
            self.config,
            self.storage,
            storage_overrides={"datasets": str(link)},
        )
        result = CS.run_validation(self.config)
        self.assertTrue(any("escape" in e.lower() for e in result.errors))

    def test_09_duplicate_canonical_path_rejected(self) -> None:
        dup = str(self.storage / "datasets")
        write_config(
            self.config,
            self.storage,
            storage_overrides={"reports": dup},
        )
        result = CS.run_validation(self.config)
        self.assertTrue(any("duplicate" in e.lower() for e in result.errors))

    def test_10_unknown_backend_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            storage_overrides={"active_backend": "nfs_mystery"},
        )
        result = CS.run_validation(self.config)
        self.assertEqual(result.exit_code, CS.EXIT_CONFIG)
        self.assertTrue(any("Unsupported active_backend" in e for e in result.errors))

    def test_11_threshold_string_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            validation_overrides={"minimum_free_bytes": "big"},
        )
        # write_config stringifies poorly; write raw yaml
        self.config.write_text(
            self.config.read_text(encoding="utf-8").replace(
                "minimum_free_bytes: 1024", 'minimum_free_bytes: "big"'
            ),
            encoding="utf-8",
        )
        result = CS.run_validation(self.config)
        self.assertEqual(result.exit_code, CS.EXIT_CONFIG)

    def test_12_minimum_ge_warning_rejected(self) -> None:
        write_config(
            self.config,
            self.storage,
            validation_overrides={
                "minimum_free_bytes": 5000,
                "warning_free_bytes": 2000,
            },
        )
        result = CS.run_validation(self.config)
        self.assertEqual(result.exit_code, CS.EXIT_CONFIG)
        self.assertTrue(any("strictly less" in e for e in result.errors))

    def test_13_missing_directory_fails(self) -> None:
        shutil.rmtree(self.storage / "datasets")
        result = CS.run_validation(self.config)
        self.assertTrue(any("datasets" in e and "does not exist" in e for e in result.errors))
        self.assertEqual(result.exit_code, CS.EXIT_FAIL)

    def test_14_readonly_default_creates_no_files(self) -> None:
        before = {p for p in self.storage.rglob("*")}
        result = CS.run_validation(self.config, do_probe=False)
        after = {p for p in self.storage.rglob("*")}
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(before, after)
        self.assertFalse(any(p.name.startswith(".storage_probe_") for p in after))

    def test_15_probe_write_read_hash_cleanup(self) -> None:
        result = CS.run_validation(self.config, do_probe=True)
        self.assertTrue(result.probe.get("passed"))
        self.assertTrue(result.probe.get("cleanup_verified"))
        leftovers = list(self.storage.glob(".storage_probe_*"))
        self.assertEqual(leftovers, [])
        self.assertEqual(result.exit_code, 0)

    def test_16_probe_refuses_overwrite(self) -> None:
        existing = self.storage / ".storage_probe_collision"
        existing.write_text("do-not-touch\n", encoding="utf-8")
        # Force probe name by patching secrets/time is hard; instead call run_probe with monkeypatch via crafting FileExists by precreating matching pattern is random.
        # Directly invoke run_probe after creating exclusive conflict by patching Path behavior is complex.
        # Simulate FileExistsError path: open O_EXCL on existing exact file through run_probe after replacing name generation.
        original_token = CS.secrets.token_hex
        original_dt = CS.datetime

        class FakeDT:
            @staticmethod
            def now(tz=None):
                class T:
                    @staticmethod
                    def strftime(_fmt):
                        return "FIXEDTS"

                return T()

        CS.datetime = FakeDT  # type: ignore
        CS.secrets.token_hex = lambda n=8: "fixedhex"  # type: ignore
        conflict = self.storage / ".storage_probe_FIXEDTS_fixedhex"
        conflict.write_bytes(b"preexisting")
        try:
            result = CS.ValidationResult()
            security = CS.run_probe(self.storage.resolve(), result)
            self.assertTrue(security or result.errors)
            self.assertTrue(conflict.exists())
            self.assertEqual(conflict.read_bytes(), b"preexisting")
        finally:
            CS.secrets.token_hex = original_token
            CS.datetime = original_dt
            if conflict.exists():
                conflict.unlink()
            if existing.exists():
                existing.unlink()

    def test_17_probe_leaves_no_residue(self) -> None:
        CS.run_validation(self.config, do_probe=True)
        residue = list(self.storage.glob(".storage_probe_*"))
        self.assertEqual(residue, [])

    def test_18_unverified_archive_missing_ok(self) -> None:
        write_config(
            self.config,
            self.storage,
            storage_overrides={
                "planned_archive_root": "/mnt/d/football_data",
                "planned_archive_status": "unverified",
            },
        )
        self.assertFalse(Path("/mnt/d/football_data").exists())
        result = CS.run_validation(self.config)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.extras["planned_archive"]["validated_as_active"], False)

    def test_19_json_output_valid(self) -> None:
        out = self.tmp / "report.json"
        code = CS.main(
            ["--config", str(self.config), "--json-out", str(out), "--quiet"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("active_root", payload)
        self.assertIn("paths", payload)

    def test_20_exit_code_contract(self) -> None:
        ok = CS.run_validation(self.config)
        self.assertEqual(ok.exit_code, 0)
        write_config(
            self.config,
            self.storage,
            storage_overrides={"active_backend": "nope"},
        )
        bad = CS.run_validation(self.config)
        self.assertEqual(bad.exit_code, CS.EXIT_CONFIG)
        # usage error via CLI
        code = CS.main([])
        self.assertEqual(code, CS.EXIT_CONFIG)

    def test_21_world_writable_warning(self) -> None:
        mode = self.storage.stat().st_mode
        try:
            self.storage.chmod(mode | stat.S_IWOTH)
            result = CS.run_validation(self.config)
            self.assertTrue(any("world-writable" in w for w in result.warnings))
            self.assertIn(result.status, {"PASS_WITH_WARNINGS", "PASS"})
        finally:
            self.storage.chmod(mode)

    def test_22_broken_yaml_config_error(self) -> None:
        broken = self.tmp / "broken.yaml"
        broken.write_text("storage: [\n  - invalid\n", encoding="utf-8")
        result = CS.run_validation(broken)
        self.assertEqual(result.exit_code, CS.EXIT_CONFIG)
        self.assertTrue(any("YAML parse error" in e or "mapping" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
