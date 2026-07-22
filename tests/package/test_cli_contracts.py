#!/usr/bin/env python3
"""CLI contracts command tests (Stage 2C)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from football_analytics.cli import main
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


class CliContractsTests(unittest.TestCase):
    def test_01_list(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "contracts", "list"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=ENV,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("videos", proc.stdout)
        self.assertIn("detections", proc.stdout)

    def test_02_show(self) -> None:
        self.assertEqual(main(["contracts", "show", "videos", "--version", "1"]), 0)

    def test_03_fingerprint_json(self) -> None:
        proc = subprocess.run(
            [
                PY,
                "-m",
                "football_analytics",
                "contracts",
                "fingerprint",
                "detections",
                "--version",
                "1",
                "--json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=ENV,
        )
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["digest"]), 64)
        self.assertNotIn("password", proc.stdout.lower())

    def test_04_validate_success(self) -> None:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        v0 = reg.load_contract("detections", 0)
        rid = generate_run_id()
        table = pa.Table.from_pylist(
            [
                {
                    "run_id": rid,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_name": "ball",
                    "confidence": 0.5,
                    "bbox_x": 1.0,
                    "bbox_y": 1.0,
                    "bbox_width": 2.0,
                    "bbox_height": 2.0,
                    "model_id": "legacy",
                }
            ],
            schema=compile_arrow_schema(v0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "v0.parquet"
            write_contract_parquet(table, path, v0, contain_root=root)
            code = main(["contracts", "validate", "detections", str(path), "--version", "0"])
            self.assertEqual(code, 0)

    def test_05_validate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.parquet"
            bad.write_bytes(b"not-parquet")
            code = main(["contracts", "validate", "videos", str(bad), "--version", "1"])
            self.assertNotEqual(code, 0)

    def test_06_migrate_success(self) -> None:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        v0 = reg.load_contract("detections", 0)
        rid = generate_run_id()
        table = pa.Table.from_pylist(
            [
                {
                    "run_id": rid,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_name": "player",
                    "confidence": 0.8,
                    "bbox_x": 10.0,
                    "bbox_y": 20.0,
                    "bbox_width": 30.0,
                    "bbox_height": 40.0,
                    "model_id": "legacy",
                }
            ],
            schema=compile_arrow_schema(v0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "v0.parquet"
            dst = root / "v1.parquet"
            write_contract_parquet(table, src, v0, contain_root=root)
            code = main(
                [
                    "contracts",
                    "migrate",
                    "detections",
                    str(src),
                    str(dst),
                    "--from-version",
                    "0",
                    "--to-version",
                    "1",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(dst.is_file())
            receipt = dst.with_suffix(dst.suffix + ".migration_receipt.json")
            self.assertTrue(receipt.is_file())

    def test_07_migrate_overwrite_rejected(self) -> None:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        v0 = reg.load_contract("detections", 0)
        rid = generate_run_id()
        table = pa.Table.from_pylist(
            [
                {
                    "run_id": rid,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_name": "ball",
                    "confidence": 0.5,
                    "bbox_x": 1.0,
                    "bbox_y": 1.0,
                    "bbox_width": 2.0,
                    "bbox_height": 2.0,
                    "model_id": "legacy",
                }
            ],
            schema=compile_arrow_schema(v0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "v0.parquet"
            dst = root / "v1.parquet"
            write_contract_parquet(table, src, v0, contain_root=root)
            dst.write_bytes(b"exists")
            code = main(
                [
                    "contracts",
                    "migrate",
                    "detections",
                    str(src),
                    str(dst),
                    "--from-version",
                    "0",
                    "--to-version",
                    "1",
                ]
            )
            self.assertNotEqual(code, 0)

    def test_08_unknown_contract_nonzero(self) -> None:
        code = main(["contracts", "show", "does_not_exist"])
        self.assertNotEqual(code, 0)

    def test_09_existing_cli_regression(self) -> None:
        self.assertEqual(main(["--version"]), 0)
        self.assertEqual(main(["info"]), 0)

    def test_10_no_eager_pyarrow_in_main_package(self) -> None:
        src = (REPO_ROOT / "src/football_analytics/__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("import pyarrow", src)
        self.assertNotIn("from pyarrow", src)


if __name__ == "__main__":
    unittest.main()
