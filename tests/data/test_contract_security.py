#!/usr/bin/env python3
"""Contract security and containment tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.core.run_id import generate_run_id
from football_analytics.data import DataContractError
from football_analytics.data.bundle import build_synthetic_bundle
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.specs import load_contract_spec


class ContractSecurityTests(unittest.TestCase):
    def test_01_spec_containment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            contain = Path(tmp) / "root"
            contain.mkdir()
            with self.assertRaises(DataContractError):
                load_contract_spec(outside, contain_root=contain)

    def test_02_oversized_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.json"
            p.write_text('{"x": "' + ("y" * 600000) + '"}', encoding="utf-8")
            with self.assertRaises(DataContractError):
                load_contract_spec(p, contain_root=Path(tmp))

    def test_03_registry_load_ok(self) -> None:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        self.assertEqual(reg.registry_version, 1)

    def test_04_write_permissions(self) -> None:
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        bundle = build_synthetic_bundle(generate_run_id())
        spec = reg.load_contract("videos", 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "videos.parquet"
            write_contract_parquet(bundle["videos"], path, spec, contain_root=root)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_05_wrong_contract_read(self) -> None:
        from football_analytics.data.parquet import read_contract_parquet

        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        bundle = build_synthetic_bundle(generate_run_id())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "videos.parquet"
            write_contract_parquet(
                bundle["videos"], path, reg.load_contract("videos", 1), contain_root=root
            )
            with self.assertRaises(DataContractError):
                read_contract_parquet(path, reg.load_contract("frames", 1), contain_root=root)

    def test_06_corrupt_parquet(self) -> None:
        from football_analytics.data.parquet import read_contract_parquet

        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bad.parquet"
            path.write_bytes(b"PAR1corrupt")
            with self.assertRaises(DataContractError):
                read_contract_parquet(path, reg.load_contract("videos", 1), contain_root=root)

    def test_07_migration_receipt_no_overwrite(self) -> None:
        import pyarrow as pa

        from football_analytics.data.compiler import compile_arrow_schema
        from football_analytics.data.migrations import migrate_parquet

        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        rid = generate_run_id()
        v0 = reg.load_contract("detections", 0)
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
            receipt = root / "receipt.json"
            write_contract_parquet(table, src, v0, contain_root=root)
            migrate_parquet(
                src,
                dst,
                registry=reg,
                contract="detections",
                from_version=0,
                to_version=1,
                receipt_path=receipt,
                contain_root=root,
            )
            dst2 = root / "v1b.parquet"
            with self.assertRaises(DataContractError):
                migrate_parquet(
                    src,
                    dst2,
                    registry=reg,
                    contract="detections",
                    from_version=0,
                    to_version=1,
                    receipt_path=receipt,
                    contain_root=root,
                )

    def test_08_package_import_no_pyarrow(self) -> None:
        import sys

        before = set(sys.modules)
        import football_analytics

        _ = football_analytics.__version__
        newly = set(sys.modules) - before
        self.assertNotIn("pyarrow", newly)


if __name__ == "__main__":
    unittest.main()
