#!/usr/bin/env python3
"""Table / bundle / parquet / migration tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.run_id import generate_run_id
from football_analytics.data import DataContractError
from football_analytics.data.bundle import build_synthetic_bundle, validate_contract_bundle
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.migrations import migrate_parquet, migrate_table, plan_migration
from football_analytics.data.parquet import (
    inspect_contract_parquet,
    read_contract_parquet,
    write_contract_parquet,
)
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.validation import validate_table

ROOT = default_project_root()


class TableParquetMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.run_id = generate_run_id()
        self.bundle = build_synthetic_bundle(self.run_id)
        self.specs = {n: self.reg.load_contract(n, 1) for n in self.bundle}

    def test_01_each_table_valid(self) -> None:
        for name, table in self.bundle.items():
            vr = validate_table(table, self.specs[name])
            self.assertEqual(vr.status, "PASS", msg=f"{name}: {vr.errors}")

    def test_02_bundle_valid(self) -> None:
        vr = validate_contract_bundle(self.bundle, self.specs)
        self.assertEqual(vr.status, "PASS", msg=vr.errors)

    def test_03_missing_column(self) -> None:
        table = self.bundle["videos"].drop(["codec"])
        vr = validate_table(table, self.specs["videos"])
        self.assertEqual(vr.status, "FAIL")

    def test_04_extra_column(self) -> None:
        table = self.bundle["videos"].append_column("extra", pa.array([1]))
        vr = validate_table(table, self.specs["videos"])
        self.assertEqual(vr.status, "FAIL")

    def test_05_duplicate_pk(self) -> None:
        rows = self.bundle["videos"].to_pylist()
        rows.append(dict(rows[0]))
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["videos"]))
        vr = validate_table(table, self.specs["videos"])
        self.assertEqual(vr.status, "FAIL")

    def test_06_confidence_out_of_range(self) -> None:
        rows = self.bundle["detections"].to_pylist()
        rows[0]["confidence"] = 1.5
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["detections"]))
        vr = validate_table(table, self.specs["detections"])
        self.assertEqual(vr.status, "FAIL")

    def test_07_bbox_invalid(self) -> None:
        rows = self.bundle["detections"].to_pylist()
        rows[0]["bbox_x2"] = rows[0]["bbox_x1"]
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["detections"]))
        vr = validate_table(table, self.specs["detections"])
        self.assertEqual(vr.status, "FAIL")

    def test_08_nan_rejected(self) -> None:
        rows = self.bundle["detections"].to_pylist()
        # Build float array with NaN bypassing from_pylist schema checks
        schema = compile_arrow_schema(self.specs["detections"])
        arrays = []
        for field in schema:
            if field.name == "confidence":
                arrays.append(
                    pa.array(
                        [float("nan")] + [r["confidence"] for r in rows[1:]], type=pa.float32()
                    )
                )
            else:
                arrays.append(pa.array([r[field.name] for r in rows], type=field.type))
        table = pa.Table.from_arrays(arrays, schema=schema)
        vr = validate_table(table, self.specs["detections"])
        self.assertEqual(vr.status, "FAIL")

    def test_09_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, table in self.bundle.items():
                path = root / f"{name}.parquet"
                write_contract_parquet(table, path, self.specs[name], contain_root=root)
                loaded = read_contract_parquet(path, self.specs[name], contain_root=root)
                self.assertEqual(loaded.to_pylist(), table.to_pylist())
                info = inspect_contract_parquet(path)
                self.assertEqual(info["contract_name"], name)
                self.assertEqual(info["schema_fingerprint"], contract_fingerprint(self.specs[name]))

    def test_10_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "videos.parquet"
            write_contract_parquet(
                self.bundle["videos"], path, self.specs["videos"], contain_root=root
            )
            with self.assertRaises(DataContractError):
                write_contract_parquet(
                    self.bundle["videos"], path, self.specs["videos"], contain_root=root
                )

    def test_11_symlink_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "videos.parquet"
            write_contract_parquet(
                self.bundle["videos"], real, self.specs["videos"], contain_root=root
            )
            link = root / "link.parquet"
            link.symlink_to(real)
            with self.assertRaises(DataContractError):
                read_contract_parquet(link, self.specs["videos"], contain_root=root)

    def test_12_fingerprint_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "videos.parquet"
            write_contract_parquet(
                self.bundle["videos"], path, self.specs["videos"], contain_root=root
            )
            # rewrite with wrong metadata via pyarrow
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            md = dict(table.schema.metadata or {})
            md[b"football_analytics.schema_fingerprint"] = b"0" * 64
            bad = table.replace_schema_metadata(md)
            bad_path = root / "bad.parquet"
            pq.write_table(bad, bad_path)
            with self.assertRaises(DataContractError):
                read_contract_parquet(bad_path, self.specs["videos"], contain_root=root)

    def test_13_orphan_detection_fk(self) -> None:
        # detection on missing frame
        rows = self.bundle["detections"].to_pylist()
        rows[0]["frame_index"] = 99
        det = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["detections"]))
        tables = dict(self.bundle)
        tables["detections"] = det
        vr = validate_contract_bundle(tables, self.specs)
        self.assertEqual(vr.status, "FAIL")

    def test_14_migration_plan(self) -> None:
        steps = plan_migration(self.reg, "detections", 0, 1)
        self.assertEqual(len(steps), 1)

    def test_15_downgrade_rejected(self) -> None:
        with self.assertRaises(DataContractError):
            plan_migration(self.reg, "detections", 1, 0)

    def test_16_v0_to_v1_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v0 = self.reg.load_contract("detections", 0)
            schema = compile_arrow_schema(v0)
            rows = [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_name": "player",
                    "confidence": 0.9,
                    "bbox_x": 1.0,
                    "bbox_y": 2.0,
                    "bbox_width": 3.0,
                    "bbox_height": 4.0,
                    "model_id": "legacy",
                }
            ]
            table = pa.Table.from_pylist(rows, schema=schema)
            src = root / "v0.parquet"
            dst = root / "v1.parquet"
            receipt = root / "receipt.json"
            write_contract_parquet(table, src, v0, contain_root=root)
            before = sha256_file(src)
            migrate_parquet(
                src,
                dst,
                registry=self.reg,
                contract="detections",
                from_version=0,
                to_version=1,
                receipt_path=receipt,
                contain_root=root,
            )
            self.assertEqual(sha256_file(src), before)
            v1 = self.reg.load_contract("detections", 1)
            out = read_contract_parquet(dst, v1, contain_root=root)
            row = out.to_pylist()[0]
            self.assertEqual(row["bbox_x1"], 1.0)
            self.assertEqual(row["bbox_x2"], 4.0)
            self.assertEqual(row["bbox_y2"], 6.0)
            self.assertFalse(row["is_interpolated"])
            self.assertEqual(row["quality_flags"], [])
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertFalse(payload["lossy"])
            self.assertEqual(payload["source_row_count"], 1)

    def test_17_invalid_width_migration(self) -> None:
        v0 = self.reg.load_contract("detections", 0)
        schema = compile_arrow_schema(v0)
        rows = [
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 0,
                "class_name": "player",
                "confidence": 0.9,
                "bbox_x": 1.0,
                "bbox_y": 2.0,
                "bbox_width": 0.0,
                "bbox_height": 4.0,
                "model_id": "legacy",
            }
        ]
        table = pa.Table.from_pylist(rows, schema=schema)
        # source semantic validation may fail first; ensure migrate_table errors
        with self.assertRaises(DataContractError):
            migrate_table(
                table, registry=self.reg, contract="detections", from_version=0, to_version=1
            )

    def test_18_destination_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v0 = self.reg.load_contract("detections", 0)
            schema = compile_arrow_schema(v0)
            table = pa.Table.from_pylist(
                [
                    {
                        "run_id": self.run_id,
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
                schema=schema,
            )
            src = root / "v0.parquet"
            dst = root / "v1.parquet"
            write_contract_parquet(table, src, v0, contain_root=root)
            dst.write_text("x", encoding="utf-8")
            with self.assertRaises(DataContractError):
                migrate_parquet(
                    src,
                    dst,
                    registry=self.reg,
                    contract="detections",
                    from_version=0,
                    to_version=1,
                    receipt_path=root / "r.json",
                    contain_root=root,
                )

    def test_19_enum_invalid(self) -> None:
        rows = self.bundle["frames"].to_pylist()
        rows[0]["decode_status"] = "nope"
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["frames"]))
        vr = validate_table(table, self.specs["frames"])
        self.assertEqual(vr.status, "FAIL")

    def test_20_input_not_mutated(self) -> None:
        table = self.bundle["videos"]
        before = table.to_pylist()
        validate_table(table, self.specs["videos"])
        self.assertEqual(table.to_pylist(), before)

    def test_21_optional_missing_table_warning(self) -> None:
        tables = {"videos": self.bundle["videos"], "frames": self.bundle["frames"]}
        specs = {k: self.specs[k] for k in tables}
        # detections absent is ok
        vr = validate_contract_bundle(tables, specs)
        self.assertIn(vr.status, {"PASS", "PASS_WITH_WARNINGS"})

    def test_22_event_actor_orphan(self) -> None:
        rows = self.bundle["events"].to_pylist()
        rows[0]["actor_track_ids"] = [999]
        events = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["events"]))
        tables = dict(self.bundle)
        tables["events"] = events
        vr = validate_contract_bundle(tables, self.specs)
        self.assertEqual(vr.status, "FAIL")

    def test_23_bbox_exceeds_video(self) -> None:
        rows = self.bundle["detections"].to_pylist()
        rows[0]["bbox_x2"] = 5000.0
        det = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["detections"]))
        tables = dict(self.bundle)
        tables["detections"] = det
        vr = validate_contract_bundle(tables, self.specs)
        self.assertEqual(vr.status, "FAIL")

    def test_24_calibration_valid_requires_h(self) -> None:
        rows = self.bundle["calibrations"].to_pylist()
        rows[0]["homography_image_to_pitch"] = None
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["calibrations"]))
        vr = validate_table(table, self.specs["calibrations"])
        self.assertEqual(vr.status, "FAIL")

    def test_25_jersey_digit_inconsistency(self) -> None:
        rows = self.bundle["jersey_observations"].to_pylist()
        rows[0]["normalized_number"] = 10
        rows[0]["digit_count"] = 1
        table = pa.Table.from_pylist(
            rows, schema=compile_arrow_schema(self.specs["jersey_observations"])
        )
        vr = validate_table(table, self.specs["jersey_observations"])
        self.assertEqual(vr.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
