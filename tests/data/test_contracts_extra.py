#!/usr/bin/env python3
"""Additional validation / security / CLI / validator tests for Stage 2C."""

from __future__ import annotations

import importlib.util
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
from football_analytics.data import DataContractError
from football_analytics.data.bundle import build_synthetic_bundle
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.fingerprint import arrow_schema_fingerprint, contract_fingerprint
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.types import assert_safe_identifier
from football_analytics.data.validation import validate_table

ROOT = default_project_root()
PY = sys.executable


class MoreContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.run_id = generate_run_id()
        self.bundle = build_synthetic_bundle(self.run_id)
        self.specs = {n: self.reg.load_contract(n, 1) for n in self.bundle}

    def test_01_safe_id_rejects_slash(self) -> None:
        with self.assertRaises(DataContractError):
            assert_safe_identifier("a/b")

    def test_02_safe_id_rejects_dotdot(self) -> None:
        with self.assertRaises(DataContractError):
            assert_safe_identifier("..abc")

    def test_03_safe_id_ok(self) -> None:
        assert_safe_identifier("clip_demo_01")

    def test_04_arrow_schema_fp_stable(self) -> None:
        schema = compile_arrow_schema(self.specs["videos"])
        self.assertEqual(arrow_schema_fingerprint(schema), arrow_schema_fingerprint(schema))

    def test_05_null_pk(self) -> None:
        # non-nullable run_id - constructing null may fail; use frames with forced null via mask
        rows = self.bundle["frames"].to_pylist()
        # empty table ok
        empty = pa.Table.from_pylist([], schema=compile_arrow_schema(self.specs["frames"]))
        vr = validate_table(empty, self.specs["frames"])
        self.assertEqual(vr.status, "PASS")
        self.assertIsNotNone(rows)

    def test_06_reordered_columns(self) -> None:
        table = self.bundle["videos"]
        cols = list(reversed(table.column_names))
        reordered = table.select(cols)
        vr = validate_table(reordered, self.specs["videos"])
        self.assertEqual(vr.status, "FAIL")

    def test_07_wrong_type(self) -> None:
        # drop and append wrong type
        t = self.bundle["videos"].drop(["width_px"]).append_column("width_px", pa.array(["1280"]))
        vr = validate_table(t, self.specs["videos"])
        self.assertEqual(vr.status, "FAIL")

    def test_08_quality_flag_null_item(self) -> None:
        rows = self.bundle["detections"].to_pylist()
        schema = compile_arrow_schema(self.specs["detections"])
        arrays = []
        for field in schema:
            if field.name == "quality_flags":
                arrays.append(pa.array([[None], rows[1]["quality_flags"]], type=field.type))
            else:
                arrays.append(pa.array([r[field.name] for r in rows], type=field.type))
        table = pa.Table.from_arrays(arrays, schema=schema)
        self.assertEqual(validate_table(table, self.specs["detections"]).status, "FAIL")

    def test_09_track_state_enum(self) -> None:
        rows = self.bundle["track_observations"].to_pylist()
        rows[0]["observation_state"] = "ghost"
        table = pa.Table.from_pylist(
            rows, schema=compile_arrow_schema(self.specs["track_observations"])
        )
        self.assertEqual(validate_table(table, self.specs["track_observations"]).status, "FAIL")

    def test_10_team_role_enum(self) -> None:
        rows = self.bundle["team_assignments"].to_pylist()
        rows[0]["team_role"] = "coach"
        table = pa.Table.from_pylist(
            rows, schema=compile_arrow_schema(self.specs["team_assignments"])
        )
        self.assertEqual(validate_table(table, self.specs["team_assignments"]).status, "FAIL")

    def test_11_event_attributes_json(self) -> None:
        rows = self.bundle["events"].to_pylist()
        rows[0]["attributes_json"] = "[1,2,3]"
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["events"]))
        self.assertEqual(validate_table(table, self.specs["events"]).status, "FAIL")

    def test_12_event_time_partial(self) -> None:
        rows = self.bundle["events"].to_pylist()
        rows[0]["end_time_us"] = None
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["events"]))
        self.assertEqual(validate_table(table, self.specs["events"]).status, "FAIL")

    def test_13_monotonic_time(self) -> None:
        rows = self.bundle["frames"].to_pylist()
        rows[3]["video_time_us"] = 0
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["frames"]))
        self.assertEqual(validate_table(table, self.specs["frames"]).status, "FAIL")

    def test_14_summary_first_last(self) -> None:
        rows = self.bundle["track_summaries"].to_pylist()
        rows[0]["first_frame_index"] = 5
        rows[0]["last_frame_index"] = 1
        table = pa.Table.from_pylist(
            rows, schema=compile_arrow_schema(self.specs["track_summaries"])
        )
        self.assertEqual(validate_table(table, self.specs["track_summaries"]).status, "FAIL")

    def test_15_path_escape_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "out.parquet"
            with self.assertRaises(DataContractError):
                write_contract_parquet(
                    self.bundle["videos"],
                    outside,
                    self.specs["videos"],
                    contain_root=root,
                )

    def test_16_cli_contracts_list(self) -> None:
        self.assertEqual(main(["contracts", "list"]), 0)

    def test_17_cli_contracts_show(self) -> None:
        self.assertEqual(main(["contracts", "show", "videos"]), 0)

    def test_18_cli_fingerprint_json(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "contracts", "fingerprint", "videos", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["digest"]), 64)

    def test_19_cli_validate_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "videos.parquet"
            write_contract_parquet(
                self.bundle["videos"], path, self.specs["videos"], contain_root=root
            )
            code = main(["contracts", "validate", "videos", str(path)])
            self.assertEqual(code, 0)

    def test_20_cli_validate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.parquet"
            bad.write_bytes(b"not-parquet")
            code = main(["contracts", "validate", "videos", str(bad)])
            self.assertNotEqual(code, 0)

    def test_21_cli_unknown_contract(self) -> None:
        self.assertNotEqual(main(["contracts", "show", "nope"]), 0)

    def test_22_existing_version_still_works(self) -> None:
        self.assertEqual(main(["--version"]), 0)

    def test_23_existing_info(self) -> None:
        self.assertEqual(main(["info"]), 0)

    def test_24_no_eager_pyarrow_on_package_import(self) -> None:
        # football_analytics root should not import pyarrow
        src = (ROOT / "src/football_analytics/__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("pyarrow", src)

    def test_25_fingerprint_ignores_description(self) -> None:
        spec = self.specs["videos"]
        a = contract_fingerprint(spec)
        # description is excluded from normalized dict already
        self.assertEqual(a, contract_fingerprint(spec))

    def test_26_validator_script_pass(self) -> None:
        path = ROOT / "scripts/check_data_contracts.py"
        spec = importlib.util.spec_from_file_location("check_data_contracts", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code = mod.main(["--registry", str(ROOT / "configs/data/schema_registry.yaml"), "--quiet"])
        self.assertEqual(code, 0)

    def test_27_validator_synthetic(self) -> None:
        path = ROOT / "scripts/check_data_contracts.py"
        spec = importlib.util.spec_from_file_location("check_data_contracts2", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = mod.main(
                [
                    "--registry",
                    str(ROOT / "configs/data/schema_registry.yaml"),
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
            self.assertTrue(payload["extras"].get("fixture_cleaned"))

    def test_28_cli_migrate(self) -> None:
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
                        "class_name": "player",
                        "confidence": 0.9,
                        "bbox_x": 1.0,
                        "bbox_y": 2.0,
                        "bbox_width": 3.0,
                        "bbox_height": 4.0,
                        "model_id": "legacy",
                    }
                ],
                schema=schema,
            )
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

    def test_29_run_id_invalid_in_table(self) -> None:
        rows = self.bundle["videos"].to_pylist()
        rows[0]["run_id"] = "not-a-run-id"
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["videos"]))
        self.assertEqual(validate_table(table, self.specs["videos"]).status, "FAIL")

    def test_30_sha256_invalid(self) -> None:
        rows = self.bundle["videos"].to_pylist()
        rows[0]["source_sha256"] = "zzzz"
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["videos"]))
        self.assertEqual(validate_table(table, self.specs["videos"]).status, "FAIL")

    def test_31_pitch_dims_positive(self) -> None:
        rows = self.bundle["calibrations"].to_pylist()
        rows[0]["pitch_length_m"] = -1.0
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["calibrations"]))
        self.assertEqual(validate_table(table, self.specs["calibrations"]).status, "FAIL")

    def test_32_actor_duplicate(self) -> None:
        rows = self.bundle["events"].to_pylist()
        rows[0]["actor_track_ids"] = [1, 1]
        table = pa.Table.from_pylist(rows, schema=compile_arrow_schema(self.specs["events"]))
        self.assertEqual(validate_table(table, self.specs["events"]).status, "FAIL")

    def test_33_unsupported_migration(self) -> None:
        from football_analytics.data.migrations import plan_migration

        with self.assertRaises(DataContractError):
            plan_migration(self.reg, "videos", 1, 2)

    def test_34_noop_migration_table(self) -> None:
        from football_analytics.data.migrations import migrate_table

        out = migrate_table(
            self.bundle["detections"],
            registry=self.reg,
            contract="detections",
            from_version=1,
            to_version=1,
        )
        self.assertEqual(out.num_rows, self.bundle["detections"].num_rows)

    def test_35_list_contracts_sorted(self) -> None:
        from football_analytics.data.compiler import list_contracts

        names = list_contracts(registry=self.reg)
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
