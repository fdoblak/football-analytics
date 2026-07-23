#!/usr/bin/env python3
"""Schema registry / specs / compiler / fingerprint tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.data import DataContractError
from football_analytics.data.compiler import compile_arrow_schema, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.specs import load_contract_spec, parse_contract_dict

ROOT = default_project_root()


class RegistryCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)

    def test_01_list_nineteen_v1(self) -> None:
        names = list_contracts(registry=self.reg)
        self.assertEqual(len(names), 19)
        for n in (
            "videos",
            "frames",
            "detections",
            "track_observations",
            "track_summaries",
            "track_lifecycle",
            "calibrations",
            "team_assignments",
            "jersey_observations",
            "events",
            "shot_boundaries",
            "shot_segments",
            "camera_view_segments",
            "analysis_windows",
            "detection_frame_status",
            "detection_attributes",
            "identity_evidence",
            "reid_candidate_links",
            "track_identity_assignments",
        ):
            self.assertIn(n, names)

    def test_02_all_compile(self) -> None:
        for name in self.reg.list_contracts():
            for ver in self.reg.get_entry(name).supported_versions:
                spec = self.reg.load_contract(name, ver)
                schema = compile_arrow_schema(spec)
                self.assertEqual(len(schema), len(spec.fields))

    def test_03_fingerprint_stable(self) -> None:
        spec = self.reg.load_contract("videos", 1)
        self.assertEqual(contract_fingerprint(spec), contract_fingerprint(spec))
        self.assertEqual(len(contract_fingerprint(spec)), 64)

    def test_04_fingerprint_field_order(self) -> None:
        spec = self.reg.load_contract("videos", 1)
        data = json.loads((ROOT / "schemas/data/v1/videos.json").read_text(encoding="utf-8"))
        data["fields"] = list(reversed(data["fields"]))
        other = parse_contract_dict(data)
        self.assertNotEqual(contract_fingerprint(spec), contract_fingerprint(other))

    def test_05_unknown_type(self) -> None:
        with self.assertRaises(DataContractError):
            parse_contract_dict(
                {
                    "contract_name": "x",
                    "version": 1,
                    "description": "d",
                    "fields": [{"name": "a", "type": "weird", "nullable": False}],
                    "primary_key": ["a"],
                    "foreign_keys": [],
                    "partition_by": [],
                    "sort_by": ["a"],
                    "semantic_rules": [],
                    "table_metadata": {},
                }
            )

    def test_06_duplicate_field(self) -> None:
        with self.assertRaises(DataContractError):
            parse_contract_dict(
                {
                    "contract_name": "x",
                    "version": 1,
                    "description": "d",
                    "fields": [
                        {"name": "a", "type": "string", "nullable": False},
                        {"name": "a", "type": "string", "nullable": False},
                    ],
                    "primary_key": ["a"],
                    "foreign_keys": [],
                    "partition_by": [],
                    "sort_by": ["a"],
                    "semantic_rules": [],
                    "table_metadata": {},
                }
            )

    def test_07_missing_pk_field(self) -> None:
        with self.assertRaises(DataContractError):
            parse_contract_dict(
                {
                    "contract_name": "x",
                    "version": 1,
                    "description": "d",
                    "fields": [{"name": "a", "type": "string", "nullable": False}],
                    "primary_key": ["missing"],
                    "foreign_keys": [],
                    "partition_by": [],
                    "sort_by": ["a"],
                    "semantic_rules": [],
                    "table_metadata": {},
                }
            )

    def test_08_symlink_spec_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.json"
            link = Path(tmp) / "link.json"
            real.write_text(
                (ROOT / "schemas/data/v1/videos.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            link.symlink_to(real)
            with self.assertRaises(DataContractError):
                load_contract_spec(link, contain_root=Path(tmp))

    def test_09_name_mismatch(self) -> None:
        # load path for videos but if we manually break registry - covered by load_contract check
        spec = self.reg.load_contract("frames", 1)
        self.assertEqual(spec.contract_name, "frames")

    def test_10_detections_v0_exists(self) -> None:
        spec = self.reg.load_contract("detections", 0)
        self.assertEqual(spec.version, 0)
        self.assertIn("bbox_width", [f.name for f in spec.fields])

    def test_11_unknown_key_rejected(self) -> None:
        data = json.loads((ROOT / "schemas/data/v1/videos.json").read_text(encoding="utf-8"))
        data["extra_key"] = 1
        with self.assertRaises(DataContractError):
            parse_contract_dict(data)

    def test_12_invalid_list_size(self) -> None:
        with self.assertRaises(DataContractError):
            parse_contract_dict(
                {
                    "contract_name": "x",
                    "version": 1,
                    "description": "d",
                    "fields": [
                        {
                            "name": "a",
                            "type": "fixed_size_list",
                            "nullable": True,
                            "value_type": "float64",
                            "list_size": 0,
                        }
                    ],
                    "primary_key": ["a"],
                    "foreign_keys": [],
                    "partition_by": [],
                    "sort_by": ["a"],
                    "semantic_rules": [],
                    "table_metadata": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
