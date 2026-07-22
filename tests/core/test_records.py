#!/usr/bin/env python3
"""Atomic record writer tests (Stage 2B)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.records import RecordError, write_json_record


class RecordsTests(unittest.TestCase):
    def test_01_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.json"
            write_json_record(target, {"z": 1, "a": 2}, contain_root=Path(tmp))
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(data["a"], 2)

    def test_02_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.json"
            write_json_record(target, {"a": 1}, contain_root=Path(tmp))
            with self.assertRaises(RecordError):
                write_json_record(target, {"a": 2}, contain_root=Path(tmp))

    def test_03_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.json"
            write_json_record(target, {"a": 1}, contain_root=Path(tmp))
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_04_secret_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RecordError):
            write_json_record(Path(tmp) / "a.json", {"password": "x"}, contain_root=Path(tmp))

    def test_05_temp_cleanup_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Invalid payload type triggers before replace
            with self.assertRaises(RecordError):
                write_json_record(Path(tmp) / "a.json", ["not", "dict"], contain_root=Path(tmp))  # type: ignore[arg-type]
            leftovers = list(Path(tmp).glob(".*.tmp"))
            self.assertEqual(leftovers, [])

    def test_06_containment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside.json"
            with self.assertRaises(RecordError):
                write_json_record(outside, {"a": 1}, contain_root=root)

    def test_07_nan_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises((ValueError, TypeError, RecordError)),
        ):
            write_json_record(
                Path(tmp) / "a.json",
                {"v": float("nan")},
                contain_root=Path(tmp),
            )


if __name__ == "__main__":
    unittest.main()
