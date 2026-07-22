#!/usr/bin/env python3
"""Tests for canonical Run ID (Stage 2B)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from football_analytics.core.run_id import (
    MAX_RUN_ID_LENGTH,
    RunIdError,
    generate_run_id,
    parse_run_id,
    validate_run_id,
)


class RunIdTests(unittest.TestCase):
    def test_01_format(self) -> None:
        rid = generate_run_id()
        self.assertTrue(rid.startswith("run_"))
        self.assertIn("T", rid)
        self.assertIn("Z_", rid)
        validate_run_id(rid)

    def test_02_utc_injection(self) -> None:
        fixed = datetime(2026, 7, 22, 17, 15, 30, 123456, tzinfo=timezone.utc)

        def now() -> datetime:
            return fixed

        rid = generate_run_id(now=now, suffix_factory=lambda: "a1b2c3d4e5f6")
        self.assertEqual(rid, "run_20260722T171530123456Z_a1b2c3d4e5f6")

    def test_03_sortable(self) -> None:
        a = generate_run_id(
            now=lambda: datetime(2026, 7, 22, 10, 0, 0, 0, tzinfo=timezone.utc),
            suffix_factory=lambda: "aaaaaaaaaaaa",
        )
        b = generate_run_id(
            now=lambda: datetime(2026, 7, 22, 11, 0, 0, 0, tzinfo=timezone.utc),
            suffix_factory=lambda: "bbbbbbbbbbbb",
        )
        self.assertLess(a, b)

    def test_04_unique_suffix_smoke(self) -> None:
        ids = {generate_run_id() for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_05_parse_roundtrip(self) -> None:
        rid = generate_run_id(suffix_factory=lambda: "0123456789ab")
        parsed = parse_run_id(rid)
        self.assertEqual(parsed.value, rid)
        self.assertEqual(parsed.suffix, "0123456789ab")

    def test_06_invalid_length(self) -> None:
        with self.assertRaises(RunIdError):
            validate_run_id("run_" + "x" * (MAX_RUN_ID_LENGTH))

    def test_07_path_traversal(self) -> None:
        for bad in (
            "../x",
            "run_../evil",
            "a/b",
            "a\\b",
            "run_20260722T171530123456Z_a1b2c3d4e5f6/../x",
        ):
            with self.assertRaises(RunIdError):
                validate_run_id(bad)

    def test_08_shell_chars(self) -> None:
        with self.assertRaises(RunIdError):
            validate_run_id("run_20260722T171530123456Z_a1b2c3d4e5f6;rm")

    def test_09_naive_clock_rejected(self) -> None:
        with self.assertRaises(RunIdError):
            generate_run_id(now=lambda: datetime(2026, 1, 1, 0, 0, 0))

    def test_10_uppercase_suffix_rejected(self) -> None:
        with self.assertRaises(RunIdError):
            validate_run_id("run_20260722T171530123456Z_A1B2C3D4E5F6")

    def test_11_space_rejected(self) -> None:
        with self.assertRaises(RunIdError):
            validate_run_id("run_20260722T171530123456Z_a1b2c3d4e5f6 ")

    def test_12_collision_smoke_same_clock_diff_suffix(self) -> None:
        clock = datetime(2026, 7, 22, 12, 0, 0, 1, tzinfo=timezone.utc)
        a = generate_run_id(now=lambda: clock, suffix_factory=lambda: "111111111111")
        b = generate_run_id(now=lambda: clock, suffix_factory=lambda: "222222222222")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
