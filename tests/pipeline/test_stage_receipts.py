#!/usr/bin/env python3
"""Stage execution receipt tests (Stage 2D)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.records import RecordError
from football_analytics.pipeline.exceptions import StageError
from football_analytics.pipeline.receipts import (
    RECEIPT_SCHEMA_VERSION,
    build_stage_execution_receipt,
    write_stage_execution_receipt,
)
from football_analytics.pipeline.types import StageResult

FP = "1" * 64
FP2 = "2" * 64
RUN = "run_20260722T120000000000Z_abcdef123456"


def _result(**kwargs) -> StageResult:
    base = dict(
        run_id=RUN,
        stage_name="echo_stage",
        stage_version=1,
        status="succeeded",
        cache_key=FP,
        cache_hit=False,
        started_at_utc="2026-07-22T12:00:00.000000Z",
        finished_at_utc="2026-07-22T12:00:01.000000Z",
        duration_ms=12,
        inputs={},
        outputs={},
        metrics={"n": 1},
        warnings=("note",),
        error=None,
        execution_fingerprint=FP2,
    )
    base.update(kwargs)
    return StageResult(**base)


class StageReceiptsTests(unittest.TestCase):
    def test_01_build_receipt(self) -> None:
        payload = build_stage_execution_receipt(_result())
        self.assertEqual(payload["schema_version"], RECEIPT_SCHEMA_VERSION)
        self.assertEqual(payload["run_id"], RUN)
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["metrics"]["n"], 1)
        self.assertEqual(payload["warnings"], ["note"])

    def test_02_write_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "stage_execution_receipt.json"
            written = write_stage_execution_receipt(path, _result(), contain_root=root)
            self.assertEqual(written, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["cache_key"], FP)
            self.assertEqual(data["execution_fingerprint"], FP2)

    def test_03_write_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "stage_execution_receipt.json"
            write_stage_execution_receipt(path, _result(), contain_root=root)
            with self.assertRaises(RecordError):
                write_stage_execution_receipt(path, _result(), contain_root=root)

    def test_04_failed_receipt_includes_error(self) -> None:
        r = _result(
            status="failed",
            error={"class": "RuntimeError", "message": "boom"},
        )
        payload = build_stage_execution_receipt(r)
        self.assertEqual(payload["error"]["class"], "RuntimeError")
        self.assertEqual(payload["error"]["message"], "boom")

    def test_05_rejects_non_result(self) -> None:
        with self.assertRaises(StageError):
            build_stage_execution_receipt({"not": "a result"})  # type: ignore[arg-type]

    def test_06_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "stage_execution_receipt.json"
            write_stage_execution_receipt(path, _result(), contain_root=root)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_07_cache_hit_receipt(self) -> None:
        payload = build_stage_execution_receipt(_result(status="cache_hit", cache_hit=True))
        self.assertTrue(payload["cache_hit"])
        self.assertEqual(payload["status"], "cache_hit")


if __name__ == "__main__":
    unittest.main()
