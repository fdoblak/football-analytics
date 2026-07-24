#!/usr/bin/env python3
"""Evidence collector / allowlist tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.evidence.collector import (
    backfill_from_workspace,
    is_safe_evidence_file,
    load_index,
    normalize_byte_identical_duplicates,
    save_index,
)


class EvidenceCollectorTests(unittest.TestCase):
    def test_01_safe_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            j = root / "a.json"
            j.write_text("{}", encoding="utf-8")
            self.assertTrue(is_safe_evidence_file(j)[0])
            p = root / "w.pt"
            p.write_bytes(b"x" * 10)
            self.assertFalse(is_safe_evidence_file(p)[0])

    def test_02_index_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            idx = {"schema_version": 1, "updated_at_utc": "t", "entries": []}
            save_index(path, idx)
            loaded = load_index(path)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertIn("entries", loaded)

    def test_03_backfill_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            d = ws / "physical_metric_contract_checks"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            # dry-run should not require project evidence dir write for copy,
            # but function still uses default project root for index — use dry_run only
            summary = backfill_from_workspace(workspace_root=ws, dry_run=True)
            self.assertIn("copied", summary)

    def test_04_normalize_byte_identical_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ev = root / "artifacts" / "evidence"
            stage = ev / "stage_test"
            stage.mkdir(parents=True)
            payload = b'{"ok": true}\n'
            digest = __import__("hashlib").sha256(payload).hexdigest()
            canonical = stage / f"{digest[:12]}_sample.json"
            plain = stage / "sample.json"
            canonical.write_bytes(payload)
            plain.write_bytes(payload)
            unique = stage / "unique.json"
            unique.write_bytes(b'{"other": 1}\n')
            index_path = ev / "index.json"
            save_index(
                index_path,
                {
                    "schema_version": 1,
                    "updated_at_utc": "t",
                    "entries": [
                        {
                            "artifact_id": "a1",
                            "stage_id": "stage_test",
                            "relative_path": str(canonical.relative_to(root)),
                            "sha256": digest,
                            "status": "present",
                        }
                    ],
                },
            )
            # Monkeypatch default_project_root via project_root arg
            result = normalize_byte_identical_duplicates(
                stage_id="stage_test", project_root=root, dry_run=False
            )
            self.assertFalse(result["data_loss"])
            self.assertEqual(len(result["removed"]), 1)
            self.assertFalse(plain.exists())
            self.assertTrue(canonical.exists())
            self.assertTrue(unique.exists())


if __name__ == "__main__":
    unittest.main()
