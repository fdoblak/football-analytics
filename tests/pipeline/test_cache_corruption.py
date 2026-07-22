#!/usr/bin/env python3
"""Cache corruption detection and quarantine tests (Stage 2D)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

from football_analytics.core.run_id import generate_run_id
from football_analytics.pipeline.artifacts import build_artifact_ref
from football_analytics.pipeline.cache import (
    entry_dir,
    load_cache_policy,
    publish_cache_entry,
    quarantine_cache_entry,
    verify_cache_entry,
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.exceptions import CacheError
from football_analytics.pipeline.stage import make_stage_identity
from football_analytics.pipeline.types import StageResult

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
FP = "8" * 64
FP2 = "9" * 64


def _stage():
    return make_stage_identity(
        name="corr_stage",
        version=1,
        code_fingerprint=FP,
        deterministic=True,
        cacheable=True,
    )


def _publish(tmp: Path, policy):
    cache_root = tmp / "cache"
    art_root = tmp / "arts"
    art_root.mkdir()
    path = art_root / "out.bin"
    path.write_bytes(b"good-payload")
    ref = build_artifact_ref("Out", path, root=art_root, media_type="application/octet-stream")
    stage = _stage()
    key = compute_cache_key(
        stage=stage,
        config_fingerprint=FP,
        compatibility_fingerprint=FP2,
        inputs={"Out": ref},
    )
    run_id = generate_run_id()
    result = StageResult(
        run_id=run_id,
        stage_name="corr_stage",
        stage_version=1,
        status="succeeded",
        cache_key=key,
        cache_hit=False,
        started_at_utc="2026-07-22T12:00:00.000000Z",
        finished_at_utc="2026-07-22T12:00:01.000000Z",
        duration_ms=1,
        inputs={},
        outputs={"Out": ref.to_dict()},
        metrics={},
        warnings=(),
        error=None,
        execution_fingerprint=FP2,
    )
    publish_cache_entry(
        cache_root=cache_root,
        cache_key=key,
        stage_identity=stage,
        config_fingerprint=FP,
        artifacts=MappingProxyType({"Out": ref}),
        artifact_root=art_root,
        stage_result=result,
        policy=policy,
        source_run_id=run_id,
    )
    return cache_root, stage, key, ref


class CacheCorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_cache_policy(POLICY_PATH)

    def test_01_corrupt_artifact_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            art = entry_dir(cache_root, key) / "artifacts" / "out.bin"
            art.write_bytes(b"TAMPERED!!!!")
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_02_corrupt_manifest_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            manifest_path = entry_dir(cache_root, key) / "cache_manifest.json"
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["cache_key"] = "0" * 64
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_03_quarantine_moves_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root, _stage, key, _ref = _publish(root, self.policy)
            qroot = root / "quarantine"
            receipt = quarantine_cache_entry(
                cache_root,
                key,
                quarantine_root=qroot,
                reason="artifact hash mismatch",
            )
            self.assertFalse(entry_dir(cache_root, key).exists())
            self.assertTrue((qroot / key).is_dir() or any(qroot.iterdir()))
            self.assertIs(receipt["permanent_delete_performed"], False)
            self.assertIn("receipt_fingerprint", receipt)
            # Quarantine receipt on disk
            dest = Path(receipt["quarantine_path"])
            self.assertTrue((dest / "quarantine_receipt.json").is_file())
            on_disk = json.loads((dest / "quarantine_receipt.json").read_text(encoding="utf-8"))
            self.assertIs(on_disk["permanent_delete_performed"], False)

    def test_04_symlink_under_artifacts_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            arts = entry_dir(cache_root, key) / "artifacts"
            real = arts / "out.bin"
            # Replace file with symlink to itself path outside
            outside = Path(tmp) / "outside.bin"
            outside.write_bytes(b"evil")
            real.unlink()
            real.symlink_to(outside)
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_05_unexpected_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            arts = entry_dir(cache_root, key) / "artifacts"
            (arts / "extra.bin").write_bytes(b"sneaky")
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_06_quarantine_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(CacheError):
            quarantine_cache_entry(
                Path(tmp) / "cache",
                "a" * 64,
                quarantine_root=Path(tmp) / "q",
                reason="missing",
            )

    def test_07_hardlink_rejected_on_verify(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            arts = entry_dir(cache_root, key) / "artifacts"
            target = arts / "out.bin"
            link = arts / "alias.bin"
            try:
                os.link(target, link)
            except OSError:
                self.skipTest("hardlink not allowed on this filesystem")
            # Unexpected file also fails; ensure hardlink path is covered via policy
            # Remove unexpected by only hardlinking same relative? Instead verify
            # reject_hardlinks on the artifact itself (nlink > 1).
            link.unlink()
            # Create hardlink outside then... actually hardlink increases nlink on target
            external = Path(tmp) / "ext.bin"
            try:
                os.link(target, external)
            except OSError:
                self.skipTest("hardlink not allowed on this filesystem")
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )


if __name__ == "__main__":
    unittest.main()
