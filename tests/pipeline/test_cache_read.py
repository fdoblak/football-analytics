#!/usr/bin/env python3
"""Cache verify / restore / hit-path tests (Stage 2D)."""

from __future__ import annotations

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
    restore_cache_entry,
    verify_cache_entry,
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.exceptions import CacheError
from football_analytics.pipeline.stage import make_stage_identity
from football_analytics.pipeline.types import StageResult

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
FP = "5" * 64
FP2 = "6" * 64


def _stage():
    return make_stage_identity(
        name="read_stage",
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
    path.write_bytes(b"cached-bytes")
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
        stage_name="read_stage",
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


class CacheReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_cache_policy(POLICY_PATH)

    def test_01_miss_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            stage = _stage()
            key = "7" * 64
            self.assertFalse(entry_dir(cache_root, key).exists())
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_02_verify_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            manifest, result = verify_cache_entry(
                cache_root,
                key,
                expected_stage=stage,
                expected_config_fp=FP,
                expected_inputs={"Out": ref},
                expected_compatibility_fp=FP2,
                policy=self.policy,
            )
            self.assertEqual(manifest["cache_key"], key)
            self.assertEqual(result["status"], "succeeded")

    def test_03_restore_copies_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root, stage, key, ref = _publish(root, self.policy)
            out = root / "restore"
            restored = restore_cache_entry(
                cache_root,
                key,
                output_directory=out,
                policy=self.policy,
                expected_stage=stage,
                expected_config_fp=FP,
                expected_inputs={"Out": ref},
                expected_compatibility_fp=FP2,
            )
            self.assertIn("Out", restored)
            self.assertEqual((out / "out.bin").read_bytes(), b"cached-bytes")

    def test_04_restore_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root, stage, key, ref = _publish(root, self.policy)
            out = root / "restore"
            out.mkdir()
            (out / "out.bin").write_bytes(b"existing")
            with self.assertRaises(CacheError):
                restore_cache_entry(
                    cache_root,
                    key,
                    output_directory=out,
                    policy=self.policy,
                    expected_stage=stage,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                )

    def test_05_verify_stage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, _stage_pub, key, ref = _publish(Path(tmp), self.policy)
            other = make_stage_identity(
                name="other_stage",
                version=1,
                code_fingerprint=FP,
            )
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=other,
                    expected_config_fp=FP,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )

    def test_06_verify_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root, stage, key, ref = _publish(Path(tmp), self.policy)
            with self.assertRaises(CacheError):
                verify_cache_entry(
                    cache_root,
                    key,
                    expected_stage=stage,
                    expected_config_fp=FP2,
                    expected_inputs={"Out": ref},
                    expected_compatibility_fp=FP2,
                    policy=self.policy,
                )


if __name__ == "__main__":
    unittest.main()
