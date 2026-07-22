#!/usr/bin/env python3
"""Cache publish tests (Stage 2D)."""

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
)
from football_analytics.pipeline.cache_key import compute_cache_key
from football_analytics.pipeline.exceptions import CacheError
from football_analytics.pipeline.stage import make_stage_identity
from football_analytics.pipeline.types import StageResult

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
FP = "3" * 64
FP2 = "4" * 64


def _stage():
    return make_stage_identity(
        name="pub_stage",
        version=1,
        code_fingerprint=FP,
        deterministic=True,
        cacheable=True,
    )


def _result(run_id: str, cache_key: str, outputs: dict) -> StageResult:
    return StageResult(
        run_id=run_id,
        stage_name="pub_stage",
        stage_version=1,
        status="succeeded",
        cache_key=cache_key,
        cache_hit=False,
        started_at_utc="2026-07-22T12:00:00.000000Z",
        finished_at_utc="2026-07-22T12:00:01.000000Z",
        duration_ms=5,
        inputs={},
        outputs=outputs,
        metrics={},
        warnings=(),
        error=None,
        execution_fingerprint=FP2,
    )


class CachePublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_cache_policy(POLICY_PATH)

    def test_01_publish_creates_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            art_root = root / "arts"
            art_root.mkdir()
            path = art_root / "out.bin"
            path.write_bytes(b"published")
            ref = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            stage = _stage()
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            result = _result(run_id, key, {"Out": ref.to_dict()})
            final = publish_cache_entry(
                cache_root=cache_root,
                cache_key=key,
                stage_identity=stage,
                config_fingerprint=FP,
                artifacts=MappingProxyType({"Out": ref}),
                artifact_root=art_root,
                stage_result=result,
                policy=self.policy,
                source_run_id=run_id,
            )
            expected = entry_dir(cache_root, key)
            self.assertEqual(final, expected)
            self.assertTrue((final / "cache_manifest.json").is_file())
            self.assertTrue((final / "stage_result.json").is_file())
            self.assertTrue((final / "artifacts" / "out.bin").is_file())
            manifest = json.loads((final / "cache_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cache_key"], key)
            self.assertEqual(manifest["source_run_id"], run_id)

    def test_02_existing_entry_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            art_root = root / "arts"
            art_root.mkdir()
            path = art_root / "out.bin"
            path.write_bytes(b"v1")
            ref = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            stage = _stage()
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            result = _result(run_id, key, {"Out": ref.to_dict()})
            publish_cache_entry(
                cache_root=cache_root,
                cache_key=key,
                stage_identity=stage,
                config_fingerprint=FP,
                artifacts=MappingProxyType({"Out": ref}),
                artifact_root=art_root,
                stage_result=result,
                policy=self.policy,
                source_run_id=run_id,
            )
            entry = entry_dir(cache_root, key)
            marker = entry / "cache_manifest.json"
            original = marker.read_bytes()
            # Mutate source and attempt re-publish — existing entry returned unchanged
            path.write_bytes(b"v2-should-not-overwrite")
            ref2 = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            # Same key path: publish short-circuits if final.exists()
            again = publish_cache_entry(
                cache_root=cache_root,
                cache_key=key,
                stage_identity=stage,
                config_fingerprint=FP,
                artifacts=MappingProxyType({"Out": ref2}),
                artifact_root=art_root,
                stage_result=result,
                policy=self.policy,
                source_run_id=run_id,
            )
            self.assertEqual(again, entry)
            self.assertEqual(marker.read_bytes(), original)
            self.assertEqual((entry / "artifacts" / "out.bin").read_bytes(), b"v1")

    def test_03_publish_refused_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disabled = self.policy.__class__(
                schema_version=self.policy.schema_version,
                enabled=False,
                algorithm=self.policy.algorithm,
                layout_version=self.policy.layout_version,
                verify_on_read=self.policy.verify_on_read,
                verify_on_publish=self.policy.verify_on_publish,
                reject_symlinks=self.policy.reject_symlinks,
                reject_special_files=self.policy.reject_special_files,
                reject_hardlinks=self.policy.reject_hardlinks,
                lock_timeout_seconds=self.policy.lock_timeout_seconds,
                max_manifest_bytes=self.policy.max_manifest_bytes,
                max_entry_files=self.policy.max_entry_files,
                max_entry_bytes=self.policy.max_entry_bytes,
                quarantine_corrupt_entries=self.policy.quarantine_corrupt_entries,
                automatic_purge=False,
            )
            art_root = root / "arts"
            art_root.mkdir()
            path = art_root / "out.bin"
            path.write_bytes(b"x")
            ref = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            stage = _stage()
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            with self.assertRaises(CacheError):
                publish_cache_entry(
                    cache_root=root / "cache",
                    cache_key=key,
                    stage_identity=stage,
                    config_fingerprint=FP,
                    artifacts=MappingProxyType({"Out": ref}),
                    artifact_root=art_root,
                    stage_result=_result(run_id, key, {"Out": ref.to_dict()}),
                    policy=disabled,
                    source_run_id=run_id,
                )

    def test_04_symlink_injection_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            art_root = root / "arts"
            art_root.mkdir()
            real = art_root / "real.bin"
            real.write_bytes(b"real")
            link = art_root / "out.bin"
            link.symlink_to(real)
            # Build ref pointing at symlink path without going through build_artifact_ref
            from football_analytics.core.hashing import sha256_bytes
            from football_analytics.pipeline.types import ArtifactRef

            ref = ArtifactRef(
                logical_name="Out",
                relative_path="out.bin",
                media_type="application/octet-stream",
                size_bytes=4,
                sha256=sha256_bytes(b"real"),
            )
            stage = _stage()
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            with self.assertRaises((CacheError, Exception)):
                publish_cache_entry(
                    cache_root=cache_root,
                    cache_key=key,
                    stage_identity=stage,
                    config_fingerprint=FP,
                    artifacts=MappingProxyType({"Out": ref}),
                    artifact_root=art_root,
                    stage_result=_result(run_id, key, {"Out": ref.to_dict()}),
                    policy=self.policy,
                    source_run_id=run_id,
                )
            self.assertFalse(entry_dir(cache_root, key).exists())

    def test_05_permissions_on_published_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            art_root = root / "arts"
            art_root.mkdir()
            path = art_root / "out.bin"
            path.write_bytes(b"perm")
            ref = build_artifact_ref(
                "Out", path, root=art_root, media_type="application/octet-stream"
            )
            stage = _stage()
            key = compute_cache_key(
                stage=stage,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Out": ref},
            )
            run_id = generate_run_id()
            final = publish_cache_entry(
                cache_root=cache_root,
                cache_key=key,
                stage_identity=stage,
                config_fingerprint=FP,
                artifacts=MappingProxyType({"Out": ref}),
                artifact_root=art_root,
                stage_result=_result(run_id, key, {"Out": ref.to_dict()}),
                policy=self.policy,
                source_run_id=run_id,
            )
            art = final / "artifacts" / "out.bin"
            self.assertEqual(art.stat().st_mode & 0o777, 0o600)
            self.assertEqual(final.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
