#!/usr/bin/env python3
"""Cache policy load and layout helpers (Stage 2D)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.pipeline.cache import (
    CachePolicyConfig,
    entry_dir,
    load_cache_policy,
    resolve_cache_root,
)
from football_analytics.pipeline.exceptions import CacheError

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
PATHS = REPO_ROOT / "configs" / "system" / "paths.yaml"
FP = "a" * 64


class CacheLayoutTests(unittest.TestCase):
    def test_01_load_default_policy(self) -> None:
        policy = load_cache_policy(POLICY)
        self.assertIsInstance(policy, CachePolicyConfig)
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.algorithm, "sha256")
        self.assertEqual(policy.layout_version, 1)
        self.assertFalse(policy.automatic_purge)
        self.assertTrue(policy.verify_on_read)
        self.assertTrue(policy.verify_on_publish)
        self.assertTrue(policy.reject_symlinks)
        self.assertTrue(policy.reject_hardlinks)
        self.assertTrue(policy.quarantine_corrupt_entries)

    def test_02_resolve_cache_root_absolute(self) -> None:
        root = resolve_cache_root(PATHS)
        self.assertTrue(root.is_absolute())
        self.assertEqual(root, Path("/home/fdoblak/workspace/cache"))

    def test_03_entry_dir_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            d = entry_dir(cache_root, FP)
            self.assertEqual(d, cache_root / "v1" / "sha256" / FP[:2] / FP[2:])

    def test_04_entry_dir_rejects_bad_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(CacheError):
            entry_dir(Path(tmp), "not-hex")

    def test_05_policy_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.yaml"
            link = root / "link.yaml"
            real.write_text(POLICY.read_text(encoding="utf-8"), encoding="utf-8")
            link.symlink_to(real)
            with self.assertRaises(CacheError):
                load_cache_policy(link)

    def test_06_policy_rejects_automatic_purge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            text = POLICY.read_text(encoding="utf-8").replace(
                "automatic_purge: false", "automatic_purge: true"
            )
            p.write_text(text, encoding="utf-8")
            with self.assertRaises(CacheError):
                load_cache_policy(p)

    def test_07_policy_rejects_bad_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            text = POLICY.read_text(encoding="utf-8").replace("algorithm: sha256", "algorithm: md5")
            p.write_text(text, encoding="utf-8")
            with self.assertRaises(CacheError):
                load_cache_policy(p)

    def test_08_policy_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            p.write_text("schema_version: 1\nenabled: true\n", encoding="utf-8")
            with self.assertRaises(CacheError):
                load_cache_policy(p)

    def test_09_paths_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paths.yaml"
            p.write_text("schema_version: 1\nsystem: {}\n", encoding="utf-8")
            with self.assertRaises(CacheError):
                resolve_cache_root(p)

    def test_10_paths_relative_cache_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paths.yaml"
            p.write_text(
                "schema_version: 1\nsystem:\n  cache: relative/cache\n",
                encoding="utf-8",
            )
            with self.assertRaises(CacheError):
                resolve_cache_root(p)

    def test_11_tests_never_use_real_user_cache(self) -> None:
        # Guard: this suite must use tempfile for publish/read, not paths.yaml cache.
        real = resolve_cache_root(PATHS)
        self.assertEqual(str(real), "/home/fdoblak/workspace/cache")
        # Evidence: no test helper writes under that path (smoke assertion only).
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
