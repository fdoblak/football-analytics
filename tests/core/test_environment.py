#!/usr/bin/env python3
"""Environment record tests (Stage 2B)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from football_analytics.core.environment import (
    GPU_CLASSIFICATION,
    PACKAGE_ALLOWLIST,
    build_environment_record,
    collect_git_metadata,
)


class EnvironmentTests(unittest.TestCase):
    def test_01_no_heavy_imports(self) -> None:
        before = set(sys.modules)
        build_environment_record(
            project_version="0.1.0.dev0",
            config_fingerprint={
                "algorithm": "sha256",
                "canonicalization_version": 1,
                "digest": "a" * 64,
            },
            repo_root=None,
        )
        heavy = {"torch", "ultralytics", "SoccerNet", "cv2"}
        self.assertFalse(heavy & (set(sys.modules) - before))

    def test_02_package_allowlist(self) -> None:
        rec = build_environment_record(
            project_version="0.1.0.dev0",
            config_fingerprint={
                "algorithm": "sha256",
                "canonicalization_version": 1,
                "digest": "b" * 64,
            },
        )
        self.assertEqual(set(rec["packages"].keys()), set(PACKAGE_ALLOWLIST))

    def test_03_gpu_classification(self) -> None:
        rec = build_environment_record(
            project_version="0.1.0.dev0",
            config_fingerprint={
                "algorithm": "sha256",
                "canonicalization_version": 1,
                "digest": "c" * 64,
            },
        )
        self.assertEqual(rec["gpu_validation"]["classification"], GPU_CLASSIFICATION)
        self.assertFalse(rec["gpu_validation"]["torch_imported"])
        self.assertFalse(rec["gpu_validation"]["cuda_initialized"])

    def test_04_no_env_dump(self) -> None:
        rec = build_environment_record(
            project_version="0.1.0.dev0",
            config_fingerprint={
                "algorithm": "sha256",
                "canonicalization_version": 1,
                "digest": "d" * 64,
            },
        )
        self.assertNotIn("environ", rec)
        self.assertNotIn("os.environ", json.dumps(rec))
        # Must not dump arbitrary process environment keys
        self.assertNotIn("PATH", rec)
        self.assertNotIn("HOME", rec)

    def test_05_git_metadata_repo(self) -> None:
        root = Path(__file__).resolve().parents[2]
        meta = collect_git_metadata(root)
        self.assertIsNotNone(meta["commit"])
        self.assertEqual(len(meta["commit"] or ""), 40)

    def test_06_git_missing_repo(self) -> None:
        meta = collect_git_metadata(None)
        self.assertIsNone(meta["commit"])

    def test_07_remote_sanitized_no_userinfo(self) -> None:
        root = Path(__file__).resolve().parents[2]
        meta = collect_git_metadata(root)
        remote = meta["remote_sanitized"] or ""
        self.assertNotIn(
            "@", remote.split("://", 1)[-1].split("/", 1)[0] if "://" in remote else ""
        )


if __name__ == "__main__":
    unittest.main()
