#!/usr/bin/env python3
"""Unit tests for scripts/check_registries.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "check_registries.py"
    name = "check_registries_under_test"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CR = load_module()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def init_git_repo(path: Path, commit_message: str = "init") -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=path, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    return head


def base_model(path: Path, digest: str, size: int, commit: str) -> dict:
    return {
        "id": "m1",
        "display_name": "M1",
        "task": "t",
        "status": "available",
        "file_path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "framework": "pytorch",
        "source_type": "third_party_checkpoint",
        "source_repo": "tp1",
        "source_commit": commit,
        "source_url": "https://example.com/repo",
        "license": None,
        "license_status": "review_required",
        "input_spec": None,
        "output_spec": None,
        "classes": None,
        "precision": None,
        "device_requirements": None,
        "tested": False,
        "test_scope": "file_integrity_only",
        "last_verified": "2026-07-22",
        "notes": "test",
    }


def base_dataset(**overrides) -> dict:
    d = {
        "id": "d1",
        "display_name": "D1",
        "task": "t",
        "status": "planned",
        "local_path": None,
        "source": "project",
        "source_url": None,
        "version": None,
        "modalities": ["video"],
        "splits": None,
        "annotation_format": None,
        "license": None,
        "license_status": "review_required",
        "access_level": "unknown",
        "nda_required": None,
        "credentials_required": None,
        "redistribution_allowed": None,
        "checksum": None,
        "size_bytes": None,
        "last_verified": "2026-07-22",
        "planned_stage": "MVP-1",
        "notes": "test",
    }
    d.update(overrides)
    return d


def make_lock(tmp: Path, n_soccer: int = 19, third: dict | None = None) -> dict:
    repos = {}
    for i in range(n_soccer):
        p = tmp / f"sn_{i}"
        head = init_git_repo(p)
        repos[f"sn_{i}"] = {
            "path": str(p),
            "remote": f"https://github.com/example/sn_{i}.git",
            "branch": "main",
            "commit": head,
            "integration": "test",
            "dirty": False,
        }
    if third is None:
        third = {}
        for name in ("tracklab", "pnlcalib", "no_bells_just_whistles"):
            p = tmp / name
            head = init_git_repo(p)
            third[name] = {
                "path": str(p),
                "remote": f"https://github.com/example/{name}.git",
                "branch": "main",
                "commit": head,
                "integration": "test",
                "dirty": False,
            }
    return {
        "schema_version": 1,
        "repositories": repos,
        "third_party_repositories": third,
    }


class CheckRegistriesTests(unittest.TestCase):
    _td = None
    _shared_lock = None
    _shared_repos = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._td = tempfile.TemporaryDirectory(prefix="fa_reg_")
        cls._shared_repos = Path(cls._td.name) / "repos"
        cls._shared_lock = make_lock(cls._shared_repos)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._td is not None:
            cls._td.cleanup()

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fa_reg_case_", dir=self._td.name))
        self.weight = self.tmp / "w.pth"
        self.payload = b"fake-weight-bytes-for-test"
        self.weight.write_bytes(self.payload)
        self.digest = sha256_bytes(self.payload)
        self.size = len(self.payload)
        # Deep-ish copy lock via yaml round-trip
        self.lock_data = yaml.safe_load(yaml.safe_dump(self._shared_lock))
        self.source_commit = self.lock_data["third_party_repositories"]["no_bells_just_whistles"][
            "commit"
        ]
        self.model = base_model(self.weight, self.digest, self.size, self.source_commit)
        self.model["source_repo"] = "no_bells_just_whistles"
        self.model_reg = {"schema_version": 1, "models": [self.model]}
        self.dataset_reg = {"schema_version": 1, "datasets": [base_dataset()]}
        self.model_path = self.tmp / "model_registry.yaml"
        self.dataset_path = self.tmp / "dataset_registry.yaml"
        self.lock_path = self.tmp / "external_repos.lock.yaml"
        self._write_all()

    def tearDown(self) -> None:
        pass

    def _write_all(self) -> None:
        self.model_path.write_text(yaml.safe_dump(self.model_reg), encoding="utf-8")
        self.dataset_path.write_text(yaml.safe_dump(self.dataset_reg), encoding="utf-8")
        self.lock_path.write_text(yaml.safe_dump(self.lock_data), encoding="utf-8")

    def _run(self, *extra: str) -> int:
        argv = [
            "--model-registry",
            str(self.model_path),
            "--dataset-registry",
            str(self.dataset_path),
            "--external-lock",
            str(self.lock_path),
            "--quiet",
            *extra,
        ]
        return CR.main(argv)

    def test_01_valid_model_registry_pass(self) -> None:
        code = self._run("--verify-files", "--verify-repos")
        self.assertEqual(code, 0)

    def test_02_duplicate_model_id_fail(self) -> None:
        dup = dict(self.model)
        dup["id"] = "m1"
        dup["file_path"] = str(self.weight)
        self.model_reg["models"].append(dup)
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_03_missing_model_file_fail(self) -> None:
        self.model["file_path"] = str(self.tmp / "missing.pth")
        self._write_all()
        self.assertEqual(self._run("--verify-files"), 3)

    def test_04_wrong_model_size_fail(self) -> None:
        self.model["size_bytes"] = self.size + 1
        self._write_all()
        self.assertEqual(self._run("--verify-files"), 3)

    def test_05_wrong_model_sha_fail(self) -> None:
        self.model["sha256"] = "0" * 64
        self._write_all()
        self.assertEqual(self._run("--verify-files"), 3)

    def test_06_short_sha_fail(self) -> None:
        self.model["sha256"] = "abcd1234"
        self._write_all()
        # without verify_files still fails required full sha for available
        code = self._run()
        self.assertIn(code, (1, 3))

    def test_07_relative_model_path_fail(self) -> None:
        self.model["file_path"] = "relative.pth"
        self._write_all()
        self.assertIn(self._run(), (1, 3))

    def test_08_unknown_model_status_fail(self) -> None:
        self.model["status"] = "ready"
        self._write_all()
        self.assertEqual(self._run(), 1)

    def test_09_secret_query_source_url_fail(self) -> None:
        self.model["source_url"] = "https://example.com/w?" + "token=" + "abc123secret"
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_10_valid_planned_dataset_pass(self) -> None:
        self.assertEqual(self._run(), 0)

    def test_11_planned_marked_available_fail(self) -> None:
        self.dataset_reg["datasets"][0]["status"] = "available"
        self.dataset_reg["datasets"][0]["local_path"] = None
        self._write_all()
        self.assertIn(self._run(), (1, 3))

    def test_12_verified_missing_path_fail(self) -> None:
        self.dataset_reg["datasets"][0].update(
            {
                "status": "verified",
                "local_path": str(self.tmp / "no_such_dataset"),
                "checksum": "a" * 64,
                "access_level": "restricted",
            }
        )
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_13_duplicate_dataset_id_fail(self) -> None:
        self.dataset_reg["datasets"].append(base_dataset(id="d1"))
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_14_unknown_access_level_fail(self) -> None:
        self.dataset_reg["datasets"][0]["access_level"] = "friends_only"
        self._write_all()
        self.assertEqual(self._run(), 1)

    def test_15_credential_field_fail(self) -> None:
        self.dataset_reg["datasets"][0]["password"] = "should-not-appear"
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_16_token_dataset_url_fail(self) -> None:
        self.dataset_reg["datasets"][0]["source_url"] = (
            "https://x/?" + "access_token=" + "leak"
        )
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_17_external_lock_short_sha_fail(self) -> None:
        rid = next(iter(self.lock_data["repositories"]))
        self.lock_data["repositories"][rid]["commit"] = "abc1234"
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_18_duplicate_repo_path_fail(self) -> None:
        paths = list(self.lock_data["repositories"].values())
        self.lock_data["third_party_repositories"]["tracklab"]["path"] = paths[0]["path"]
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_19_head_lock_mismatch_fail(self) -> None:
        rid = next(iter(self.lock_data["repositories"]))
        bad = "0" * 40
        self.lock_data["repositories"][rid]["commit"] = bad
        self._write_all()
        self.assertEqual(self._run("--verify-repos"), 3)

    def test_20_model_source_commit_lock_mismatch_fail(self) -> None:
        self.model["source_commit"] = "1" * 40
        self._write_all()
        self.assertEqual(self._run(), 3)

    def test_21_atomic_json_output_valid(self) -> None:
        out = self.tmp / "out.json"
        code = self._run("--json-out", str(out))
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("status", payload)
        self.assertIn("exit_code", payload)

    def test_22_exit_code_contract(self) -> None:
        # config error
        code = CR.main(
            [
                "--model-registry",
                str(self.tmp / "missing.yaml"),
                "--dataset-registry",
                str(self.dataset_path),
                "--external-lock",
                str(self.lock_path),
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
