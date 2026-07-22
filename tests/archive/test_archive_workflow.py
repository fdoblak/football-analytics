#!/usr/bin/env python3
"""Archive / verify / restore / cleanup workflow tests (Stage 1D)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_analytics.utils import archive_safety as safety  # noqa: E402


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    mod_name = f"{name.replace('.', '_')}_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


AR = load_script("archive_run.py")
VA = load_script("verify_archive.py")
RR = load_script("restore_run.py")
CU = load_script("cleanup_run.py")

RUN_ID = "run_20260722_120000_a1d001"


def write_policy(root: Path) -> Path:
    workspace = root / "workspace"
    runs = workspace / "runs"
    archive = root / "archive"
    quarantine = workspace / "quarantine"
    for p in (runs, archive, quarantine, workspace):
        p.mkdir(parents=True, exist_ok=True)
    policy = {
        "schema_version": 1,
        "paths": {
            "runs_root": str(runs),
            "archive_root": str(archive),
            "quarantine_root": str(quarantine),
            "workspace_root": str(workspace),
            "current_symlink": str(workspace / "current"),
            "planned_external_archive_root": "/mnt/d/football_data/experiments_archive",
        },
        "policy": {
            "active_archive_backend": "wsl_local",
            "failure_domain": "same_wsl_vhdx",
            "independent_backup": False,
            "checksum_algorithm": "sha256",
            "require_completed_status": True,
            "require_archive_verification": True,
            "reject_symlinks": True,
            "reject_special_files": True,
            "reject_path_escape": True,
            "cleanup_mode": "quarantine",
            "quarantine_retention_days": 30,
            "minimum_archive_free_bytes": 1024,
        },
        "run_id": {"pattern": r"^run_[0-9]{8}_[0-9]{6}_[a-z0-9]{6,12}$"},
        "required_source_files": ["run_manifest.json"],
        "tool_version": "1.0.0-stage1d-test",
    }
    path = root / "archive_policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return path


def make_completed_run(
    runs_root: Path,
    run_id: str = RUN_ID,
    *,
    status: str = "completed",
    with_artifact: bool = True,
    fixture_marker: str = "stage1d_synthetic_fixture",
) -> Path:
    run = runs_root / run_id
    (run / "00_ingest").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    artifact = "00_ingest/video_manifest.json"
    if with_artifact:
        (run / artifact).write_text('{"ok": true}\n', encoding="utf-8")
    (run / "logs" / "smoke.jsonl").write_text('{"event":"smoke"}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "created_at": "2026-07-22T12:00:00Z",
        "completed_at": "2026-07-22T12:01:00Z" if status == "completed" else None,
        "input": {"source": "synthetic", "sha256": None},
        "stages": {},
        "required_artifacts": [artifact] if with_artifact else [],
        "notes": "test",
        "fixture_marker": fixture_marker,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run


class ArchiveWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="fa_arch_")
        self.root = Path(self._td.name)
        self.policy = write_policy(self.root)
        self.pol = yaml.safe_load(self.policy.read_text())
        self.runs = Path(self.pol["paths"]["runs_root"])
        self.archive = Path(self.pol["paths"]["archive_root"])
        self.quarantine = Path(self.pol["paths"]["quarantine_root"])
        self.workspace = Path(self.pol["paths"]["workspace_root"])

    def tearDown(self) -> None:
        self._td.cleanup()

    def _archive(self, *extra: str, run_id: str = RUN_ID) -> int:
        return AR.main(["--run-id", run_id, "--policy", str(self.policy), "--quiet", *extra])

    def _verify(self, *extra: str, run_id: str = RUN_ID) -> int:
        return VA.main(["--run-id", run_id, "--policy", str(self.policy), "--quiet", *extra])

    def _restore(self, *extra: str, run_id: str = RUN_ID) -> int:
        return RR.main(["--run-id", run_id, "--policy", str(self.policy), "--quiet", *extra])

    def _cleanup(self, *extra: str, run_id: str = RUN_ID) -> int:
        return CU.main(["--run-id", run_id, "--policy", str(self.policy), "--quiet", *extra])

    def test_01_valid_completed_archive_pass(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertTrue((self.archive / RUN_ID / "archive_manifest.json").is_file())
        self.assertTrue((self.runs / RUN_ID / "archive_receipt.json").is_file())

    def test_02_pending_rejected(self) -> None:
        make_completed_run(self.runs, status="pending")
        self.assertEqual(self._archive("--execute"), 1)

    def test_03_running_rejected(self) -> None:
        make_completed_run(self.runs, status="running")
        self.assertEqual(self._archive("--execute"), 1)

    def test_04_failed_rejected(self) -> None:
        make_completed_run(self.runs, status="failed")
        self.assertEqual(self._archive("--execute"), 1)

    def test_05_invalid_run_id_rejected(self) -> None:
        make_completed_run(self.runs)
        code = AR.main(
            ["--run-id", "../escape", "--policy", str(self.policy), "--quiet", "--execute"]
        )
        self.assertEqual(code, 3)

    def test_06_source_outside_runs_root(self) -> None:
        # Craft outside path; archive must not treat it as runs_root child.
        outside = self.root / "outside" / RUN_ID
        outside.mkdir(parents=True)
        (outside / "run_manifest.json").write_text("{}", encoding="utf-8")
        # attempting archive of run that doesn't exist under runs_root
        self.assertIn(self._archive("--execute"), (1, 2))

    def test_07_missing_run_manifest(self) -> None:
        run = self.runs / RUN_ID
        run.mkdir(parents=True)
        (run / "x.txt").write_text("x", encoding="utf-8")
        self.assertEqual(self._archive("--execute"), 1)

    def test_08_manifest_run_id_mismatch(self) -> None:
        run = make_completed_run(self.runs)
        data = json.loads((run / "run_manifest.json").read_text())
        data["run_id"] = "run_20260722_120000_other01"
        (run / "run_manifest.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self._archive("--execute"), 1)

    def test_09_missing_required_artifact(self) -> None:
        make_completed_run(self.runs, with_artifact=False)
        run = self.runs / RUN_ID
        data = json.loads((run / "run_manifest.json").read_text())
        data["required_artifacts"] = ["00_ingest/missing.json"]
        (run / "run_manifest.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self._archive("--execute"), 1)

    def test_10_source_symlink_rejected(self) -> None:
        target = self.runs / "realdir"
        make_completed_run(self.runs, run_id="realdir")
        link = self.runs / RUN_ID
        link.symlink_to(target)
        self.assertEqual(self._archive("--execute"), 3)

    def test_11_inner_symlink_rejected(self) -> None:
        run = make_completed_run(self.runs)
        (run / "evil").symlink_to("/tmp")
        self.assertEqual(self._archive("--execute"), 3)

    def test_12_fifo_rejected(self) -> None:
        run = make_completed_run(self.runs)
        fifo = run / "pipe.fifo"
        os.mkfifo(fifo)
        self.assertEqual(self._archive("--execute"), 3)

    def test_13_unsafe_relative_path_helper(self) -> None:
        with self.assertRaises(safety.ArchiveError):
            safety.normalize_rel_path("../x")
        with self.assertRaises(safety.ArchiveError):
            safety.normalize_rel_path("/abs")

    def test_14_archive_overwrite_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        # modify source so not identical idempotent
        (self.runs / RUN_ID / "logs" / "smoke.jsonl").write_text(
            '{"event":"changed"}\n', encoding="utf-8"
        )
        self.assertEqual(self._archive("--execute"), 1)

    def test_15_temp_archive_failure_cleanup(self) -> None:
        make_completed_run(self.runs)
        # Force copy failure; temporary archive dirs must be cleaned up.
        orig = safety.copy_file_verified

        def boom(*a, **k):
            raise safety.ArchiveError("forced copy fail", safety.EXIT_INTEGRITY)

        safety.copy_file_verified = boom  # type: ignore
        try:
            code = self._archive("--execute")
            self.assertEqual(code, 1)
            leftovers = list(self.archive.glob(".archive_tmp_*"))
            self.assertEqual(leftovers, [])
        finally:
            safety.copy_file_verified = orig  # type: ignore

    def test_16_archive_file_hash_exact(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        manifest = json.loads((self.archive / RUN_ID / "archive_manifest.json").read_text())
        for entry in manifest["files"]:
            p = self.archive / RUN_ID / entry["relative_path"]
            self.assertEqual(safety.sha256_file(p), entry["sha256"])
            self.assertEqual(p.stat().st_size, entry["size_bytes"])

    def test_17_totals_exact(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        manifest = json.loads((self.archive / RUN_ID / "archive_manifest.json").read_text())
        files = [e for e in manifest["files"]]
        self.assertEqual(manifest["total_files"], len(files))
        self.assertEqual(manifest["total_bytes"], sum(e["size_bytes"] for e in files))

    def test_18_manifest_sorted(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        files = json.loads((self.archive / RUN_ID / "archive_manifest.json").read_text())["files"]
        paths = [f["relative_path"] for f in files]
        self.assertEqual(paths, sorted(paths))

    def test_19_tampered_archive_verify_fail(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        target = self.archive / RUN_ID / "logs" / "smoke.jsonl"
        target.write_text("tampered\n", encoding="utf-8")
        self.assertEqual(self._verify(), 1)

    def test_20_missing_archive_file_verify_fail(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        (self.archive / RUN_ID / "logs" / "smoke.jsonl").unlink()
        self.assertEqual(self._verify(), 1)

    def test_21_extra_archive_file_verify_fail(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        (self.archive / RUN_ID / "extra.txt").write_text("x", encoding="utf-8")
        self.assertEqual(self._verify(), 1)

    def test_22_unsafe_manifest_path_verify_fail(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        mpath = self.archive / RUN_ID / "archive_manifest.json"
        data = json.loads(mpath.read_text())
        data["files"].append(
            {
                "relative_path": "../escape.txt",
                "size_bytes": 1,
                "sha256": "a" * 64,
                "file_type": "regular",
            }
        )
        data["total_files"] = len(data["files"])
        mpath.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn(self._verify(), (1, 3))

    def test_23_restore_verified_pass(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        self.assertEqual(self._restore("--execute"), 0)
        self.assertTrue((self.runs / RUN_ID / "run_manifest.json").is_file())

    def test_24_restore_overwrite_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._restore("--execute"), 1)

    def test_25_restore_tampered_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        (self.archive / RUN_ID / "logs" / "smoke.jsonl").write_text("bad", encoding="utf-8")
        self.assertEqual(self._restore("--execute"), 1)

    def test_26_restore_hashes_match(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        arch_manifest = json.loads((self.archive / RUN_ID / "archive_manifest.json").read_text())
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        self.assertEqual(self._restore("--execute"), 0)
        for entry in arch_manifest["files"]:
            p = self.runs / RUN_ID / entry["relative_path"]
            self.assertEqual(safety.sha256_file(p), entry["sha256"])

    def test_27_cleanup_dry_run_no_mutation(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        before = list(self.runs.iterdir())
        self.assertEqual(self._cleanup(), 0)
        self.assertEqual(list(self.runs.iterdir()), before)
        self.assertTrue((self.runs / RUN_ID).is_dir())

    def test_28_cleanup_without_receipt_rejected(self) -> None:
        make_completed_run(self.runs)
        # archive without leaving receipt — remove receipt
        self.assertEqual(self._archive("--execute"), 0)
        (self.runs / RUN_ID / "archive_receipt.json").unlink()
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 1)

    def test_29_cleanup_confirm_mismatch(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(
            self._cleanup("--execute", "--confirm-run-id", "run_20260722_120000_zzzzzz"),
            3,
        )

    def test_30_cleanup_tampered_archive_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        (self.archive / RUN_ID / "logs" / "smoke.jsonl").write_text("x", encoding="utf-8")
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 1)

    def test_31_cleanup_current_symlink_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        current = self.workspace / "current"
        current.symlink_to(self.runs / RUN_ID)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 3)

    def test_32_cleanup_moves_to_quarantine(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        self.assertFalse((self.runs / RUN_ID).exists())
        q = list(self.quarantine.glob(f"{RUN_ID}_*"))
        self.assertEqual(len(q), 1)

    def test_33_quarantine_receipt(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        q = next(self.quarantine.glob(f"{RUN_ID}_*"))
        receipt = json.loads((q / "quarantine_receipt.json").read_text())
        self.assertEqual(receipt["purge_status"], "not_performed")
        self.assertEqual(receipt["run_id"], RUN_ID)
        self.assertTrue(receipt["quarantine_path"].endswith(q.name))

    def test_34_cleanup_no_permanent_delete(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 0)
        self.assertTrue((self.archive / RUN_ID).is_dir())
        self.assertTrue(any(self.quarantine.iterdir()))

    def test_35_quarantine_collision_rejected(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)

        # Pre-create colliding quarantine name via datetime monkeypatch.
        class FixedDateTime:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 7, 22, 0, 0, 0, tzinfo=timezone.utc)

            @staticmethod
            def strftime(*a, **k):
                raise NotImplementedError

        # patch cleanup_run.datetime

        # module already loaded as CU; patch CU.datetime
        real = CU.datetime

        class D:
            timezone = timezone

            @staticmethod
            def now(tz=None):
                return datetime(2026, 7, 22, 0, 0, 0, tzinfo=timezone.utc)

        CU.datetime = D  # type: ignore
        try:
            qpath = self.quarantine / f"{RUN_ID}_20260722T000000Z"
            qpath.mkdir()
            self.assertEqual(self._cleanup("--execute", "--confirm-run-id", RUN_ID), 1)
        finally:
            CU.datetime = real  # type: ignore

    def test_36_dangerous_broad_paths_rejected(self) -> None:
        with self.assertRaises(safety.ArchiveError):
            safety.assert_not_dangerous_operation_root(Path("/home/fdoblak"))
        with self.assertRaises(safety.ArchiveError):
            safety.assert_contained(Path("/tmp/x"), Path(self.runs), label="x")

    def test_37_planned_mnt_d_absence_ok(self) -> None:
        make_completed_run(self.runs)
        self.assertFalse(Path("/mnt/d/football_data/experiments_archive").exists())
        self.assertEqual(self._archive("--execute"), 0)

    def test_38_independent_backup_false(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        manifest = json.loads((self.archive / RUN_ID / "archive_manifest.json").read_text())
        self.assertIs(manifest["independent_backup"], False)
        self.assertEqual(manifest["failure_domain"], "same_wsl_vhdx")

    def test_39_json_runtime_outputs_valid(self) -> None:
        make_completed_run(self.runs)
        out = self.root / "out.json"
        self.assertEqual(self._archive("--execute", "--json-out", str(out)), 0)
        payload = json.loads(out.read_text())
        self.assertIn("status", payload)
        self.assertEqual(payload["exit_code"], 0)

    def test_40_exit_code_contracts(self) -> None:
        code = AR.main(["--run-id", "bad", "--policy", str(self.policy), "--quiet"])
        self.assertEqual(code, 3)
        code = VA.main(["--policy", str(self.policy), "--quiet"])
        self.assertEqual(code, 2)

    def test_41_atomic_write_failure_safe(self) -> None:
        target = self.root / "nope" / "x.json"
        with self.assertRaises(safety.ArchiveError):
            safety.write_json_atomic(target, {"a": 1})

    def test_42_no_fixture_residue_outside_temp(self) -> None:
        make_completed_run(self.runs)
        self.assertEqual(self._archive("--execute"), 0)
        # ensure no files created at real workspace roots by this test
        self.assertFalse(Path("/home/fdoblak/workspace/runs").joinpath(RUN_ID).exists())


if __name__ == "__main__":
    unittest.main()
