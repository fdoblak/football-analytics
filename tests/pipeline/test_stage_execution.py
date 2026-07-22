#!/usr/bin/env python3
"""Stage execution lifecycle tests (Stage 2D)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.core.run_id import generate_run_id
from football_analytics.pipeline.artifacts import build_artifact_ref
from football_analytics.pipeline.cache import entry_dir, load_cache_policy
from football_analytics.pipeline.execution import execute_stage
from football_analytics.pipeline.stage import SyntheticEchoStage, make_stage_identity
from football_analytics.pipeline.types import StageRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "system" / "cache_policy.yaml"
FP = "d" * 64
FP2 = "e" * 64


def _prepare(root: Path, stage: SyntheticEchoStage, *, config_fp: str = FP):
    work = root / "work"
    out = root / "out"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Fresh output each call
    for child in out.iterdir():
        if child.is_file():
            child.unlink()
    inp = work / "payload.bin"
    if not inp.exists():
        inp.write_bytes(b"exec-payload")
    ref = build_artifact_ref("Payload", inp, root=work, media_type="application/octet-stream")
    req = StageRequest(
        run_id=generate_run_id(),
        stage_identity=stage.identity,
        config_fingerprint=config_fp,
        compatibility_fingerprint=FP2,
        inputs={"Payload": ref},
        working_directory=work,
        output_directory=out,
        requested_at_utc="2026-07-22T12:00:00.000000Z",
    )
    return req, work, out


class StageExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_cache_policy(POLICY_PATH)

    def test_01_miss_execute_publish(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req, _work, out = _prepare(root / "run1", stage)
            result = execute_stage(
                stage,
                req,
                cache_root=cache_root,
                policy=self.policy,
                quarantine_root=root / "q",
            )
            self.assertEqual(result.status, "succeeded")
            self.assertFalse(result.cache_hit)
            self.assertEqual(stage.executions, 1)
            self.assertTrue(entry_dir(cache_root, result.cache_key).is_dir())
            self.assertTrue((out / "echo.bin").is_file())
            self.assertTrue((out / "stage_execution_receipt.json").is_file())

    def test_02_hit_no_execute(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req1, _w1, _o1 = _prepare(root / "run1", stage)
            r1 = execute_stage(stage, req1, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r1.status, "succeeded")
            self.assertEqual(stage.executions, 1)

            req2, _w2, out2 = _prepare(root / "run2", stage)
            # Same fingerprints/inputs content for cache key
            req2 = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs=req1.inputs,
                working_directory=req2.working_directory,
                output_directory=out2,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            # Ensure input file exists under new working dir with same relative path
            import shutil

            src_name = next(iter(req1.inputs.values())).relative_path
            dst = req2.working_directory / src_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(root / "run1" / "work" / src_name, dst)

            r2 = execute_stage(stage, req2, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r2.status, "cache_hit")
            self.assertTrue(r2.cache_hit)
            self.assertEqual(stage.executions, 1)
            self.assertTrue((out2 / "echo.bin").is_file())

    def test_03_cache_disabled_always_executes(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
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
            req1, _, out1 = _prepare(root / "run1", stage)
            r1 = execute_stage(stage, req1, cache_root=cache_root, policy=disabled)
            self.assertEqual(r1.status, "succeeded")
            self.assertFalse(r1.cache_hit)
            self.assertEqual(stage.executions, 1)
            self.assertFalse(list(cache_root.rglob("cache_manifest.json")))

            # Second run still executes
            for child in out1.iterdir():
                if child.is_file():
                    child.unlink()
            req2, _, _ = _prepare(root / "run1", stage)
            r2 = execute_stage(stage, req2, cache_root=cache_root, policy=disabled)
            self.assertEqual(r2.status, "succeeded")
            self.assertEqual(stage.executions, 2)

    def test_04_request_cache_policy_disabled(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req, _, out = _prepare(root / "run1", stage)
            req = StageRequest(
                run_id=req.run_id,
                stage_identity=req.stage_identity,
                config_fingerprint=req.config_fingerprint,
                compatibility_fingerprint=req.compatibility_fingerprint,
                inputs=dict(req.inputs),
                working_directory=req.working_directory,
                output_directory=out,
                requested_at_utc=req.requested_at_utc,
                cache_policy_enabled=False,
            )
            r = execute_stage(stage, req, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r.status, "succeeded")
            self.assertFalse(list((cache_root).rglob("cache_manifest.json")))

    def test_05_stage_exception_failed_no_publish(self) -> None:
        class BoomStage:
            def __init__(self) -> None:
                self._identity = make_stage_identity(
                    name="boom_stage",
                    version=1,
                    code_fingerprint=FP,
                    deterministic=True,
                    cacheable=True,
                )
                self.executions = 0

            @property
            def identity(self):
                return self._identity

            def execute(self, request):
                self.executions += 1
                raise RuntimeError("intentional failure")

        stage = BoomStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            inp = work / "x.bin"
            inp.write_bytes(b"x")
            ref = build_artifact_ref("X", inp, root=work, media_type="application/octet-stream")
            req = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"X": ref},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            result = execute_stage(stage, req, cache_root=cache_root, policy=self.policy)
            self.assertEqual(result.status, "failed")
            self.assertIsNotNone(result.error)
            self.assertEqual(result.error["class"], "RuntimeError")
            self.assertFalse(list(cache_root.rglob("cache_manifest.json")))
            self.assertTrue((out / "stage_execution_receipt.json").is_file())

    def test_06_nondeterministic_no_cache(self) -> None:
        stage = SyntheticEchoStage()

        # Replace identity flags via wrapper
        class Nondet:
            def __init__(self, inner: SyntheticEchoStage) -> None:
                self._inner = inner
                self._identity = make_stage_identity(
                    name=inner.identity.name,
                    version=inner.identity.version,
                    code_fingerprint=inner.identity.code_fingerprint,
                    deterministic=False,
                    cacheable=True,
                )

            @property
            def identity(self):
                return self._identity

            @property
            def executions(self):
                return self._inner.executions

            def execute(self, request):
                # SyntheticEchoStage checks request.stage_identity.name
                return self._inner.execute(
                    StageRequest(
                        run_id=request.run_id,
                        stage_identity=self._inner.identity,
                        config_fingerprint=request.config_fingerprint,
                        compatibility_fingerprint=request.compatibility_fingerprint,
                        inputs=dict(request.inputs),
                        working_directory=request.working_directory,
                        output_directory=request.output_directory,
                        requested_at_utc=request.requested_at_utc,
                        cache_policy_enabled=request.cache_policy_enabled,
                    )
                )

        wrapped = Nondet(stage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req, _, _ = _prepare(root / "run1", stage)
            req = StageRequest(
                run_id=req.run_id,
                stage_identity=wrapped.identity,
                config_fingerprint=req.config_fingerprint,
                compatibility_fingerprint=req.compatibility_fingerprint,
                inputs=dict(req.inputs),
                working_directory=req.working_directory,
                output_directory=req.output_directory,
                requested_at_utc=req.requested_at_utc,
            )
            r = execute_stage(wrapped, req, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r.status, "succeeded")
            self.assertFalse(list(cache_root.rglob("cache_manifest.json")))

    def test_07_non_cacheable_no_cache(self) -> None:
        stage = SyntheticEchoStage()

        class Uncacheable:
            def __init__(self, inner: SyntheticEchoStage) -> None:
                self._inner = inner
                self._identity = make_stage_identity(
                    name=inner.identity.name,
                    version=inner.identity.version,
                    code_fingerprint=inner.identity.code_fingerprint,
                    deterministic=True,
                    cacheable=False,
                )

            @property
            def identity(self):
                return self._identity

            def execute(self, request):
                return self._inner.execute(
                    StageRequest(
                        run_id=request.run_id,
                        stage_identity=self._inner.identity,
                        config_fingerprint=request.config_fingerprint,
                        compatibility_fingerprint=request.compatibility_fingerprint,
                        inputs=dict(request.inputs),
                        working_directory=request.working_directory,
                        output_directory=request.output_directory,
                        requested_at_utc=request.requested_at_utc,
                        cache_policy_enabled=request.cache_policy_enabled,
                    )
                )

        wrapped = Uncacheable(stage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req, _, _ = _prepare(root / "run1", stage)
            req = StageRequest(
                run_id=req.run_id,
                stage_identity=wrapped.identity,
                config_fingerprint=req.config_fingerprint,
                compatibility_fingerprint=req.compatibility_fingerprint,
                inputs=dict(req.inputs),
                working_directory=req.working_directory,
                output_directory=req.output_directory,
                requested_at_utc=req.requested_at_utc,
            )
            r = execute_stage(wrapped, req, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r.status, "succeeded")
            self.assertFalse(list(cache_root.rglob("cache_manifest.json")))

    def test_08_config_change_miss(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req1, _, _ = _prepare(root / "run1", stage, config_fp=FP)
            execute_stage(stage, req1, cache_root=cache_root, policy=self.policy)
            self.assertEqual(stage.executions, 1)
            req2, _, out2 = _prepare(root / "run2", stage, config_fp=FP2)
            r2 = execute_stage(stage, req2, cache_root=cache_root, policy=self.policy)
            self.assertEqual(r2.status, "succeeded")
            self.assertFalse(r2.cache_hit)
            self.assertEqual(stage.executions, 2)
            self.assertTrue((out2 / "echo.bin").is_file())

    def test_09_force_miss(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            req1, _, _ = _prepare(root / "run1", stage)
            execute_stage(stage, req1, cache_root=cache_root, policy=self.policy)
            req2, work2, out2 = _prepare(root / "run2", stage)
            import shutil

            shutil.copy2(root / "run1" / "work" / "payload.bin", work2 / "payload.bin")
            req2 = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs=req1.inputs,
                working_directory=work2,
                output_directory=out2,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            r2 = execute_stage(
                stage,
                req2,
                cache_root=cache_root,
                policy=self.policy,
                force_miss=True,
            )
            self.assertEqual(r2.status, "succeeded")
            self.assertEqual(stage.executions, 2)

    def test_10_corruption_quarantine_then_miss(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            qroot = root / "quarantine"
            req1, _, _ = _prepare(root / "run1", stage)
            r1 = execute_stage(
                stage,
                req1,
                cache_root=cache_root,
                policy=self.policy,
                quarantine_root=qroot,
            )
            entry = entry_dir(cache_root, r1.cache_key)
            art = entry / "artifacts" / "echo.bin"
            art.write_bytes(b"CORRUPT")
            req2, work2, out2 = _prepare(root / "run2", stage)
            import shutil

            shutil.copy2(root / "run1" / "work" / "payload.bin", work2 / "payload.bin")
            req2 = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs=req1.inputs,
                working_directory=work2,
                output_directory=out2,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            r2 = execute_stage(
                stage,
                req2,
                cache_root=cache_root,
                policy=self.policy,
                quarantine_root=qroot,
            )
            self.assertEqual(r2.status, "succeeded")
            self.assertFalse(r2.cache_hit)
            self.assertEqual(stage.executions, 2)
            self.assertTrue(any("corrupt" in w.lower() for w in r2.warnings))
            # Quarantine receipt permanent_delete_performed False
            receipts = list(qroot.rglob("quarantine_receipt.json"))
            self.assertTrue(receipts)
            import json

            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertIs(receipt["permanent_delete_performed"], False)

    def test_11_identity_mismatch_raises(self) -> None:
        from football_analytics.pipeline.exceptions import StageError

        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = make_stage_identity(name="other_stage", version=1, code_fingerprint=FP)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            inp = work / "p.bin"
            inp.write_bytes(b"p")
            ref = build_artifact_ref("P", inp, root=work, media_type="application/octet-stream")
            req = StageRequest(
                run_id=generate_run_id(),
                stage_identity=other,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"P": ref},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            with self.assertRaises(StageError):
                execute_stage(stage, req, cache_root=root / "cache", policy=self.policy)

    def test_12_no_eager_heavy_imports(self) -> None:
        import importlib
        import sys

        # Do not delete already-loaded pipeline modules (breaks isinstance across tests).
        before = set(sys.modules)
        importlib.import_module("football_analytics.pipeline")
        importlib.import_module("football_analytics.pipeline.types")
        importlib.import_module("football_analytics.pipeline.cache")
        newly = set(sys.modules) - before
        self.assertNotIn("torch", newly)
        self.assertNotIn("pyarrow", newly)
        init = (REPO_ROOT / "src/football_analytics/pipeline/__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(init, r"(?m)^import torch\b")
        self.assertNotRegex(init, r"(?m)^from torch\b")
        self.assertNotRegex(init, r"(?m)^import pyarrow\b")
        self.assertNotRegex(init, r"(?m)^from pyarrow\b")
        # Top-level modules must not eagerly import pyarrow; lazy import inside
        # _verify_parquet_schema_fingerprint is allowed.
        for rel in (
            "src/football_analytics/pipeline/types.py",
            "src/football_analytics/pipeline/cache.py",
            "src/football_analytics/pipeline/execution.py",
            "src/football_analytics/pipeline/stage.py",
            "src/football_analytics/pipeline/cache_key.py",
            "src/football_analytics/pipeline/receipts.py",
        ):
            src = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotRegex(src, r"(?m)^import pyarrow\b")
            self.assertNotRegex(src, r"(?m)^from pyarrow\b")
            self.assertNotRegex(src, r"(?m)^import torch\b")
        art_src = (REPO_ROOT / "src/football_analytics/pipeline/artifacts.py").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(art_src, r"(?m)^import pyarrow\b")
        self.assertIn("import pyarrow.parquet", art_src)


if __name__ == "__main__":
    unittest.main()
