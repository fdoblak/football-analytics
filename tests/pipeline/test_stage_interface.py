#!/usr/bin/env python3
"""Stage protocol, registry, and SyntheticEchoStage tests (Stage 2D)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import sha256_bytes
from football_analytics.core.run_id import generate_run_id
from football_analytics.pipeline.artifacts import build_artifact_ref
from football_analytics.pipeline.exceptions import StageError
from football_analytics.pipeline.stage import (
    Stage,
    StageRegistry,
    SyntheticEchoStage,
    make_stage_identity,
)
from football_analytics.pipeline.types import ArtifactRef, StageRequest

FP = "c" * 64
FP2 = "d" * 64


def _make_request(
    stage: SyntheticEchoStage,
    work: Path,
    out: Path,
    *,
    config_fp: str = FP,
    cacheable: bool | None = None,
) -> StageRequest:
    inp = work / "payload.bin"
    inp.write_bytes(b"echo-payload")
    ref = build_artifact_ref(
        "Payload",
        inp,
        root=work,
        media_type="application/octet-stream",
    )
    identity = stage.identity
    if cacheable is not None:
        identity = make_stage_identity(
            name=identity.name,
            version=identity.version,
            code_fingerprint=identity.code_fingerprint,
            deterministic=identity.deterministic,
            cacheable=cacheable,
        )
    return StageRequest(
        run_id=generate_run_id(),
        stage_identity=identity,
        config_fingerprint=config_fp,
        compatibility_fingerprint=FP2,
        inputs={"Payload": ref},
        working_directory=work,
        output_directory=out,
        requested_at_utc="2026-07-22T12:00:00.000000Z",
    )


class StageInterfaceTests(unittest.TestCase):
    def test_01_make_stage_identity(self) -> None:
        ident = make_stage_identity(
            name="demo_stage",
            version=2,
            code_fingerprint=FP,
        )
        self.assertEqual(ident.name, "demo_stage")
        self.assertEqual(ident.version, 2)

    def test_02_synthetic_echo_is_stage(self) -> None:
        stage = SyntheticEchoStage()
        self.assertIsInstance(stage, Stage)
        self.assertEqual(stage.identity.name, "synthetic_echo")
        self.assertTrue(stage.identity.cacheable)
        self.assertTrue(stage.identity.deterministic)

    def test_03_registry_register_and_get(self) -> None:
        reg = StageRegistry()
        stage = SyntheticEchoStage()
        reg.register(stage)
        got = reg.get("synthetic_echo", 1)
        self.assertIs(got, stage)
        self.assertEqual(len(reg.list()), 1)

    def test_04_registry_duplicate_rejected(self) -> None:
        reg = StageRegistry()
        reg.register(SyntheticEchoStage())
        with self.assertRaises(StageError):
            reg.register(SyntheticEchoStage())

    def test_05_registry_missing(self) -> None:
        reg = StageRegistry()
        with self.assertRaises(StageError):
            reg.get("missing_stage", 1)

    def test_06_registry_latest_version(self) -> None:
        class V2:
            def __init__(self) -> None:
                self._identity = make_stage_identity(
                    name="synthetic_echo",
                    version=2,
                    code_fingerprint=FP,
                )

            @property
            def identity(self):
                return self._identity

            def execute(self, request):
                raise NotImplementedError

        reg = StageRegistry()
        s1 = SyntheticEchoStage()
        s2 = V2()
        reg.register(s1)
        reg.register(s2)
        self.assertIs(reg.get("synthetic_echo"), s2)

    def test_07_echo_execution_count(self) -> None:
        stage = SyntheticEchoStage()
        self.assertEqual(stage.executions, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            req = _make_request(stage, work, out)
            result = stage.execute(req)
            self.assertEqual(stage.executions, 1)
            self.assertIn("echo", result.outputs)
            self.assertEqual(result.metrics["executions"], 1)
            echo_path = out / "echo.bin"
            self.assertTrue(echo_path.is_file())
            expected = sha256_bytes(b"echo-payload" + FP.encode("utf-8"))
            # echo stores digest of (bytes || config_fp) as hex ascii
            self.assertEqual(echo_path.read_text(encoding="ascii"), expected)

    def test_08_echo_requires_one_input(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            req = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            with self.assertRaises(StageError):
                stage.execute(req)

    def test_09_echo_rejects_existing_output(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            (out / "echo.bin").write_bytes(b"exists")
            req = _make_request(stage, work, out)
            with self.assertRaises(StageError):
                stage.execute(req)

    def test_10_echo_rejects_symlink_input(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            # Symlink that escapes working_directory (resolve follows; relative_to fails
            # OR is_symlink check on the unresolved path via is_file/is_symlink).
            outside = root / "outside.bin"
            outside.write_bytes(b"data")
            link = work / "payload.bin"
            link.symlink_to(outside)
            ref = ArtifactRef(
                logical_name="Payload",
                relative_path="payload.bin",
                media_type="application/octet-stream",
                size_bytes=4,
                sha256=sha256_bytes(b"data"),
            )
            req = StageRequest(
                run_id=generate_run_id(),
                stage_identity=stage.identity,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Payload": ref},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            with self.assertRaises(StageError):
                stage.execute(req)

    def test_11_identity_mismatch(self) -> None:
        stage = SyntheticEchoStage()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            other = make_stage_identity(
                name="other_stage",
                version=1,
                code_fingerprint=FP,
            )
            inp = work / "payload.bin"
            inp.write_bytes(b"x")
            ref = build_artifact_ref(
                "Payload", inp, root=work, media_type="application/octet-stream"
            )
            req = StageRequest(
                run_id=generate_run_id(),
                stage_identity=other,
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Payload": ref},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            with self.assertRaises(StageError):
                stage.execute(req)


if __name__ == "__main__":
    unittest.main()
