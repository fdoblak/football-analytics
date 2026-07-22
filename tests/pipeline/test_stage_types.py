#!/usr/bin/env python3
"""StageIdentity / StageRequest / StageResult type tests (Stage 2D)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

from football_analytics.pipeline.exceptions import ArtifactError, PipelineError, StageError
from football_analytics.pipeline.types import (
    ArtifactRef,
    CacheManifest,
    ContractRef,
    StageExecutionOutput,
    StageIdentity,
    StageRequest,
    StageResult,
)

FP = "a" * 64
FP2 = "b" * 64
RUN = "run_20260722T120000000000Z_abcdef123456"


def _identity(**kwargs) -> StageIdentity:
    base = dict(
        name="echo_stage",
        version=1,
        code_fingerprint=FP,
        input_contracts=(),
        output_contracts=(),
        deterministic=True,
        cacheable=True,
    )
    base.update(kwargs)
    return StageIdentity(**base)


def _artifact(**kwargs) -> ArtifactRef:
    base = dict(
        logical_name="Input",
        relative_path="in.bin",
        media_type="application/octet-stream",
        size_bytes=3,
        sha256=FP,
        metadata={},
    )
    base.update(kwargs)
    return ArtifactRef(**base)


def _result(**kwargs) -> StageResult:
    base = dict(
        run_id=RUN,
        stage_name="echo_stage",
        stage_version=1,
        status="succeeded",
        cache_key=FP,
        cache_hit=False,
        started_at_utc="2026-07-22T12:00:00.000000Z",
        finished_at_utc="2026-07-22T12:00:01.000000Z",
        duration_ms=10,
        inputs={},
        outputs={},
        metrics={},
        warnings=(),
        error=None,
        execution_fingerprint=FP2,
    )
    base.update(kwargs)
    return StageResult(**base)


class StageTypesTests(unittest.TestCase):
    def test_01_valid_identity(self) -> None:
        ident = _identity()
        self.assertEqual(ident.name, "echo_stage")
        self.assertTrue(ident.deterministic)
        self.assertTrue(ident.cacheable)

    def test_02_identity_rejects_slash(self) -> None:
        with self.assertRaises(StageError):
            _identity(name="bad/name")

    def test_03_identity_rejects_uppercase(self) -> None:
        with self.assertRaises(StageError):
            _identity(name="BadName")

    def test_04_identity_rejects_dotdot(self) -> None:
        with self.assertRaises(StageError):
            _identity(name="..evil")

    def test_05_identity_rejects_single_char(self) -> None:
        with self.assertRaises(StageError):
            _identity(name="a")

    def test_06_identity_rejects_bad_fingerprint(self) -> None:
        with self.assertRaises(PipelineError):
            _identity(code_fingerprint="nothex")

    def test_07_identity_rejects_zero_version(self) -> None:
        with self.assertRaises(StageError):
            _identity(version=0)

    def test_08_identity_to_dict(self) -> None:
        d = _identity(output_contracts=(ContractRef("events", 1),)).to_dict()
        self.assertEqual(d["name"], "echo_stage")
        self.assertEqual(d["output_contracts"][0]["name"], "events")

    def test_09_contract_ref_rejects_path(self) -> None:
        with self.assertRaises(StageError):
            ContractRef(name="../x", version=1)
        with self.assertRaises(StageError):
            ContractRef(name="a/b", version=1)

    def test_10_contract_ref_rejects_negative_version(self) -> None:
        with self.assertRaises(StageError):
            ContractRef(name="events", version=-1)

    def test_11_request_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            req = StageRequest(
                run_id=RUN,
                stage_identity=_identity(),
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Input": _artifact()},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            with self.assertRaises((TypeError, AttributeError)):
                req.run_id = "mutated"  # type: ignore[misc]
            with self.assertRaises(TypeError):
                req.inputs["other"] = _artifact(logical_name="other")  # type: ignore[index]

    def test_12_request_invalid_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            with self.assertRaises(StageError):
                StageRequest(
                    run_id="not-a-run-id",
                    stage_identity=_identity(),
                    config_fingerprint=FP,
                    compatibility_fingerprint=FP2,
                    inputs={},
                    working_directory=work,
                    output_directory=out,
                    requested_at_utc="2026-07-22T12:00:00.000000Z",
                )

    def test_13_request_rejects_same_work_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            both = root / "same"
            both.mkdir()
            with self.assertRaises(StageError):
                StageRequest(
                    run_id=RUN,
                    stage_identity=_identity(),
                    config_fingerprint=FP,
                    compatibility_fingerprint=FP2,
                    inputs={},
                    working_directory=both,
                    output_directory=both,
                    requested_at_utc="2026-07-22T12:00:00.000000Z",
                )

    def test_14_request_input_key_must_match_logical_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            with self.assertRaises(StageError):
                StageRequest(
                    run_id=RUN,
                    stage_identity=_identity(),
                    config_fingerprint=FP,
                    compatibility_fingerprint=FP2,
                    inputs={"Wrong": _artifact(logical_name="Input")},
                    working_directory=work,
                    output_directory=out,
                    requested_at_utc="2026-07-22T12:00:00.000000Z",
                )

    def test_15_request_to_dict_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out = root / "out"
            work.mkdir()
            out.mkdir()
            req = StageRequest(
                run_id=RUN,
                stage_identity=_identity(),
                config_fingerprint=FP,
                compatibility_fingerprint=FP2,
                inputs={"Input": _artifact()},
                working_directory=work,
                output_directory=out,
                requested_at_utc="2026-07-22T12:00:00.000000Z",
            )
            blob = str(req.to_dict()).lower()
            self.assertNotIn("password=", blob)

    def test_16_result_status_invariants_cache_hit(self) -> None:
        with self.assertRaises(StageError):
            _result(status="cache_hit", cache_hit=False)

    def test_17_result_failed_requires_error(self) -> None:
        with self.assertRaises(StageError):
            _result(status="failed", error=None)

    def test_18_result_success_forbids_error(self) -> None:
        with self.assertRaises(StageError):
            _result(status="succeeded", error={"class": "X", "message": "y"})

    def test_19_result_rejects_nan_metric(self) -> None:
        with self.assertRaises(StageError):
            _result(metrics={"score": float("nan")})

    def test_20_result_rejects_inf_metric(self) -> None:
        with self.assertRaises(StageError):
            _result(metrics={"score": math.inf})

    def test_21_result_rejects_secret_metric_key(self) -> None:
        with self.assertRaises(StageError):
            _result(metrics={"api_password": "x"})

    def test_22_result_error_rejects_traceback(self) -> None:
        with self.assertRaises(StageError):
            _result(
                status="failed",
                error={"class": "E", "message": "Traceback (most recent)"},
            )

    def test_23_result_error_requires_exact_keys(self) -> None:
        with self.assertRaises(StageError):
            _result(status="failed", error={"class": "E"})

    def test_24_result_finished_before_started(self) -> None:
        with self.assertRaises(StageError):
            _result(
                started_at_utc="2026-07-22T12:00:02.000000Z",
                finished_at_utc="2026-07-22T12:00:01.000000Z",
            )

    def test_25_result_immutable(self) -> None:
        r = _result()
        with self.assertRaises((TypeError, AttributeError)):
            r.status = "failed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            r.metrics["x"] = 1  # type: ignore[index]

    def test_26_result_to_dict(self) -> None:
        d = _result(metrics={"n": 1}).to_dict()
        self.assertEqual(d["status"], "succeeded")
        self.assertEqual(d["metrics"]["n"], 1)

    def test_27_execution_output_freezes(self) -> None:
        out = StageExecutionOutput(
            outputs={"Input": _artifact()},
            metrics={"n": 2},
            warnings=("w",),
        )
        self.assertIsInstance(out.outputs, MappingProxyType)
        with self.assertRaises(TypeError):
            out.metrics["x"] = 1  # type: ignore[index]

    def test_28_cache_manifest_valid(self) -> None:
        m = CacheManifest(
            cache_key=FP,
            layout_version=1,
            stage_name="echo_stage",
            stage_version=1,
            config_fingerprint=FP2,
            artifacts=(_artifact().to_dict(),),
            created_at_utc="2026-07-22T12:00:00.000000Z",
            source_run_id=RUN,
        )
        self.assertEqual(m.schema_version, 1)
        self.assertIn("cache_key", m.to_dict())

    def test_29_cache_manifest_bad_schema(self) -> None:
        with self.assertRaises(PipelineError):
            CacheManifest(
                cache_key=FP,
                layout_version=1,
                stage_name="echo_stage",
                stage_version=1,
                config_fingerprint=FP2,
                artifacts=(),
                created_at_utc="2026-07-22T12:00:00.000000Z",
                source_run_id=RUN,
                schema_version=99,
            )

    def test_30_artifact_secret_metadata_rejected(self) -> None:
        with self.assertRaises(PipelineError):
            _artifact(metadata={"password": "secret"})

    def test_31_artifact_path_traversal_rejected(self) -> None:
        with self.assertRaises(ArtifactError):
            _artifact(relative_path="../escape.bin")

    def test_32_valid_statuses(self) -> None:
        for status in ("succeeded", "skipped", "cancelled"):
            r = _result(status=status)
            self.assertEqual(r.status, status)
        r = _result(
            status="failed",
            error={"class": "E", "message": "boom"},
        )
        self.assertEqual(r.status, "failed")
        r = _result(status="cache_hit", cache_hit=True)
        self.assertTrue(r.cache_hit)


if __name__ == "__main__":
    unittest.main()
