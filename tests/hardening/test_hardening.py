"""Unit tests for Stage 15 hardening package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.hardening.artifacts import (
    ArtifactPolicyError,
    assert_redacted,
    assert_safe_evidence_candidate,
)
from football_analytics.hardening.cache_gc import execute_cache_gc, plan_cache_gc
from football_analytics.hardening.ci_parity import remote_ci_status
from football_analytics.hardening.disk_gate import free_bytes, gate_pipeline_disk
from football_analytics.hardening.fingerprints import (
    assert_deterministic_cache_key,
    assert_deterministic_fingerprint,
)
from football_analytics.hardening.gpu_profile import (
    resolve_device_request,
    rtx3050_bounded_batch_profile,
)
from football_analytics.hardening.licensing import (
    fallback_no_model_behavior,
    scan_model_registry_approvals,
)
from football_analytics.hardening.materialize import MaterializeBoundError, assert_pylist_bounds
from football_analytics.hardening.network import (
    NetworkPolicyError,
    assert_download_allowed,
    assert_no_network_default,
)
from football_analytics.hardening.performance import deterministic_repeat, streaming_parquet_notes
from football_analytics.hardening.policy import (
    hardening_policy_fingerprint,
    load_hardening_policy,
)
from football_analytics.hardening.recovery import (
    clear_interrupted,
    is_interrupted,
    mark_interrupted,
    write_failure_receipt,
)
from football_analytics.hardening.storage_readiness import validate_storage_readiness

REPO = Path(__file__).resolve().parents[2]


class TestHardeningPolicy(unittest.TestCase):
    def test_load_and_fingerprint(self) -> None:
        pol = load_hardening_policy()
        fp1 = hardening_policy_fingerprint(pol)
        fp2 = hardening_policy_fingerprint(pol)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)
        self.assertFalse(pol.automatic_purge)
        self.assertFalse(pol.permanent_delete_by_default)


class TestMaterialize(unittest.TestCase):
    def test_bounds(self) -> None:
        pol = load_hardening_policy()
        assert_pylist_bounds(10, policy=pol)
        with self.assertRaises(MaterializeBoundError):
            assert_pylist_bounds(pol.max_pylist_rows + 1, policy=pol)


class TestCacheGc(unittest.TestCase):
    def test_dry_run_no_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            (cache / "v1" / "sha256" / "ab").mkdir(parents=True)
            plan = plan_cache_gc(cache, max_age_days=0)
            self.assertIn("plan_fingerprint", plan)
            self.assertFalse(plan["permanent_delete_performed"])
            dry = execute_cache_gc(cache, quarantine_root=root / "q", mode="dry_run")
            self.assertFalse(dry["permanent_delete_performed"])
            self.assertEqual(dry["mode"], "dry_run")


class TestDiskAndGpu(unittest.TestCase):
    def test_disk_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.assertGreater(free_bytes(p), 0)
            gate_pipeline_disk(p)
            profile = rtx3050_bounded_batch_profile()
            self.assertEqual(profile["max_batch_size"], 1)
            device = resolve_device_request(cuda_available=False)
            self.assertEqual(device["device"], "cpu")
            self.assertTrue(device["agent_gpu_unverifiable"])


class TestLicensingAndNetwork(unittest.TestCase):
    def test_registry_and_network(self) -> None:
        scan = scan_model_registry_approvals(REPO / "model_registry.yaml")
        self.assertEqual(scan["status"], "PASS")
        self.assertFalse(scan["production_approved_any"])
        self.assertEqual(fallback_no_model_behavior()["production_approved"], False)
        assert_no_network_default()
        with self.assertRaises(NetworkPolicyError):
            assert_download_allowed("video")


class TestRecoveryArtifacts(unittest.TestCase):
    def test_interrupt_redact_safe_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mark_interrupted(root / "run")
            self.assertTrue(is_interrupted(root / "run"))
            clear_interrupted(root / "run")
            self.assertFalse(is_interrupted(root / "run"))
            receipt = write_failure_receipt(
                root / "fail",
                run_id="run_x",
                stage_name="s",
                error="synthetic failure",
                extra={"password": "x"},
            )
            self.assertIn(
                "password", (receipt.get("extra") or {}).get("sensitive_keys_stripped", [])
            )
            self.assertNotIn("password", receipt.get("extra") or {})
            red = assert_redacted({"password": "x", "n": 1})
            self.assertEqual(red["password"], "[REDACTED]")
            good = root / "ok.json"
            good.write_text("{}\n", encoding="utf-8")
            assert_safe_evidence_candidate(good)
            bad = root / "x.mp4"
            bad.write_bytes(b"abc")
            with self.assertRaises(ArtifactPolicyError):
                assert_safe_evidence_candidate(bad)


class TestFingerprintsPerfStorage(unittest.TestCase):
    def test_misc(self) -> None:
        assert_deterministic_fingerprint({"a": 1})
        assert_deterministic_cache_key()
        val, meta = deterministic_repeat(lambda: 42, runs=2)
        self.assertEqual(val, 42)
        self.assertTrue(meta["deterministic"])
        self.assertIn("write_contract_parquet_streaming", streaming_parquet_notes()["api"])
        storage = validate_storage_readiness(REPO)
        self.assertFalse(storage["mnt_d_claimed_ready"])
        self.assertFalse(storage["independent_backup"])
        remote = remote_ci_status()
        self.assertEqual(remote["remote_ci_status"], "UNVERIFIABLE_AGENT_API_CONTEXT")
        self.assertFalse(remote["invented_green_remote_ci"])


class TestGateConstant(unittest.TestCase):
    def test_gate_hint(self) -> None:
        from football_analytics.hardening import GATE_HINT

        self.assertIn("STAGE 15 PRE-RELEASE COMPLETE", GATE_HINT)
        self.assertIn("STAGE 16", GATE_HINT)


if __name__ == "__main__":
    unittest.main()
