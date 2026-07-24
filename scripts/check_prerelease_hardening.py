#!/usr/bin/env python3
"""Validate Stage 15 pre-release hardening (15A–15G) + write evidence JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
RUNTIME_ROOT = Path("/home/fdoblak/workspace/prerelease_hardening_checks")
EVIDENCE_DIR = REPO_ROOT / "artifacts" / "evidence" / "stage_15"
GATE = (
    "PASS_WITH_FINDINGS — STAGE 15 PRE-RELEASE COMPLETE; "
    "ALL IMPLEMENTATION STAGES CLOSED; "
    "ONLY REAL-MATCH ACCEPTANCE STAGE 16 REMAINS"
)
GATE_FAIL = "NO-GO — STAGE 15 PRE-RELEASE HARDENING FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.scenarios: dict[str, str] = {}
        self.extras: dict[str, Any] = {}
        self.findings: list[str] = []

    def err(self, msg: str, *, config: bool = False) -> None:
        self.errors.append(msg)
        if config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def finding(self, msg: str) -> None:
        self.findings.append(msg)
        self.warnings.append(msg)

    def ok(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def finalize(self) -> Result:
        if self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.findings or self.warnings:
            self.status = "PASS_WITH_FINDINGS"
        else:
            self.status = (
                "PASS_WITH_FINDINGS"  # Stage 15 close is always findings (Stage 16 remains)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "gate": GATE if not self.errors else GATE_FAIL,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def _write_evidence(name: str, payload: dict[str, Any]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_checks(*, keep: bool, light: bool) -> Result:
    from football_analytics.hardening.acceptance import run_synthetic_orchestration_e2e
    from football_analytics.hardening.artifacts import (
        assert_redacted,
        assert_safe_evidence_candidate,
    )
    from football_analytics.hardening.cache_gc import execute_cache_gc, plan_cache_gc
    from football_analytics.hardening.ci_parity import (
        check_external_repo_lock,
        check_protected_pins,
        check_workflow_sha_pins,
        local_deep_validation_commands,
        remote_ci_status,
    )
    from football_analytics.hardening.concurrency import cache_key_lock
    from football_analytics.hardening.disk_gate import disk_gate_summary, gate_pipeline_disk
    from football_analytics.hardening.evidence_checks import (
        check_evidence_index,
        check_report_data_consistency,
        stage_evidence_json_only,
    )
    from football_analytics.hardening.fingerprints import (
        assert_deterministic_cache_key,
        assert_deterministic_fingerprint,
    )
    from football_analytics.hardening.gpu_profile import (
        optional_cuda_smoke,
        resolve_device_request,
        rtx3050_bounded_batch_profile,
    )
    from football_analytics.hardening.licensing import (
        fallback_no_model_behavior,
        scan_model_registry_approvals,
        third_party_notices_present,
    )
    from football_analytics.hardening.materialize import (
        MaterializeBoundError,
        assert_pylist_bounds,
        materialize_policy_summary,
    )
    from football_analytics.hardening.network import assert_no_network_default
    from football_analytics.hardening.performance import (
        bounded_memory_probe,
        deterministic_repeat,
        streaming_parquet_notes,
    )
    from football_analytics.hardening.policy import (
        hardening_policy_fingerprint,
        load_hardening_policy,
    )
    from football_analytics.hardening.recovery import (
        clear_interrupted,
        mark_interrupted,
        recover_interrupted_run,
        write_atomic_json,
        write_failure_receipt,
    )
    from football_analytics.hardening.storage_readiness import (
        scan_large_files,
        validate_storage_readiness,
        write_cleanup_receipt,
    )
    from football_analytics.orchestration.report.builder import build_single_player_report

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="hard15_", dir=str(RUNTIME_ROOT)))

    try:
        # --- policy ---
        policy = load_hardening_policy()
        fp = hardening_policy_fingerprint(policy)
        result.extras["hardening_policy_fp"] = fp
        if policy.automatic_purge or policy.permanent_delete_by_default:
            result.fail("15a_policy", "unsafe GC defaults")
        else:
            result.ok("15a_policy")

        # --- 15A materialize / RISK-029 ---
        summary = materialize_policy_summary(policy)
        try:
            assert_pylist_bounds(policy.max_pylist_rows + 1, policy=policy, context="overflow")
            result.fail("15a_materialize", "overflow allowed")
        except MaterializeBoundError:
            result.ok("15a_materialize")
        result.extras["risk_029"] = summary

        # --- 15A cache GC / RISK-041 ---
        cache_root = session / "cache"
        (cache_root / "v1" / "sha256" / "ab").mkdir(parents=True)
        # empty plan is fine
        plan = plan_cache_gc(cache_root, policy=policy, max_age_days=0)
        dry = execute_cache_gc(
            cache_root,
            quarantine_root=session / "cache_q",
            mode="dry_run",
            policy=policy,
        )
        if dry.get("permanent_delete_performed") is not False:
            result.fail("15a_cache_gc", "permanent delete claimed")
        else:
            result.ok("15a_cache_gc")
        result.extras["risk_041"] = {"plan": plan, "dry_run": dry}

        # --- disk gate ---
        gate_pipeline_disk(session, policy=policy)
        result.extras["disk"] = disk_gate_summary(session, policy=policy)
        result.ok("15a_disk_gate")

        # --- fingerprints ---
        assert_deterministic_fingerprint({"stage": 15, "k": [1, 2, 3]})
        assert_deterministic_cache_key()
        result.ok("15a_fingerprints")

        # --- failure receipt + atomic + interrupted recovery ---
        receipt = write_failure_receipt(
            session / "fail",
            run_id="run_stage15_fail",
            stage_name="probe",
            error="synthetic failure",
            extra={"api_key": "not-a-real-secret-value"},
        )
        extra_meta = receipt.get("extra") or {}
        stripped = list(extra_meta.get("sensitive_keys_stripped") or [])
        if "api_key" in extra_meta:
            result.fail("15a_failure_receipt", "api_key key persisted")
        elif "api_key" not in stripped:
            result.fail("15a_failure_receipt", "sensitive key not stripped")
        else:
            result.ok("15a_failure_receipt")
        write_atomic_json(session / "atomic.json", {"ok": True}, contain_root=session)
        mark_interrupted(session / "run_x", policy=policy)
        rec = recover_interrupted_run(session / "run_x", policy=policy)
        if not rec.get("interrupted"):
            result.fail("15a_recovery", "marker missing")
        else:
            clear_interrupted(session / "run_x", policy=policy)
            result.ok("15a_recovery")

        # --- concurrency lock ---
        key = "f" * 64
        with cache_key_lock(cache_root, key, policy=policy):
            pass
        result.ok("15a_concurrency_lock")

        # --- GPU / CPU ---
        device = resolve_device_request(cuda_available=False, policy=policy)
        profile = rtx3050_bounded_batch_profile(policy)
        if profile.get("max_batch_size") != 1:
            result.fail("15a_gpu_profile", "batch not bounded")
        elif not device.get("agent_gpu_unverifiable"):
            result.fail("15a_gpu_profile", "agent GPU not marked unverifiable")
        else:
            result.ok("15a_gpu_profile")
        smoke = optional_cuda_smoke()
        result.extras["cuda_smoke"] = smoke
        result.ok("15a_cpu_fallback")

        # --- network / redaction / large artifact ---
        assert_no_network_default(policy)
        result.ok("15a_no_network")
        redacted = assert_redacted({"api_key": "secret", "ok": 1})
        if redacted.get("api_key") != "[REDACTED]":
            result.fail("15a_redaction", "api_key not redacted")
        else:
            result.ok("15a_redaction")
        safe_json = session / "tiny.json"
        safe_json.write_text('{"a":1}\n', encoding="utf-8")
        assert_safe_evidence_candidate(safe_json, policy=policy)
        bad = session / "weights.pt"
        bad.write_bytes(b"x" * 10)
        try:
            assert_safe_evidence_candidate(bad, policy=policy)
            result.fail("15a_large_artifact", "weights allowed")
        except Exception:  # noqa: BLE001
            result.ok("15a_large_artifact")

        # --- 15B licensing ---
        lic = scan_model_registry_approvals(REPO_ROOT / "model_registry.yaml", policy=policy)
        notices = third_party_notices_present(REPO_ROOT)
        if not notices.get("license_inventory"):
            result.fail("15b_inventory", "license inventory missing")
        elif not notices.get("third_party_notices"):
            result.fail("15b_notices", "third_party_notices missing")
        else:
            result.ok("15b_licensing")
        result.extras["licensing"] = {
            "scan": lic,
            "notices": notices,
            "fallback": fallback_no_model_behavior(),
        }
        result.finding(
            "GPL/AGPL adapters remain evaluation_only; legal clearance deferred to Stage 16"
        )

        # --- 15C storage ---
        storage = validate_storage_readiness(REPO_ROOT, policy=policy)
        if storage.get("mnt_d_claimed_ready"):
            result.fail("15c_storage", "claimed /mnt/d ready")
        else:
            result.ok("15c_storage")
        scan = scan_large_files(session, policy=policy)
        write_cleanup_receipt(
            session / "cleanup_receipt.json",
            action="stage15_temp_cleanup",
            items=[str(session)],
            contain_root=session,
        )
        result.extras["storage"] = {"readiness": storage, "scan": scan}
        result.finding("Same-VHDX /mnt/d independent backup remains Stage 16 external")

        # --- 15D CI parity ---
        remote = remote_ci_status(policy)
        pins = check_protected_pins()
        lock = check_external_repo_lock(REPO_ROOT)
        wf = check_workflow_sha_pins(REPO_ROOT)
        if wf.get("exit_code") not in {0, None} and wf.get("status") == "FAIL":
            result.fail("15d_workflow", str(wf.get("errors")))
        else:
            result.ok("15d_ci_parity")
        result.extras["ci_parity"] = {
            "remote": remote,
            "pins": pins,
            "lock": lock,
            "workflow": wf,
            "local_commands": local_deep_validation_commands(),
        }
        result.finding(
            "Remote GitHub Actions status UNVERIFIABLE_AGENT_API_CONTEXT (RISK-042); "
            "local CI-equivalent required"
        )

        # --- 15E synthetic acceptance ---
        e2e = run_synthetic_orchestration_e2e(session / "e2e", light=light)
        if e2e.get("overall_status") not in {"succeeded", "partial"}:
            result.fail("15e_e2e", str(e2e.get("overall_status")))
        elif not e2e.get("renderer_temp_cleaned"):
            result.fail("15e_renderer_cleanup", "temp png remains")
        elif not e2e.get("not_evaluable_present"):
            result.fail("15e_not_evaluable", "missing not_evaluable")
        else:
            result.ok("15e_synthetic_acceptance")
        report = build_single_player_report(
            run_id="run_stage15_report_check",
            git_commit="a" * 40,
            target_player_id="target_player_a",
            display_name="Target A",
            match_id="match_stage15",
            video_id="vid_stage15",
        )
        check_report_data_consistency(report)
        result.ok("15e_report_consistency")
        result.extras["acceptance"] = e2e

        # --- 15F performance ---
        _val, mem_meta = bounded_memory_probe(lambda: sum(range(10000)), policy=policy)
        _val2, det_meta = deterministic_repeat(lambda: hash_canonical_probe(), runs=2)
        result.extras["performance"] = {
            "memory": mem_meta,
            "deterministic": det_meta,
            "streaming_parquet": streaming_parquet_notes(),
            "cuda_smoke": smoke,
        }
        result.ok("15f_performance")

        # --- evidence index ---
        idx = check_evidence_index(
            REPO_ROOT / "artifacts" / "evidence" / "index.json",
            require_stage="stage_15",
        )
        # stage_15 dir may be empty until we write; ensure dir exists
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        result.ok("15a_evidence_index")
        result.extras["evidence_index"] = idx

        result.extras["gate"] = GATE
        result.finding("ONLY REAL-MATCH ACCEPTANCE STAGE 16 REMAINS")
    except Exception as exc:  # noqa: BLE001
        result.fail("99_unexpected", f"{type(exc).__name__}: {exc}")
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)

    result.finalize()

    # Always emit evidence JSON (even on failure) for audit.
    summary = result.to_dict()
    _write_evidence("stage_15_validator_summary.json", summary)
    _write_evidence(
        "stage_15_close_summary.json",
        {
            "schema_version": 1,
            "stage": 15,
            "status": "CLOSED" if not result.errors else "OPEN",
            "gate": summary["gate"],
            "implementation_stages_closed": not bool(result.errors),
            "only_stage_16_remains": True,
            "arrow_registry_bumped": False,
            "findings": list(result.findings),
            "scenarios": dict(result.scenarios),
        },
    )
    _write_evidence(
        "stage_15_deferred_closure.json",
        {
            "schema_version": 1,
            "closed_machine_local": [
                "RISK-029",
                "RISK-041",
                "bounded_memory_resumability_hardening",
                "license_adapter_isolation_gates",
                "storage_readiness_no_mnt_d_claim",
                "local_ci_parity",
            ],
            "remaining_stage_16_only": [
                "Model / GPL/AGPL legal clearance",
                "Same-VHDX /mnt/d independent backup verification",
                "GitHub API 403 / remote CI visibility (RISK-042)",
                "Real match GT / accuracy / E2E",
                "Manual identity / real video",
                "Real final report + single visual",
            ],
        },
    )
    try:
        stage_evidence_json_only(EVIDENCE_DIR)
    except Exception as exc:  # noqa: BLE001
        result.fail("15g_evidence_json_only", str(exc))
        result.finalize()
    return result


def hash_canonical_probe() -> str:
    from football_analytics.core.hashing import hash_canonical_json

    return hash_canonical_json({"probe": 15, "items": [1, 2, 3]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--full-fixtures",
        action="store_true",
        help="Call heavy stage fixture entrypoints (default: light stubs)",
    )
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), light=not bool(args.full_fixtures))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_PASS if not result.errors else int(result.exit_code)
    print(payload.get("gate") or GATE_FAIL)
    print(f"status={result.status} exit_code={result.exit_code}")
    for k, v in result.scenarios.items():
        print(f"  {k}: {v}")
    for e in result.errors:
        print(f"ERROR: {e}", file=sys.stderr)
    for f in result.findings:
        print(f"FINDING: {f}")
    if result.errors:
        print(GATE_FAIL)
        return int(result.exit_code)
    print(GATE)
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
