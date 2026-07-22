#!/usr/bin/env python3
"""Validate Stage 3A video ingest contracts, policy, and schema/python alignment.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/video_contract_checks")


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "extras": self.extras,
        }


def _sample_sha() -> str:
    return "a" * 64


def _run_id() -> str:
    from football_analytics.core.run_id import generate_run_id

    return generate_run_id()


def _valid_source_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_id": "src_sample_one",
        "source_kind": "synthetic_fixture",
        "original_filename": "sample.mp4",
        "source_path": "/home/fdoblak/workspace/video_contract_checks/sample.mp4",
        "source_size_bytes": 1024,
        "source_sha256": _sample_sha(),
        "media_type": "video/mp4",
        "container_hint": "mp4",
        "created_at_utc": "2026-07-22T21:00:00Z",
        "registered_at_utc": "2026-07-22T21:00:01Z",
        "immutability_policy": "detect_mutation",
        "provenance": {
            "origin": "synthetic_generated",
            "label": "stage3a_sample",
            "notes": None,
        },
    }


def _valid_request_payload(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": "req_sample_one",
        "run_id": run_id,
        "source_id": "src_sample_one",
        "source_path": "/home/fdoblak/workspace/video_contract_checks/sample.mp4",
        "requested_at_utc": "2026-07-22T21:00:02Z",
        "ingest_mode": "validate_only",
        "policy_version": "video_ingest_policy_v1",
        "probe_requested": False,
        "normalization_requested": False,
        "expected_source_sha256": _sample_sha(),
        "expected_source_size_bytes": 1024,
        "output_root": "/home/fdoblak/workspace/video_contract_checks/out",
        "fixture_mode": True,
    }


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.video.contracts import (
        SCHEMA_FILES,
        assert_schema_python_enum_alignment,
        build_normalize_plan,
        load_all_video_schemas,
        load_ingest_policy,
        validate_payload_against_schema,
    )
    from football_analytics.video.fixtures import SCENARIOS, metadata_fixture
    from football_analytics.video.types import (
        ContractFingerprints,
        IngestReceipt,
        IngestRequest,
        Rational,
        ReceiptProvenance,
        ReceiptStatus,
        VideoProbe,
        VideoSource,
    )
    from football_analytics.video.validation import (
        assert_request_source_compatibility,
        reject_unsafe_path_string,
    )

    result = Result()
    policy_path = Path(args.policy)
    schema_root = Path(args.schema_root)
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    if not schema_root.is_absolute():
        schema_root = REPO_ROOT / schema_root

    try:
        policy = load_ingest_policy(policy_path)
        schemas = load_all_video_schemas(schema_root)
        assert_schema_python_enum_alignment(policy)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config/schema failure: {type(exc).__name__}: {exc}", config=True)
        return result.finalize(strict=args.strict)

    result.extras["schemas"] = sorted(schemas)
    result.extras["policy_version"] = policy["policy_version"]
    result.extras["schema_files"] = list(SCHEMA_FILES)

    if set(schemas) != set(SCHEMA_FILES):
        result.err("schema set mismatch", config=True)

    run_id = _run_id()
    source_payload = _valid_source_payload()
    request_payload = _valid_request_payload(run_id)

    # Schema validate core payloads
    try:
        validate_payload_against_schema(source_payload, schemas["video_source.schema.json"])
        validate_payload_against_schema(request_payload, schemas["ingest_request.schema.json"])
        source = VideoSource.from_dict(source_payload)
        request = IngestRequest.from_dict(request_payload)
        assert_request_source_compatibility(request, source, policy)
        if source.fingerprint() != source.fingerprint():
            result.err("source fingerprint nondeterministic", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"source/request validation failed: {exc}")

    # Probe metadata fixtures
    try:
        for name in ("rotation_metadata", "vfr_metadata", "unknown_frame_count"):
            payload = metadata_fixture(name, source_sha256=_sample_sha())
            validate_payload_against_schema(payload, schemas["video_probe.schema.json"])
            VideoProbe.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        result.err(f"probe fixture validation failed: {exc}")

    # Normalize plan
    try:
        plan = build_normalize_plan(
            plan_id="plan_sample_one",
            source_id="src_sample_one",
            source_sha256=_sample_sha(),
            policy_version=policy["policy_version"],
            required=False,
            reasons=("source_already_compliant",),
            target_container="mp4",
            target_video_codec="h264",
            target_audio_policy="copy_if_present_else_drop",
            target_pixel_format="yuv420p",
            target_width=None,
            target_height=None,
            resize_policy="none",
            target_frame_rate=None,
            frame_rate_policy="preserve",
            target_time_base=Rational(1, 90_000),
            rotation_policy="preserve_metadata",
            sar_policy="preserve",
            audio_policy="copy_if_present_else_drop",
            copy_metadata_policy="copy_safe",
            estimated_output_path="/home/fdoblak/workspace/video_contract_checks/out/norm.mp4",
            overwrite_policy=False,
        )
        validate_payload_against_schema(plan.to_dict(), schemas["normalize_plan.schema.json"])
    except Exception as exc:  # noqa: BLE001
        result.err(f"normalize plan validation failed: {exc}")

    # Receipt (validated — no false success)
    try:
        receipt = IngestReceipt(
            receipt_id="rcpt_sample_one",
            request_id="req_sample_one",
            run_id=run_id,
            source_id="src_sample_one",
            source_sha256=_sample_sha(),
            source_size_bytes=1024,
            status=ReceiptStatus.VALIDATED,
            started_at_utc="2026-07-22T21:00:03Z",
            completed_at_utc="2026-07-22T21:00:04Z",
            probe_record_ref=None,
            normalize_plan_ref=None,
            artifact_refs=(),
            policy_version=policy["policy_version"],
            contract_fingerprints=ContractFingerprints(
                source=source.fingerprint(),
                request=request.fingerprint(),
            ),
            warnings=(),
            errors=(),
            provenance=ReceiptProvenance(stage="3A", label="contract_validation", notes=None),
        )
        validate_payload_against_schema(receipt.to_dict(), schemas["ingest_receipt.schema.json"])
    except Exception as exc:  # noqa: BLE001
        result.err(f"receipt validation failed: {exc}")

    # Negative path checks
    for bad in ("../escape.mp4", "~/video.mp4", "/tmp/${HOME}/x.mp4", "https://example/x.mp4"):
        try:
            reject_unsafe_path_string(bad, label="path")
            result.err(f"unsafe path accepted: {bad}", integrity=True)
        except Exception:
            pass

    # Invalid payload rejection
    bad_source = dict(source_payload)
    bad_source["source_sha256"] = "abc"
    try:
        validate_payload_against_schema(bad_source, schemas["video_source.schema.json"])
        result.err("invalid sha accepted by schema")
    except Exception:
        pass

    result.extras["fixture_scenarios"] = [s.name for s in SCENARIOS]
    result.extras["runtime_root"] = str(RUNTIME_ROOT)
    return result.finalize(strict=args.strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="configs/video/ingest_policy.yaml",
        help="Path to ingest policy YAML",
    )
    parser.add_argument(
        "--schema-root",
        default="schemas/video",
        help="Directory containing video *.schema.json files",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON report output path")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_checks(args)
    payload = result.to_dict()

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            out = RUNTIME_ROOT / out.name
        if out.exists():
            print(f"json-out already exists: {out}", file=sys.stderr)
            return EXIT_CONFIG
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)
        result.extras["json_out"] = str(out)

    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code}")
        for err in result.errors:
            print(f"ERROR: {err}")
        for warn in result.warnings:
            print(f"WARNING: {warn}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
