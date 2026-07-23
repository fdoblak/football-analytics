#!/usr/bin/env python3
"""Validate Stage 5E detection fusion + quality pipeline (synthetic E2E).

Exit codes:
  0 PASS / PASS_WITH_FINDINGS
  1 validation finding / NO-GO content
  2 configuration failure
  3 integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/detection_pipeline_checks")
EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
GATE = (
    "PASS_WITH_FINDINGS — DETECTION PIPELINE ACTIVE; STAGE 5 CLOSED; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
NOT_EVAL = "NOT_EVALUATED_NO_REVIEWED_DETECTION_GROUND_TRUTH"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.findings: list[str] = []
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

    def finding(self, msg: str) -> None:
        self.findings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "NO-GO" if self.exit_code == EXIT_INTEGRITY else "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.findings or self.warnings:
            self.status = "PASS_WITH_FINDINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": GATE if self.status in {"PASS", "PASS_WITH_FINDINGS"} else self.status,
        }
        body.update(self.extras)
        return body


def _tbl(contract_name: str, rows: list[dict[str, Any]]) -> Any:
    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.core.run_id import generate_run_id
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import write_contract_parquet
    from football_analytics.perception.contracts import (
        detection_schema_fingerprints,
        load_perception_json_schema,
        validate_against_json_schema,
    )
    from football_analytics.perception.detection_pipeline import run_detection_integrate
    from football_analytics.perception.detection_pipeline_config import (
        detection_pipeline_config_fingerprint,
        load_detection_pipeline_config,
    )
    from football_analytics.perception.detection_pipeline_fixtures import (
        SOURCE_SHA_A,
        TIMELINE_FP_A,
        assert_runtime_root,
        build_minimal_fusion_inputs,
        write_json,
    )
    from football_analytics.perception.detection_quality import NOT_EVALUATED_DETECTION

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_detection_pipeline_config(cfg_path)
        result.extras["config_fingerprint"] = detection_pipeline_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    fps = detection_schema_fingerprints()
    result.extras["detections_fingerprint"] = fps["detections"]
    if fps["detections"] != EXPECTED_DETECTIONS_FP:
        result.err("detections v1 fingerprint changed", integrity=True)

    # Lazy import: perception must not pull ultralytics
    before = {k for k in sys.modules if "ultralytics" in k.lower()}
    import football_analytics.perception as perc  # noqa: F401

    after = {k for k in sys.modules if "ultralytics" in k.lower()}
    if after - before:
        result.err("import football_analytics.perception loaded ultralytics", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    session = Path(tempfile.mkdtemp(prefix="pipe_val_", dir=str(RUNTIME_ROOT)))
    try:
        rid = generate_run_id()
        inputs = build_minimal_fusion_inputs(rid, n_frames=min(int(args.frames), 20))
        # Write artifacts
        for key, contract in (
            ("human_detections", "detections"),
            ("human_frame_status", "detection_frame_status"),
            ("human_attributes", "detection_attributes"),
            ("ball_detections", "detections"),
            ("ball_frame_status", "detection_frame_status"),
            ("ball_attributes", "detection_attributes"),
            ("role_attributes", "detection_attributes"),
            ("frames", "frames"),
            ("analysis_windows", "analysis_windows"),
        ):
            rows = list(inputs[key])
            write_contract_parquet(
                _tbl(contract, rows),
                session / f"{key}.parquet",
                get_contract(contract, 1),
                contain_root=RUNTIME_ROOT,
            )
        write_json(session / "human_receipt.json", inputs["human_receipt"])
        write_json(session / "ball_receipt.json", inputs["ball_receipt"])
        write_json(session / "role_receipt.json", inputs["role_receipt"])

        out1 = session / "out1"
        out1.mkdir()
        res1 = run_detection_integrate(
            human_detections=str(session / "human_detections.parquet"),
            human_frame_status=str(session / "human_frame_status.parquet"),
            human_attributes=str(session / "human_attributes.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_detections=str(session / "ball_detections.parquet"),
            ball_frame_status=str(session / "ball_frame_status.parquet"),
            ball_attributes=str(session / "ball_attributes.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            role_attributes=str(session / "role_attributes.parquet"),
            role_receipt=str(session / "role_receipt.json"),
            output_dir=str(out1),
            config=config,
            contain_root=RUNTIME_ROOT,
            analysis_windows=str(session / "analysis_windows.parquet"),
            frames=str(session / "frames.parquet"),
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=TIMELINE_FP_A,
        )
        if not res1.accepted:
            result.err(f"integrate failed: {res1.error_code}")
            return result.finalize()

        receipt = json.loads(Path(str(res1.pipeline_receipt_json)).read_text(encoding="utf-8"))
        quality = json.loads(Path(str(res1.quality_report_json)).read_text(encoding="utf-8"))
        try:
            validate_against_json_schema(
                receipt, load_perception_json_schema("detection_pipeline_receipt")
            )
            validate_against_json_schema(
                quality, load_perception_json_schema("detection_quality_report")
            )
            result.extras["schema_validation"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result.err(f"schema validation failed: {exc}", integrity=True)

        if receipt.get("ground_truth_evaluation_status") != NOT_EVALUATED_DETECTION:
            result.err("expected NOT_EVALUATED detection GT status")
        if quality.get("status") not in {"pass", "pass_with_findings"}:
            result.err(f"unexpected quality status: {quality.get('status')}")

        # Deterministic repeat
        out2 = session / "out2"
        out2.mkdir()
        res2 = run_detection_integrate(
            human_detections=str(session / "human_detections.parquet"),
            human_frame_status=str(session / "human_frame_status.parquet"),
            human_attributes=str(session / "human_attributes.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_detections=str(session / "ball_detections.parquet"),
            ball_frame_status=str(session / "ball_frame_status.parquet"),
            ball_attributes=str(session / "ball_attributes.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            role_attributes=str(session / "role_attributes.parquet"),
            role_receipt=str(session / "role_receipt.json"),
            output_dir=str(out2),
            config=config,
            contain_root=RUNTIME_ROOT,
            analysis_windows=str(session / "analysis_windows.parquet"),
            frames=str(session / "frames.parquet"),
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=TIMELINE_FP_A,
        )
        if not res2.accepted:
            result.err(f"repeat integrate failed: {res2.error_code}")
        elif res1.total_detection_count != res2.total_detection_count:
            result.err("non-deterministic detection counts")
        else:
            result.extras["deterministic"] = True

        # Negative: overwrite forbidden
        res_ow = run_detection_integrate(
            human_detections=str(session / "human_detections.parquet"),
            human_frame_status=str(session / "human_frame_status.parquet"),
            human_attributes=str(session / "human_attributes.parquet"),
            human_receipt=str(session / "human_receipt.json"),
            ball_detections=str(session / "ball_detections.parquet"),
            ball_frame_status=str(session / "ball_frame_status.parquet"),
            ball_attributes=str(session / "ball_attributes.parquet"),
            ball_receipt=str(session / "ball_receipt.json"),
            role_attributes=str(session / "role_attributes.parquet"),
            role_receipt=str(session / "role_receipt.json"),
            output_dir=str(out1),
            config=config,
            contain_root=RUNTIME_ROOT,
            expected_source_sha=SOURCE_SHA_A,
            expected_timeline_fp=TIMELINE_FP_A,
        )
        if res_ow.accepted or res_ow.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite control failed")

        result.finding(NOT_EVAL)
        result.finding("real football detection accuracy not validated")
        result.extras["total_detection_count"] = res1.total_detection_count
        result.extras["quality_status"] = res1.quality_status
        result.extras["review_count"] = res1.review_count
        result.extras["cuda_required"] = False
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleanup"] = "removed_session"
        else:
            result.extras["session"] = str(session)
            result.extras["cleanup"] = "kept"

    return result.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/perception/detection_pipeline.yaml",
        help="Pipeline config path",
    )
    parser.add_argument("--frames", type=int, default=8, help="Synthetic frame count (≤20)")
    parser.add_argument("--keep", action="store_true", help="Keep session directory")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON path")
    args = parser.parse_args()
    if args.frames > 20:
        print("frames must be <= 20", file=sys.stderr)
        return EXIT_CONFIG

    result = run_checks(args)
    payload = result.to_dict()
    out_path = args.json_out
    if out_path is None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = str(RUNTIME_ROOT / f"detection_pipeline_validation_{stamp}.json")
    Path(out_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status: {payload['status']}")
    print(f"gate: {payload.get('gate')}")
    print(f"report: {out_path}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
