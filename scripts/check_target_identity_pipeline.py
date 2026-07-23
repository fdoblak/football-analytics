#!/usr/bin/env python3
"""Validate Stage 7E target identity fusion + manual approval pipeline.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/target_identity_checks")
GATE_PASS = "PASS — TARGET PLAYER IDENTITY WORKFLOW ACTIVE; STAGE 7 CLOSED"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — TARGET PLAYER IDENTITY WORKFLOW ACTIVE; "
    "STAGE 7 CLOSED; REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — TARGET IDENTITY PIPELINE FAILURE"


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
        gate = GATE_FAIL
        if self.status in {"PASS", "PASS_WITH_FINDINGS"}:
            gate = GATE_FINDINGS if self.findings or self.warnings else GATE_PASS
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": gate,
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.identity.contracts import (
        assert_frozen_upstream_fingerprints,
        assert_identity_contracts_registered,
    )
    from football_analytics.identity.target_fusion_config import (
        load_target_fusion_config,
        target_fusion_config_fingerprint,
    )
    from football_analytics.identity.target_fusion_evaluation import NOT_EVALUATED_TARGET_IDENTITY
    from football_analytics.identity.target_fusion_fixtures import assert_runtime_root
    from football_analytics.identity.target_fusion_service import (
        prepare_review,
        resolve_fusion,
        run_fixture_decision,
        validate_fusion_outputs,
    )

    result = Result()
    try:
        assert_runtime_root()
        assert_identity_contracts_registered()
        assert_frozen_upstream_fingerprints()
    except Exception as exc:  # noqa: BLE001
        result.err(f"contract pin failure: {exc}", integrity=True)
        return result.finalize()

    try:
        config = load_target_fusion_config()
        cfg_fp = target_fusion_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config failure: {exc}", config=True)
        return result.finalize()

    if config["safety"]["auto_confirm"] is not False:
        result.err("auto_confirm must be false", integrity=True)
    if config["safety"]["face_recognition"] is not False:
        result.err("face_recognition must be false", integrity=True)
    if config["safety"]["cross_video_auto_link"] is not False:
        result.err("cross_video_auto_link must be false", integrity=True)

    keep = bool(args.keep)
    work = Path(tempfile.mkdtemp(prefix="target_identity_", dir=str(RUNTIME_ROOT)))
    try:
        out = work / "e2e"
        prep = prepare_review(
            output_dir=out,
            config=config,
            contain_root=RUNTIME_ROOT,
            fixture_name="e2e_bundle",
        )
        if not prep.accepted:
            result.err(f"prepare-review failed: {prep.error_code}", integrity=True)
            return result.finalize()
        for decision, track, start, end, did in (
            ("confirm", 4, 0, 40, "dec_val_confirm"),
            ("reject", 5, 50, 80, "dec_val_reject"),
            ("revoke", 4, 0, 40, "dec_val_revoke"),
        ):
            d = run_fixture_decision(
                output_dir=out,
                config=config,
                decision=decision,
                track_id=track,
                start=start,
                end=end,
                decision_id=did,
                contain_root=RUNTIME_ROOT,
            )
            if not d.accepted:
                result.err(f"decide {decision} failed: {d.error_code}", integrity=True)
                return result.finalize()
        res = resolve_fusion(output_dir=out, config=config, contain_root=RUNTIME_ROOT)
        if not res.accepted:
            result.err(f"resolve failed: {res.error_code}", integrity=True)
            return result.finalize()
        val = validate_fusion_outputs(output_dir=out, config=config, contain_root=RUNTIME_ROOT)
        if not val.accepted:
            result.err(f"validate failed: {val.error_code}", integrity=True)
            return result.finalize()

        receipt = json.loads(Path(res.receipt_json).read_text(encoding="utf-8"))
        if receipt["ground_truth_evaluation_status"] != NOT_EVALUATED_TARGET_IDENTITY:
            result.err("expected NOT_EVALUATED target identity status", integrity=True)
        if receipt["provenance"]["auto_confirm_forbidden"] is not True:
            result.err("auto_confirm_forbidden missing", integrity=True)
        result.extras["config_fingerprint"] = cfg_fp
        result.extras["assignment_counts"] = receipt["assignment_counts"]
        result.extras["evaluation_status"] = receipt["ground_truth_evaluation_status"]
        result.finding(
            "No reviewed target-identity ground truth; real football accuracy not validated"
        )
        result.finding("Stage 7 closed as technical identity workflow baseline")
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            result.extras["kept_runtime_dir"] = str(work)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["gate"])
        for e in payload["errors"]:
            print(f"ERROR: {e}", file=sys.stderr)
        for f in payload["findings"]:
            print(f"FINDING: {f}")
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
