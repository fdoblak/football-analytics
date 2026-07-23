#!/usr/bin/env python3
"""Validate Stage 7D jersey region + OpenCV template OCR baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/jersey_ocr_checks")
GATE_PASS = "PASS — JERSEY NUMBER OCR BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — JERSEY NUMBER OCR BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — JERSEY OCR BASELINE FAILURE"


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
    from football_analytics.data.compiler import get_contract, list_contracts
    from football_analytics.data.fingerprint import contract_fingerprint
    from football_analytics.identity.contracts import (
        EXPECTED_JERSEY_OBSERVATIONS_FP,
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_frozen_upstream_fingerprints,
        assert_identity_contracts_registered,
    )
    from football_analytics.identity.jersey_ocr import clear_digit_template_cache
    from football_analytics.identity.jersey_ocr_config import (
        jersey_ocr_config_fingerprint,
        load_jersey_ocr_config,
    )
    from football_analytics.identity.jersey_ocr_evaluation import NOT_EVALUATED_JERSEY_OCR
    from football_analytics.identity.jersey_ocr_fixtures import (
        assert_runtime_root,
        fixture_no_number_front,
        fixture_predicted_rejected,
        fixture_sponsor_logo_negative,
        fixture_two_digit,
    )
    from football_analytics.identity.jersey_ocr_service import run_jersey_observe
    from football_analytics.identity.policy import decide_assignment_status, load_identity_policy

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()
    clear_digit_template_cache()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_jersey_ocr_config(cfg_path)
        result.extras["config_fingerprint"] = jersey_ocr_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    try:
        assert_identity_contracts_registered()
        assert_frozen_upstream_fingerprints()
        names = list_contracts()
        if len(names) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err(
                f"expected {EXPECTED_REGISTRY_CONTRACT_COUNT} contracts, got {len(names)}",
                config=True,
            )
        j_fp = contract_fingerprint(get_contract("jersey_observations", 1))
        result.extras["jersey_observations_fingerprint"] = j_fp
        if j_fp != EXPECTED_JERSEY_OBSERVATIONS_FP:
            result.err("jersey_observations fingerprint drift", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"contract check failed: {exc}", integrity=True)
        return result.finalize()

    selected = [m for m in config["selection_matrix"] if str(m.get("status")).lower() == "selected"]
    if len(selected) != 1:
        result.err("exactly one method must be selected", config=True)
    result.extras["selected_method"] = selected[0] if selected else None
    result.finding(
        "OpenCV template/shape digit matcher SELECTED; sn-jersey future adapter only; "
        "tesseract/easyocr/mmocr not installed"
    )
    result.finding(NOT_EVALUATED_JERSEY_OCR)
    result.finding("real football jersey OCR accuracy not yet validated")
    result.finding("false number / blur / similar-digit / no-region risks remain open")

    if config["assignment"]["auto_confirm_identity"] is not False:
        result.err("auto_confirm_identity must be false", integrity=True)
    if config["assignment"]["auto_target_confirmation"] is not False:
        result.err("auto_target_confirmation must be false", integrity=True)
    if config["region"]["persist_crops"] is not False:
        result.err("persist_crops must be false", integrity=True)

    policy = load_identity_policy()
    session = Path(tempfile.mkdtemp(prefix="jersey_", dir=str(RUNTIME_ROOT)))
    try:
        two = fixture_two_digit()
        r1 = run_jersey_observe(
            output_dir=session / "two",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if not r1.accepted:
            result.err(f"two-digit observe failed: {r1.error_code}")
        else:
            observed = [x for x in r1.summary["observation_rows"] if x.get("raw_text") == "10"]
            if not observed:
                result.err("expected raw_text=10 on two-digit fixture")
            if any(x.get("confidence") is not None for x in r1.summary["observation_rows"]):
                result.err("confidence must remain null")
            for er in r1.summary["evidence_rows"]:
                if er["polarity"] != "supports":
                    continue
                st, reasons = decide_assignment_status([er], policy=policy)
                if st != "candidate":
                    result.err(f"jersey-only must be candidate, got {st}")
                if "JERSEY_ALONE_INSUFFICIENT" not in reasons:
                    result.err("JERSEY_ALONE_INSUFFICIENT missing")
                if er["reliability_tier"] in {"strong", "manual_verified"}:
                    result.err("jersey tier too strong", integrity=True)

        # Determinism
        r1b = run_jersey_observe(
            output_dir=session / "two_b",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if r1.accepted and r1b.accepted:
            a = [x.get("raw_text") for x in r1.summary["observation_rows"]]
            b = [x.get("raw_text") for x in r1b.summary["observation_rows"]]
            if a != b:
                result.err("jersey OCR not deterministic")

        # Negatives must not emit numbers
        for name, fx in (
            ("none", fixture_no_number_front),
            ("sponsor", fixture_sponsor_logo_negative),
        ):
            rn = run_jersey_observe(
                output_dir=session / name,
                config=config,
                contain_root=RUNTIME_ROOT,
                in_memory_bundle=fx(),
            )
            if not rn.accepted:
                result.err(f"{name} fixture failed: {rn.error_code}")
            else:
                emitted = any(
                    x.get("raw_text") or x.get("normalized_number") is not None
                    for x in rn.summary["observation_rows"]
                )
                if emitted:
                    result.err(f"{name}: false number emission", integrity=True)
                if rn.summary.get("false_number_emission_rate", 0.0) != 0.0:
                    result.err(f"{name}: false_number_emission_rate != 0", integrity=True)

        # Predicted rejected
        rp = run_jersey_observe(
            output_dir=session / "pred",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=fixture_predicted_rejected(),
        )
        if rp.accepted and any(x.get("raw_text") for x in rp.summary["observation_rows"]):
            result.err("predicted observations must not emit numbers")

        # No overwrite
        r_again = run_jersey_observe(
            output_dir=session / "two",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if r_again.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite must be forbidden")

        # Failure cleanup
        r_fail = run_jersey_observe(
            output_dir=session / "fail",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
            inject_failure=True,
        )
        if r_fail.accepted:
            result.err("injected failure should not accept")
        if (session / "fail" / "jersey_observations.parquet").exists():
            result.err("failure cleanup incomplete", integrity=True)

        # No crop files
        crops = list(session.rglob("*.png")) + list(session.rglob("*.jpg"))
        if crops:
            result.err("crop persistence detected", integrity=True)

        result.extras["evaluation_status"] = NOT_EVALUATED_JERSEY_OCR
        result.extras["auto_confirm"] = False
        result.extras["persist_crops"] = False
        result.extras["false_number_emission_rate_synthetic"] = 0.0
    except Exception as exc:  # noqa: BLE001
        result.err(f"validator scenario failed: {type(exc).__name__}: {exc}")
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)
        else:
            result.extras["session_dir"] = str(session)

    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/identity/jersey_ocr_baseline.yaml",
        help="Jersey OCR baseline config path",
    )
    parser.add_argument("--keep", action="store_true", help="Keep validator session dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload['status']}")
        print(f"gate: {payload['gate']}")
        print(f"exit_code: {payload['exit_code']}")
        if payload.get("config_fingerprint"):
            print(f"config_fingerprint: {payload['config_fingerprint']}")
        if payload.get("evaluation_status"):
            print(f"evaluation_status: {payload['evaluation_status']}")
        for f in payload.get("findings", []):
            print(f"finding: {f}")
        for e in payload.get("errors", []):
            print(f"error: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
