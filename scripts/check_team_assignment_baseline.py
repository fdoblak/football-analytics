#!/usr/bin/env python3
"""Validate Stage 7C anonymous team appearance clustering + team_assignments baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/team_assignment_checks")
GATE_PASS = "PASS — TEAM ASSIGNMENT BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — TEAM ASSIGNMENT BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — TEAM ASSIGNMENT BASELINE FAILURE"


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
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        EXPECTED_TEAM_ASSIGNMENTS_FP,
        assert_frozen_upstream_fingerprints,
        assert_identity_contracts_registered,
    )
    from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
    from football_analytics.identity.team_assignment_config import (
        load_team_assignment_config,
        team_assignment_config_fingerprint,
    )
    from football_analytics.identity.team_assignment_evaluation import NOT_EVALUATED_TEAM_ASSIGNMENT
    from football_analytics.identity.team_assignment_fixtures import (
        assert_runtime_root,
        fixture_goalkeeper_different_kit,
        fixture_insufficient_seeds,
        fixture_similar_kit_hard,
        fixture_third_color_outlier,
        fixture_two_distinct_teams,
        fixture_with_referee,
    )
    from football_analytics.identity.team_assignment_service import run_team_classify

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_team_assignment_config(cfg_path)
        result.extras["config_fingerprint"] = team_assignment_config_fingerprint(config)
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
        ta_fp = contract_fingerprint(get_contract("team_assignments", 1))
        result.extras["team_assignments_fingerprint"] = ta_fp
        if ta_fp != EXPECTED_TEAM_ASSIGNMENTS_FP:
            result.err("team_assignments fingerprint drift", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"contract check failed: {exc}", integrity=True)
        return result.finalize()

    selected = [m for m in config["selection_matrix"] if str(m.get("status")).lower() == "selected"]
    if len(selected) != 1:
        result.err("exactly one method must be selected", config=True)
    result.extras["selected_method"] = selected[0] if selected else None
    result.finding("anonymous 2-cluster color SELECTED; real team naming / home-away forbidden")
    result.finding(NOT_EVALUATED_TEAM_ASSIGNMENT)
    result.finding("real football team-assignment accuracy not yet validated")
    result.finding("similar-kit / lighting / GK-ref contamination risks remain open")

    if config["assignment"]["auto_target_confirmation"] is not False:
        result.err("auto_target_confirmation must be false", integrity=True)
    if config["assignment"]["auto_real_team_naming"] is not False:
        result.err("auto_real_team_naming must be false", integrity=True)
    if config["assignment"]["auto_home_away"] is not False:
        result.err("auto_home_away must be false", integrity=True)
    if config["clustering"]["cross_video_auto_transfer"] is not False:
        result.err("cross_video_auto_transfer must be false", integrity=True)

    policy = load_identity_policy()
    session = Path(tempfile.mkdtemp(prefix="team_", dir=str(RUNTIME_ROOT)))
    try:
        two = fixture_two_distinct_teams()
        r1 = run_team_classify(
            output_dir=session / "two",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if not r1.accepted:
            result.err(f"two-team classify failed: {r1.error_code}")
        else:
            model = r1.summary["model"]
            if model.status != "ok":
                result.err(f"expected ok clusters, got {model.status}")
            teams = {d.team_id for d in r1.summary["decisions"] if d.status == "assigned"}
            if teams != {"team_a", "team_b"}:
                result.err(f"expected team_a/team_b assigned, got {teams}")
            for er in r1.summary["evidence_rows"]:
                st, reasons = decide_assignment_status([er], policy=policy)
                if st != "candidate":
                    result.err(f"team-only must be candidate, got {st}")
                if "TEAM_ALONE_INSUFFICIENT" not in reasons:
                    result.err("TEAM_ALONE_INSUFFICIENT missing")
                if er["reliability_tier"] in {"strong", "manual_verified"}:
                    result.err("team tier too strong", integrity=True)

        # Determinism
        r1b = run_team_classify(
            output_dir=session / "two_b",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if (
            r1.accepted
            and r1b.accepted
            and (
                r1.summary["model"].centroid_fingerprints
                != r1b.summary["model"].centroid_fingerprints
            )
        ):
            result.err("clustering not deterministic")

        # Insufficient seeds
        few = fixture_insufficient_seeds()
        rf = run_team_classify(
            output_dir=session / "few",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=few,
        )
        if not rf.accepted or rf.summary["model"].status != "insufficient_team_evidence":
            result.err("insufficient seeds must yield insufficient_team_evidence")

        # Referee not eligible
        ref = fixture_with_referee()
        rr = run_team_classify(
            output_dir=session / "ref",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=ref,
        )
        if rr.accepted:
            d = next(x for x in rr.summary["decisions"] if x.track_id == 99)
            if d.status != "not_eligible" or d.team_role != "official":
                result.err("referee must be not_eligible official")

        # GK unbound
        gk = fixture_goalkeeper_different_kit()
        rg = run_team_classify(
            output_dir=session / "gk",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=gk,
        )
        if rg.accepted:
            d = next(x for x in rg.summary["decisions"] if x.track_id == 77)
            if d.team_id != "unknown" or "GOALKEEPER_NO_AUTO_TEAM_FROM_KIT" not in d.reason_codes:
                result.err("goalkeeper must not auto-bind from kit")

        # Outlier
        out = fixture_third_color_outlier()
        ro = run_team_classify(
            output_dir=session / "out",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=out,
        )
        if ro.accepted:
            d = next(x for x in ro.summary["decisions"] if x.track_id == 66)
            if d.team_id != "unknown":
                result.err("third-color outlier must be unknown")

        # Similar kit abstain path
        sim = fixture_similar_kit_hard()
        rs = run_team_classify(
            output_dir=session / "sim",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=sim,
        )
        if not rs.accepted:
            result.err(f"similar-kit run failed: {rs.error_code}")
        else:
            if rs.summary["model"].status == "ok":
                result.warn(
                    "similar-kit produced clusters; ambiguity/unknown still expected on margin"
                )
            else:
                result.extras["similar_kit_status"] = rs.summary["model"].status

        # No overwrite
        r_again = run_team_classify(
            output_dir=session / "two",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
        )
        if r_again.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite must be forbidden")

        # Failure cleanup
        r_fail = run_team_classify(
            output_dir=session / "fail",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=two,
            inject_failure=True,
        )
        if r_fail.accepted:
            result.err("injected failure should not accept")
        if (session / "fail" / "team_assignments.parquet").exists():
            result.err("failure cleanup incomplete", integrity=True)

        result.extras["evaluation_status"] = NOT_EVALUATED_TEAM_ASSIGNMENT
        result.extras["auto_confirm"] = False
        result.extras["anonymous_labels_only"] = True
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
        default="configs/identity/team_assignment_baseline.yaml",
        help="Team assignment baseline config path",
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
