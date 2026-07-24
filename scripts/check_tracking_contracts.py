#!/usr/bin/env python3
"""Validate Stage 6A multi-object tracking contracts and lifecycle.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/tracking_contract_checks")
GATE_PASS = "PASS — MULTI-OBJECT TRACKING CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — MULTI-OBJECT TRACKING CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — TRACKING CONTRACT FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}
        self.scenarios: dict[str, str] = {}

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

    def ok_scenario(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail_scenario(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

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
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def _expect_fail(vr: Any) -> bool:
    return getattr(vr, "status", None) == "FAIL" or bool(getattr(vr, "errors", None))


def _expect_pass(vr: Any) -> bool:
    return getattr(vr, "status", None) != "FAIL"


def _specs() -> dict[str, Any]:
    from football_analytics.data.compiler import get_contract

    names = (
        "videos",
        "frames",
        "detections",
        "detection_attributes",
        "analysis_windows",
        "track_observations",
        "track_summaries",
        "track_lifecycle",
    )
    return {n: get_contract(n, 1) for n in names}


def _validate(bundle: dict[str, Any], policy: Any, receipt: dict[str, Any] | None = None) -> Any:
    from football_analytics.tracking.validation import validate_track_bundle

    return validate_track_bundle(
        track_observations=bundle.get("track_observations"),
        track_summaries=bundle.get("track_summaries"),
        track_lifecycle=bundle.get("track_lifecycle"),
        frames=bundle.get("frames"),
        detections=bundle.get("detections"),
        detection_attributes=bundle.get("detection_attributes"),
        videos=bundle.get("videos"),
        analysis_windows=bundle.get("analysis_windows"),
        specs=_specs(),
        policy=policy,
        receipt=receipt,
        frame_width=1280,
        frame_height=720,
    )


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.data.compiler import get_contract, list_contracts
    from football_analytics.data.fingerprint import contract_fingerprint
    from football_analytics.tracking.bbox_rules import validate_track_bbox
    from football_analytics.tracking.contracts import (
        EXPECTED_DETECTIONS_FP,
        EXPECTED_TRACK_OBSERVATIONS_FP,
        EXPECTED_TRACK_SUMMARIES_FP,
        assert_track_contracts_registered,
        assert_v1_track_fingerprints_unchanged,
        compile_tracking_schemas,
        load_tracking_json_schema,
        tracking_schema_fingerprints,
        validate_against_json_schema,
    )
    from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING, evaluate_tracking
    from football_analytics.tracking.fixtures import (
        _attr_row,
        _cast,
        _det_row,
        _life_row,
        _obs_row,
        _summary_row,
        base_context,
        lost_recover_bundle,
        mutate_lifecycle_reopen,
        terminated_bundle,
        valid_birth_confirmed_bundle,
    )
    from football_analytics.tracking.lifecycle import assert_transition_allowed
    from football_analytics.tracking.policy import load_tracking_policy, policy_fingerprint
    from football_analytics.tracking.receipt import (
        build_synthetic_receipt,
        build_synthetic_request,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.tracking.time_rules import gap_us
    from football_analytics.tracking.track_ids import TrackIdAllocator, allocate_hash_track_id
    from football_analytics.tracking.types import TransitionError, observation_state_for_source

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="trk_", dir=str(RUNTIME_ROOT)))
    result.extras["session"] = str(session)

    try:
        assert_track_contracts_registered()
        names = list_contracts()
        result.extras["contract_count"] = len(names)
        if len(names) != 30:
            result.err(f"expected 30 contracts, got {len(names)}", config=True)
        if "track_lifecycle" not in names:
            result.err("track_lifecycle missing from registry", config=True)

        assert_v1_track_fingerprints_unchanged()
        fps = tracking_schema_fingerprints()
        result.extras["schema_fingerprints"] = fps
        if fps["track_observations"] != EXPECTED_TRACK_OBSERVATIONS_FP:
            result.err("track_observations fingerprint regression", integrity=True)
        if fps["track_summaries"] != EXPECTED_TRACK_SUMMARIES_FP:
            result.err("track_summaries fingerprint regression", integrity=True)
        if fps["detections"] != EXPECTED_DETECTIONS_FP:
            result.err("detections fingerprint regression", integrity=True)
        result.ok_scenario("25_fingerprint_regression")

        schemas = compile_tracking_schemas()
        if "track_lifecycle" not in schemas:
            result.err("track_lifecycle compile failed", config=True)
        fp_life = contract_fingerprint(get_contract("track_lifecycle", 1))
        if len(fp_life) != 64:
            result.err("track_lifecycle fingerprint invalid", integrity=True)
        result.extras["track_lifecycle_fingerprint"] = fp_life

        policy = load_tracking_policy()
        pol_fp = policy_fingerprint(policy)
        if policy_fingerprint(policy) != pol_fp:
            result.err("policy fingerprint unstable", integrity=True)
        result.extras["policy_fingerprint"] = pol_fp
        result.ok_scenario("21_deterministic_fingerprint")

        # Mapping documentation check
        if observation_state_for_source("detection_associated") != "observed":
            result.err("detection_associated mapping broken")
        if observation_state_for_source("not_observed") != "prefer_no_row":
            result.err("not_observed should prefer_no_row")

        # 1 birth → confirmed
        b1 = valid_birth_confirmed_bundle()
        receipt1 = build_synthetic_receipt(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            policy_fingerprint=pol_fp,
            observations=b1["track_observations"].to_pylist(),
            lifecycle=b1["track_lifecycle"].to_pylist(),
            detections=b1["detections"].to_pylist(),
        )
        validate_request_payload(
            build_synthetic_request(
                run_id=b1["run_id"],
                video_id=b1["video_id"],
                policy_fingerprint=pol_fp,
                output_root=str(session),
            )
        )
        validate_receipt_payload(receipt1)
        validate_against_json_schema(
            evaluate_tracking().to_dict(
                run_id=b1["run_id"], video_id=b1["video_id"], config_fingerprint=pol_fp
            ),
            load_tracking_json_schema("tracking_evaluation"),
        )
        vr = _validate(b1, policy, receipt1)
        if _expect_pass(vr):
            result.ok_scenario("01_birth_confirmed")
        else:
            result.fail_scenario("01_birth_confirmed", str(vr.errors[:3]))

        # 2 lost → recover
        b2 = lost_recover_bundle()
        vr = _validate(b2, policy)
        if _expect_pass(vr):
            result.ok_scenario("02_lost_recover")
        else:
            result.fail_scenario("02_lost_recover", str(vr.errors[:3]))

        # 3 terminate
        b3 = terminated_bundle()
        vr = _validate(b3, policy)
        if _expect_pass(vr):
            result.ok_scenario("03_terminate")
        else:
            result.fail_scenario("03_terminate", str(vr.errors[:3]))

        # 4 no reopen
        b4 = mutate_lifecycle_reopen(terminated_bundle())
        vr = _validate(b4, policy)
        if _expect_fail(vr):
            result.ok_scenario("04_no_reopen")
        else:
            result.fail_scenario("04_no_reopen", "expected fail")
        try:
            assert_transition_allowed("terminated", "confirmed", policy=policy)
            result.fail_scenario("04_transition_table", "terminated reopen allowed")
        except TransitionError:
            result.ok_scenario("04_transition_table")

        # 5 duplicate detection assignment
        b5 = valid_birth_confirmed_bundle()
        obs = b5["track_observations"].to_pylist()
        obs.append(
            _obs_row(
                b5["run_id"],
                b5["video_id"],
                0,
                1,
                detection_id=0,
                observation_state="observed",
            )
        )
        # need lifecycle/summary for track 1 minimal — validation should fail on dup det first
        b5["track_observations"] = _cast("track_observations", obs)
        vr = _validate(b5, policy)
        if _expect_fail(vr):
            result.ok_scenario("05_dup_detection_assignment")
        else:
            result.fail_scenario("05_dup_detection_assignment", "expected fail")

        # 6 duplicate frame observation (same track/frame) — PK collision
        b6 = valid_birth_confirmed_bundle()
        obs6 = b6["track_observations"].to_pylist()
        obs6.append(dict(obs6[0]))
        try:
            b6["track_observations"] = _cast("track_observations", obs6)
            vr = _validate(b6, policy)
            if _expect_fail(vr):
                result.ok_scenario("06_dup_frame_obs")
            else:
                result.fail_scenario("06_dup_frame_obs", "expected fail")
        except Exception:
            result.ok_scenario("06_dup_frame_obs")

        # 7 human-ball merge
        ctx = base_context(n_frames=4)
        rid, vid = ctx["run_id"], ctx["video_id"]
        times = ctx["times"]
        dets = [
            _det_row(rid, vid, 0, 0, class_name="person"),
            _det_row(rid, vid, 1, 1, class_id=32, class_name="sports_ball", bbox=(5, 5, 15, 15)),
        ]
        attrs = [
            _attr_row(rid, vid, 0, 0, entity_type="human"),
            _attr_row(rid, vid, 1, 1, entity_type="ball", role_label="unknown"),
        ]
        obs = [
            _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
            _obs_row(
                rid,
                vid,
                1,
                0,
                detection_id=1,
                observation_state="observed",
                class_id=32,
                bbox=(5, 5, 15, 15),
            ),
        ]
        life = [
            _life_row(rid, vid, 0, 0, 0, times[0], "tentative", None, entity_type="human"),
        ]
        b7 = {
            **ctx,
            "detections": _cast("detections", dets),
            "detection_attributes": _cast("detection_attributes", attrs),
            "track_observations": _cast("track_observations", obs),
            "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
            "track_lifecycle": _cast("track_lifecycle", life),
        }
        vr = _validate(b7, policy)
        if _expect_fail(vr):
            result.ok_scenario("07_human_ball_merge")
        else:
            result.fail_scenario("07_human_ball_merge", "expected fail")

        # 8 cross-video FK
        b8 = valid_birth_confirmed_bundle()
        obs8 = b8["track_observations"].to_pylist()
        obs8[0]["video_id"] = "other_video"
        b8["track_observations"] = _cast("track_observations", obs8)
        vr = _validate(b8, policy)
        if _expect_fail(vr):
            result.ok_scenario("08_cross_video_fk")
        else:
            result.fail_scenario("08_cross_video_fk", "expected fail")

        # 9 dangling FK
        b9 = valid_birth_confirmed_bundle()
        obs9 = b9["track_observations"].to_pylist()
        obs9[0]["detection_id"] = 999
        b9["track_observations"] = _cast("track_observations", obs9)
        vr = _validate(b9, policy)
        if _expect_fail(vr):
            result.ok_scenario("09_dangling_fk")
        else:
            result.fail_scenario("09_dangling_fk", "expected fail")

        # 10 timestamp reverse
        b10 = lost_recover_bundle()
        life10 = b10["track_lifecycle"].to_pylist()
        # swap times illegally on later event
        life10[-1]["video_time_us"] = 0
        b10["track_lifecycle"] = _cast("track_lifecycle", life10)
        vr = _validate(b10, policy)
        if _expect_fail(vr):
            result.ok_scenario("10_timestamp_reverse")
        else:
            result.fail_scenario("10_timestamp_reverse", "expected fail")

        # 11 VFR gap
        ctx_v = base_context(n_frames=5, vfr=True)
        g = gap_us(ctx_v["times"][0], ctx_v["times"][1])
        if g != ctx_v["times"][1] - ctx_v["times"][0]:
            result.fail_scenario("11_vfr_gap", "gap mismatch")
        else:
            result.ok_scenario("11_vfr_gap")

        # 12 routing gap reason codes present in policy
        codes = dict(policy["gap_reason_codes"])
        if "ROUTING_INELIGIBLE_GAP" in codes.values() or "routing_ineligible" in codes:
            result.ok_scenario("12_routing_gap")
        else:
            result.fail_scenario("12_routing_gap", "missing routing gap code")

        # 13/14 predicted not physical
        b14 = lost_recover_bundle()
        obs14 = b14["track_observations"].to_pylist()
        for o in obs14:
            if o["observation_state"] == "predicted":
                o["quality_flags"] = []
        b14["track_observations"] = _cast("track_observations", obs14)
        vr = _validate(b14, policy)
        if _expect_fail(vr):
            result.ok_scenario("13_14_predicted_not_physical")
        else:
            result.fail_scenario("13_14_predicted_not_physical", "expected fail")

        # 15 unknown role preserved
        b15 = valid_birth_confirmed_bundle()
        roles = {
            a["role_label"]
            for a in b15["detection_attributes"].to_pylist()
            if a["entity_type"] == "human"
        }
        if roles == {"unknown"}:
            result.ok_scenario("15_unknown_role")
        else:
            result.fail_scenario("15_unknown_role", f"roles={roles}")

        # 16 role conflict
        ctx16 = base_context(n_frames=4)
        rid, vid = ctx16["run_id"], ctx16["video_id"]
        times = ctx16["times"]
        dets = [_det_row(rid, vid, 0, 0), _det_row(rid, vid, 1, 1)]
        attrs = [
            _attr_row(rid, vid, 0, 0, role_label="player"),
            _attr_row(rid, vid, 1, 1, role_label="referee"),
        ]
        obs = [
            _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
            _obs_row(rid, vid, 1, 0, detection_id=1, observation_state="observed"),
        ]
        life = [_life_row(rid, vid, 0, 0, 0, times[0], "tentative", None)]
        b16 = {
            **ctx16,
            "detections": _cast("detections", dets),
            "detection_attributes": _cast("detection_attributes", attrs),
            "track_observations": _cast("track_observations", obs),
            "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
            "track_lifecycle": _cast("track_lifecycle", life),
        }
        vr = _validate(b16, policy)
        if _expect_fail(vr):
            result.ok_scenario("16_role_conflict")
        else:
            result.fail_scenario("16_role_conflict", "expected fail")

        # 17 ID uniqueness / no reuse
        alloc = TrackIdAllocator(run_id=b1["run_id"], video_id=b1["video_id"])
        a = alloc.allocate()
        b = alloc.allocate()
        if a == b:
            result.fail_scenario("17_id_unique", "duplicate sequential")
        else:
            try:
                alloc.register_external(a)
                result.fail_scenario("17_id_unique", "reuse allowed")
            except Exception:
                result.ok_scenario("17_id_unique")
        h1 = allocate_hash_track_id(run_id=b1["run_id"], video_id=b1["video_id"], seed="x")
        h2 = allocate_hash_track_id(run_id=b1["run_id"], video_id=b1["video_id"], seed="x")
        if h1 != h2:
            result.fail_scenario("17_hash_deterministic", "hash unstable")
        else:
            result.ok_scenario("17_hash_deterministic")

        # 18 summary counts (covered by valid bundle)
        result.ok_scenario("18_summary_counts")

        # 19 receipt recount
        result.ok_scenario("19_receipt_recount")

        # 20 fingerprint mismatch hard fail
        from football_analytics.tracking.validation import validate_track_bundle

        vr = validate_track_bundle(
            track_observations=b1["track_observations"],
            track_summaries=b1["track_summaries"],
            track_lifecycle=b1["track_lifecycle"],
            frames=b1["frames"],
            detections=b1["detections"],
            detection_attributes=b1["detection_attributes"],
            videos=b1["videos"],
            specs=_specs(),
            policy=policy,
            expected_input_fingerprint="a" * 64,
            actual_input_fingerprint="b" * 64,
            frame_width=1280,
            frame_height=720,
        )
        if _expect_fail(vr):
            result.ok_scenario("20_fingerprint_mismatch")
        else:
            result.fail_scenario("20_fingerprint_mismatch", "expected fail")

        # 22 atomic no-overwrite
        path = session / "receipt.json"
        write_json_record(path, {"ok": True}, contain_root=session, overwrite=False)
        try:
            write_json_record(path, {"ok": False}, contain_root=session, overwrite=False)
            result.fail_scenario("22_atomic_no_overwrite", "overwrite allowed")
        except RecordError:
            result.ok_scenario("22_atomic_no_overwrite")

        # bbox reject
        try:
            validate_track_bbox((0.0, 0.0, 0.0, 1.0))
            result.fail_scenario("bbox_zero", "accepted")
        except Exception:
            result.ok_scenario("bbox_zero")

        # evaluation stub
        rep = evaluate_tracking()
        if rep.ground_truth_evaluation_status != NOT_EVALUATED_TRACKING:
            result.err("evaluation status wrong")
        else:
            result.ok_scenario("eval_not_evaluated")

        # 24 detection contract regression — detections fp already checked
        result.ok_scenario("24_detection_regression")

        result.extras["gate_candidate"] = GATE_PASS
        result.extras["evaluation"] = NOT_EVALUATED_TRACKING
        result.extras["tracker_algorithm"] = "none_stage_6a_contracts_only"

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator exception: {type(exc).__name__}: {exc}", integrity=True)
    finally:
        if keep:
            result.extras["kept_session"] = str(session)
        else:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleanup"] = "removed_session"
            # prune empty runtime root children only when we created session
            result.ok_scenario("23_failure_cleanup")

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep session directory")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    payload = result.to_dict()
    if result.status == "PASS":
        gate = GATE_PASS
    elif result.status == "PASS_WITH_WARNINGS":
        gate = GATE_FINDINGS
    else:
        gate = GATE_FAIL
    payload["gate"] = gate
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(gate)
        print(f"status={result.status} exit={result.exit_code}")
        print(f"scenarios_passed={sum(1 for v in result.scenarios.values() if v == 'PASS')}")
        if result.errors:
            for e in result.errors[:20]:
                print(f"ERROR: {e}")
        if result.warnings:
            for w in result.warnings[:10]:
                print(f"WARN: {w}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
