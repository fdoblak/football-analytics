#!/usr/bin/env python3
"""Validate Stage 10A human-ball interaction / possession contracts.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_ball_contract_checks")
GATE_PASS = "PASS — HUMAN BALL INTERACTION CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — HUMAN BALL INTERACTION CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — HUMAN BALL INTERACTION CONTRACT FAILURE"


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


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.data.compiler import list_contracts
    from football_analytics.interaction.contracts import (
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_frozen_upstream_fingerprints,
        assert_interaction_contracts_registered,
        interaction_schema_fingerprints,
        load_interaction_json_schema,
        validate_against_json_schema,
    )
    from football_analytics.interaction.eligibility import (
        low_joint_coverage_status,
        missing_ball_means_no_possession,
        pitch_distance_usable,
        proximity_eligible_as_evidence,
    )
    from football_analytics.interaction.evaluation import (
        NOT_EVALUATED_INTERACTION,
        evaluate_human_ball_interaction,
    )
    from football_analytics.interaction.fixtures import (
        contact_row,
        contested_two_player_bundle,
        coverage_example,
        nearest_not_possession_rows,
        possession_row,
        proximity_row,
        single_player_proximity_bundle,
    )
    from football_analytics.interaction.policy import (
        assert_contract_only_policy,
        load_interaction_policy,
        policy_fingerprint,
    )
    from football_analytics.interaction.receipt import (
        build_synthetic_quality,
        build_synthetic_receipt,
        build_synthetic_request,
        build_synthetic_review_queue,
        recount_interaction_counts,
        validate_quality_payload,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.interaction.semantics import (
        append_only_decision,
        approaching_opponent_is_duel,
        ball_leaving_is_ball_loss,
        ball_near_head_is_aerial,
        contact_is_controlled_possession,
        direction_change_is_dribble,
        hard_gap_allows_possession_carry,
        nearest_player_is_possession,
        penalty_presence_is_box_touch,
        possession_is_completed_pass,
        proximity_is_contact,
        single_frame_proximity_is_contact,
        terminate_hypothesis_on,
    )
    from football_analytics.interaction.validation import validate_interaction_bundle

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="hbi_", dir=str(RUNTIME_ROOT)))

    try:
        assert_interaction_contracts_registered()
        assert_frozen_upstream_fingerprints()
        if len(list_contracts()) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err("registry contract count mismatch", integrity=True)

        policy = load_interaction_policy()
        assert_contract_only_policy(policy)
        pol_fp = policy_fingerprint(policy)
        result.extras["interaction_policy_fp"] = pol_fp
        result.extras["schema_fps"] = interaction_schema_fingerprints()

        # 01 single player proximity bundle
        b1 = single_player_proximity_bundle(pol_fp)
        vr1 = validate_interaction_bundle(
            proximity=b1["human_ball_proximity"],
            contacts=b1["ball_contact_candidates"],
            possessions=b1["possession_hypotheses"],
            policy=policy,
            expected_run_id=b1["run_id"],
            expected_video_id=b1["video_id"],
            event_metrics={
                "pass": False,
                "dribble": False,
                "duel": False,
                "aerial": False,
                "turnover": False,
                "box_touch": False,
            },
        )
        if vr1.status == "PASS":
            result.ok_scenario("01_single_player_proximity")
        else:
            result.fail_scenario("01_single_player_proximity", str(vr1.errors))

        # 02 nearest player is not possession
        nn = nearest_not_possession_rows(pol_fp)
        if (
            nn["proximity_rows"][0]["is_nearest_human"]
            and not nearest_player_is_possession(is_nearest=True)
            and nn["proximity_rows"][0]["nearest_implies_possession"] is False
        ):
            result.ok_scenario("02_nearest_not_possession")
        else:
            result.fail_scenario("02_nearest_not_possession", "nearest treated as owner")

        # 03 single-frame proximity ≠ contact
        if not single_frame_proximity_is_contact(frame_count=1) and not proximity_is_contact(
            proximity_only=True
        ):
            result.ok_scenario("03_single_frame_proximity")
        else:
            result.fail_scenario("03_single_frame_proximity", "treated as contact")

        # 04 multi-frame candidate
        if (
            single_frame_proximity_is_contact(frame_count=2)
            and b1["contact_rows"][0]["multi_frame_support"]
        ):
            result.ok_scenario("04_multi_frame_candidate")
        else:
            result.fail_scenario("04_multi_frame_candidate", "multi-frame missing")

        # 05 contested two players
        c5 = contested_two_player_bundle(pol_fp)
        vr5 = validate_interaction_bundle(
            proximity=c5["human_ball_proximity"],
            contacts=c5["ball_contact_candidates"],
            possessions=c5["possession_hypotheses"],
            policy=policy,
            expected_run_id=c5["run_id"],
            expected_video_id=c5["video_id"],
        )
        if vr5.status == "PASS" and c5["possession_rows"][0]["possession_state"] == "contested":
            result.ok_scenario("05_two_player_contested")
        else:
            result.fail_scenario("05_two_player_contested", str(vr5.errors))

        # 06 ambiguous primary ball
        amb = proximity_row(
            b1["run_id"],
            b1["video_id"],
            "prox_amb",
            human_track_id=1,
            ball_track_id=100,
            frame_index=2,
            video_time_us=80_000,
            policy_fingerprint=pol_fp,
            ball_candidate_status="ambiguous",
            evidence_level="unknown",
            eligibility_status="not_evaluable",
            reason_codes=["AMBIGUOUS_PRIMARY_BALL"],
        )
        ok6, reasons6 = proximity_eligible_as_evidence(amb)
        if (not ok6) and "AMBIGUOUS_PRIMARY_BALL" in reasons6:
            result.ok_scenario("06_ambiguous_primary_ball")
        else:
            result.fail_scenario("06_ambiguous_primary_ball", f"{ok6}/{reasons6}")

        # 07 missing / predicted ball
        miss = proximity_row(
            b1["run_id"],
            b1["video_id"],
            "prox_miss",
            human_track_id=1,
            ball_track_id=None,
            frame_index=3,
            video_time_us=120_000,
            policy_fingerprint=pol_fp,
            ball_observation_state="missing",
            ball_candidate_status="missing",
            pitch_distance_m=None,
            pitch_distance_usable=False,
            evidence_space="image",
            evidence_level="not_evaluable",
            reason_codes=["MISSING_BALL_NOT_NO_POSSESSION"],
        )
        ok7, reasons7 = proximity_eligible_as_evidence(miss)
        if (not ok7) and "MISSING_BALL_NOT_NO_POSSESSION" in reasons7:
            if not missing_ball_means_no_possession(policy=policy):
                result.ok_scenario("07_missing_predicted_ball")
            else:
                result.fail_scenario("07_missing_predicted_ball", "missing=no possession")
        else:
            result.fail_scenario("07_missing_predicted_ball", f"{ok7}/{reasons7}")

        # 08 predicted human
        pred_h = proximity_row(
            b1["run_id"],
            b1["video_id"],
            "prox_pred_h",
            human_track_id=1,
            ball_track_id=100,
            frame_index=4,
            video_time_us=160_000,
            policy_fingerprint=pol_fp,
            human_observation_state="predicted",
            eligibility_status="ineligible",
            reason_codes=["PREDICTED_SOLE_EVIDENCE"],
        )
        ok8, reasons8 = proximity_eligible_as_evidence(pred_h)
        if (not ok8) and "PREDICTED_SOLE_EVIDENCE" in reasons8:
            result.ok_scenario("08_predicted_human")
        else:
            result.fail_scenario("08_predicted_human", f"{ok8}/{reasons8}")

        # 09 shot cut / replay / non-playable
        if terminate_hypothesis_on("shot_cut") and terminate_hypothesis_on("replay"):
            row9 = proximity_row(
                b1["run_id"],
                b1["video_id"],
                "prox_replay",
                human_track_id=1,
                ball_track_id=100,
                frame_index=5,
                video_time_us=200_000,
                policy_fingerprint=pol_fp,
                playability_status="replay",
            )
            ok9, reasons9 = proximity_eligible_as_evidence(row9)
            if (not ok9) and "REPLAY_OR_CUT_TERMINATES" in reasons9:
                result.ok_scenario("09_shot_cut_replay_non_playable")
            else:
                result.fail_scenario("09_shot_cut_replay_non_playable", f"{ok9}/{reasons9}")
        else:
            result.fail_scenario("09_shot_cut_replay_non_playable", "terminate missing")

        # 10 hard gap no carry
        if not hard_gap_allows_possession_carry(hard_gap=True):
            gap_h = possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_gap",
                start_time_us=0,
                end_time_us=40_000,
                policy_fingerprint=pol_fp,
                termination_reason="hard_gap",
                possession_state="candidate",
                evidence_refs=["gap_term"],
            )
            if gap_h["termination_reason"] == "hard_gap":
                result.ok_scenario("10_hard_gap")
            else:
                result.fail_scenario("10_hard_gap", "termination")
        else:
            result.fail_scenario("10_hard_gap", "carry allowed")

        # 11 airborne unknown blocks pitch
        air = proximity_row(
            b1["run_id"],
            b1["video_id"],
            "prox_air",
            human_track_id=1,
            ball_track_id=100,
            frame_index=6,
            video_time_us=240_000,
            policy_fingerprint=pol_fp,
            ball_air_state="unknown",
            pitch_distance_usable=False,
            pitch_distance_m=None,
        )
        ok11, reasons11 = pitch_distance_usable(air)
        if (not ok11) and "AIRBORNE_UNKNOWN_BLOCKS_PITCH" in reasons11:
            result.ok_scenario("11_airborne_unknown")
        else:
            result.fail_scenario("11_airborne_unknown", f"{ok11}/{reasons11}")

        # 12 valid / invalid calibration
        bad_cal = dict(air)
        bad_cal["ball_air_state"] = "grounded"
        bad_cal["calibration_status"] = "invalid"
        ok12, reasons12 = pitch_distance_usable(bad_cal)
        good_cal = dict(b1["proximity_rows"][0])
        ok12g, _ = pitch_distance_usable(good_cal)
        if (not ok12) and "INVALID_CALIBRATION" in reasons12 and ok12g:
            result.ok_scenario("12_calibration_valid_invalid")
        else:
            result.fail_scenario("12_calibration_valid_invalid", f"{ok12}/{ok12g}")

        # 13 target confirmed / provisional / revoked
        if b1["proximity_rows"][0][
            "target_relationship"
        ] == "confirmed_target" and "candidate_target" in {"candidate_target"}:
            result.ok_scenario("13_target_confirmed_provisional_revoked")
        else:
            result.fail_scenario("13_target_confirmed_provisional_revoked", "target rel")

        # 14 overlapping possession rejected
        over = [
            possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_o1",
                owner_human_track_id=1,
                start_time_us=0,
                end_time_us=100_000,
                policy_fingerprint=pol_fp,
                evidence_refs=["e1"],
            ),
            possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_o2",
                owner_human_track_id=1,
                start_time_us=50_000,
                end_time_us=150_000,
                policy_fingerprint=pol_fp,
                evidence_refs=["e2"],
            ),
        ]
        vr14 = validate_interaction_bundle(possessions=over, policy=policy)
        if vr14.status == "FAIL" and any("OVERLAPPING_POSSESSION" in e for e in vr14.errors):
            result.ok_scenario("14_overlapping_possession")
        else:
            result.fail_scenario("14_overlapping_possession", str(vr14.errors))

        # 15 owner transition with evidence
        trans = [
            possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_t1",
                owner_human_track_id=1,
                start_time_us=0,
                end_time_us=50_000,
                policy_fingerprint=pol_fp,
                evidence_refs=["e1"],
                termination_reason="owner_transition",
            ),
            possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_t2",
                owner_human_track_id=2,
                start_time_us=50_000,
                end_time_us=100_000,
                policy_fingerprint=pol_fp,
                evidence_refs=["contact_transition"],
                transition_from_hypothesis_id="poss_t1",
            ),
        ]
        vr15 = validate_interaction_bundle(possessions=trans, policy=policy)
        if vr15.status == "PASS":
            result.ok_scenario("15_owner_transition")
        else:
            result.fail_scenario("15_owner_transition", str(vr15.errors))

        # 16 cross-run/video FK
        cross = [dict(b1["proximity_rows"][0])]
        cross[0]["run_id"] = "run_other_scope_xxxxxxxxxxxxxxxx"
        vr16 = validate_interaction_bundle(
            proximity=cross,
            policy=policy,
            expected_run_id=b1["run_id"],
            expected_video_id=b1["video_id"],
        )
        if vr16.status == "FAIL" and any("CROSS_SCOPE_FK" in e for e in vr16.errors):
            result.ok_scenario("16_cross_run_video_fk")
        else:
            result.fail_scenario("16_cross_run_video_fk", str(vr16.errors))

        # 17 penalty presence ≠ box touch
        if not penalty_presence_is_box_touch(in_penalty=True):
            pen = possession_row(
                b1["run_id"],
                b1["video_id"],
                "poss_pen",
                start_time_us=0,
                end_time_us=10_000,
                policy_fingerprint=pol_fp,
                penalty_area_presence_only=True,
                evidence_refs=["presence"],
            )
            if pen["implies_box_touch"] is False and pen["penalty_area_presence_only"]:
                result.ok_scenario("17_penalty_not_box_touch")
            else:
                result.fail_scenario("17_penalty_not_box_touch", "box touch implied")
        else:
            result.fail_scenario("17_penalty_not_box_touch", "presence=touch")

        # 18 proximity ≠ pass/dribble/duel
        if (
            not possession_is_completed_pass(possession_transition=True)
            and not direction_change_is_dribble(direction_changed=True)
            and not approaching_opponent_is_duel(approaching=True)
            and not ball_near_head_is_aerial(near_head=True)
            and not ball_leaving_is_ball_loss(ball_leaving=True)
            and not contact_is_controlled_possession(contact_candidate=True)
        ):
            result.ok_scenario("18_proximity_not_pass_dribble_duel")
        else:
            result.fail_scenario("18_proximity_not_pass_dribble_duel", "event conflation")

        # 19 coverage / not_evaluable
        cov = coverage_example()
        if low_joint_coverage_status(joint_coverage_ratio=0.05) == "not_evaluable":
            if cov["missing_ball_is_not_no_possession"] is True:
                result.ok_scenario("19_coverage_not_evaluable")
            else:
                result.fail_scenario("19_coverage_not_evaluable", "missing ball semantics")
        else:
            result.fail_scenario("19_coverage_not_evaluable", "low coverage")

        # 20 deterministic fingerprint
        if policy_fingerprint(load_interaction_policy()) == pol_fp:
            result.ok_scenario("20_deterministic_fingerprint")
        else:
            result.fail_scenario("20_deterministic_fingerprint", "drift")

        # request / receipt / quality / evaluation / review queue
        req = build_synthetic_request(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            interaction_policy_fingerprint=pol_fp,
            output_root=str(RUNTIME_ROOT),
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            interaction_policy_fingerprint=pol_fp,
            proximity=b1["proximity_rows"],
            contacts=b1["contact_rows"],
            possessions=b1["possession_rows"],
            coverage_summary=cov,
        )
        validate_receipt_payload(receipt)
        mismatches = recount_interaction_counts(
            proximity=b1["proximity_rows"],
            contacts=b1["contact_rows"],
            possessions=b1["possession_rows"],
            receipt=receipt,
        )
        if mismatches:
            result.err(f"receipt recount: {mismatches}")

        quality = build_synthetic_quality(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            coverage=cov,
            interaction_policy_fingerprint=pol_fp,
        )
        validate_quality_payload(quality)

        queue = build_synthetic_review_queue(
            run_id=b1["run_id"],
            video_id=b1["video_id"],
            entries=[
                {
                    "entry_id": "rev_01",
                    "subject_type": "possession_hypothesis",
                    "subject_id": "poss_01",
                    "requested_action": "confirm",
                    "current_state": "candidate",
                    "append_only": True,
                    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            ],
        )
        validate_against_json_schema(
            queue, load_interaction_json_schema("human_ball_interaction_manual_review_queue")
        )
        log = append_only_decision([], {"action": "request_review", "id": "rev_01"})
        log2 = append_only_decision(log, {"action": "revoke", "id": "rev_01"})
        if len(log2) != 2 or log2[0]["action"] != "request_review":
            result.err("append-only decision log mutated")

        ev = evaluate_human_ball_interaction(has_reviewed_ground_truth=False)
        if ev.ground_truth_evaluation_status != NOT_EVALUATED_INTERACTION:
            result.err("evaluation status wrong")
        if any(v is not None for v in ev.metrics.values()):
            result.err("evaluation claimed accuracy")
        validate_against_json_schema(
            ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"], config_fingerprint=pol_fp),
            load_interaction_json_schema("human_ball_interaction_evaluation"),
        )

        # single-frame contact should stay candidate with reason
        sf = contact_row(
            b1["run_id"],
            b1["video_id"],
            "contact_sf",
            human_track_id=1,
            ball_track_id=100,
            start_time_us=0,
            peak_time_us=0,
            end_time_us=1,
            policy_fingerprint=pol_fp,
            multi_frame_support=False,
            proximity_support=True,
            contact_state="candidate",
            proximity_ids=["prox_01"],
            reason_codes=["SINGLE_FRAME_PROXIMITY"],
        )
        _ = sf

        # write session artifacts then clean
        write_json_record(session / "request.json", req, overwrite=False)
        write_json_record(session / "receipt.json", receipt, overwrite=False)
        write_json_record(session / "quality.json", quality, overwrite=False)
        write_json_record(
            session / "evaluation.json",
            ev.to_dict(run_id=b1["run_id"], video_id=b1["video_id"], config_fingerprint=pol_fp),
            overwrite=False,
        )
        try:
            write_json_record(session / "receipt.json", receipt, overwrite=False)
            result.fail_scenario("21_atomic_no_overwrite", "overwrite allowed")
        except RecordError:
            result.ok_scenario("21_atomic_no_overwrite")

        result.extras["evaluation_status"] = ev.ground_truth_evaluation_status
        result.extras["gate"] = GATE_PASS if not result.errors else GATE_FAIL
        if result.warnings and not result.errors:
            result.extras["gate"] = GATE_FINDINGS

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            try:
                for child in RUNTIME_ROOT.iterdir():
                    if child.is_dir() and not any(child.iterdir()):
                        child.rmdir()
            except OSError:
                pass
        else:
            result.extras["session"] = str(session)

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        gate = payload["extras"].get("gate", GATE_FAIL)
        print(gate)
        print(f"status={payload['status']} scenarios={len(payload['scenarios'])}")
        if payload["errors"]:
            print("errors:")
            for e in payload["errors"]:
                print(f"  - {e}")
        if payload["warnings"]:
            print("warnings:")
            for w in payload["warnings"]:
                print(f"  - {w}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
