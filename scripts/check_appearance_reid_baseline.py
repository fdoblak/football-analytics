#!/usr/bin/env python3
"""Validate Stage 7B appearance embedding + tracklet ReID baseline.

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

RUNTIME_ROOT = Path("/home/fdoblak/workspace/appearance_reid_checks")
GATE_PASS = "PASS — APPEARANCE REID BASELINE ACTIVE"
GATE_FINDINGS = (
    "PASS_WITH_FINDINGS — APPEARANCE REID BASELINE ACTIVE; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)
GATE_FAIL = "NO-GO — APPEARANCE REID BASELINE FAILURE"


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
    from football_analytics.identity.appearance_descriptor import cosine_similarity
    from football_analytics.identity.appearance_reid_config import (
        appearance_reid_config_fingerprint,
        load_appearance_reid_config,
    )
    from football_analytics.identity.appearance_reid_evaluation import NOT_EVALUATED_APPEARANCE_REID
    from football_analytics.identity.appearance_reid_fixtures import (
        assert_runtime_root,
        fixture_cross_video_reject,
        fixture_different_appearance,
        fixture_human_ball_reject,
        fixture_same_appearance_different_tracklets,
        fixture_single_crop_insufficient,
        fixture_temporal_overlap,
    )
    from football_analytics.identity.appearance_reid_service import (
        build_profiles_from_bundle,
        run_reid_candidates,
    )
    from football_analytics.identity.contracts import (
        EXPECTED_DETECTIONS_FP,
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        EXPECTED_TRACK_OBSERVATIONS_FP,
        TRACKLET_APPEARANCE_PROFILES_CONTRACT,
        assert_frozen_upstream_fingerprints,
        assert_identity_contracts_registered,
    )
    from football_analytics.identity.policy import decide_assignment_status, load_identity_policy

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_appearance_reid_config(cfg_path)
        result.extras["config_fingerprint"] = appearance_reid_config_fingerprint(config)
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
        fps = {
            "tracklet_appearance_profiles": contract_fingerprint(
                get_contract(TRACKLET_APPEARANCE_PROFILES_CONTRACT, 1)
            ),
            "detections": contract_fingerprint(get_contract("detections", 1)),
            "track_observations": contract_fingerprint(get_contract("track_observations", 1)),
        }
        result.extras["schema_fingerprints"] = fps
        if fps["detections"] != EXPECTED_DETECTIONS_FP:
            result.err("detections fingerprint drift", integrity=True)
        if fps["track_observations"] != EXPECTED_TRACK_OBSERVATIONS_FP:
            result.err("track_observations fingerprint drift", integrity=True)
        if len(fps["tracklet_appearance_profiles"]) != 64:
            result.err("tracklet_appearance_profiles fingerprint invalid", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"contract check failed: {exc}", integrity=True)
        return result.finalize()

    # Selection matrix
    selected = [m for m in config["selection_matrix"] if str(m.get("status")).lower() == "selected"]
    if len(selected) != 1 or "handcrafted" not in str(selected[0].get("candidate", "")).lower():
        result.err("handcrafted extractor not selected", config=True)
    result.extras["selected_extractor"] = selected[0] if selected else None
    result.finding(
        "handcrafted HSV/Lab/edge SELECTED; sn-reid/TrackLab future; "
        "no local learned ReID weights; torchvision unused"
    )
    result.finding("NOT_EVALUATED_NO_REVIEWED_APPEARANCE_REID_GROUND_TRUTH")
    result.finding("real football appearance ReID accuracy not yet validated")
    result.finding("same-kit false-match risk remains open")

    if config["matching"]["auto_confirm"] is not False:
        result.err("auto_confirm must be false", integrity=True)
    if config["matching"]["face_regions_use"] is not False:
        result.err("face_regions_use must be false", integrity=True)
    if config["sampling"]["persist_crops"] is not False:
        result.err("persist_crops must be false", integrity=True)

    policy = load_identity_policy()
    session = Path(tempfile.mkdtemp(prefix="areid_", dir=str(RUNTIME_ROOT)))
    try:
        # Same appearance → candidate
        same = fixture_same_appearance_different_tracklets()
        out1 = session / "same"
        r1 = run_reid_candidates(
            output_dir=out1, config=config, contain_root=RUNTIME_ROOT, in_memory_bundle=same
        )
        if not r1.accepted:
            result.err(f"same-appearance run failed: {r1.error_code}")
        else:
            if r1.summary["counts"].get("candidate_link_count", 0) < 1:
                result.err("expected candidate link for same appearance")
            for er in r1.summary["evidence_rows"]:
                st, _ = decide_assignment_status([er], policy=policy)
                if st != "candidate":
                    result.err(f"appearance-only must be candidate, got {st}")
                if er["reliability_tier"] in {"strong", "manual_verified"}:
                    result.err("appearance tier too strong", integrity=True)

        # Different appearance lower similarity
        diff = fixture_different_appearance()
        ps, _, _ = build_profiles_from_bundle(
            bundle=same,
            config=config,
            config_fingerprint=result.extras["config_fingerprint"],
        )
        pd, _, _ = build_profiles_from_bundle(
            bundle=diff,
            config=config,
            config_fingerprint=result.extras["config_fingerprint"],
        )
        if cosine_similarity(ps[0].embedding, ps[1].embedding) <= cosine_similarity(
            pd[0].embedding, pd[1].embedding
        ):
            result.err("same-appearance similarity should exceed different-appearance")

        # Temporal overlap reject
        ov = fixture_temporal_overlap()
        r_ov = run_reid_candidates(
            output_dir=session / "overlap",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=ov,
        )
        if not any(
            "TEMPORAL_OVERLAP_FORBIDDEN" in m.reason_codes for m in r_ov.summary.get("matches", [])
        ):
            result.err("temporal overlap must reject")

        # Cross-video / human-ball
        xv = fixture_cross_video_reject()
        r_xv = run_reid_candidates(
            output_dir=session / "xvideo",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=xv,
        )
        if not any(
            "CROSS_VIDEO_AUTO_LINK_FORBIDDEN" in m.reason_codes
            for m in r_xv.summary.get("matches", [])
        ):
            result.err("cross-video must reject")
        hb = fixture_human_ball_reject()
        r_hb = run_reid_candidates(
            output_dir=session / "ball",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=hb,
        )
        if not any(
            "HUMAN_BALL_LINK_FORBIDDEN" in m.reason_codes for m in r_hb.summary.get("matches", [])
        ):
            result.err("human-ball must reject")

        # Insufficient
        single = fixture_single_crop_insufficient()
        p_single, _, stats = build_profiles_from_bundle(
            bundle=single,
            config=config,
            config_fingerprint=result.extras["config_fingerprint"],
        )
        if p_single[0].status != "insufficient_appearance_evidence":
            result.err("single crop must be insufficient_appearance_evidence")

        # No crop persistence
        if list((session / "same").glob("**/*.png")) or list((session / "same").glob("**/*.jpg")):
            result.err("crops must not be persisted", integrity=True)

        # Atomic no-overwrite
        r_again = run_reid_candidates(
            output_dir=session / "same",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=same,
        )
        if r_again.error_code != "OVERWRITE_FORBIDDEN":
            result.err("overwrite must be forbidden")

        # Failure cleanup
        r_fail = run_reid_candidates(
            output_dir=session / "fail",
            config=config,
            contain_root=RUNTIME_ROOT,
            in_memory_bundle=same,
            inject_failure=True,
        )
        if r_fail.accepted:
            result.err("injected failure should not accept")
        if (session / "fail" / "identity_evidence.parquet").exists():
            result.err("failure cleanup incomplete", integrity=True)

        result.extras["evaluation_status"] = NOT_EVALUATED_APPEARANCE_REID
        result.extras["insufficient_check"] = stats
        result.extras["auto_confirm"] = False
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
        default="configs/identity/appearance_reid_baseline.yaml",
        help="Appearance ReID baseline config path",
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
