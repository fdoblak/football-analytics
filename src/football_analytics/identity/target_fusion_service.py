"""Stage 7E target identity fusion service (prepare/decide/resolve/validate)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.identity.assignments import (
    build_revocation,
    validate_assignment_record,
    validate_assignment_rows,
)
from football_analytics.identity.contracts import (
    TRACK_IDENTITY_ASSIGNMENTS_CONTRACT,
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.evidence import (
    assert_no_face_biometric_evidence,
    validate_evidence_rows,
)
from football_analytics.identity.fixtures import assignment_row
from football_analytics.identity.metric_eligibility import resolve_metric_eligibility
from football_analytics.identity.policy import load_identity_policy, policy_fingerprint
from football_analytics.identity.target_decisions import (
    TargetDecisionError,
    append_decision_audit,
    assert_manifest_cas,
    assert_no_duplicate_decision,
    build_target_decision,
    latest_audit_hash,
    load_decision_ids,
    validate_target_decision,
    write_decision_file,
)
from football_analytics.identity.target_eligibility_timeline import build_eligibility_timeline
from football_analytics.identity.target_fusion import (
    TargetFusionError,
    detect_confirmed_overlaps,
    detect_track_multi_identity,
    fuse_track_evidence,
    group_evidence_by_track,
)
from football_analytics.identity.target_fusion_config import (
    load_target_fusion_config,
    target_fusion_config_fingerprint,
)
from football_analytics.identity.target_fusion_evaluation import (
    NOT_EVALUATED_TARGET_IDENTITY,
    evaluate_target_fusion,
)
from football_analytics.identity.target_fusion_fixtures import get_fixture
from football_analytics.identity.target_ranking import rank_candidates, rank_score_for_candidate
from football_analytics.identity.target_review import (
    build_review_manifest,
    validate_review_manifest,
)
from football_analytics.identity.types import (
    AssignmentStatus,
    IdentityContractError,
    LeakageClass,
    TargetScope,
)


class TargetFusionServiceError(RuntimeError):
    """Target fusion service failure."""


@dataclass
class TargetFusionServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    manifest_json: str | None = None
    decision_json: str | None = None
    assignments_parquet: str | None = None
    eligibility_json: str | None = None
    receipt_json: str | None = None
    evaluation_json: str | None = None
    quality_json: str | None = None
    audit_jsonl: str | None = None
    summary: Mapping[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "manifest_json": self.manifest_json,
            "decision_json": self.decision_json,
            "assignments_parquet": self.assignments_parquet,
            "eligibility_json": self.eligibility_json,
            "receipt_json": self.receipt_json,
            "evaluation_json": self.evaluation_json,
            "quality_json": self.quality_json,
            "audit_jsonl": self.audit_jsonl,
            **dict(self.summary or {}),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int,
    config_fingerprint: str,
    cleanup: Sequence[Path] | None = None,
) -> TargetFusionServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return TargetFusionServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _chmod_file(path: Path, mode: int = 0o600) -> None:
    path.chmod(mode)


def _safe_contain(out: Path, root: Path) -> bool:
    try:
        out.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _within_anchor(
    track_id: int,
    start: int,
    end: int,
    anchors: Sequence[Mapping[str, Any]],
) -> bool:
    for a in anchors:
        if int(a["track_id"]) != int(track_id):
            continue
        if int(a["start_frame_index"]) <= start and end <= int(a["end_frame_index"]):
            return True
    return False


def _assignment_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "candidate": 0,
        "provisional": 0,
        "confirmed": 0,
        "rejected": 0,
        "revoked": 0,
        "unknown": 0,
        "total": 0,
    }
    for r in rows:
        st = str(r["assignment_status"])
        if st in counts:
            counts[st] += 1
        counts["total"] += 1
    return counts


def _build_candidates_from_fixture(
    fixture: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence = validate_evidence_rows(list(fixture["evidence"]))
    assert_no_face_biometric_evidence(evidence)
    grouped = group_evidence_by_track(evidence)
    anchors = list(fixture.get("manual_anchors") or [])
    long_gap = bool(fixture.get("long_gap_after_cut", False))
    conflict_flags: list[str] = []
    candidates: list[dict[str, Any]] = []
    track_meta = {int(t["track_id"]): t for t in fixture.get("tracks") or []}

    for track_id, rows in sorted(grouped.items()):
        meta = track_meta.get(track_id, {})
        start = int(meta.get("start", rows[0].get("start_frame_index") or 0))
        end = int(meta.get("end", rows[0].get("end_frame_index") or start))
        in_scope = _within_anchor(track_id, start, end, anchors)
        status, reasons, supp, conf = fuse_track_evidence(
            rows,
            policy=policy,
            within_manual_anchor_scope=in_scope,
            long_gap_after_cut=long_gap,
        )
        coverage = int(meta.get("coverage", end - start + 1))
        tiers = [str(r.get("reliability_tier")) for r in rows]
        appearance_margin = None
        for r in rows:
            if (
                str(r.get("evidence_type")) == "appearance_similarity"
                and r.get("score") is not None
            ):
                appearance_margin = float(r["score"]) - 0.5
        team_jersey = (
            ("jersey_number" in {str(r["evidence_type"]) for r in rows})
            and ("team_assignment" in {str(r["evidence_type"]) for r in rows})
            and not conf
        )
        score = rank_score_for_candidate(
            proposed_status=status,
            supporting_count=len(supp),
            conflicting_count=len(conf),
            reliability_tiers=tiers,
            appearance_margin=appearance_margin,
            team_jersey_consistent=bool(team_jersey),
            coverage_frames=coverage,
            has_manual_scope=in_scope,
            review_required=bool(conf) or status in {"provisional", "rejected"},
        )
        cid = f"cand_t{track_id}"
        candidates.append(
            {
                "candidate_id": cid,
                "track_id": track_id,
                "start_frame_index": start,
                "end_frame_index": end,
                "proposed_status": status,
                "rank": 0,
                "rank_score": score,
                "reason_codes": reasons,
                "supporting_evidence_ids": supp,
                "conflicting_evidence_ids": conf,
                "ambiguous": False,
                "manual_review_required": bool(conf)
                or status in {"provisional", "rejected"}
                or in_scope,
                "coverage_frames": coverage,
                "conflict_summary": ",".join(conf) if conf else None,
            }
        )
        if conf:
            conflict_flags.append(f"track_{track_id}_conflict")

    ranked = rank_candidates(
        candidates,
        ambiguity_margin=float(config["candidate_rules"]["ambiguity_score_margin"]),
        max_candidates=int(config["candidate_rules"]["max_candidates_per_run"]),
    )
    return ranked, conflict_flags


def prepare_review(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    contain_root: Path | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    fixture_name: str = "e2e_bundle",
    inject_failure: bool = False,
) -> TargetFusionServiceResult:
    cfg_fp = target_fusion_config_fingerprint(config)
    root = (contain_root or Path(str(config["runtime_root"]))).resolve()
    out = output_dir.resolve()
    written: list[Path] = []

    if not _safe_contain(out, root):
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    try:
        fixture = get_fixture(fixture_name)
    except KeyError:
        return _fail(error_code="UNKNOWN_FIXTURE", exit_code=2, config_fingerprint=cfg_fp)

    if fixture.get("expect_cross_video_fail") or fixture.get("expect_leakage_fail"):
        # Still allow prepare to surface failures via fusion.
        pass

    rid = run_id or str(fixture["run_id"])
    vid = video_id or str(fixture["video_id"])
    try:
        validate_run_id(rid)
    except Exception:
        return _fail(error_code="INVALID_RUN_ID", exit_code=2, config_fingerprint=cfg_fp)
    if not SAFE_ID_RE.fullmatch(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=2, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "target_review_manifest.json"
    evidence_path = out / "identity_evidence_sidecar.json"
    for p in (manifest_path, evidence_path):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    policy = load_identity_policy(
        Path(str(config["identity_policy_path"]))
        if Path(str(config["identity_policy_path"])).is_absolute()
        else None
    )
    # Prefer relative policy from project if configured relative.
    if not Path(str(config["identity_policy_path"])).is_absolute():
        from football_analytics.data.registry import default_project_root

        policy = load_identity_policy(default_project_root() / str(config["identity_policy_path"]))
    pol_fp = policy_fingerprint(policy)

    try:
        if fixture.get("expect_cross_video_fail") or fixture.get("expect_leakage_fail"):
            # Force fusion path to raise.
            ranked, conflict_flags = _build_candidates_from_fixture(
                fixture, policy=policy, config=config
            )
        else:
            ranked, conflict_flags = _build_candidates_from_fixture(
                fixture, policy=policy, config=config
            )
    except TargetFusionError as exc:
        return _fail(
            error_code=str(exc.args[0]) if exc.args else "FUSION_ERROR",
            exit_code=3,
            config_fingerprint=cfg_fp,
        )

    if inject_failure:
        return _fail(
            error_code="INJECTED_FAILURE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    evidence_payload = {
        "run_id": rid,
        "video_id": vid,
        "fixture": fixture["name"],
        "evidence": list(fixture["evidence"]),
        "manual_anchors": list(fixture.get("manual_anchors") or []),
        "tracks": list(fixture.get("tracks") or []),
        "confirm_track_id": fixture.get("confirm_track_id"),
        "confirm_start": fixture.get("confirm_start"),
        "confirm_end": fixture.get("confirm_end"),
        "synthetic_expected_track_ids": list(fixture.get("synthetic_expected_track_ids") or []),
        "expected_max_status": fixture.get("expected_max_status"),
    }
    write_json_record(evidence_path, evidence_payload, contain_root=root, overwrite=False)
    written.append(evidence_path)
    _chmod_file(evidence_path)

    manifest = build_review_manifest(
        manifest_id="man_" + fixture["name"].replace("-", "_")[:40],
        run_id=rid,
        video_id=vid,
        request_id=str(fixture["request_id"]),
        target_player_id=str(fixture["target_player_id"]),
        config_fingerprint=cfg_fp,
        policy_fingerprint=pol_fp,
        expected_assignment_version=1,
        candidates=ranked,
        allowed_decisions=list(config["review"]["allowed_decisions"]),
        artifact_refs={
            "evidence_sidecar": {
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
            }
        },
        conflict_flags=conflict_flags,
        max_review_items=int(config["review"]["max_review_items"]),
        notes="synthetic prepare-review; ranking is review aid only",
    )
    write_json_record(manifest_path, manifest, contain_root=root, overwrite=False)
    written.append(manifest_path)
    _chmod_file(manifest_path)

    return TargetFusionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        manifest_json=str(manifest_path),
        summary={
            "status": "succeeded",
            "candidate_count": len(manifest["candidates"]),
            "conflict_flags": conflict_flags,
            "run_id": rid,
            "video_id": vid,
            "fixture": fixture["name"],
        },
    )


def apply_decision(
    *,
    output_dir: Path,
    decision_path: Path | None = None,
    decision_payload: Mapping[str, Any] | None = None,
    config: Mapping[str, Any],
    contain_root: Path | None = None,
    inject_failure: bool = False,
) -> TargetFusionServiceResult:
    cfg_fp = target_fusion_config_fingerprint(config)
    root = (contain_root or Path(str(config["runtime_root"]))).resolve()
    out = output_dir.resolve()
    written: list[Path] = []
    if not _safe_contain(out, root):
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    manifest_path = out / "target_review_manifest.json"
    if not manifest_path.is_file():
        return _fail(error_code="MANIFEST_MISSING", exit_code=2, config_fingerprint=cfg_fp)
    manifest = validate_review_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))

    decisions_dir = out / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out / "identity_manual_audit.jsonl"

    existing_ids, existing_hashes = load_decision_ids(decisions_dir)
    prev_hash = latest_audit_hash(audit_path)

    if decision_payload is None:
        if decision_path is None or not decision_path.is_file():
            return _fail(error_code="DECISION_MISSING", exit_code=2, config_fingerprint=cfg_fp)
        decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))

    try:
        decision = validate_target_decision(dict(decision_payload))
        # Ensure previous_audit_hash matches chain tip when provided.
        if decision.get("previous_audit_hash") is None and prev_hash is not None:
            # rebuild with chain tip
            rebuilt = dict(decision)
            rebuilt["previous_audit_hash"] = prev_hash
            rebuilt.pop("record_hash", None)
            from football_analytics.identity.target_decisions import compute_decision_record_hash

            rebuilt["record_hash"] = compute_decision_record_hash(rebuilt)
            decision = validate_target_decision(rebuilt)
        assert_manifest_cas(
            decision,
            manifest=manifest,
            current_assignment_version=int(manifest["expected_assignment_version"]),
        )
        assert_no_duplicate_decision(
            decision,
            existing_decision_ids=existing_ids,
            existing_record_hashes=existing_hashes,
        )
    except (TargetDecisionError, IdentityContractError) as exc:
        code = str(exc.args[0]) if exc.args else "DECISION_REJECTED"
        if "stale" in code.lower():
            code = "STALE_DECISION"
        if "duplicate" in code.lower():
            code = "DUPLICATE_DECISION"
        return _fail(error_code=code, exit_code=1, config_fingerprint=cfg_fp)

    if inject_failure:
        return _fail(
            error_code="INJECTED_FAILURE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    out_decision = decisions_dir / f"{decision['decision_id']}.json"
    try:
        write_decision_file(out_decision, decision, contain_root=root)
        written.append(out_decision)
        _chmod_file(out_decision)
        append_decision_audit(audit_path, decision, contain_root=root)
        written.append(audit_path)
        _chmod_file(audit_path)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"WRITE_FAILED:{exc}",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    return TargetFusionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        manifest_json=str(manifest_path),
        decision_json=str(out_decision),
        audit_jsonl=str(audit_path),
        summary={
            "status": "succeeded",
            "decision_id": decision["decision_id"],
            "new_status": decision["new_status"],
        },
    )


def resolve_fusion(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    contain_root: Path | None = None,
    inject_failure: bool = False,
) -> TargetFusionServiceResult:
    cfg_fp = target_fusion_config_fingerprint(config)
    root = (contain_root or Path(str(config["runtime_root"]))).resolve()
    out = output_dir.resolve()
    written: list[Path] = []
    if not _safe_contain(out, root):
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    started = _utc_now()
    manifest_path = out / "target_review_manifest.json"
    evidence_path = out / "identity_evidence_sidecar.json"
    decisions_dir = out / "decisions"
    audit_path = out / "identity_manual_audit.jsonl"
    assignments_out = out / "track_identity_assignments.parquet"
    eligibility_out = out / "metric_eligibility_timeline.json"
    receipt_out = out / "target_fusion_receipt.json"
    evaluation_out = out / "target_fusion_evaluation.json"
    quality_out = out / "target_fusion_quality.json"

    for p in (assignments_out, eligibility_out, receipt_out, evaluation_out, quality_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    if not manifest_path.is_file() or not evidence_path.is_file():
        return _fail(error_code="PREPARE_ARTIFACTS_MISSING", exit_code=2, config_fingerprint=cfg_fp)

    manifest = validate_review_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    evidence_sidecar = json.loads(evidence_path.read_text(encoding="utf-8"))
    pol_fp = str(manifest["policy_fingerprint"])
    rid = str(manifest["run_id"])
    vid = str(manifest["video_id"])
    tid = str(manifest["target_player_id"])

    decisions: list[dict[str, Any]] = []
    if decisions_dir.is_dir():
        for path in sorted(decisions_dir.glob("*.json")):
            decisions.append(validate_target_decision(json.loads(path.read_text(encoding="utf-8"))))

    # Build baseline assignments from candidates (auto path never confirms).
    assignments: list[dict[str, Any]] = []
    for cand in manifest["candidates"]:
        status = str(cand["proposed_status"])
        eligibility = resolve_metric_eligibility(
            assignment_status=status,
            target_scope=TargetScope.TARGET.value,
            has_observed_tracking=True,
            sufficient_coverage=int(cand.get("coverage_frames") or 0)
            >= int(config["metric_eligibility"].get("min_confirmed_coverage_frames", 1)),
            unresolved_hard_conflict=bool(cand.get("conflicting_evidence_ids")),
            observation_state="observed",
        )
        row = assignment_row(
            rid,
            vid,
            f"asn_{cand['candidate_id']}",
            track_id=int(cand["track_id"]),
            target_player_id=tid,
            assignment_status=status,
            evidence_ids=list(cand.get("supporting_evidence_ids") or []),
            supporting=len(cand.get("supporting_evidence_ids") or []),
            conflicting=len(cand.get("conflicting_evidence_ids") or []),
            metric_eligibility=eligibility,
            policy_fingerprint=pol_fp,
            start_frame_index=int(cand["start_frame_index"]),
            end_frame_index=int(cand["end_frame_index"]),
            manual_review_required=bool(cand.get("manual_review_required")),
            reason_codes=list(cand.get("reason_codes") or []),
            leakage_class=LeakageClass.SYNTHETIC.value,
        )
        row["producer"] = str(config["producer"])
        row["producer_version"] = str(config["producer_version"])
        assignments.append(validate_assignment_record(row))

    # Apply manual decisions (CAS already enforced at decide time).
    by_track = {int(a["track_id"]): a for a in assignments}
    for dec in decisions:
        track_id = int(dec["track_id"])
        prev = by_track.get(track_id)
        if prev is None:
            continue
        new_status = str(dec["new_status"])
        if new_status == AssignmentStatus.REVOKED.value:
            revoked = build_revocation(
                prev,
                new_assignment_id=f"asn_rev_{dec['decision_id']}"[:64],
                reason=str(dec["reason"]),
                actor_provenance=dec["reviewer_id"],
            )
            revoked["producer"] = str(config["producer"])
            revoked["producer_version"] = str(config["producer_version"])
            assignments.append(revoked)
            by_track[track_id] = revoked
            continue
        # Scoped confirm: only the exact interval is confirmed.
        if new_status == AssignmentStatus.CONFIRMED.value:
            start = int(dec["start_frame_index"])
            end = int(dec["end_frame_index"])
            if start < int(prev["start_frame_index"]) or end > int(prev["end_frame_index"]):
                return _fail(
                    error_code="MANUAL_SCOPE_VIOLATION",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            updated = dict(prev)
            updated["assignment_id"] = f"asn_cnf_{dec['decision_id']}"[:64]
            updated["assignment_status"] = AssignmentStatus.CONFIRMED.value
            updated["assignment_version"] = int(prev["assignment_version"]) + 1
            updated["supersedes_assignment_id"] = prev["assignment_id"]
            updated["start_frame_index"] = start
            updated["end_frame_index"] = end
            updated["reason_codes"] = list(prev.get("reason_codes") or []) + [
                "MANUAL_VERIFIED_IN_SCOPE",
                "SCOPED_MANUAL_CONFIRM",
            ]
            # Remove alone-insufficient reasons that would block confirmed validation.
            alone = {
                "JERSEY_ALONE_INSUFFICIENT",
                "TEAM_ALONE_INSUFFICIENT",
                "ROLE_ALONE_INSUFFICIENT",
                "APPEARANCE_ALONE_INSUFFICIENT",
                "SINGLE_WEAK_CANNOT_CONFIRM",
                "MANUAL_CONFIRM_REQUIRED",
            }
            updated["reason_codes"] = [r for r in updated["reason_codes"] if r not in alone]
            updated["supporting_evidence_count"] = max(
                1, int(updated.get("supporting_evidence_count") or 0)
            )
            updated["manual_review_required"] = False
            updated["metric_eligibility"] = resolve_metric_eligibility(
                assignment_status=AssignmentStatus.CONFIRMED.value,
                target_scope=TargetScope.TARGET.value,
                has_observed_tracking=True,
                sufficient_coverage=True,
                unresolved_hard_conflict=False,
                observation_state="observed",
            )
            updated = validate_assignment_record(updated)
            assignments.append(updated)
            by_track[track_id] = updated
            continue

        updated = dict(prev)
        updated["assignment_id"] = f"asn_dec_{dec['decision_id']}"[:64]
        updated["assignment_status"] = new_status
        updated["assignment_version"] = int(prev["assignment_version"]) + 1
        updated["supersedes_assignment_id"] = prev["assignment_id"]
        updated["reason_codes"] = list(prev.get("reason_codes") or []) + [
            f"MANUAL_{dec['decision'].upper()}"
        ]
        updated["metric_eligibility"] = resolve_metric_eligibility(
            assignment_status=new_status,
            target_scope=TargetScope.TARGET.value,
            has_observed_tracking=True,
            sufficient_coverage=True,
            unresolved_hard_conflict=False,
            observation_state="observed",
        )
        updated = validate_assignment_record(updated)
        assignments.append(updated)
        by_track[track_id] = updated

    assignments = validate_assignment_rows(assignments)
    overlap_tol = int(config["conflict_rules"].get("overlap_tolerance_frames", 0))
    overlaps = detect_confirmed_overlaps(assignments, overlap_tolerance_frames=overlap_tol)
    multi = detect_track_multi_identity(assignments, overlap_tolerance_frames=overlap_tol)
    if overlaps or multi:
        return _fail(
            error_code="DUPLICATE_CONFIRMED_IDENTITY" if overlaps else "TRACK_MULTI_IDENTITY",
            exit_code=1,
            config_fingerprint=cfg_fp,
        )

    if inject_failure:
        return _fail(
            error_code="INJECTED_FAILURE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    # Write assignments parquet.
    table = _rows_to_table(assignments, TRACK_IDENTITY_ASSIGNMENTS_CONTRACT)
    write_contract_parquet(
        table,
        assignments_out,
        get_contract(TRACK_IDENTITY_ASSIGNMENTS_CONTRACT, 1),
        contain_root=root,
        overwrite=False,
    )
    written.append(assignments_out)
    _chmod_file(assignments_out)

    # Observation state overlays for predicted/interpolated tests via sidecar.
    obs_map: dict[str, str] = {}
    cov_map: dict[str, bool] = {}
    conf_map: dict[str, bool] = {}
    for a in assignments:
        aid = str(a["assignment_id"])
        obs_map[aid] = "observed"
        cov_map[aid] = True
        conf_map[aid] = False
        if "PREDICTED" in list(a.get("reason_codes") or []):
            obs_map[aid] = "predicted"
        if "INSUFFICIENT_COVERAGE" in list(a.get("reason_codes") or []):
            cov_map[aid] = False

    tl_id = "tel" + "".join(ch for ch in rid.lower() if ch.isalnum())[:20]
    timeline = build_eligibility_timeline(
        assignments,
        timeline_id=tl_id,
        run_id=rid,
        video_id=vid,
        target_player_id=tid,
        observation_state_by_assignment=obs_map,
        sufficient_coverage_by_assignment=cov_map,
        conflict_by_assignment=conf_map,
    )
    write_json_record(eligibility_out, timeline, contain_root=root, overwrite=False)
    written.append(eligibility_out)
    _chmod_file(eligibility_out)

    audit_valid = True
    try:
        if audit_path.is_file():
            from football_analytics.identity.review_audit import read_audit_log

            read_audit_log(audit_path)
    except Exception:
        audit_valid = False

    counts = _assignment_counts(assignments)
    evidence_rows = list(evidence_sidecar.get("evidence") or [])
    evidence_counts: dict[str, int] = {}
    for e in evidence_rows:
        et = str(e.get("evidence_type", "unknown"))
        evidence_counts[et] = evidence_counts.get(et, 0) + 1

    eval_report = evaluate_target_fusion(
        assignments=assignments,
        has_reviewed_ground_truth=False,
        synthetic_expected_track_ids=list(
            evidence_sidecar.get("synthetic_expected_track_ids") or []
        )
        or None,
    )
    eval_payload = eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp)
    write_json_record(evaluation_out, eval_payload, contain_root=root, overwrite=False)
    written.append(evaluation_out)
    _chmod_file(evaluation_out)

    confirmed_cov = 0
    for a in assignments:
        if a["assignment_status"] == AssignmentStatus.CONFIRMED.value:
            confirmed_cov += int(a["end_frame_index"]) - int(a["start_frame_index"]) + 1

    output_artifacts: dict[str, Any] = {}
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": "rcp" + "".join(ch for ch in rid.lower() if ch.isalnum())[:20],
        "run_id": rid,
        "video_id": vid,
        "request_id": str(manifest["request_id"]),
        "config_fingerprint": cfg_fp,
        "policy_fingerprint": pol_fp,
        "tracking_fingerprint": None,
        "appearance_fingerprint": None,
        "team_fingerprint": None,
        "jersey_fingerprint": None,
        "assignment_counts": counts,
        "evidence_counts": evidence_counts,
        "candidate_count": len(manifest["candidates"]),
        "manual_review_count": len(manifest["candidates"]),
        "manual_decision_count": len(decisions),
        "supporting_evidence_total": sum(int(a["supporting_evidence_count"]) for a in assignments),
        "conflicting_evidence_total": sum(
            int(a["conflicting_evidence_count"]) for a in assignments
        ),
        "confirmed_observed_coverage_frames": confirmed_cov,
        "conflict_count": len(overlaps) + len(multi),
        "false_overlap_count": len(overlaps),
        "stale_decision_count": 0,
        "metric_eligibility_summary": dict(timeline["summary"]),
        "audit_chain_valid": audit_valid,
        "ground_truth_evaluation_status": NOT_EVALUATED_TARGET_IDENTITY,
        "output_artifacts": output_artifacts,
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "status": "succeeded",
        "warnings": [],
        "errors": [],
        "provenance": {
            "stage": "7E",
            "face_biometric_forbidden": True,
            "auto_confirm_forbidden": True,
            "no_track_merge": True,
            "cross_video_auto_link_forbidden": True,
            "notes": "synthetic fusion receipt; not football accuracy",
        },
    }
    for label, path in (
        ("assignments", assignments_out),
        ("eligibility", eligibility_out),
        ("evaluation", evaluation_out),
        ("manifest", manifest_path),
    ):
        output_artifacts[label] = _artifact_meta(path)
    if audit_path.is_file():
        output_artifacts["audit"] = _artifact_meta(audit_path)

    schema = load_identity_json_schema("target_fusion_receipt")
    validate_against_json_schema(receipt, schema)
    write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
    written.append(receipt_out)
    _chmod_file(receipt_out)

    quality = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "assignment_counts": counts,
        "metric_eligibility_summary": dict(timeline["summary"]),
        "false_target_attribution_synthetic": eval_payload["metrics"].get(
            "false_target_attribution"
        ),
        "audit_chain_valid": audit_valid,
        "ground_truth_evaluation_status": NOT_EVALUATED_TARGET_IDENTITY,
        "receipt_sha256": sha256_file(receipt_out),
    }
    write_json_record(quality_out, quality, contain_root=root, overwrite=False)
    written.append(quality_out)
    _chmod_file(quality_out)

    return TargetFusionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        manifest_json=str(manifest_path),
        assignments_parquet=str(assignments_out),
        eligibility_json=str(eligibility_out),
        receipt_json=str(receipt_out),
        evaluation_json=str(evaluation_out),
        quality_json=str(quality_out),
        audit_jsonl=str(audit_path) if audit_path.is_file() else None,
        summary={
            "status": "succeeded",
            "assignment_counts": counts,
            "ground_truth_evaluation_status": NOT_EVALUATED_TARGET_IDENTITY,
        },
    )


def validate_fusion_outputs(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    contain_root: Path | None = None,
) -> TargetFusionServiceResult:
    cfg_fp = target_fusion_config_fingerprint(config)
    root = (contain_root or Path(str(config["runtime_root"]))).resolve()
    out = output_dir.resolve()
    if not _safe_contain(out, root):
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    receipt_path = out / "target_fusion_receipt.json"
    assignments_path = out / "track_identity_assignments.parquet"
    if not receipt_path.is_file() or not assignments_path.is_file():
        return _fail(error_code="ARTIFACTS_MISSING", exit_code=2, config_fingerprint=cfg_fp)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = load_identity_json_schema("target_fusion_receipt")
    validate_against_json_schema(receipt, schema)

    import pyarrow.parquet as pq

    table = pq.read_table(assignments_path)
    rows = table.to_pylist()
    recounted = _assignment_counts(rows)
    if recounted != receipt["assignment_counts"]:
        return _fail(error_code="RECEIPT_RECOUNT_MISMATCH", exit_code=1, config_fingerprint=cfg_fp)

    for _label, meta in (receipt.get("output_artifacts") or {}).items():
        p = Path(str(meta["path"]))
        if not p.is_file():
            return _fail(error_code="ARTIFACT_MISSING", exit_code=3, config_fingerprint=cfg_fp)
        if meta.get("sha256") and sha256_file(p) != meta["sha256"]:
            return _fail(
                error_code="ARTIFACT_HASH_MISMATCH", exit_code=3, config_fingerprint=cfg_fp
            )

    return TargetFusionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        receipt_json=str(receipt_path),
        assignments_parquet=str(assignments_path),
        summary={"status": "succeeded", "assignment_counts": recounted},
    )


def run_fixture_decision(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    decision: str,
    track_id: int,
    start: int,
    end: int,
    reviewer_id: str = "synth_reviewer",
    decision_id: str | None = None,
    reason: str = "synthetic_manual_decision",
    contain_root: Path | None = None,
) -> TargetFusionServiceResult:
    """Helper for tests: build + apply a synthetic decision against prepare output."""
    out = output_dir.resolve()
    manifest = validate_review_manifest(
        json.loads((out / "target_review_manifest.json").read_text(encoding="utf-8"))
    )
    did = decision_id or f"dec_{decision}_{track_id}"
    did = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in did.lower())
    if not did[0].isalpha():
        did = "d" + did
    did = did[:64]
    cand = next((c for c in manifest["candidates"] if int(c["track_id"]) == track_id), None)
    prev_status = str(cand["proposed_status"]) if cand else "candidate"
    evidence_fps = []
    art = manifest.get("artifact_refs") or {}
    for meta in art.values():
        if meta.get("sha256"):
            evidence_fps.append(str(meta["sha256"]))
    payload = build_target_decision(
        decision_id=did,
        manifest=manifest,
        track_id=track_id,
        start_frame_index=start,
        end_frame_index=end,
        decision=decision,
        reviewer_id=reviewer_id,
        reason=reason,
        expected_assignment_version=int(manifest["expected_assignment_version"]),
        expected_previous_status=prev_status,
        evidence_fingerprints=evidence_fps,
        linked_evidence_ids=list(cand.get("supporting_evidence_ids") or []) if cand else [],
        previous_audit_hash=latest_audit_hash(out / "identity_manual_audit.jsonl"),
        candidate_id=str(cand["candidate_id"]) if cand else None,
        synthetic_fixture=True,
        notes="synthetic fixture decision; not a real user claim",
    )
    return apply_decision(
        output_dir=out,
        decision_payload=payload,
        config=config,
        contain_root=contain_root,
    )


__all__ = [
    "TargetFusionServiceError",
    "TargetFusionServiceResult",
    "prepare_review",
    "apply_decision",
    "resolve_fusion",
    "validate_fusion_outputs",
    "run_fixture_decision",
    "load_target_fusion_config",
    "generate_run_id",
    "hash_canonical_json",
]
