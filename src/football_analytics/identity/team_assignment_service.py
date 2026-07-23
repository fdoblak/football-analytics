"""Stage 7C team appearance clustering + team_assignments service."""

from __future__ import annotations

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
from football_analytics.identity.appearance_reid_config import (
    appearance_reid_config_fingerprint,
    load_appearance_reid_config,
)
from football_analytics.identity.appearance_reid_service import build_profiles_from_bundle
from football_analytics.identity.contracts import (
    IDENTITY_EVIDENCE_CONTRACT,
    TEAM_ASSIGNMENTS_CONTRACT,
)
from football_analytics.identity.evidence import validate_evidence_rows
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
from football_analytics.identity.team_assignment import (
    PRODUCER,
    PRODUCER_VERSION,
    build_team_decisions,
    decisions_to_assignment_rows,
    decisions_to_evidence_rows,
)
from football_analytics.identity.team_assignment_config import team_assignment_config_fingerprint
from football_analytics.identity.team_assignment_evaluation import (
    NOT_EVALUATED_TEAM_ASSIGNMENT,
    evaluate_team_assignment,
    permutation_matched_accuracy,
)
from football_analytics.identity.team_clustering import (
    TeamClusterModel,
    align_centroids_across_shots,
    collect_seed_tracks,
    fit_two_team_clusters,
)
from football_analytics.identity.types import LeakageClass


class TeamAssignmentServiceError(RuntimeError):
    """Team assignment service failure."""


@dataclass
class TeamAssignmentServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    team_assignments_parquet: str | None
    evidence_parquet: str | None
    cluster_provenance_json: str | None
    receipt_json: str | None
    evaluation_json: str | None
    quality_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "team_assignments_parquet": self.team_assignments_parquet,
            "evidence_parquet": self.evidence_parquet,
            "cluster_provenance_json": self.cluster_provenance_json,
            "receipt_json": self.receipt_json,
            "evaluation_json": self.evaluation_json,
            "quality_json": self.quality_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int,
    config_fingerprint: str,
    cleanup: Sequence[Path] | None = None,
) -> TeamAssignmentServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return TeamAssignmentServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        team_assignments_parquet=None,
        evidence_parquet=None,
        cluster_provenance_json=None,
        receipt_json=None,
        evaluation_json=None,
        quality_json=None,
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


def _resolve_appearance_config(
    team_config: Mapping[str, Any], *, repo_root: Path | None = None
) -> tuple[Mapping[str, Any], str]:
    root = repo_root or Path(__file__).resolve().parents[3]
    path = Path(str(team_config["appearance_reid_config_path"]))
    if not path.is_absolute():
        path = root / path
    app_cfg = load_appearance_reid_config(path)
    return app_cfg, appearance_reid_config_fingerprint(app_cfg)


def _fit_model_for_bundle(
    profiles: Sequence[Any],
    *,
    config: Mapping[str, Any],
    role_by_track: Mapping[int, str],
    shot_by_track: Mapping[int, str] | None,
) -> tuple[TeamClusterModel, list[dict[str, Any]], dict[str, Any]]:
    seeds, rejected = collect_seed_tracks(
        profiles,
        config=config,
        role_by_track=role_by_track,
        shot_by_track=shot_by_track,
    )
    # Optional per-shot clustering then alignment
    if shot_by_track and config["clustering"]["cross_shot_alignment_enabled"]:
        shots = sorted({s for s in shot_by_track.values() if s})
        if len(shots) >= 2:
            models: dict[str, TeamClusterModel] = {}
            for shot in shots:
                shot_seeds = [s for s in seeds if s.shot_id == shot]
                models[shot] = fit_two_team_clusters(shot_seeds, config=config)
            # Use first OK model as reference; align others for provenance note.
            ref = next((models[s] for s in shots if models[s].status == "ok"), None)
            if ref is not None:
                aligned = ref
                for shot in shots[1:]:
                    if models[shot].status == "ok":
                        aligned = align_centroids_across_shots(models[shot], ref, config=config)
                # Prefer global fit when enough seeds; keep alignment provenance.
                global_model = fit_two_team_clusters(seeds, config=config)
                if global_model.status == "ok":
                    prov = dict(global_model.provenance)
                    prov["cross_shot"] = {
                        s: {
                            "status": models[s].status,
                            "reasons": list(models[s].reason_codes),
                            "alignment": dict(aligned.provenance).get("cross_shot_alignment"),
                        }
                        for s in shots
                    }
                    global_model = TeamClusterModel(
                        status=global_model.status,
                        reason_codes=global_model.reason_codes,
                        centroids=global_model.centroids,
                        centroid_fingerprints=global_model.centroid_fingerprints,
                        seed_track_ids=global_model.seed_track_ids,
                        cluster_sizes=global_model.cluster_sizes,
                        separation=global_model.separation,
                        label_order=global_model.label_order,
                        provenance=prov,
                    )
                return global_model, rejected, {"seeds": len(seeds), "shots": shots}

    model = fit_two_team_clusters(seeds, config=config)
    return model, rejected, {"seeds": len(seeds)}


def run_team_classify(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    in_memory_bundle: Mapping[str, Any] | None = None,
    leakage_class: str | None = None,
    inject_failure: bool = False,
    repo_root: Path | None = None,
) -> TeamAssignmentServiceResult:
    """Cluster anonymous teams and write team_assignments + identity_evidence."""
    cfg_fp = team_assignment_config_fingerprint(config)
    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    out = Path(output_dir)
    written: list[Path] = []

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not str(out.resolve()).startswith(str(root.resolve())):
            return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    if in_memory_bundle is None:
        return _fail(error_code="INPUT_MISSING", exit_code=3, config_fingerprint=cfg_fp)

    # Cross-video auto transfer forbidden
    if (
        in_memory_bundle.get("cross_video_probe")
        and config["clustering"]["cross_video_auto_transfer"]
    ):
        return _fail(
            error_code="CROSS_VIDEO_AUTO_TRANSFER_FORBIDDEN",
            exit_code=1,
            config_fingerprint=cfg_fp,
        )

    teams_out = out / "team_assignments.parquet"
    evidence_out = out / "identity_evidence.parquet"
    provenance_out = out / "team_cluster_provenance.json"
    receipt_out = out / "team_assignment_receipt.json"
    quality_out = out / "team_assignment_quality.json"
    eval_out = out / "team_assignment_evaluation.json"
    for p in (teams_out, evidence_out, provenance_out, receipt_out, quality_out, eval_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    rid = run_id or str(in_memory_bundle.get("run_id") or generate_run_id())
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INVALID_RUN_ID", exit_code=3, config_fingerprint=cfg_fp)
    vid = video_id or str(in_memory_bundle.get("video_id") or "video_01")
    if not SAFE_ID_RE.match(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=3, config_fingerprint=cfg_fp)

    leak = leakage_class or str(
        in_memory_bundle.get("force_leakage_class") or LeakageClass.SYNTHETIC.value
    )
    started = _utc_now()

    try:
        if config["assignment"]["auto_target_confirmation"]:
            return _fail(
                error_code="AUTO_CONFIRM_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp
            )
        if config["assignment"]["auto_real_team_naming"]:
            return _fail(
                error_code="REAL_TEAM_NAMING_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp
            )
        if config["assignment"]["auto_home_away"]:
            return _fail(error_code="HOME_AWAY_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

        app_cfg, app_fp = _resolve_appearance_config(config, repo_root=repo_root)
        policy_path = Path(str(config["identity_policy_path"]))
        if not policy_path.is_absolute():
            policy_path = (repo_root or Path(__file__).resolve().parents[3]) / policy_path
        policy = load_identity_policy(policy_path)

        bundle = dict(in_memory_bundle)
        bundle["run_id"] = rid
        bundle["video_id"] = vid
        profiles, profile_rows, stats = build_profiles_from_bundle(
            bundle=bundle,
            config=app_cfg,
            config_fingerprint=app_fp,
            leakage_class=leak,
        )
        if len(profiles) > int(config["safety_limits"]["max_tracks_per_video"]):
            return _fail(error_code="MAX_TRACKS", exit_code=3, config_fingerprint=cfg_fp)

        role_by_track = dict(bundle.get("role_by_track") or {})
        shot_by_track = dict(bundle.get("shot_by_track") or {})
        prior = dict(bundle.get("prior_team_by_track") or {})

        model, rejected_seeds, seed_stats = _fit_model_for_bundle(
            profiles,
            config=config,
            role_by_track=role_by_track,
            shot_by_track=shot_by_track or None,
        )
        decisions, _ = build_team_decisions(
            profiles,
            config=config,
            role_by_track=role_by_track,
            model=model,
            prior_team_by_track=prior,
        )
        assignment_rows = decisions_to_assignment_rows(
            decisions, run_id=rid, video_id=vid, config=config
        )
        if len(assignment_rows) > int(config["safety_limits"]["max_assignments_per_run"]):
            return _fail(error_code="MAX_ASSIGNMENTS", exit_code=3, config_fingerprint=cfg_fp)

        cluster_fp = hash_canonical_json(
            {
                "centroids": {k: list(v) for k, v in model.centroids.items()},
                "fingerprints": dict(model.centroid_fingerprints),
                "status": model.status,
                "separation": model.separation,
            }
        )
        evidence_rows = decisions_to_evidence_rows(
            decisions,
            run_id=rid,
            video_id=vid,
            config_fingerprint=cfg_fp,
            cluster_fingerprint=cluster_fp,
            leakage_class=leak,
        )
        validate_evidence_rows(evidence_rows)

        # Team evidence alone must not auto-confirm identity.
        for er in evidence_rows:
            st, reasons = decide_assignment_status([er], policy=policy)
            if st in {"confirmed", "provisional"}:
                return _fail(
                    error_code="TEAM_ALONE_AUTO_CONFIRM",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            if "TEAM_ALONE_INSUFFICIENT" not in reasons and er["reliability_tier"] not in {
                "unavailable",
                "conflicting",
                "weak",
                "supporting",
            }:
                return _fail(
                    error_code="TEAM_EVIDENCE_POLICY",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            if str(er.get("reliability_tier")) in {"strong", "manual_verified"}:
                return _fail(
                    error_code="TEAM_TIER_TOO_STRONG",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )

        if inject_failure:
            raise TeamAssignmentServiceError("injected_failure")

        # Cross-video: never copy team ids across videos.
        if bundle.get("cross_video_probe"):
            # Explicit reject marker in quality; no silent transfer.
            pass

        ttable = _rows_to_table(assignment_rows, TEAM_ASSIGNMENTS_CONTRACT)
        write_contract_parquet(
            ttable,
            teams_out,
            get_contract(TEAM_ASSIGNMENTS_CONTRACT, 1),
            contain_root=root,
            overwrite=False,
        )
        _chmod_file(teams_out, int(config["output_policy"]["chmod_mode"]))
        written.append(teams_out)

        etable = _rows_to_table(list(evidence_rows), IDENTITY_EVIDENCE_CONTRACT)
        write_contract_parquet(
            etable,
            evidence_out,
            get_contract(IDENTITY_EVIDENCE_CONTRACT, 1),
            contain_root=root,
            overwrite=False,
        )
        _chmod_file(evidence_out, int(config["output_policy"]["chmod_mode"]))
        written.append(evidence_out)

        provenance = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "method_id": config["method_id"],
            "config_fingerprint": cfg_fp,
            "appearance_config_fingerprint": app_fp,
            "cluster_fingerprint": cluster_fp,
            "status": model.status,
            "reason_codes": list(model.reason_codes),
            "centroids": {k: list(v) for k, v in model.centroids.items()},
            "centroid_fingerprints": dict(model.centroid_fingerprints),
            "cluster_sizes": dict(model.cluster_sizes),
            "separation": model.separation,
            "label_order": list(model.label_order),
            "seed_track_ids": list(model.seed_track_ids),
            "rejected_seeds": rejected_seeds,
            "seed_stats": seed_stats,
            "cross_video_auto_transfer": False,
            "auto_real_team_naming": False,
            "auto_home_away": False,
            "auto_target_confirmation": False,
            "provenance": dict(model.provenance),
            "created_at_utc": _utc_now(),
        }
        if config["output_policy"]["write_cluster_provenance"]:
            write_json_record(provenance_out, provenance, overwrite=False)
            _chmod_file(provenance_out)
            written.append(provenance_out)

        # Synthetic diagnostic metrics only (never claim real accuracy).
        synth_metrics = None
        gt = bundle.get("synthetic_gt_team")
        if gt and model.status == "ok":
            # Map synthetic arbitrary labels onto anonymous predictions via permutation.
            pred = []
            truth_raw = []
            # Build consistent anonymous→gt pairing for diagnostic.
            anon_by_track = {d.track_id: d.team_id for d in decisions}
            # Use cluster purity style: group tracks by gt label, map to majority anon.
            from collections import defaultdict

            gt_groups: dict[str, list[str]] = defaultdict(list)
            for tid, g in gt.items():
                pred_lab = anon_by_track.get(int(tid), "unknown")
                if pred_lab in {"team_a", "team_b"}:
                    gt_groups[str(g)].append(pred_lab)
            # Remap arbitrary GT labels alphabetically for permutation diagnostic.
            gt_labels = sorted(set(gt.values()))
            if len(gt_labels) == 2:
                mapping_guess = {gt_labels[0]: "team_a", gt_labels[1]: "team_b"}
                for tid in sorted(gt.keys()):
                    pred.append(anon_by_track.get(int(tid), "unknown"))
                    truth_raw.append(mapping_guess[str(gt[tid])])
                acc = permutation_matched_accuracy(pred, truth_raw)
                synth_metrics = {"permutation_matched_accuracy": acc}

        eval_report = evaluate_team_assignment(
            assignments=assignment_rows,
            has_reviewed_ground_truth=False,
            synthetic_metrics=synth_metrics,
        )
        eval_payload = eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp)
        write_json_record(eval_out, eval_payload, overwrite=False)
        _chmod_file(eval_out)
        written.append(eval_out)

        assigned_n = sum(1 for d in decisions if d.status == "assigned")
        cand_n = sum(1 for d in decisions if d.status == "candidate")
        unk_n = sum(1 for d in decisions if d.status == "unknown")
        ne_n = sum(1 for d in decisions if d.status == "not_eligible")
        amb_n = sum(1 for d in decisions if d.status == "ambiguous")
        conf_n = sum(1 for d in decisions if d.status == "conflict")
        review_n = sum(1 for d in decisions if d.review_required)

        quality = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "appearance_config_fingerprint": app_fp,
            "appearance_profile_count": stats.get("profile_count"),
            "eligible_rejected_seed_counts": {
                "seeds": seed_stats.get("seeds"),
                "rejected": len(rejected_seeds),
            },
            "cluster_status": model.status,
            "cluster_sizes": dict(model.cluster_sizes),
            "separation": model.separation,
            "counts": {
                "assigned": assigned_n,
                "candidate": cand_n,
                "unknown": unk_n,
                "not_eligible": ne_n,
                "ambiguous": amb_n,
                "conflict": conf_n,
                "review_required": review_n,
            },
            "cross_video_auto_transfer": False,
            "auto_confirm": False,
            "evaluation_status": NOT_EVALUATED_TEAM_ASSIGNMENT,
            "created_at_utc": _utc_now(),
        }
        write_json_record(quality_out, quality, overwrite=False)
        _chmod_file(quality_out)
        written.append(quality_out)

        receipt = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "request_id": "team_classify_01",
            "status": "succeeded",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "config_fingerprint": cfg_fp,
            "appearance_config_fingerprint": app_fp,
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "method_id": config["method_id"],
            "team_assignments_ref": str(teams_out),
            "identity_evidence_ref": str(evidence_out),
            "cluster_provenance_ref": str(provenance_out) if provenance_out.exists() else None,
            "artifacts": {
                "team_assignments": _artifact_meta(teams_out),
                "identity_evidence": _artifact_meta(evidence_out),
                "evaluation": _artifact_meta(eval_out),
                "quality": _artifact_meta(quality_out),
            },
            "counts": quality["counts"],
            "cluster_status": model.status,
            "auto_confirm": False,
            "auto_real_team_naming": False,
            "auto_home_away": False,
            "face_recognition_used": False,
            "cross_video_auto_transfer": False,
            "evaluation_status": NOT_EVALUATED_TEAM_ASSIGNMENT,
            "reason_codes": list(model.reason_codes),
            "quality_flags": ["anonymous_team_labels", "real_accuracy_unvalidated"],
        }
        write_json_record(receipt_out, receipt, overwrite=False)
        _chmod_file(receipt_out)
        written.append(receipt_out)

        return TeamAssignmentServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            team_assignments_parquet=str(teams_out),
            evidence_parquet=str(evidence_out),
            cluster_provenance_json=str(provenance_out) if provenance_out.exists() else None,
            receipt_json=str(receipt_out),
            evaluation_json=str(eval_out),
            quality_json=str(quality_out),
            summary={
                "status": "succeeded",
                "decisions": decisions,
                "assignment_rows": assignment_rows,
                "evidence_rows": evidence_rows,
                "model": model,
                "profile_rows": profile_rows,
                "rejected_seeds": rejected_seeds,
                "quality": quality,
                "receipt": receipt,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"TEAM_CLASSIFY_FAIL:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )


def run_team_evaluate(
    *,
    config: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    run_id: str = "run_eval",
    video_id: str = "video_eval",
) -> dict[str, Any]:
    cfg_fp = team_assignment_config_fingerprint(config)
    report = evaluate_team_assignment(
        assignments=assignments,
        ground_truth=ground_truth,
        has_reviewed_ground_truth=has_reviewed_ground_truth,
    )
    return report.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp)


__all__ = [
    "TeamAssignmentServiceError",
    "TeamAssignmentServiceResult",
    "run_team_classify",
    "run_team_evaluate",
]
