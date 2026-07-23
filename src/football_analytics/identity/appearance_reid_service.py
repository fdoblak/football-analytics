"""Stage 7B appearance embedding + tracklet ReID service."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.identity.appearance_matching import (
    AppearanceMatchingError,
    match_to_evidence_and_link_rows,
    propose_reid_candidates,
    score_profile_pair,
)
from football_analytics.identity.appearance_profiles import (
    AppearanceProfile,
    aggregate_tracklet_profile,
    validate_profile_embedding_row,
)
from football_analytics.identity.appearance_reid_config import appearance_reid_config_fingerprint
from football_analytics.identity.appearance_reid_evaluation import (
    NOT_EVALUATED_APPEARANCE_REID,
    evaluate_appearance_reid,
)
from football_analytics.identity.appearance_sampling import sample_tracklet_crops
from football_analytics.identity.contracts import (
    IDENTITY_EVIDENCE_CONTRACT,
    REID_CANDIDATE_LINKS_CONTRACT,
    TRACKLET_APPEARANCE_PROFILES_CONTRACT,
)
from football_analytics.identity.policy import load_identity_policy
from football_analytics.identity.types import LeakageClass


class AppearanceReidServiceError(RuntimeError):
    """Appearance ReID service failure."""


@dataclass
class AppearanceReidServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    profiles_parquet: str | None
    evidence_parquet: str | None
    links_parquet: str | None
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
            "profiles_parquet": self.profiles_parquet,
            "evidence_parquet": self.evidence_parquet,
            "links_parquet": self.links_parquet,
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
) -> AppearanceReidServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return AppearanceReidServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        profiles_parquet=None,
        evidence_parquet=None,
        links_parquet=None,
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


def build_profiles_from_bundle(
    *,
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
    config_fingerprint: str,
    leakage_class: str = LeakageClass.SYNTHETIC.value,
) -> tuple[list[AppearanceProfile], list[dict[str, Any]], dict[str, Any]]:
    run_id = str(bundle["run_id"])
    video_id = str(bundle["video_id"])
    observations = list(bundle["observations"])
    attributes = list(bundle.get("attributes") or [])
    crops = bundle.get("synthetic_crops")
    frames_bgr = bundle.get("frames_bgr")
    frame_times = bundle.get("frame_times_us") or {}
    entity_by_track = dict(bundle.get("entity_by_track") or {})

    track_ids = sorted({int(o["track_id"]) for o in observations})
    max_tracks = int(config["safety_limits"]["max_tracks_per_video"])
    if len(track_ids) > max_tracks:
        raise AppearanceReidServiceError("MAX_TRACKS_EXCEEDED")

    profiles: list[AppearanceProfile] = []
    rows: list[dict[str, Any]] = []
    sampled = 0
    rejected_crops = 0
    insufficient = 0

    for tid in track_ids:
        samp = sample_tracklet_crops(
            track_id=tid,
            observations=observations,
            frames_bgr=frames_bgr,
            synthetic_crops=crops,
            attributes=attributes,
            summaries=None,
            config=config,
            frame_times_us=frame_times,
        )
        if not entity_by_track.get(tid):
            entity_by_track[tid] = samp.entity_type
        sampled += len(samp.accepted)
        rejected_crops += samp.rejected_count
        prof = aggregate_tracklet_profile(samp, config=config, run_id=run_id, video_id=video_id)
        if prof.status == "insufficient_appearance_evidence":
            insufficient += 1
        profiles.append(prof)
        row = prof.to_row(
            run_id=run_id,
            video_id=video_id,
            config=config,
            config_fingerprint=config_fingerprint,
            leakage_class=leakage_class,
            provenance={"stage": "7B", "crops_persisted": False},
        )
        validate_profile_embedding_row(row, expected_dim=int(config["descriptor"]["embedding_dim"]))
        rows.append(row)

    stats = {
        "eligible_tracklets": len(track_ids),
        "profile_count": len(rows),
        "sampled_crop_count": sampled,
        "rejected_crop_count": rejected_crops,
        "insufficient_evidence_count": insufficient,
        "entity_by_track": entity_by_track,
    }
    return profiles, rows, stats


def run_appearance_extract(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    in_memory_bundle: Mapping[str, Any] | None = None,
    leakage_class: str = LeakageClass.SYNTHETIC.value,
) -> AppearanceReidServiceResult:
    """Extract tracklet appearance profiles; atomic no-overwrite publish."""
    cfg_fp = appearance_reid_config_fingerprint(config)
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

    profiles_out = out / "tracklet_appearance_profiles.parquet"
    receipt_out = out / "appearance_extract_receipt.json"
    quality_out = out / "appearance_extract_quality.json"
    for p in (profiles_out, receipt_out, quality_out):
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

    # Ensure crop persistence disabled
    if config["sampling"]["persist_crops"] or config["sampling"]["debug_crop_output"]:
        return _fail(error_code="CROP_PERSIST_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    started = _utc_now()
    try:
        bundle = dict(in_memory_bundle)
        bundle["run_id"] = rid
        bundle["video_id"] = vid
        profiles, rows, stats = build_profiles_from_bundle(
            bundle=bundle,
            config=config,
            config_fingerprint=cfg_fp,
            leakage_class=leakage_class,
        )
        if len(rows) > int(config["safety_limits"]["max_profiles_per_run"]):
            return _fail(error_code="MAX_PROFILES", exit_code=3, config_fingerprint=cfg_fp)

        table = _rows_to_table(rows, TRACKLET_APPEARANCE_PROFILES_CONTRACT)
        write_contract_parquet(
            table,
            profiles_out,
            get_contract(TRACKLET_APPEARANCE_PROFILES_CONTRACT, 1),
            contain_root=root,
            overwrite=False,
        )
        _chmod_file(profiles_out, int(config["output_policy"]["chmod_mode"]))
        written.append(profiles_out)

        # Assert no crop files were written
        crop_hits = (
            list(out.glob("**/*crop*")) + list(out.glob("**/*.png")) + list(out.glob("**/*.jpg"))
        )
        if crop_hits:
            return _fail(
                error_code="CROP_PERSISTED",
                exit_code=3,
                config_fingerprint=cfg_fp,
                cleanup=written,
            )

        quality = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "extractor_id": config["extractor_id"],
            "extractor_type": config["extractor_type"],
            "embedding_dim": int(config["descriptor"]["embedding_dim"]),
            "counts": stats,
            "norm_checks_ok": True,
            "crops_persisted": False,
            "created_at_utc": _utc_now(),
        }
        write_json_record(quality_out, quality, overwrite=False)
        _chmod_file(quality_out)
        written.append(quality_out)

        receipt = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "request_id": "appearance_extract_01",
            "status": "succeeded",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "config_fingerprint": cfg_fp,
            "extractor_id": config["extractor_id"],
            "extractor_version": config["extractor_version"],
            "appearance_evidence_ref": None,
            "profiles_ref": str(profiles_out),
            "artifact": _artifact_meta(profiles_out),
            "counts": {
                "profile_count": stats["profile_count"],
                "sampled_crop_count": stats["sampled_crop_count"],
                "rejected_crop_count": stats["rejected_crop_count"],
                "insufficient_evidence_count": stats["insufficient_evidence_count"],
            },
            "auto_confirm": False,
            "face_recognition_used": False,
            "crops_persisted": False,
            "evaluation_status": NOT_EVALUATED_APPEARANCE_REID,
            "reason_codes": [],
            "quality_flags": ["same_kit_hard_negative_risk"],
        }
        write_json_record(receipt_out, receipt, overwrite=False)
        _chmod_file(receipt_out)
        written.append(receipt_out)

        return AppearanceReidServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            profiles_parquet=str(profiles_out),
            evidence_parquet=None,
            links_parquet=None,
            receipt_json=str(receipt_out),
            evaluation_json=None,
            quality_json=str(quality_out),
            summary={
                "status": "succeeded",
                "profiles": profiles,
                "profile_rows": rows,
                "stats": stats,
                "entity_by_track": stats["entity_by_track"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"EXTRACT_FAIL:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )


def run_reid_candidates(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    profiles: Sequence[AppearanceProfile] | None = None,
    profile_rows: Sequence[Mapping[str, Any]] | None = None,
    entity_by_track: Mapping[int, str] | None = None,
    role_conflict_pairs: set[tuple[int, int]] | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    in_memory_bundle: Mapping[str, Any] | None = None,
    leakage_class: str = LeakageClass.SYNTHETIC.value,
    inject_failure: bool = False,
) -> AppearanceReidServiceResult:
    """Produce reid_candidate_links + identity_evidence from profiles."""
    cfg_fp = appearance_reid_config_fingerprint(config)
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

    evidence_out = out / "identity_evidence.parquet"
    links_out = out / "reid_candidate_links.parquet"
    profiles_out = out / "tracklet_appearance_profiles.parquet"
    receipt_out = out / "reid_candidates_receipt.json"
    quality_out = out / "reid_candidates_quality.json"
    eval_out = out / "appearance_reid_evaluation.json"
    for p in (evidence_out, links_out, receipt_out, quality_out, eval_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    rid = run_id or (str(in_memory_bundle["run_id"]) if in_memory_bundle else generate_run_id())
    vid = video_id or (str(in_memory_bundle["video_id"]) if in_memory_bundle else "video_01")
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INVALID_RUN_ID", exit_code=3, config_fingerprint=cfg_fp)
    if not SAFE_ID_RE.match(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=3, config_fingerprint=cfg_fp)

    started = _utc_now()
    try:
        policy_path = Path(str(config["identity_policy_path"]))
        if not policy_path.is_absolute():
            policy_path = Path(__file__).resolve().parents[3] / policy_path
        policy = load_identity_policy(policy_path)

        local_profiles: list[AppearanceProfile]
        local_rows: list[dict[str, Any]]
        entities: dict[int, str]
        if profiles is None:
            if in_memory_bundle is None:
                return _fail(error_code="INPUT_MISSING", exit_code=3, config_fingerprint=cfg_fp)
            bundle = dict(in_memory_bundle)
            bundle["run_id"] = rid
            bundle["video_id"] = vid
            local_profiles, local_rows, stats = build_profiles_from_bundle(
                bundle=bundle,
                config=config,
                config_fingerprint=cfg_fp,
                leakage_class=leakage_class,
            )
            entities = dict(stats["entity_by_track"])
            role_conflicts = set(bundle.get("role_conflict_pairs") or ())
        else:
            local_profiles = list(profiles)
            local_rows = [dict(r) for r in (profile_rows or [])]
            entities = dict(entity_by_track or {})
            role_conflicts = set(role_conflict_pairs or ())
            stats = {
                "profile_count": len(local_profiles),
                "sampled_crop_count": sum(p.observed_sample_count for p in local_profiles),
                "rejected_crop_count": sum(p.rejected_sample_count for p in local_profiles),
                "insufficient_evidence_count": sum(
                    1 for p in local_profiles if p.status == "insufficient_appearance_evidence"
                ),
            }

        matches = propose_reid_candidates(
            local_profiles,
            config=config,
            video_id=vid,
            entity_by_track=entities,
            role_conflict_pairs=role_conflicts,
        )

        # Explicit cross-video reject probe if present
        if in_memory_bundle and in_memory_bundle.get("cross_video_probe"):
            ok = [p for p in local_profiles if p.status == "ok"]
            if len(ok) >= 2:
                cross = score_profile_pair(
                    ok[0],
                    ok[1],
                    config=config,
                    same_video=False,
                )
                if cross is not None:
                    matches.append(cross)

        if len(matches) > int(config["safety_limits"]["max_candidates_per_run"]):
            return _fail(error_code="MAX_CANDIDATES", exit_code=3, config_fingerprint=cfg_fp)

        evidence_rows, link_rows = match_to_evidence_and_link_rows(
            matches,
            local_profiles,
            run_id=rid,
            video_id=vid,
            config_fingerprint=cfg_fp,
            policy=policy,
            leakage_class=leakage_class,
        )

        # Leakage guard: evaluation-class evidence must not claim production confirmation.
        for er in evidence_rows:
            if (
                str(er.get("leakage_class")) == LeakageClass.EVALUATION.value
                and leakage_class != LeakageClass.EVALUATION.value
            ):
                return _fail(
                    error_code="LEAKAGE_SEPARATION_VIOLATION",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            # Appearance tiers must stay supporting/weak/unavailable/conflicting.
            if str(er.get("reliability_tier")) in {"strong", "manual_verified"}:
                return _fail(
                    error_code="APPEARANCE_TIER_TOO_STRONG",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )

        if inject_failure:
            raise AppearanceMatchingError("injected_failure")

        if local_rows and not profiles_out.exists():
            ptable = _rows_to_table(local_rows, TRACKLET_APPEARANCE_PROFILES_CONTRACT)
            write_contract_parquet(
                ptable,
                profiles_out,
                get_contract(TRACKLET_APPEARANCE_PROFILES_CONTRACT, 1),
                contain_root=root,
                overwrite=False,
            )
            _chmod_file(profiles_out, int(config["output_policy"]["chmod_mode"]))
            written.append(profiles_out)

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

        ltable = _rows_to_table(list(link_rows), REID_CANDIDATE_LINKS_CONTRACT)
        write_contract_parquet(
            ltable,
            links_out,
            get_contract(REID_CANDIDATE_LINKS_CONTRACT, 1),
            contain_root=root,
            overwrite=False,
        )
        _chmod_file(links_out, int(config["output_policy"]["chmod_mode"]))
        written.append(links_out)

        cand_n = sum(1 for m in matches if m.decision_status == "candidate")
        rej_n = sum(1 for m in matches if m.decision_status == "rejected")
        amb_n = sum(1 for m in matches if m.decision_status == "review_required")
        review_n = sum(1 for m in matches if m.manual_review_required)

        eval_report = evaluate_appearance_reid(
            profiles=local_rows,
            links=link_rows,
            has_reviewed_ground_truth=False,
        )
        eval_payload = eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp)
        write_json_record(eval_out, eval_payload, overwrite=False)
        _chmod_file(eval_out)
        written.append(eval_out)

        quality = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "counts": {
                **stats,
                "candidate_link_count": cand_n,
                "rejected_link_count": rej_n,
                "ambiguous_link_count": amb_n,
                "review_count": review_n,
                "evidence_count": len(evidence_rows),
            },
            "auto_confirm": False,
            "physical_merge": False,
            "crops_persisted": False,
            "evaluation_status": NOT_EVALUATED_APPEARANCE_REID,
            "created_at_utc": _utc_now(),
        }
        write_json_record(quality_out, quality, overwrite=False)
        _chmod_file(quality_out)
        written.append(quality_out)

        receipt = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "request_id": "reid_candidates_01",
            "status": "succeeded",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "config_fingerprint": cfg_fp,
            "appearance_evidence_ref": str(evidence_out),
            "reid_links_ref": str(links_out),
            "profiles_ref": str(profiles_out) if profiles_out.exists() else None,
            "artifacts": {
                "evidence": _artifact_meta(evidence_out),
                "links": _artifact_meta(links_out),
                "evaluation": _artifact_meta(eval_out),
            },
            "counts": quality["counts"],
            "auto_confirm": False,
            "face_recognition_used": False,
            "physical_track_merge": False,
            "evaluation_status": NOT_EVALUATED_APPEARANCE_REID,
            "selection_matrix_selected": "handcrafted",
            "reason_codes": [],
            "quality_flags": ["same_kit_false_match_risk"],
        }
        write_json_record(receipt_out, receipt, overwrite=False)
        _chmod_file(receipt_out)
        written.append(receipt_out)

        return AppearanceReidServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            profiles_parquet=str(profiles_out) if profiles_out.exists() else None,
            evidence_parquet=str(evidence_out),
            links_parquet=str(links_out),
            receipt_json=str(receipt_out),
            evaluation_json=str(eval_out),
            quality_json=str(quality_out),
            summary={
                "status": "succeeded",
                "matches": matches,
                "evidence_rows": evidence_rows,
                "link_rows": link_rows,
                "counts": quality["counts"],
                "evaluation_status": NOT_EVALUATED_APPEARANCE_REID,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"CANDIDATES_FAIL:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )


def run_reid_evaluate(
    *,
    links: Sequence[Mapping[str, Any]] | None = None,
    profiles: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    run_id: str = "20260723T000000Z_eval01",
    video_id: str = "video_eval",
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    report = evaluate_appearance_reid(
        profiles=profiles,
        links=links,
        ground_truth=ground_truth,
        has_reviewed_ground_truth=has_reviewed_ground_truth,
    )
    return report.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=config_fingerprint)


__all__ = [
    "AppearanceReidServiceError",
    "AppearanceReidServiceResult",
    "build_profiles_from_bundle",
    "run_appearance_extract",
    "run_reid_candidates",
    "run_reid_evaluate",
]
