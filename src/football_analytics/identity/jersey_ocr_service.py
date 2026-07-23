"""Stage 7D jersey region + OCR baseline service."""

from __future__ import annotations

import json
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
from football_analytics.identity.contracts import (
    IDENTITY_EVIDENCE_CONTRACT,
    JERSEY_OBSERVATIONS_CONTRACT,
)
from football_analytics.identity.evidence import (
    assert_no_face_biometric_evidence,
    validate_evidence_rows,
)
from football_analytics.identity.jersey_consensus import (
    JerseyObservationVote,
    TrackJerseyConsensus,
    build_track_consensus,
)
from football_analytics.identity.jersey_ocr import (
    PRODUCER,
    PRODUCER_VERSION,
    TEMPLATE_VERSION,
    recognize_jersey_number,
)
from football_analytics.identity.jersey_ocr_config import jersey_ocr_config_fingerprint
from football_analytics.identity.jersey_ocr_evaluation import (
    NOT_EVALUATED_JERSEY_OCR,
    evaluate_jersey_ocr,
    false_number_emission_rate,
)
from football_analytics.identity.jersey_region import (
    extract_region_crop,
    propose_torso_regions,
    region_metrics_payload,
)
from football_analytics.identity.policy import decide_assignment_status, load_identity_policy
from football_analytics.identity.types import (
    CONTRACT_VERSION,
    EvidencePolarity,
    EvidenceType,
    LeakageClass,
    ReliabilityTier,
    ReviewStatus,
)


class JerseyOcrServiceError(RuntimeError):
    """Jersey OCR service failure."""


@dataclass
class JerseyOcrServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    jersey_observations_parquet: str | None
    evidence_parquet: str | None
    consensus_sidecar_json: str | None
    region_provenance_json: str | None
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
            "jersey_observations_parquet": self.jersey_observations_parquet,
            "evidence_parquet": self.evidence_parquet,
            "consensus_sidecar_json": self.consensus_sidecar_json,
            "region_provenance_json": self.region_provenance_json,
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
) -> JerseyOcrServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return JerseyOcrServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        jersey_observations_parquet=None,
        evidence_parquet=None,
        consensus_sidecar_json=None,
        region_provenance_json=None,
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


def _is_eligible_sample(
    sample: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[bool, str]:
    elig = config["eligibility"]
    if elig["human_only"] and str(sample.get("entity_type", "human")) != "human":
        return False, "not_eligible_entity"
    state = str(sample.get("observation_state", "observed")).lower()
    if elig["observed_only"] and elig["exclude_predicted"] and state != "observed":
        return False, "not_eligible_predicted"
    if elig["exclude_ball"] and str(sample.get("entity_type", "")).lower() == "ball":
        return False, "not_eligible_ball"
    role = str(sample.get("role", "unknown")).lower()
    if role in {r.lower() for r in elig["exclude_roles"]}:
        return False, f"not_eligible_role_{role}"
    allow = {r.lower() for r in elig["allow_roles"]}
    if role == "unknown" and elig["unknown_role_conservative"]:
        return True, "unknown_role_conservative"
    if role not in allow and role != "unknown":
        return False, f"not_eligible_role_{role}"
    return True, "eligible"


def _observation_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    observation_id: int,
    track_id: int,
    ocr: Any,
    status_override: str | None = None,
    extra_flags: Sequence[str] | None = None,
) -> dict[str, Any]:
    status = status_override or ocr.status
    flags = list(ocr.quality_flags)
    if extra_flags:
        flags.extend(extra_flags)
    # Map pipeline status into contract fields; keep confidence null.
    visibility = ocr.visibility
    readability = ocr.readability
    review = ReviewStatus.UNREVIEWED.value
    if status in {"ambiguous", "low_quality"} or "conflict" in flags:
        review = ReviewStatus.NEEDS_REVIEW.value
    if status == "not_eligible" and "not_eligible" not in flags:
        flags.append("not_eligible")
    if status == "low_quality" and "low_quality" not in flags:
        flags.append("low_quality")
    # Do not emit numbers for non-observed statuses.
    raw_text = ocr.raw_text if status == "observed" else None
    normalized = ocr.normalized_number if status == "observed" else None
    digit_count = ocr.digit_count
    if status in {"no_region", "not_eligible", "failed"}:
        digit_count = None if status != "no_digits" else 0
    if status == "no_digits":
        digit_count = 0
        readability = "none"
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "observation_id": int(observation_id),
        "track_id": int(track_id),
        "raw_text": raw_text,
        "normalized_number": normalized,
        "digit_count": digit_count,
        "visibility": visibility,
        "readability": readability,
        "confidence": None,
        "source": ocr.source,
        "review_status": review,
        "crop_artifact_id": None,
        "quality_flags": sorted(set(flags)),
    }


def _evidence_row(
    *,
    run_id: str,
    video_id: str,
    evidence_id: str,
    track_id: int,
    reliability_tier: str,
    polarity: str,
    review_status: str,
    reason_codes: Sequence[str],
    quality_flags: Sequence[str],
    leakage_class: str,
    config_fingerprint: str,
    observed_value_ref: str | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "evidence_id": evidence_id,
        "track_id": int(track_id),
        "frame_index": None,
        "start_frame_index": None,
        "end_frame_index": None,
        "start_time_us": None,
        "end_time_us": None,
        "evidence_type": EvidenceType.JERSEY_NUMBER.value,
        "source_artifact_ref": "jersey_observations",
        "source_fingerprint": config_fingerprint,
        "observed_value_ref": observed_value_ref,
        "score": None,
        "reliability_tier": reliability_tier,
        "polarity": polarity,
        "review_status": review_status,
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "reason_codes": list(reason_codes),
        "quality_flags": list(quality_flags),
        "leakage_class": leakage_class,
        "provenance_json": json.dumps(dict(provenance), sort_keys=True),
        "contract_version": CONTRACT_VERSION,
    }


def _evidence_from_consensus(
    consensus: Sequence[TrackJerseyConsensus],
    *,
    run_id: str,
    video_id: str,
    config_fingerprint: str,
    leakage_class: str,
    team_by_track: Mapping[int, str] | None,
    sample_team_hints: Mapping[int, str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in consensus:
        if c.status != "observed" or not c.raw_text:
            if c.status == "ambiguous" and c.review_required:
                rows.append(
                    _evidence_row(
                        run_id=run_id,
                        video_id=video_id,
                        evidence_id=f"jev_{video_id}_t{c.track_id}_conflict",
                        track_id=int(c.track_id),
                        reliability_tier=ReliabilityTier.CONFLICTING.value,
                        polarity=EvidencePolarity.CONFLICTS.value,
                        review_status=ReviewStatus.NEEDS_REVIEW.value,
                        reason_codes=list(c.reason_codes),
                        quality_flags=list(c.quality_flags),
                        leakage_class=leakage_class,
                        config_fingerprint=config_fingerprint,
                        observed_value_ref=None,
                        provenance={
                            "status": c.status,
                            "observation_ids": list(c.observation_ids),
                            "auto_confirm": False,
                        },
                    )
                )
            continue
        polarity = EvidencePolarity.SUPPORTS.value
        tier = ReliabilityTier.SUPPORTING.value
        reasons = list(c.reason_codes) + ["JERSEY_ALONE_INSUFFICIENT"]
        flags = list(c.quality_flags)
        # Team/jersey conflict → review/conflicting evidence
        team = team_by_track.get(int(c.track_id)) if team_by_track else None
        hint = sample_team_hints.get(int(c.track_id)) if sample_team_hints else None
        if team and hint and team != hint:
            polarity = EvidencePolarity.CONFLICTS.value
            tier = ReliabilityTier.CONFLICTING.value
            reasons.append("TEAM_JERSEY_CONFLICT")
            flags.append("team_jersey_conflict")
        rows.append(
            _evidence_row(
                run_id=run_id,
                video_id=video_id,
                evidence_id=f"jev_{video_id}_t{c.track_id}",
                track_id=int(c.track_id),
                reliability_tier=tier,
                polarity=polarity,
                review_status=(
                    ReviewStatus.NEEDS_REVIEW.value
                    if polarity == EvidencePolarity.CONFLICTS.value
                    else ReviewStatus.UNREVIEWED.value
                ),
                reason_codes=reasons,
                quality_flags=flags,
                leakage_class=leakage_class,
                config_fingerprint=config_fingerprint,
                observed_value_ref=c.raw_text,
                provenance={
                    "status": c.status,
                    "raw_text": c.raw_text,
                    "normalized_number": c.normalized_number,
                    "observation_ids": list(c.observation_ids),
                    "template_version": TEMPLATE_VERSION,
                    "auto_confirm": False,
                },
            )
        )
    return rows


def run_jersey_observe(
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
) -> JerseyOcrServiceResult:
    """Observe jersey numbers from synthetic/in-memory frame crops; write artifacts."""
    cfg_fp = jersey_ocr_config_fingerprint(config)
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

    jerseys_out = out / "jersey_observations.parquet"
    evidence_out = out / "identity_evidence.parquet"
    consensus_out = out / "jersey_track_consensus.json"
    region_out = out / "jersey_region_provenance.json"
    receipt_out = out / "jersey_ocr_receipt.json"
    quality_out = out / "jersey_ocr_quality.json"
    eval_out = out / "jersey_ocr_evaluation.json"
    for p in (
        jerseys_out,
        evidence_out,
        consensus_out,
        region_out,
        receipt_out,
        quality_out,
        eval_out,
    ):
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

    auto_confirm = config["assignment"]["auto_confirm_identity"]
    auto_target = config["assignment"]["auto_target_confirmation"]
    if auto_confirm or auto_target:
        return _fail(error_code="AUTO_CONFIRM_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)
    if config["region"]["persist_crops"] or config["assignment"]["persist_crops"]:
        return _fail(error_code="CROP_PERSIST_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    leak = leakage_class or str(
        in_memory_bundle.get("force_leakage_class")
        or in_memory_bundle.get("leakage_class")
        or LeakageClass.SYNTHETIC.value
    )
    if leak == LeakageClass.EVALUATION.value and in_memory_bundle.get("evaluation_label"):
        # Evaluation labels must not enter OCR features — reject leakage into production path.
        return _fail(
            error_code="LEAKAGE_SEPARATION_VIOLATION",
            exit_code=1,
            config_fingerprint=cfg_fp,
        )

    started = _utc_now()
    policy_path = Path(str(config["identity_policy_path"]))
    if not policy_path.is_absolute():
        policy_path = (repo_root or Path(__file__).resolve().parents[3]) / policy_path

    try:
        policy = load_identity_policy(policy_path)
        samples = list(in_memory_bundle.get("samples") or [])
        observation_rows: list[dict[str, Any]] = []
        votes: list[JerseyObservationVote] = []
        region_prov: list[dict[str, Any]] = []
        counts = {
            "eligible": 0,
            "rejected": 0,
            "observed": 0,
            "no_region": 0,
            "no_digits": 0,
            "ambiguous": 0,
            "low_quality": 0,
            "not_eligible": 0,
            "failed": 0,
        }
        # Deterministic sampling per track
        by_track: dict[int, list[Mapping[str, Any]]] = {}
        for s in samples:
            by_track.setdefault(int(s["track_id"]), []).append(s)
        obs_id = 0
        negative_obs_ids: list[int] = []
        sample_team_hints: dict[int, str] = {}
        stride = int(config["eligibility"]["sample_stride"])
        max_per = int(config["eligibility"]["max_samples_per_track"])

        for tid in sorted(by_track):
            track_samples = sorted(by_track[tid], key=lambda x: int(x["frame_index"]))
            taken = 0
            for idx, sample in enumerate(track_samples):
                if idx % stride != 0:
                    continue
                if taken >= max_per:
                    break
                taken += 1
                ok, reason = _is_eligible_sample(sample, config=config)
                frame = sample["frame_image"]
                bbox = sample["bbox"]
                fi = int(sample["frame_index"])
                if sample.get("team_id"):
                    sample_team_hints[int(tid)] = str(sample["team_id"])
                if not ok:
                    counts["rejected"] += 1
                    counts["not_eligible"] += 1
                    from football_analytics.identity.jersey_ocr import JerseyOcrResult

                    dummy = JerseyOcrResult(
                        status="not_eligible",
                        raw_text=None,
                        normalized_number=None,
                        digit_count=None,
                        number_score=None,
                        number_margin=None,
                        digit_scores=(),
                        quality_flags=("not_eligible", reason),
                        reason_codes=(reason.upper(),),
                        visibility="unknown",
                        readability="none",
                        source=str(config["ocr"]["source"]),
                    )
                    observation_rows.append(
                        _observation_row(
                            run_id=rid,
                            video_id=vid,
                            frame_index=fi,
                            observation_id=obs_id,
                            track_id=tid,
                            ocr=dummy,
                            status_override="not_eligible",
                        )
                    )
                    if sample.get("is_negative"):
                        negative_obs_ids.append(obs_id)
                    obs_id += 1
                    continue

                counts["eligible"] += 1
                # Unknown role: more conservative thresholds via suitability gate
                regions = propose_torso_regions(frame, bbox, config=config)
                if not regions:
                    from football_analytics.identity.jersey_ocr import JerseyOcrResult

                    ocr = JerseyOcrResult(
                        status="no_region",
                        raw_text=None,
                        normalized_number=None,
                        digit_count=None,
                        number_score=None,
                        number_margin=None,
                        digit_scores=(),
                        quality_flags=("no_region",),
                        reason_codes=("NO_REGION_CANDIDATE",),
                        visibility="unknown",
                        readability="none",
                        source=str(config["ocr"]["source"]),
                    )
                    counts["no_region"] += 1
                    observation_rows.append(
                        _observation_row(
                            run_id=rid,
                            video_id=vid,
                            frame_index=fi,
                            observation_id=obs_id,
                            track_id=tid,
                            ocr=ocr,
                        )
                    )
                    if sample.get("is_negative"):
                        negative_obs_ids.append(obs_id)
                    obs_id += 1
                    continue

                best = regions[0]
                region_prov.append(
                    {
                        "track_id": tid,
                        "frame_index": fi,
                        "observation_id": obs_id,
                        "candidates": [region_metrics_payload(c) for c in regions[:3]],
                        "selected": region_metrics_payload(best),
                    }
                )
                extra: list[str] = []
                if best.suitability == "not_suitable":
                    # Prefer low_quality over inventing from bad region.
                    from football_analytics.identity.jersey_ocr import JerseyOcrResult

                    ocr = JerseyOcrResult(
                        status="low_quality",
                        raw_text=None,
                        normalized_number=None,
                        digit_count=None,
                        number_score=None,
                        number_margin=None,
                        digit_scores=(),
                        quality_flags=("low_quality", *best.reason_codes),
                        reason_codes=tuple(best.reason_codes) or ("LOW_QUALITY_REGION",),
                        visibility="partial",
                        readability="illegible",
                        source=str(config["ocr"]["source"]),
                    )
                    counts["low_quality"] += 1
                else:
                    if reason == "unknown_role_conservative":
                        extra.append("unknown_role_conservative")
                    crop = extract_region_crop(frame, best)
                    # Never persist crop by default — keep in memory only.
                    ocr = recognize_jersey_number(crop, config=config)
                    counts[ocr.status] = counts.get(ocr.status, 0) + 1
                    if ocr.status == "observed":
                        votes.append(
                            JerseyObservationVote(
                                track_id=tid,
                                frame_index=fi,
                                observation_id=obs_id,
                                raw_text=str(ocr.raw_text),
                                normalized_number=ocr.normalized_number,
                                quality=float(best.quality),
                                score=ocr.number_score,
                                status="observed",
                            )
                        )
                observation_rows.append(
                    _observation_row(
                        run_id=rid,
                        video_id=vid,
                        frame_index=fi,
                        observation_id=obs_id,
                        track_id=tid,
                        ocr=ocr,
                        extra_flags=extra,
                    )
                )
                if sample.get("is_negative"):
                    negative_obs_ids.append(obs_id)
                obs_id += 1

        if len(observation_rows) > int(config["safety_limits"]["max_observations_per_run"]):
            return _fail(error_code="MAX_OBSERVATIONS", exit_code=3, config_fingerprint=cfg_fp)

        consensus = build_track_consensus(votes, config=config)
        team_by_track = dict(in_memory_bundle.get("team_by_track") or {})
        evidence_rows = _evidence_from_consensus(
            consensus,
            run_id=rid,
            video_id=vid,
            config_fingerprint=cfg_fp,
            leakage_class=leak,
            team_by_track=team_by_track,
            sample_team_hints=sample_team_hints,
        )
        validate_evidence_rows(evidence_rows)
        assert_no_face_biometric_evidence(evidence_rows)

        # Jersey alone → candidate only; never auto-confirm.
        for er in evidence_rows:
            if er["polarity"] != EvidencePolarity.SUPPORTS.value:
                continue
            st, reasons = decide_assignment_status([er], policy=policy)
            if st in {"confirmed", "provisional"}:
                return _fail(
                    error_code="JERSEY_ALONE_AUTO_CONFIRM",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            if "JERSEY_ALONE_INSUFFICIENT" not in reasons:
                return _fail(
                    error_code="JERSEY_EVIDENCE_POLICY",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )
            if str(er.get("reliability_tier")) in {"strong", "manual_verified"}:
                return _fail(
                    error_code="JERSEY_TIER_TOO_STRONG",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                )

        if inject_failure:
            raise JerseyOcrServiceError("injected_failure")

        jtable = _rows_to_table(observation_rows, JERSEY_OBSERVATIONS_CONTRACT)
        write_contract_parquet(
            jtable,
            jerseys_out,
            get_contract(JERSEY_OBSERVATIONS_CONTRACT, 1),
            contain_root=root,
            overwrite=False,
        )
        _chmod_file(jerseys_out, int(config["output_policy"]["chmod_mode"]))
        written.append(jerseys_out)

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

        consensus_payload = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "template_version": TEMPLATE_VERSION,
            "tracks": [
                {
                    "track_id": c.track_id,
                    "status": c.status,
                    "raw_text": c.raw_text,
                    "normalized_number": c.normalized_number,
                    "digit_count": c.digit_count,
                    "vote_weight": c.vote_weight,
                    "margin": c.margin,
                    "observation_count": c.observation_count,
                    "temporal_spread": c.temporal_spread,
                    "observation_ids": list(c.observation_ids),
                    "reason_codes": list(c.reason_codes),
                    "quality_flags": list(c.quality_flags),
                    "review_required": c.review_required,
                }
                for c in consensus
            ],
            "created_at_utc": _utc_now(),
        }
        if config["output_policy"]["write_consensus_sidecar"]:
            write_json_record(consensus_out, consensus_payload, overwrite=False)
            _chmod_file(consensus_out)
            written.append(consensus_out)

        region_payload = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "persist_crops": False,
            "regions": region_prov,
            "created_at_utc": _utc_now(),
        }
        if config["output_policy"]["write_region_provenance"]:
            write_json_record(region_out, region_payload, overwrite=False)
            _chmod_file(region_out)
            written.append(region_out)

        fp_rate = false_number_emission_rate(
            observation_rows, negative_observation_ids=negative_obs_ids
        )
        synth_metrics = {"false_number_emission_rate": fp_rate} if negative_obs_ids else None
        eval_report = evaluate_jersey_ocr(
            observations=observation_rows,
            has_reviewed_ground_truth=False,
            synthetic_metrics=synth_metrics,
        )
        eval_payload = eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp)
        write_json_record(eval_out, eval_payload, overwrite=False)
        _chmod_file(eval_out)
        written.append(eval_out)

        cons_observed = sum(1 for c in consensus if c.status == "observed")
        cons_amb = sum(1 for c in consensus if c.status == "ambiguous")
        review_n = sum(1 for c in consensus if c.review_required)

        quality = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "config_fingerprint": cfg_fp,
            "template_version": TEMPLATE_VERSION,
            "counts": counts,
            "consensus_counts": {
                "observed": cons_observed,
                "ambiguous": cons_amb,
                "review_required": review_n,
            },
            "false_number_emission_rate_synthetic": fp_rate if negative_obs_ids else None,
            "negative_control_count": len(negative_obs_ids),
            "persist_crops": False,
            "auto_confirm": False,
            "face_recognition_used": False,
            "evaluation_status": NOT_EVALUATED_JERSEY_OCR,
            "created_at_utc": _utc_now(),
        }
        write_json_record(quality_out, quality, overwrite=False)
        _chmod_file(quality_out)
        written.append(quality_out)

        receipt = {
            "schema_version": 1,
            "run_id": rid,
            "video_id": vid,
            "request_id": "jersey_observe_01",
            "status": "succeeded",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "config_fingerprint": cfg_fp,
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "method_id": config["method_id"],
            "template_version": TEMPLATE_VERSION,
            "jersey_observations_ref": str(jerseys_out),
            "identity_evidence_ref": str(evidence_out),
            "consensus_sidecar_ref": str(consensus_out) if consensus_out.exists() else None,
            "region_provenance_ref": str(region_out) if region_out.exists() else None,
            "artifacts": {
                "jersey_observations": _artifact_meta(jerseys_out),
                "identity_evidence": _artifact_meta(evidence_out),
                "evaluation": _artifact_meta(eval_out),
                "quality": _artifact_meta(quality_out),
            },
            "counts": counts,
            "consensus_counts": quality["consensus_counts"],
            "auto_confirm": False,
            "persist_crops": False,
            "face_recognition_used": False,
            "evaluation_status": NOT_EVALUATED_JERSEY_OCR,
            "quality_flags": [
                "jersey_supporting_only",
                "real_accuracy_unvalidated",
                "confidence_null_raw_scores_in_flags",
            ],
        }
        write_json_record(receipt_out, receipt, overwrite=False)
        _chmod_file(receipt_out)
        written.append(receipt_out)

        return JerseyOcrServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            jersey_observations_parquet=str(jerseys_out),
            evidence_parquet=str(evidence_out),
            consensus_sidecar_json=str(consensus_out) if consensus_out.exists() else None,
            region_provenance_json=str(region_out) if region_out.exists() else None,
            receipt_json=str(receipt_out),
            evaluation_json=str(eval_out),
            quality_json=str(quality_out),
            summary={
                "status": "succeeded",
                "observation_rows": observation_rows,
                "evidence_rows": evidence_rows,
                "consensus": consensus,
                "quality": quality,
                "receipt": receipt,
                "negative_observation_ids": negative_obs_ids,
                "false_number_emission_rate": fp_rate if negative_obs_ids else 0.0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"JERSEY_OBSERVE_FAIL:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )


def run_jersey_evaluate(
    *,
    config: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    has_reviewed_ground_truth: bool = False,
    run_id: str = "run_eval",
    video_id: str = "video_eval",
) -> dict[str, Any]:
    cfg_fp = jersey_ocr_config_fingerprint(config)
    report = evaluate_jersey_ocr(
        observations=observations,
        ground_truth=ground_truth,
        has_reviewed_ground_truth=has_reviewed_ground_truth,
    )
    return report.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp)


__all__ = [
    "JerseyOcrServiceError",
    "JerseyOcrServiceResult",
    "run_jersey_observe",
    "run_jersey_evaluate",
]
