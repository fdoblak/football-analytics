"""Tracklet appearance profile aggregation (Stage 7B)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.identity.appearance_descriptor import (
    AppearanceDescriptorError,
    cosine_similarity,
    l2_normalize,
    validate_embedding,
)
from football_analytics.identity.appearance_sampling import TrackSamplingResult
from football_analytics.identity.types import CONTRACT_VERSION, LeakageClass

PRODUCER = "appearance_reid_baseline"
PRODUCER_VERSION = "0.1.0"


class AppearanceProfileError(ValueError):
    """Profile aggregation failure."""


@dataclass(frozen=True)
class AppearanceProfile:
    track_id: int
    profile_id: str
    embedding: tuple[float, ...]
    vector_dimension: int
    aggregation_method: str
    observed_sample_count: int
    rejected_sample_count: int
    coverage_score: float | None
    quality_score: float | None
    status: str
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    review_required: bool
    start_frame_index: int | None
    end_frame_index: int | None
    start_time_us: int | None
    end_time_us: int | None
    profile_fingerprint: str

    def to_row(
        self,
        *,
        run_id: str,
        video_id: str,
        config: Mapping[str, Any],
        config_fingerprint: str,
        source_track_fingerprint: str | None = None,
        source_frames_fingerprint: str | None = None,
        leakage_class: str = LeakageClass.SYNTHETIC.value,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "video_id": video_id,
            "track_id": int(self.track_id),
            "profile_id": self.profile_id,
            "extractor_id": str(config["extractor_id"]),
            "extractor_version": str(config["extractor_version"]),
            "extractor_type": str(config["extractor_type"]),
            "vector_dimension": int(self.vector_dimension),
            "embedding": list(self.embedding),
            "aggregation_method": self.aggregation_method,
            "observed_sample_count": int(self.observed_sample_count),
            "rejected_sample_count": int(self.rejected_sample_count),
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "source_track_fingerprint": source_track_fingerprint,
            "source_frames_fingerprint": source_frames_fingerprint,
            "config_fingerprint": config_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "quality_flags": list(self.quality_flags),
            "review_required": bool(self.review_required),
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "leakage_class": leakage_class,
            "provenance_json": json.dumps(dict(provenance or {}), sort_keys=True),
            "contract_version": CONTRACT_VERSION,
        }


def _quality_weighted_mean(
    vectors: Sequence[Sequence[float]], weights: Sequence[float]
) -> np.ndarray:
    mats = [np.asarray(list(v), dtype=np.float64) for v in vectors]
    w = np.asarray(list(weights), dtype=np.float64)
    w = np.maximum(w, 1e-6)
    w = w / float(np.sum(w))
    stacked = np.stack(mats, axis=0)
    return np.sum(stacked * w[:, None], axis=0)


def _median_aggregate(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    stacked = np.stack([np.asarray(list(v), dtype=np.float64) for v in vectors], axis=0)
    return np.median(stacked, axis=0)


def aggregate_tracklet_profile(
    sampling: TrackSamplingResult,
    *,
    config: Mapping[str, Any],
    run_id: str,
    video_id: str,
) -> AppearanceProfile:
    """Aggregate crop descriptors into one L2-normalized tracklet profile."""
    agg = config["aggregation"]
    sampling_cfg = config["sampling"]
    dim = int(config["descriptor"]["embedding_dim"])
    method = str(agg["method"])
    min_samples = int(sampling_cfg["min_samples_for_profile"])
    profile_id = f"ap_{video_id}_t{int(sampling.track_id)}"

    accepted = list(sampling.accepted)
    rejected = int(sampling.rejected_count)
    flags: list[str] = ["same_kit_hard_negative_risk"]
    reasons: list[str] = list(sampling.reject_reasons)

    if sampling.entity_type != "human":
        empty = validate_embedding(l2_normalize(np.zeros(dim)).tolist(), expected_dim=dim)
        fp = hash_canonical_json(
            {
                "track_id": sampling.track_id,
                "status": "rejected",
                "entity": sampling.entity_type,
            }
        )
        return AppearanceProfile(
            track_id=sampling.track_id,
            profile_id=profile_id,
            embedding=empty,
            vector_dimension=dim,
            aggregation_method=method,
            observed_sample_count=0,
            rejected_sample_count=rejected,
            coverage_score=0.0,
            quality_score=0.0,
            status="rejected",
            reason_codes=("NON_HUMAN_ENTITY", *reasons),
            quality_flags=tuple(flags),
            review_required=False,
            start_frame_index=sampling.start_frame,
            end_frame_index=sampling.end_frame,
            start_time_us=sampling.start_time_us,
            end_time_us=sampling.end_time_us,
            profile_fingerprint=fp,
        )

    if len(accepted) < min_samples:
        empty = validate_embedding(l2_normalize(np.zeros(dim)).tolist(), expected_dim=dim)
        status = "insufficient_appearance_evidence"
        reasons = ["insufficient_appearance_evidence", *reasons]
        fp = hash_canonical_json(
            {
                "track_id": sampling.track_id,
                "status": status,
                "n": len(accepted),
                "rejected": rejected,
            }
        )
        return AppearanceProfile(
            track_id=sampling.track_id,
            profile_id=profile_id,
            embedding=empty,
            vector_dimension=dim,
            aggregation_method=method,
            observed_sample_count=len(accepted),
            rejected_sample_count=rejected,
            coverage_score=0.0,
            quality_score=float(np.mean([c.quality for c in accepted])) if accepted else 0.0,
            status=status,
            reason_codes=tuple(dict.fromkeys(reasons)),
            quality_flags=tuple(flags),
            review_required=True,
            start_frame_index=sampling.start_frame,
            end_frame_index=sampling.end_frame,
            start_time_us=sampling.start_time_us,
            end_time_us=sampling.end_time_us,
            profile_fingerprint=fp,
        )

    # Provisional aggregate for outlier rejection
    vecs = [c.descriptor for c in accepted]
    weights = [c.quality for c in accepted]
    if method == "median":
        provisional = l2_normalize(_median_aggregate(vecs))
    else:
        provisional = l2_normalize(_quality_weighted_mean(vecs, weights))

    keep_idx: list[int] = []
    outlier_thr = float(agg["outlier_cosine_reject"])
    for i, v in enumerate(vecs):
        sim = cosine_similarity(v, provisional.tolist())
        # Reject far outliers (low similarity).
        if sim < (1.0 - outlier_thr):
            rejected += 1
            reasons.append("OUTLIER_CROP_REJECTED")
            continue
        keep_idx.append(i)

    if len(keep_idx) < min_samples:
        empty = validate_embedding(l2_normalize(np.zeros(dim)).tolist(), expected_dim=dim)
        status = "insufficient_appearance_evidence"
        fp = hash_canonical_json(
            {"track_id": sampling.track_id, "status": status, "kept": len(keep_idx)}
        )
        return AppearanceProfile(
            track_id=sampling.track_id,
            profile_id=profile_id,
            embedding=empty,
            vector_dimension=dim,
            aggregation_method=method,
            observed_sample_count=len(keep_idx),
            rejected_sample_count=rejected,
            coverage_score=0.0,
            quality_score=0.0,
            status=status,
            reason_codes=tuple(dict.fromkeys(("insufficient_appearance_evidence", *reasons))),
            quality_flags=tuple(flags),
            review_required=True,
            start_frame_index=sampling.start_frame,
            end_frame_index=sampling.end_frame,
            start_time_us=sampling.start_time_us,
            end_time_us=sampling.end_time_us,
            profile_fingerprint=fp,
        )

    kept_vecs = [vecs[i] for i in keep_idx]
    kept_w = [weights[i] for i in keep_idx]
    if method == "median":
        final = l2_normalize(_median_aggregate(kept_vecs))
    else:
        final = l2_normalize(_quality_weighted_mean(kept_vecs, kept_w))
    embedding = validate_embedding(final.tolist(), expected_dim=dim)

    max_samples = int(sampling_cfg["max_samples_per_track"])
    coverage = min(1.0, len(keep_idx) / max(max_samples, 1))
    quality = float(np.mean(kept_w))
    min_cov = float(agg["min_coverage"])
    status = "ok"
    review = False
    if coverage < min_cov:
        flags.append("low_coverage")
        review = True
    if len(keep_idx) == 1 and agg["single_crop_strong_forbidden"]:
        # Should not happen given min_samples>=2, but keep invariant.
        status = "insufficient_appearance_evidence"
        reasons.append("SINGLE_CROP_FORBIDDEN")
        review = True

    fp_payload = {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": sampling.track_id,
        "embedding": list(embedding),
        "method": method,
        "n": len(keep_idx),
        "quality": round(quality, 6),
        "frames": [accepted[i].frame_index for i in keep_idx],
    }
    fp = hash_canonical_json(fp_payload)
    return AppearanceProfile(
        track_id=sampling.track_id,
        profile_id=profile_id,
        embedding=embedding,
        vector_dimension=dim,
        aggregation_method=method,
        observed_sample_count=len(keep_idx),
        rejected_sample_count=rejected,
        coverage_score=float(coverage),
        quality_score=float(quality),
        status=status,
        reason_codes=tuple(dict.fromkeys(reasons)),
        quality_flags=tuple(dict.fromkeys(flags)),
        review_required=review,
        start_frame_index=sampling.start_frame,
        end_frame_index=sampling.end_frame,
        start_time_us=sampling.start_time_us,
        end_time_us=sampling.end_time_us,
        profile_fingerprint=fp,
    )


def validate_profile_embedding_row(row: Mapping[str, Any], *, expected_dim: int) -> None:
    emb = row.get("embedding")
    if not isinstance(emb, list):
        raise AppearanceProfileError("embedding must be a list")
    if int(row.get("vector_dimension", -1)) != expected_dim:
        raise AppearanceProfileError("vector_dimension mismatch")
    try:
        validate_embedding(emb, expected_dim=expected_dim)
    except AppearanceDescriptorError as exc:
        raise AppearanceProfileError(str(exc)) from exc


__all__ = [
    "PRODUCER",
    "PRODUCER_VERSION",
    "AppearanceProfileError",
    "AppearanceProfile",
    "aggregate_tracklet_profile",
    "validate_profile_embedding_row",
]
