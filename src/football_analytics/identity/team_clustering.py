"""Anonymous team appearance clustering (Stage 7C) — team_a / team_b / unknown."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.identity.appearance_descriptor import l2_normalize
from football_analytics.identity.appearance_profiles import AppearanceProfile

ANONYMOUS_TEAM_IDS = frozenset({"team_a", "team_b", "unknown"})
COLOR_DIM_DEFAULT = 64  # HSV/Lab upper+lower; excludes edge/texture (88-64)


class TeamClusteringError(ValueError):
    """Team clustering failure."""


@dataclass(frozen=True)
class SeedTrack:
    track_id: int
    vector: tuple[float, ...]
    role: str
    coverage: float
    quality: float
    shot_id: str | None
    start_frame_index: int | None
    end_frame_index: int | None


@dataclass(frozen=True)
class TeamClusterModel:
    status: str
    reason_codes: tuple[str, ...]
    centroids: Mapping[str, tuple[float, ...]]
    centroid_fingerprints: Mapping[str, str]
    seed_track_ids: tuple[int, ...]
    cluster_sizes: Mapping[str, int]
    separation: float | None
    label_order: tuple[str, ...]
    provenance: Mapping[str, Any]


def team_feature_vector(
    embedding: Sequence[float],
    *,
    config: Mapping[str, Any],
) -> tuple[float, ...]:
    """Color-focused vector for kit clustering (downweight identity texture/edge)."""
    arr = np.asarray(list(embedding), dtype=np.float64)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        raise TeamClusteringError("invalid embedding for team feature")
    cl = config["clustering"]
    if cl["color_dims_only"]:
        n = min(COLOR_DIM_DEFAULT, int(arr.size))
        color = arr[:n].copy()
        w = float(cl["edge_texture_weight"])
        if w > 0.0 and arr.size > n:
            color = np.concatenate([color, w * arr[n:]])
        out = l2_normalize(color)
    else:
        out = l2_normalize(arr)
    return tuple(float(x) for x in out.tolist())


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(list(a), dtype=np.float64)
    vb = np.asarray(list(b), dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    sim = float(np.dot(va, vb) / (na * nb))
    if not math.isfinite(sim):
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - sim)))


def _centroid_fp(vec: Sequence[float]) -> str:
    rounded = [round(float(x), 8) for x in vec]
    return hash_canonical_json({"centroid": rounded})


def resolve_track_role(
    track_id: int,
    *,
    attributes: Sequence[Mapping[str, Any]] | None,
    role_by_track: Mapping[int, str] | None,
) -> str:
    if role_by_track and track_id in role_by_track:
        return str(role_by_track[track_id]).lower()
    if not attributes:
        return "unknown"
    labels: list[str] = []
    for a in attributes:
        if int(a.get("detection_id", -1)) < 0:
            continue
        # Role may be keyed by track via caller; attributes often lack track_id.
        if "track_id" in a and int(a["track_id"]) != track_id:
            continue
        lab = str(a.get("role_label") or "unknown").lower()
        labels.append(lab)
    if not labels and role_by_track is None:
        # Fall back: any attributes without track_id are not usable alone.
        return "unknown"
    if not labels:
        return "unknown"
    # Majority vote with deterministic tie-break.
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best


def collect_seed_tracks(
    profiles: Sequence[AppearanceProfile],
    *,
    config: Mapping[str, Any],
    role_by_track: Mapping[int, str],
    shot_by_track: Mapping[int, str] | None = None,
) -> tuple[list[SeedTrack], list[dict[str, Any]]]:
    """Select player seeds only; referee/staff/GK/unknown excluded from centroids."""
    elig = config["eligibility"]
    seed_roles = {str(x).lower() for x in elig["seed_roles"]}
    exclude = {str(x).lower() for x in elig["exclude_roles_from_seed"]}
    seeds: list[SeedTrack] = []
    rejected: list[dict[str, Any]] = []

    for p in profiles:
        role = str(role_by_track.get(p.track_id, "unknown")).lower()
        reasons: list[str] = []
        if p.status != "ok":
            reasons.append("INSUFFICIENT_APPEARANCE")
        cov = float(p.coverage_score if p.coverage_score is not None else 0.0)
        qual = float(p.quality_score if p.quality_score is not None else 0.0)
        if cov < float(elig["min_coverage"]):
            reasons.append("LOW_COVERAGE")
        if qual < float(elig["min_quality"]):
            reasons.append("LOW_QUALITY")
        if int(p.observed_sample_count) < int(elig["min_observed_samples"]):
            reasons.append("TOO_FEW_SAMPLES")
        if role in exclude or role not in seed_roles:
            reasons.append(f"ROLE_NOT_SEED:{role}")
        if role == "goalkeeper" and elig["exclude_confirmed_goalkeeper_from_seed"]:
            reasons.append("GOALKEEPER_EXCLUDED")
        if role == "unknown" and not elig["unknown_role_may_seed"]:
            reasons.append("UNKNOWN_ROLE_NO_SEED")
        if reasons:
            rejected.append({"track_id": p.track_id, "role": role, "reasons": reasons})
            continue
        vec = team_feature_vector(p.embedding, config=config)
        seeds.append(
            SeedTrack(
                track_id=p.track_id,
                vector=vec,
                role=role,
                coverage=cov,
                quality=qual,
                shot_id=(shot_by_track or {}).get(p.track_id),
                start_frame_index=p.start_frame_index,
                end_frame_index=p.end_frame_index,
            )
        )
    seeds.sort(key=lambda s: s.track_id)
    return seeds, rejected


def _farthest_pair_init(vectors: np.ndarray) -> np.ndarray:
    n = vectors.shape[0]
    if n < 2:
        raise TeamClusteringError("need >=2 vectors for farthest-pair init")
    # Deterministic: max pairwise cosine distance; tie-break by indices.
    best = (-1.0, 0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_distance(vectors[i], vectors[j])
            key = (d, -i, -j)
            if key > (best[0], -best[1], -best[2]):
                best = (d, i, j)
    return np.stack([vectors[best[1]], vectors[best[2]]], axis=0)


def _lloyd_two_cluster(
    vectors: np.ndarray,
    init: np.ndarray,
    *,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    cents = init.copy()
    labels = np.zeros(vectors.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        # Assign
        d0 = np.asarray([_cosine_distance(v, cents[0]) for v in vectors])
        d1 = np.asarray([_cosine_distance(v, cents[1]) for v in vectors])
        # Tie-break: prefer cluster 0 when distances equal (deterministic).
        new_labels = np.where(d1 < d0, 1, 0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        new_cents = cents.copy()
        for k in (0, 1):
            members = vectors[labels == k]
            if members.shape[0] == 0:
                continue
            mean = np.mean(members, axis=0)
            new_cents[k] = l2_normalize(mean)
        cents = new_cents
    return labels, cents


def _order_labels_by_centroid_fp(centroids: np.ndarray) -> dict[int, str]:
    fps = [_centroid_fp(centroids[i]) for i in range(2)]
    # Ascending fingerprint → team_a then team_b
    order = sorted(range(2), key=lambda i: (fps[i], i))
    return {order[0]: "team_a", order[1]: "team_b"}


def fit_two_team_clusters(
    seeds: Sequence[SeedTrack],
    *,
    config: Mapping[str, Any],
) -> TeamClusterModel:
    """Deterministic 2-cluster baseline; abstain when separation/seeds insufficient."""
    cl = config["clustering"]
    min_seeds = int(cl["min_seeds"])
    if len(seeds) < min_seeds:
        return TeamClusterModel(
            status="insufficient_team_evidence",
            reason_codes=("INSUFFICIENT_SEEDS",),
            centroids={},
            centroid_fingerprints={},
            seed_track_ids=tuple(s.track_id for s in seeds),
            cluster_sizes={},
            separation=None,
            label_order=(),
            provenance={"seed_count": len(seeds), "min_seeds": min_seeds},
        )

    all_seeds = list(seeds)
    vectors = np.stack([np.asarray(s.vector, dtype=np.float64) for s in all_seeds], axis=0)
    init = _farthest_pair_init(vectors)
    labels, cents = _lloyd_two_cluster(vectors, init, max_iters=int(cl["max_lloyd_iters"]))

    # Reject third-color / far seeds, then optionally refit once on inliers.
    outlier_thr = float(cl["outlier_distance"])
    inlier_mask = np.ones(len(all_seeds), dtype=bool)
    outlier_track_ids: list[int] = []
    for i, s in enumerate(all_seeds):
        d0 = _cosine_distance(vectors[i], cents[0])
        d1 = _cosine_distance(vectors[i], cents[1])
        if min(d0, d1) > outlier_thr:
            inlier_mask[i] = False
            outlier_track_ids.append(s.track_id)
    if outlier_track_ids and int(np.sum(inlier_mask)) >= min_seeds:
        inlier_seeds = [s for s, keep in zip(all_seeds, inlier_mask, strict=True) if keep]
        vectors = np.stack([np.asarray(s.vector, dtype=np.float64) for s in inlier_seeds], axis=0)
        init = _farthest_pair_init(vectors)
        labels, cents = _lloyd_two_cluster(vectors, init, max_iters=int(cl["max_lloyd_iters"]))
        all_seeds = inlier_seeds
    elif outlier_track_ids and int(np.sum(inlier_mask)) < min_seeds:
        # Keep original fit but record outliers for assignment-time unknown.
        pass

    sizes = {0: int(np.sum(labels == 0)), 1: int(np.sum(labels == 1))}
    if sizes[0] < int(cl["min_cluster_size"]) or sizes[1] < int(cl["min_cluster_size"]):
        return TeamClusterModel(
            status="insufficient_team_evidence",
            reason_codes=("CLUSTER_COLLAPSE", "MIN_CLUSTER_SIZE"),
            centroids={},
            centroid_fingerprints={},
            seed_track_ids=tuple(s.track_id for s in seeds),
            cluster_sizes={},
            separation=None,
            label_order=(),
            provenance={"sizes_raw": sizes, "outlier_track_ids": outlier_track_ids},
        )

    sep = _cosine_distance(cents[0], cents[1])
    if sep < float(cl["similar_kit_separation_floor"]):
        return TeamClusterModel(
            status="insufficient_team_evidence",
            reason_codes=("SIMILAR_KIT_ABSTAIN", "LOW_SEPARATION"),
            centroids={},
            centroid_fingerprints={},
            seed_track_ids=tuple(s.track_id for s in seeds),
            cluster_sizes={},
            separation=float(sep),
            label_order=(),
            provenance={"separation": sep, "outlier_track_ids": outlier_track_ids},
        )
    if sep < float(cl["min_centroid_separation"]):
        return TeamClusterModel(
            status="insufficient_team_evidence",
            reason_codes=("INSUFFICIENT_SEPARATION",),
            centroids={},
            centroid_fingerprints={},
            seed_track_ids=tuple(s.track_id for s in seeds),
            cluster_sizes={},
            separation=float(sep),
            label_order=(),
            provenance={"separation": sep, "outlier_track_ids": outlier_track_ids},
        )

    # Intra-cluster spread check
    for k in (0, 1):
        members = vectors[labels == k]
        spreads = [_cosine_distance(m, cents[k]) for m in members]
        if spreads and max(spreads) > float(cl["max_intra_cluster_spread"]):
            return TeamClusterModel(
                status="insufficient_team_evidence",
                reason_codes=("HIGH_INTRA_SPREAD",),
                centroids={},
                centroid_fingerprints={},
                seed_track_ids=tuple(s.track_id for s in seeds),
                cluster_sizes={},
                separation=float(sep),
                label_order=(),
                provenance={
                    "spread_cluster": k,
                    "max_spread": max(spreads),
                    "outlier_track_ids": outlier_track_ids,
                },
            )

    label_map = _order_labels_by_centroid_fp(cents)
    named_cents: dict[str, tuple[float, ...]] = {}
    named_fps: dict[str, str] = {}
    named_sizes: dict[str, int] = {}
    for raw_idx, name in label_map.items():
        named_cents[name] = tuple(float(x) for x in cents[raw_idx].tolist())
        named_fps[name] = _centroid_fp(named_cents[name])
        named_sizes[name] = sizes[raw_idx]

    order = tuple(sorted(named_cents.keys(), key=lambda n: (named_fps[n], n)))
    # Enforce team_a before team_b by fingerprint rule already applied.
    if order != ("team_a", "team_b"):
        # Re-map if somehow inverted (should not happen with asc fp ordering).
        order = ("team_a", "team_b")

    return TeamClusterModel(
        status="ok",
        reason_codes=(),
        centroids=named_cents,
        centroid_fingerprints=named_fps,
        seed_track_ids=tuple(s.track_id for s in all_seeds),
        cluster_sizes=named_sizes,
        separation=float(sep),
        label_order=order,
        provenance={
            "init": "farthest_pair",
            "seed_count": len(all_seeds),
            "input_seed_count": len(seeds),
            "separation": sep,
            "centroid_fingerprints": dict(named_fps),
            "label_ordering": "centroid_fingerprint_asc",
            "outlier_track_ids": outlier_track_ids,
        },
    )


def align_centroids_across_shots(
    local: TeamClusterModel,
    reference: TeamClusterModel,
    *,
    config: Mapping[str, Any],
) -> TeamClusterModel:
    """Align local labels to reference only with strong centroid evidence; else no swap."""
    cl = config["clustering"]
    if not cl["cross_shot_alignment_enabled"]:
        return local
    if local.status != "ok" or reference.status != "ok":
        return local
    if not local.centroids or not reference.centroids:
        return local
    min_sim = float(cl["cross_shot_alignment_min_similarity"])
    # Compare team_a local to both reference centroids
    d_aa = 1.0 - _cosine_distance(local.centroids["team_a"], reference.centroids["team_a"])
    d_ab = 1.0 - _cosine_distance(local.centroids["team_a"], reference.centroids["team_b"])
    # Cosine distance → similarity = 1 - dist
    # Strong match required
    if max(d_aa, d_ab) < min_sim:
        # Ambiguous alignment — keep local labels (already fingerprint-ordered); no silent swap.
        return TeamClusterModel(
            status=local.status,
            reason_codes=tuple([*local.reason_codes, "CROSS_SHOT_ALIGNMENT_SKIPPED"]),
            centroids=local.centroids,
            centroid_fingerprints=local.centroid_fingerprints,
            seed_track_ids=local.seed_track_ids,
            cluster_sizes=local.cluster_sizes,
            separation=local.separation,
            label_order=local.label_order,
            provenance={
                **dict(local.provenance),
                "cross_shot_alignment": "skipped_weak",
                "sim_aa": d_aa,
                "sim_ab": d_ab,
            },
        )
    if d_ab > d_aa and d_ab >= min_sim:
        # Need swap to match reference — only when clearly better
        swapped_cents = {
            "team_a": local.centroids["team_b"],
            "team_b": local.centroids["team_a"],
        }
        swapped_fps = {k: _centroid_fp(v) for k, v in swapped_cents.items()}
        swapped_sizes = {
            "team_a": local.cluster_sizes.get("team_b", 0),
            "team_b": local.cluster_sizes.get("team_a", 0),
        }
        return TeamClusterModel(
            status=local.status,
            reason_codes=tuple([*local.reason_codes, "CROSS_SHOT_ALIGNED_SWAP"]),
            centroids=swapped_cents,
            centroid_fingerprints=swapped_fps,
            seed_track_ids=local.seed_track_ids,
            cluster_sizes=swapped_sizes,
            separation=local.separation,
            label_order=("team_a", "team_b"),
            provenance={
                **dict(local.provenance),
                "cross_shot_alignment": "swapped",
                "sim_aa": d_aa,
                "sim_ab": d_ab,
            },
        )
    return TeamClusterModel(
        status=local.status,
        reason_codes=tuple([*local.reason_codes, "CROSS_SHOT_ALIGNED_KEEP"]),
        centroids=local.centroids,
        centroid_fingerprints=local.centroid_fingerprints,
        seed_track_ids=local.seed_track_ids,
        cluster_sizes=local.cluster_sizes,
        separation=local.separation,
        label_order=local.label_order,
        provenance={
            **dict(local.provenance),
            "cross_shot_alignment": "kept",
            "sim_aa": d_aa,
            "sim_ab": d_ab,
        },
    )


def score_against_clusters(
    vector: Sequence[float],
    model: TeamClusterModel,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Nearest anonymous team with margin/outlier handling."""
    if model.status != "ok" or not model.centroids:
        return {
            "team_id": "unknown",
            "distance": None,
            "margin": None,
            "status": "unknown",
            "reason_codes": ["NO_CLUSTER_MODEL"],
        }
    cl = config["clustering"]
    d_a = _cosine_distance(vector, model.centroids["team_a"])
    d_b = _cosine_distance(vector, model.centroids["team_b"])
    if d_a <= d_b:
        best, second, best_id = d_a, d_b, "team_a"
    else:
        best, second, best_id = d_b, d_a, "team_b"
    margin = float(second - best)
    reasons: list[str] = []
    if best > float(cl["outlier_distance"]):
        return {
            "team_id": "unknown",
            "distance": best,
            "margin": margin,
            "status": "unknown",
            "reason_codes": ["THIRD_COLOR_OUTLIER", "OUTLIER_DISTANCE"],
        }
    if best > float(cl["assignment_max_distance"]):
        return {
            "team_id": "unknown",
            "distance": best,
            "margin": margin,
            "status": "unknown",
            "reason_codes": ["ASSIGNMENT_DISTANCE"],
        }
    if margin < float(cl["ambiguity_margin"]):
        return {
            "team_id": "unknown",
            "distance": best,
            "margin": margin,
            "status": "ambiguous",
            "reason_codes": ["AMBIGUOUS_MARGIN", "SIMILAR_KIT_RISK"],
        }
    return {
        "team_id": best_id,
        "distance": best,
        "margin": margin,
        "status": "assigned",
        "reason_codes": reasons,
    }


__all__ = [
    "ANONYMOUS_TEAM_IDS",
    "COLOR_DIM_DEFAULT",
    "TeamClusteringError",
    "SeedTrack",
    "TeamClusterModel",
    "team_feature_vector",
    "resolve_track_role",
    "collect_seed_tracks",
    "fit_two_team_clusters",
    "align_centroids_across_shots",
    "score_against_clusters",
]
