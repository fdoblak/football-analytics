"""Deterministic kit-color clustering (max 2 outfield clusters, no team names)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.role_features import RoleFeatures, kit_color_distance


@dataclass(frozen=True)
class KitCluster:
    cluster_id: int
    member_detection_ids: tuple[int, ...]
    centroid: tuple[float, ...]
    size: int
    stability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_detection_ids": list(self.member_detection_ids),
            "centroid": list(self.centroid),
            "size": self.size,
            "stability": self.stability,
            # Explicitly no team identity.
            "team_id": None,
            "team_name": None,
        }


@dataclass(frozen=True)
class ClusterAssignment:
    detection_id: int
    cluster_id: int | None
    distance_to_nearest: float
    distance_to_second: float | None
    margin: float
    is_outfield_member: bool


def _centroid(sigs: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not sigs:
        return ()
    dim = len(sigs[0])
    means = []
    for i in range(dim):
        means.append(sum(s[i] for s in sigs) / float(len(sigs)))
    return tuple(means)


def _sort_key(feat: RoleFeatures) -> tuple[Any, ...]:
    # Deterministic order: signature then detection_id.
    return (*feat.color_signature, int(feat.detection_id))


def cluster_kit_colors(
    features: Sequence[RoleFeatures],
    *,
    config: Mapping[str, Any],
) -> tuple[tuple[KitCluster, ...], dict[int, ClusterAssignment]]:
    """Greedy agglomerative kit clustering with max 2 outfield clusters.

    Clusters are sorted deterministically by centroid lexicographic order.
    No team_id / team_name is ever assigned.
    """
    clus_cfg = config["clustering"]
    max_out = int(clus_cfg["max_outfield_clusters"])
    min_size = int(clus_cfg["min_cluster_size"])
    thr = float(clus_cfg["color_distance_threshold"])
    min_stab = float(clus_cfg["min_cluster_stability"])

    ordered = sorted(features, key=_sort_key)
    # provisional clusters as lists of RoleFeatures
    buckets: list[list[RoleFeatures]] = []
    for feat in ordered:
        placed = False
        for bucket in buckets:
            bucket_cent = _centroid([f.color_signature for f in bucket])
            if kit_color_distance(feat.color_signature, bucket_cent) <= thr:
                bucket.append(feat)
                placed = True
                break
        if not placed:
            buckets.append([feat])

    # Rank buckets by size desc, then centroid lex, keep up to max_out as outfield.
    ranked = sorted(
        buckets,
        key=lambda b: (
            -len(b),
            _centroid([f.color_signature for f in b]),
            min(f.detection_id for f in b),
        ),
    )
    outfield_raw: list[list[RoleFeatures]] = []
    for bucket in ranked:
        if len(outfield_raw) >= max_out:
            break
        if len(bucket) >= min_size:
            outfield_raw.append(bucket)

    clusters: list[KitCluster] = []
    for i, bucket in enumerate(
        sorted(
            outfield_raw,
            key=lambda b: (
                _centroid([f.color_signature for f in b]),
                min(x.detection_id for x in b),
            ),
        )
    ):
        cent = _centroid([f.color_signature for f in bucket])
        # Stability: fraction of members within threshold of centroid.
        inside = sum(1 for f in bucket if kit_color_distance(f.color_signature, cent) <= thr)
        stability = inside / float(len(bucket))
        if stability < min_stab:
            continue
        members = tuple(sorted(int(f.detection_id) for f in bucket))
        clusters.append(
            KitCluster(
                cluster_id=i,
                member_detection_ids=members,
                centroid=cent,
                size=len(members),
                stability=float(stability),
            )
        )

    # Re-index after stability filter, keep deterministic centroid sort.
    clusters = sorted(clusters, key=lambda c: (c.centroid, c.member_detection_ids))
    clusters = [
        KitCluster(
            cluster_id=i,
            member_detection_ids=c.member_detection_ids,
            centroid=c.centroid,
            size=c.size,
            stability=c.stability,
        )
        for i, c in enumerate(clusters[:max_out])
    ]

    assignments: dict[int, ClusterAssignment] = {}
    for feat in features:
        dists: list[tuple[float, int]] = []
        for c in clusters:
            dists.append((kit_color_distance(feat.color_signature, c.centroid), c.cluster_id))
        dists.sort(key=lambda t: (t[0], t[1]))
        if not dists:
            assignments[feat.detection_id] = ClusterAssignment(
                detection_id=feat.detection_id,
                cluster_id=None,
                distance_to_nearest=1.0,
                distance_to_second=None,
                margin=0.0,
                is_outfield_member=False,
            )
            continue
        d0, cid = dists[0]
        d1 = dists[1][0] if len(dists) > 1 else None
        margin = (d1 - d0) if d1 is not None else (1.0 - d0)
        member_ids = {m for c in clusters if c.cluster_id == cid for m in c.member_detection_ids}
        assignments[feat.detection_id] = ClusterAssignment(
            detection_id=feat.detection_id,
            cluster_id=cid if feat.detection_id in member_ids or d0 <= thr else None,
            distance_to_nearest=float(d0),
            distance_to_second=None if d1 is None else float(d1),
            margin=float(margin),
            is_outfield_member=feat.detection_id in member_ids and d0 <= thr,
        )
    return tuple(clusters), assignments


__all__ = [
    "KitCluster",
    "ClusterAssignment",
    "cluster_kit_colors",
]
