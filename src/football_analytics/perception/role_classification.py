"""Conservative human role assignment (Stage 5D).

Canonical RoleLabel uses ``staff`` (not user-facing ``other``).
Color difference alone never assigns goalkeeper or referee.
Generic human never auto-maps to player.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from football_analytics.perception.role_clustering import ClusterAssignment, KitCluster
from football_analytics.perception.role_features import RoleFeatures, kit_color_distance
from football_analytics.perception.types import ReviewStatus, RoleLabel, RoleSource


class AssignmentStatus(str, Enum):
    CLASSIFIED = "classified"
    ABSTAINED = "abstained"
    NOT_ELIGIBLE = "not_eligible"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class RoleAssignment:
    run_id: str
    video_id: str
    frame_index: int
    detection_id: int
    role_label: RoleLabel
    role_source: RoleSource
    role_score: float | None
    assignment_status: AssignmentStatus
    evidence_codes: tuple[str, ...]
    review_status: ReviewStatus
    review_required: bool
    crop_quality: float
    margin: float
    cluster_id: int | None
    config_fingerprint: str
    classifier_id: str
    classifier_version: str
    provenance: Mapping[str, Any]

    def to_attribute_row(self) -> dict[str, Any]:
        import json

        prov = {
            **dict(self.provenance),
            "assignment_status": self.assignment_status.value,
            "evidence_codes": list(self.evidence_codes),
            "raw_margin": self.margin,
            "cluster_id": self.cluster_id,
            "crop_quality": self.crop_quality,
            "review_required": self.review_required,
            "config_fingerprint": self.config_fingerprint,
            "classifier_id": self.classifier_id,
            "classifier_version": self.classifier_version,
            "other_maps_to": "staff",
            "team_id": None,
            "team_name": None,
        }
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "entity_type": "human",
            "role_label": self.role_label.value,
            "role_source": self.role_source.value,
            "role_score": self.role_score,
            "occlusion": None,
            "truncation": None,
            "visibility": None if self.crop_quality is None else float(self.crop_quality),
            "review_status": self.review_status.value,
            "attribute_source_ref": f"role:{self.classifier_id}:{self.detection_id}",
            "provenance_json": json.dumps(prov, sort_keys=True, separators=(",", ":")),
            "contract_version": 1,
        }


def _size_zscore(area: float, areas: Sequence[float]) -> float:
    if len(areas) < 2:
        return 0.0
    mean = statistics.fmean(areas)
    stdev = statistics.pstdev(areas)
    if stdev <= 1e-9:
        return 0.0
    return (area - mean) / stdev


def assign_roles_for_frame(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    features: Sequence[RoleFeatures],
    clusters: Sequence[KitCluster],
    cluster_assignments: Mapping[int, ClusterAssignment],
    config: Mapping[str, Any],
    config_fingerprint: str,
) -> list[RoleAssignment]:
    """Assign roles for one frame/window context. Conservative + abstention."""
    thr = config["thresholds"]
    elig = config["eligibility"]
    out_pol = config["output_policy"]
    classifier_id = str(config["classifier_id"])
    classifier_version = str(config["classifier_version"])
    role_source = RoleSource(str(out_pol["role_source"]))
    areas = [f.bbox_area for f in features]
    results: list[RoleAssignment] = []

    n_humans = len(features)
    staff_budget = max(1, int(math.floor(float(thr["staff_max_fraction"]) * max(n_humans, 1))))

    for feat in sorted(features, key=lambda f: int(f.detection_id)):
        ca = cluster_assignments.get(feat.detection_id)
        margin = float(ca.margin) if ca is not None else 0.0
        codes: list[str] = []
        role = RoleLabel.UNKNOWN
        status = AssignmentStatus.ABSTAINED
        review_required = True
        review = ReviewStatus.NEEDS_REVIEW

        # Quality / geometry gates → abstain (still eligible entity).
        dist_out: float | None = float(ca.distance_to_nearest) if ca else None
        if feat.crop_quality < float(elig["min_crop_quality"]) or feat.crop_quality < float(
            thr["player_min_quality"]
        ):
            codes.append("LOW_CROP_QUALITY")
            status = AssignmentStatus.ABSTAINED
            role = RoleLabel.UNKNOWN
        elif feat.bbox_area < float(elig["min_crop_area_px"]):
            codes.append("CROP_TOO_SMALL")
            status = AssignmentStatus.ABSTAINED
            role = RoleLabel.UNKNOWN
        elif feat.aspect_ratio < float(elig["min_aspect_ratio"]) or feat.aspect_ratio > float(
            elig["max_aspect_ratio"]
        ):
            codes.append("INVALID_ASPECT")
            status = AssignmentStatus.ABSTAINED
            role = RoleLabel.UNKNOWN
        else:
            # Candidate evidence
            is_outfield = bool(ca and ca.is_outfield_member)
            dist_out = float(ca.distance_to_nearest) if ca else 1.0
            dark = feat.mean_saturation <= float(
                thr["referee_max_saturation"]
            ) and feat.mean_value <= float(thr["referee_max_value"])
            min_dist_to_clusters = 1.0
            if clusters:
                min_dist_to_clusters = min(
                    kit_color_distance(feat.color_signature, c.centroid) for c in clusters
                )
            distinct_from_both = True
            if len(clusters) >= 1:
                distinct_from_both = all(
                    kit_color_distance(feat.color_signature, c.centroid)
                    >= float(thr["goalkeeper_color_margin"])
                    for c in clusters
                )
            lateral = feat.norm_cx <= float(thr["goalkeeper_lateral_edge"]) or feat.norm_cx >= (
                1.0 - float(thr["goalkeeper_lateral_edge"])
            )
            size_z = _size_zscore(feat.bbox_area, areas)
            size_outlier = abs(size_z) >= float(thr["goalkeeper_size_zscore"])
            extra_gk = lateral or size_outlier

            # Conflict / weak margin → unknown
            if margin < float(thr["conflict_margin"]) and is_outfield and dark:
                codes.append("CONFLICTING_EVIDENCE")
                role = RoleLabel.UNKNOWN
                status = AssignmentStatus.ABSTAINED
            elif is_outfield and margin >= float(thr["player_cluster_margin"]) and not dark:
                role = RoleLabel.PLAYER
                status = AssignmentStatus.CLASSIFIED
                codes.append("OUTFIELD_CLUSTER_MEMBER")
                review_required = False
                review = ReviewStatus.UNREVIEWED
            elif distinct_from_both and extra_gk and (not is_outfield):
                # Color alone insufficient — extra_gk required by config.
                # Cluster pairwise margin is unreliable for true outliers (both
                # outfield centroids may be similarly far); color margin + extra
                # evidence is the gate.
                role = RoleLabel.GOALKEEPER
                status = AssignmentStatus.CLASSIFIED
                codes.extend(["COLOR_DISTINCT_FROM_OUTFIELD", "GK_EXTRA_EVIDENCE"])
                if lateral:
                    codes.append("LATERAL_POSITION")
                if size_outlier:
                    codes.append("SIZE_OUTLIER")
                review_required = True
                review = ReviewStatus.NEEDS_REVIEW
            elif (
                dark
                and (not is_outfield)
                and min_dist_to_clusters >= float(thr["referee_color_margin"])
            ):
                role = RoleLabel.REFEREE
                status = AssignmentStatus.CLASSIFIED
                codes.extend(["DARK_LOW_SAT_KIT", "NOT_OUTFIELD_CLUSTER"])
                review_required = True
                review = ReviewStatus.NEEDS_REVIEW
            elif (
                (not is_outfield)
                and (not dark)
                and distinct_from_both
                and not extra_gk
                and n_humans > 0
            ):
                # Rare residual → staff (maps user "other"); weak official evidence.
                # Cap staff fraction; otherwise abstain.
                existing_staff = sum(
                    1
                    for r in results
                    if r.role_label == RoleLabel.STAFF
                    and r.assignment_status == AssignmentStatus.CLASSIFIED
                )
                if existing_staff < staff_budget and margin >= float(thr["abstain_margin"]):
                    role = RoleLabel.STAFF
                    status = AssignmentStatus.CLASSIFIED
                    codes.append("RESIDUAL_STAFF_OTHER_MAP")
                    review_required = True
                    review = ReviewStatus.NEEDS_REVIEW
                else:
                    role = RoleLabel.UNKNOWN
                    status = AssignmentStatus.ABSTAINED
                    codes.append("WEAK_EVIDENCE_ABSTAIN")
            else:
                role = RoleLabel.UNKNOWN
                status = AssignmentStatus.ABSTAINED
                if (
                    dark
                    and not extra_gk
                    and not (
                        min_dist_to_clusters >= float(thr["referee_color_margin"])
                        and not is_outfield
                    )
                ):
                    codes.append("DARK_COLOR_ALONE_INSUFFICIENT")
                if distinct_from_both and not extra_gk:
                    codes.append("COLOR_ALONE_INSUFFICIENT_FOR_GK")
                if margin < float(thr["abstain_margin"]):
                    codes.append("LOW_MARGIN_ABSTAIN")
                if not codes:
                    codes.append("INSUFFICIENT_EVIDENCE")

        # Never treat generic human as automatic player without cluster evidence.
        if role == RoleLabel.PLAYER and "OUTFIELD_CLUSTER_MEMBER" not in codes:
            role = RoleLabel.UNKNOWN
            status = AssignmentStatus.ABSTAINED
            codes.append("NO_AUTO_PLAYER_FROM_HUMAN")

        role_score = None if out_pol.get("role_score_null", True) else min(1.0, max(0.0, margin))
        results.append(
            RoleAssignment(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                detection_id=feat.detection_id,
                role_label=role,
                role_source=role_source,
                role_score=role_score,
                assignment_status=status,
                evidence_codes=tuple(codes),
                review_status=review,
                review_required=review_required,
                crop_quality=float(feat.crop_quality),
                margin=margin,
                cluster_id=None if ca is None else ca.cluster_id,
                config_fingerprint=config_fingerprint,
                classifier_id=classifier_id,
                classifier_version=classifier_version,
                provenance={
                    "stage": "5D",
                    "feature_source": feat.feature_source,
                    "dist_outfield": dist_out,
                    "mean_saturation": feat.mean_saturation,
                    "mean_value": feat.mean_value,
                    "norm_cx": feat.norm_cx,
                    "norm_cy": feat.norm_cy,
                },
            )
        )
    return results


def make_non_human_skip(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    entity_type: str,
    config: Mapping[str, Any],
    config_fingerprint: str,
) -> RoleAssignment:
    """Ball/unknown entities are skipped (not classified as roles)."""
    return RoleAssignment(
        run_id=run_id,
        video_id=video_id,
        frame_index=frame_index,
        detection_id=detection_id,
        role_label=RoleLabel.UNKNOWN,
        role_source=RoleSource.DOWNSTREAM_CLASSIFIER,
        role_score=None,
        assignment_status=AssignmentStatus.SKIPPED,
        evidence_codes=("NON_HUMAN_ENTITY", f"entity:{entity_type}"),
        review_status=ReviewStatus.UNREVIEWED,
        review_required=False,
        crop_quality=0.0,
        margin=0.0,
        cluster_id=None,
        config_fingerprint=config_fingerprint,
        classifier_id=str(config["classifier_id"]),
        classifier_version=str(config["classifier_version"]),
        provenance={"stage": "5D", "skipped_entity_type": entity_type},
    )


def make_not_eligible(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    reason: str,
    config: Mapping[str, Any],
    config_fingerprint: str,
) -> RoleAssignment:
    return RoleAssignment(
        run_id=run_id,
        video_id=video_id,
        frame_index=frame_index,
        detection_id=detection_id,
        role_label=RoleLabel.UNKNOWN,
        role_source=RoleSource.DOWNSTREAM_CLASSIFIER,
        role_score=None,
        assignment_status=AssignmentStatus.NOT_ELIGIBLE,
        evidence_codes=(reason,),
        review_status=ReviewStatus.UNREVIEWED,
        review_required=False,
        crop_quality=0.0,
        margin=0.0,
        cluster_id=None,
        config_fingerprint=config_fingerprint,
        classifier_id=str(config["classifier_id"]),
        classifier_version=str(config["classifier_version"]),
        provenance={"stage": "5D", "not_eligible_reason": reason},
    )


__all__ = [
    "AssignmentStatus",
    "RoleAssignment",
    "assign_roles_for_frame",
    "make_non_human_skip",
    "make_not_eligible",
]
