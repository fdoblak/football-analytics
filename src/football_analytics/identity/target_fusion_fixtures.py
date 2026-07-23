"""Synthetic fixtures for Stage 7E target identity fusion (not accuracy claims)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.identity.fixtures import evidence_row
from football_analytics.identity.types import (
    EvidencePolarity,
    EvidenceType,
    LeakageClass,
    ReliabilityTier,
)

RUNTIME_ROOT = Path("/home/fdoblak/workspace/target_identity_checks")


def assert_runtime_root() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    return RUNTIME_ROOT


def _ids(prefix: str = "tf") -> dict[str, str]:
    return {
        "run_id": generate_run_id(),
        "video_id": f"video_{prefix}_01",
        "target_player_id": "target_player_01",
        "request_id": f"req_{prefix}_01",
    }


def fixture_appearance_only() -> dict[str, Any]:
    ids = _ids("app")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "appearance_only",
        **ids,
        "tracks": [{"track_id": 0, "start": 0, "end": 20, "coverage": 21}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_app_01",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=0,
                start_frame_index=0,
                end_frame_index=20,
                score=0.82,
                reason_codes=["APPEARANCE_SUPPORT"],
            )
        ],
        "manual_anchors": [],
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_jersey_only() -> dict[str, Any]:
    ids = _ids("jer")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "jersey_only",
        **ids,
        "tracks": [{"track_id": 1, "start": 0, "end": 15, "coverage": 16}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_jer_01",
                evidence_type=EvidenceType.JERSEY_NUMBER.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=1,
                start_frame_index=0,
                end_frame_index=15,
                reason_codes=["JERSEY_SUPPORT"],
            )
        ],
        "manual_anchors": [],
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_team_only() -> dict[str, Any]:
    ids = _ids("team")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "team_only",
        **ids,
        "tracks": [{"track_id": 2, "start": 0, "end": 12, "coverage": 13}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_team_01",
                evidence_type=EvidenceType.TEAM_ASSIGNMENT.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=2,
                start_frame_index=0,
                end_frame_index=12,
                reason_codes=["TEAM_SUPPORT"],
            )
        ],
        "manual_anchors": [],
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_two_auto_provisional() -> dict[str, Any]:
    ids = _ids("two")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "two_auto_provisional",
        **ids,
        "tracks": [{"track_id": 3, "start": 0, "end": 30, "coverage": 31}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_app_02",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=3,
                start_frame_index=0,
                end_frame_index=30,
            ),
            evidence_row(
                rid,
                vid,
                "ev_jer_02",
                evidence_type=EvidenceType.JERSEY_NUMBER.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=3,
                start_frame_index=0,
                end_frame_index=30,
            ),
        ],
        "manual_anchors": [],
        "expected_max_status": "provisional",
        "synthetic_expected_track_ids": [],
    }


def fixture_scoped_manual_confirm() -> dict[str, Any]:
    ids = _ids("man")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "scoped_manual_confirm",
        **ids,
        "tracks": [
            {"track_id": 4, "start": 0, "end": 40, "coverage": 41},
            {"track_id": 5, "start": 50, "end": 80, "coverage": 31},
        ],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_man_01",
                evidence_type=EvidenceType.MANUAL_TRACK_ANCHOR.value,
                reliability_tier=ReliabilityTier.MANUAL_VERIFIED.value,
                track_id=4,
                start_frame_index=0,
                end_frame_index=40,
            ),
            evidence_row(
                rid,
                vid,
                "ev_app_linked",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=5,
                start_frame_index=50,
                end_frame_index=80,
            ),
            evidence_row(
                rid,
                vid,
                "ev_team_linked",
                evidence_type=EvidenceType.TEAM_ASSIGNMENT.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=5,
                start_frame_index=50,
                end_frame_index=80,
            ),
        ],
        "manual_anchors": [
            {
                "track_id": 4,
                "start_frame_index": 0,
                "end_frame_index": 40,
                "anchor_type": "track_interval",
            }
        ],
        "confirm_track_id": 4,
        "confirm_start": 0,
        "confirm_end": 40,
        "expected_max_status": "provisional",  # before manual decide
        "synthetic_expected_track_ids": [4],
    }


def fixture_conflict_jersey_team() -> dict[str, Any]:
    ids = _ids("cnf")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "conflict_jersey_team",
        **ids,
        "tracks": [{"track_id": 6, "start": 0, "end": 25, "coverage": 26}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_jer_c",
                evidence_type=EvidenceType.JERSEY_NUMBER.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=6,
                polarity=EvidencePolarity.SUPPORTS.value,
            ),
            evidence_row(
                rid,
                vid,
                "ev_team_c",
                evidence_type=EvidenceType.TEAM_ASSIGNMENT.value,
                reliability_tier=ReliabilityTier.CONFLICTING.value,
                track_id=6,
                polarity=EvidencePolarity.CONFLICTS.value,
                reason_codes=["TEAM_JERSEY_MISMATCH"],
            ),
        ],
        "manual_anchors": [],
        "expected_max_status": "rejected",
        "synthetic_expected_track_ids": [],
    }


def fixture_long_gap_candidate() -> dict[str, Any]:
    ids = _ids("gap")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "long_gap_candidate",
        **ids,
        "tracks": [{"track_id": 7, "start": 200, "end": 240, "coverage": 41}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_app_gap",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=7,
                start_frame_index=200,
                end_frame_index=240,
            ),
            evidence_row(
                rid,
                vid,
                "ev_jer_gap",
                evidence_type=EvidenceType.JERSEY_NUMBER.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=7,
                start_frame_index=200,
                end_frame_index=240,
            ),
        ],
        "manual_anchors": [],
        "long_gap_after_cut": True,
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_cross_video_forbidden() -> dict[str, Any]:
    ids = _ids("xvid")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "cross_video_forbidden",
        **ids,
        "tracks": [{"track_id": 8, "start": 0, "end": 10, "coverage": 11}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_xvid",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=8,
                reason_codes=["CROSS_VIDEO_AUTO_LINK_FORBIDDEN"],
                quality_flags=["cross_video_auto_link"],
            )
        ],
        "manual_anchors": [],
        "expect_cross_video_fail": True,
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_evaluation_leakage() -> dict[str, Any]:
    ids = _ids("leak")
    rid, vid = ids["run_id"], ids["video_id"]
    return {
        "name": "evaluation_leakage",
        **ids,
        "tracks": [{"track_id": 9, "start": 0, "end": 10, "coverage": 11}],
        "evidence": [
            evidence_row(
                rid,
                vid,
                "ev_leak",
                evidence_type=EvidenceType.APPEARANCE_SIMILARITY.value,
                reliability_tier=ReliabilityTier.SUPPORTING.value,
                track_id=9,
                leakage_class=LeakageClass.EVALUATION.value,
                reason_codes=["evaluation_label"],
            )
        ],
        "manual_anchors": [],
        "expect_leakage_fail": True,
        "expected_max_status": "candidate",
        "synthetic_expected_track_ids": [],
    }


def fixture_e2e_bundle() -> dict[str, Any]:
    """Full synthetic E2E: prepare → confirm/reject/revoke (synthetic decisions)."""
    base = fixture_scoped_manual_confirm()
    base["name"] = "e2e_bundle"
    return base


FIXTURE_REGISTRY: dict[str, Any] = {
    "appearance_only": fixture_appearance_only,
    "jersey_only": fixture_jersey_only,
    "team_only": fixture_team_only,
    "two_auto_provisional": fixture_two_auto_provisional,
    "scoped_manual_confirm": fixture_scoped_manual_confirm,
    "conflict_jersey_team": fixture_conflict_jersey_team,
    "long_gap_candidate": fixture_long_gap_candidate,
    "cross_video_forbidden": fixture_cross_video_forbidden,
    "evaluation_leakage": fixture_evaluation_leakage,
    "e2e_bundle": fixture_e2e_bundle,
}


def get_fixture(name: str) -> Mapping[str, Any]:
    if name not in FIXTURE_REGISTRY:
        raise KeyError(f"unknown fixture: {name}")
    return FIXTURE_REGISTRY[name]()


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "FIXTURE_REGISTRY",
    "get_fixture",
    "fixture_appearance_only",
    "fixture_jersey_only",
    "fixture_team_only",
    "fixture_two_auto_provisional",
    "fixture_scoped_manual_confirm",
    "fixture_conflict_jersey_team",
    "fixture_long_gap_candidate",
    "fixture_cross_video_forbidden",
    "fixture_evaluation_leakage",
    "fixture_e2e_bundle",
]
