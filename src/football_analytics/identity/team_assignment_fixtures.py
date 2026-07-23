"""Synthetic team assignment fixtures (Stage 7C) — no real video required."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from football_analytics.identity.appearance_reid_fixtures import (
    _attr,
    _bundle,
    _kit_crop,
    _obs,
)

RUNTIME_ROOT = Path("/home/fdoblak/workspace/team_assignment_checks")


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not str(RUNTIME_ROOT).startswith("/home/fdoblak/workspace/"):
        raise RuntimeError("team assignment runtime root escape")


def _role_by_detections(attrs: list[dict[str, Any]], obs: list[dict[str, Any]]) -> dict[int, str]:
    """Map track_id → majority role via detection_id join."""
    det_role = {
        (int(a["frame_index"]), int(a["detection_id"])): str(a.get("role_label") or "unknown")
        for a in attrs
    }
    track_labels: dict[int, list[str]] = {}
    for o in obs:
        did = o.get("detection_id")
        if did is None:
            continue
        key = (int(o["frame_index"]), int(did))
        lab = det_role.get(key, "unknown")
        track_labels.setdefault(int(o["track_id"]), []).append(lab)
    out: dict[int, str] = {}
    for tid, labs in track_labels.items():
        counts: dict[str, int] = {}
        for lab in labs:
            counts[lab] = counts.get(lab, 0) + 1
        out[tid] = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return out


def _multi_track_bundle(
    *,
    run_id: str,
    video_id: str,
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    """tracks: list of {track_id, role, crop, frames: list[int], shot_id?}."""
    obs: list[dict[str, Any]] = []
    attrs: list[dict[str, Any]] = []
    crops: dict[tuple[int, int], np.ndarray] = {}
    shot_by_track: dict[int, str] = {}
    did = 1
    for t in tracks:
        tid = int(t["track_id"])
        role = str(t.get("role") or "player")
        crop = t["crop"]
        frames = list(t["frames"])
        if t.get("shot_id"):
            shot_by_track[tid] = str(t["shot_id"])
        for fi in frames:
            obs.append(
                _obs(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=int(fi),
                    track_id=tid,
                    detection_id=did,
                    bbox=(10.0 + tid, 10.0, 42.0 + tid, 74.0),
                )
            )
            attrs.append(
                _attr(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=int(fi),
                    detection_id=did,
                    role_label=role,
                )
            )
            crops[(int(fi), did)] = crop.copy()
            did += 1
    bundle = _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )
    bundle["role_by_track"] = _role_by_detections(attrs, obs)
    bundle["shot_by_track"] = shot_by_track
    if any("prior_team" in t for t in tracks):
        bundle["prior_team_by_track"] = {
            int(t["track_id"]): str(t["prior_team"]) for t in tracks if t.get("prior_team")
        }
    if any("gt_team" in t for t in tracks):
        bundle["synthetic_gt_team"] = {
            int(t["track_id"]): str(t["gt_team"]) for t in tracks if t.get("gt_team")
        }
    return bundle


def fixture_two_distinct_teams() -> dict[str, Any]:
    """1. Clearly different two kits → team_a / team_b separation."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000001", "video_two_teams"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 4, i * 4 + 1, i * 4 + 2],
                "gt_team": "team_red",
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 4, 21 + i * 4, 22 + i * 4],
                "gt_team": "team_blue",
            }
        )
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_similar_kit_hard() -> dict[str, Any]:
    """2. Near-identical kits → abstain / ambiguous."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000002", "video_similar_kit"
    a = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(30, 30, 30), seed=21)
    b = _kit_crop(upper_bgr=(42, 42, 198), lower_bgr=(32, 32, 32), seed=22)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {"track_id": tid, "role": "player", "crop": a, "frames": [i * 3, i * 3 + 1, i * 3 + 2]}
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": b,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_same_upper_diff_shorts() -> dict[str, Any]:
    """3. Same upper, different shorts."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000003", "video_shorts"
    a = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=31)
    b = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 180, 20), seed=32)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {"track_id": tid, "role": "player", "crop": a, "frames": [i * 3, i * 3 + 1, i * 3 + 2]}
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": b,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_brightness_shift_teams() -> dict[str, Any]:
    """4. Lighting shift within same kits."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000004", "video_bright"
    dark = _kit_crop(upper_bgr=(30, 30, 200), lower_bgr=(20, 20, 20), seed=41, brightness=0.7)
    light = _kit_crop(upper_bgr=(30, 30, 200), lower_bgr=(20, 20, 20), seed=42, brightness=1.3)
    other = _kit_crop(upper_bgr=(200, 40, 40), lower_bgr=(20, 20, 20), seed=43)
    tracks = [
        {"track_id": 1, "role": "player", "crop": dark, "frames": [0, 1, 2]},
        {"track_id": 2, "role": "player", "crop": light, "frames": [3, 4, 5]},
        {"track_id": 3, "role": "player", "crop": dark, "frames": [6, 7, 8]},
        {"track_id": 4, "role": "player", "crop": other, "frames": [20, 21, 22]},
        {"track_id": 5, "role": "player", "crop": other, "frames": [23, 24, 25]},
        {"track_id": 6, "role": "player", "crop": other, "frames": [26, 27, 28]},
    ]
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_with_referee() -> dict[str, Any]:
    """5. Referee present — not seeded / not_eligible."""
    base = fixture_two_distinct_teams()
    run_id, video_id = "run_20260723T200000000000Z_b11100000005", "video_ref"
    ref = _kit_crop(upper_bgr=(0, 200, 200), lower_bgr=(0, 200, 200), seed=51)
    tracks = []
    # Rebuild from two teams + referee
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    tracks.append({"track_id": 99, "role": "referee", "crop": ref, "frames": [40, 41, 42]})
    _ = base
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_with_staff() -> dict[str, Any]:
    """6. Staff present — not_eligible."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000006", "video_staff"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    staff = _kit_crop(upper_bgr=(180, 180, 180), lower_bgr=(100, 100, 100), seed=61)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    tracks.append({"track_id": 88, "role": "staff", "crop": staff, "frames": [50, 51, 52]})
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_goalkeeper_different_kit() -> dict[str, Any]:
    """7. Goalkeeper different kit — no auto team from kit."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000007", "video_gk"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    gk = _kit_crop(upper_bgr=(0, 220, 220), lower_bgr=(0, 180, 180), seed=71)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    tracks.append({"track_id": 77, "role": "goalkeeper", "crop": gk, "frames": [60, 61, 62]})
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_unknown_role() -> dict[str, Any]:
    """8. Unknown role — no seed; candidate at most."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000008", "video_unk_role"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    tracks.append({"track_id": 55, "role": "unknown", "crop": red, "frames": [70, 71, 72]})
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_insufficient_seeds() -> dict[str, Any]:
    """9. Too few player seeds."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000009", "video_few_seeds"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    tracks = [
        {"track_id": 1, "role": "player", "crop": red, "frames": [0, 1, 2]},
        {"track_id": 2, "role": "player", "crop": blue, "frames": [10, 11, 12]},
        {"track_id": 3, "role": "referee", "crop": red, "frames": [20, 21, 22]},
    ]
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_cluster_collapse() -> dict[str, Any]:
    """10. Single appearance / collapse."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000010", "video_collapse"
    one = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(30, 30, 30), seed=101)
    tracks = []
    for i, tid in enumerate([1, 2, 3, 4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": one,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_third_color_outlier() -> dict[str, Any]:
    """11. Third color outlier → unknown."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000011", "video_outlier"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    green = _kit_crop(upper_bgr=(40, 255, 40), lower_bgr=(0, 255, 0), seed=111)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    tracks.append({"track_id": 66, "role": "player", "crop": green, "frames": [80, 81, 82]})
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_cross_shot_alignment() -> dict[str, Any]:
    """12. Two shots with same kits — strong centroid alignment."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000012", "video_shots"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    tracks = [
        {"track_id": 1, "role": "player", "crop": red, "frames": [0, 1, 2], "shot_id": "shot_a"},
        {"track_id": 2, "role": "player", "crop": red, "frames": [3, 4, 5], "shot_id": "shot_a"},
        {"track_id": 3, "role": "player", "crop": blue, "frames": [6, 7, 8], "shot_id": "shot_a"},
        {"track_id": 4, "role": "player", "crop": blue, "frames": [9, 10, 11], "shot_id": "shot_a"},
        {"track_id": 5, "role": "player", "crop": red, "frames": [30, 31, 32], "shot_id": "shot_b"},
        {"track_id": 6, "role": "player", "crop": red, "frames": [33, 34, 35], "shot_id": "shot_b"},
        {
            "track_id": 7,
            "role": "player",
            "crop": blue,
            "frames": [36, 37, 38],
            "shot_id": "shot_b",
        },
        {
            "track_id": 8,
            "role": "player",
            "crop": blue,
            "frames": [39, 40, 41],
            "shot_id": "shot_b",
        },
    ]
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_label_permutation_gt() -> dict[str, Any]:
    """13. Synthetic GT for permutation-invariant eval (labels arbitrary)."""
    b = fixture_two_distinct_teams()
    b["video_id"] = "video_perm_gt"
    b["run_id"] = "run_20260723T200000000000Z_b11100000013"
    return b


def fixture_assignment_ambiguity() -> dict[str, Any]:
    """14. Near decision boundary ambiguity."""
    return fixture_similar_kit_hard()


def fixture_team_switch_conflict() -> dict[str, Any]:
    """15. Prior team differs → conflict/review."""
    run_id, video_id = "run_20260723T200000000000Z_b11100000015", "video_switch"
    red = _kit_crop(upper_bgr=(30, 30, 220), lower_bgr=(20, 20, 20), seed=11)
    blue = _kit_crop(upper_bgr=(220, 40, 40), lower_bgr=(20, 20, 20), seed=12)
    tracks = []
    for i, tid in enumerate([1, 2, 3]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": red,
                "frames": [i * 3, i * 3 + 1, i * 3 + 2],
            }
        )
    for i, tid in enumerate([4, 5, 6]):
        tracks.append(
            {
                "track_id": tid,
                "role": "player",
                "crop": blue,
                "frames": [20 + i * 3, 21 + i * 3, 22 + i * 3],
            }
        )
    # Track 1 looks red but prior says other anonymous team
    tracks[0]["prior_team"] = "team_b"
    return _multi_track_bundle(run_id=run_id, video_id=video_id, tracks=tracks)


def fixture_cross_video_reject() -> dict[str, Any]:
    """16. Cross-video auto transfer must be rejected by policy."""
    b = fixture_two_distinct_teams()
    b["cross_video_probe"] = True
    b["foreign_video_id"] = "video_other"
    return b


def fixture_hash_fk_mismatch_probe() -> dict[str, Any]:
    """17. Bundle with intentional track gap for FK checks (service still writes)."""
    b = fixture_two_distinct_teams()
    b["fk_probe_missing_track"] = 999
    return b


def fixture_evaluation_leakage_negative() -> dict[str, Any]:
    """18. Evaluation leakage class must not auto-confirm."""
    b = fixture_two_distinct_teams()
    b["force_leakage_class"] = "evaluation"
    return b


def fixture_deterministic_repeat() -> dict[str, Any]:
    """19. Same as two distinct teams for deterministic repeat."""
    return fixture_two_distinct_teams()


def all_team_fixtures() -> dict[str, Any]:
    return {
        "two_teams": fixture_two_distinct_teams,
        "similar_kit": fixture_similar_kit_hard,
        "same_upper_diff_shorts": fixture_same_upper_diff_shorts,
        "brightness": fixture_brightness_shift_teams,
        "referee": fixture_with_referee,
        "staff": fixture_with_staff,
        "goalkeeper": fixture_goalkeeper_different_kit,
        "unknown_role": fixture_unknown_role,
        "insufficient_seeds": fixture_insufficient_seeds,
        "collapse": fixture_cluster_collapse,
        "outlier": fixture_third_color_outlier,
        "cross_shot": fixture_cross_shot_alignment,
        "perm_gt": fixture_label_permutation_gt,
        "ambiguity": fixture_assignment_ambiguity,
        "team_switch": fixture_team_switch_conflict,
        "cross_video": fixture_cross_video_reject,
        "fk_probe": fixture_hash_fk_mismatch_probe,
        "leakage": fixture_evaluation_leakage_negative,
        "deterministic": fixture_deterministic_repeat,
    }


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "fixture_two_distinct_teams",
    "fixture_similar_kit_hard",
    "fixture_same_upper_diff_shorts",
    "fixture_brightness_shift_teams",
    "fixture_with_referee",
    "fixture_with_staff",
    "fixture_goalkeeper_different_kit",
    "fixture_unknown_role",
    "fixture_insufficient_seeds",
    "fixture_cluster_collapse",
    "fixture_third_color_outlier",
    "fixture_cross_shot_alignment",
    "fixture_label_permutation_gt",
    "fixture_assignment_ambiguity",
    "fixture_team_switch_conflict",
    "fixture_cross_video_reject",
    "fixture_hash_fk_mismatch_probe",
    "fixture_evaluation_leakage_negative",
    "fixture_deterministic_repeat",
    "all_team_fixtures",
]
