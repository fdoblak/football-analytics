"""Synthetic appearance ReID fixtures (Stage 7B) — no real video required."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_ROOT = Path("/home/fdoblak/workspace/appearance_reid_checks")


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not str(RUNTIME_ROOT).startswith("/home/fdoblak/workspace/"):
        raise RuntimeError("appearance ReID runtime root escape")


def _solid_bgr(
    h: int,
    w: int,
    bgr: tuple[int, int, int],
    *,
    noise: int = 0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = np.array(bgr, dtype=np.uint8)
    if noise > 0:
        n = rng.integers(-noise, noise + 1, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def _kit_crop(
    *,
    upper_bgr: tuple[int, int, int],
    lower_bgr: tuple[int, int, int],
    h: int = 64,
    w: int = 32,
    seed: int = 0,
    brightness: float = 1.0,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    split = h // 2
    img[:split, :] = np.array(upper_bgr, dtype=np.uint8)
    img[split:, :] = np.array(lower_bgr, dtype=np.uint8)
    if brightness != 1.0:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    # Mild texture so edge/texture bins are non-degenerate.
    rng = np.random.default_rng(seed)
    tex = rng.integers(0, 8, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)
    return img


def _obs(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    track_id: int,
    detection_id: int | None,
    bbox: tuple[float, float, float, float],
    observation_state: str = "observed",
    class_id: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "track_id": track_id,
        "detection_id": detection_id,
        "class_id": class_id,
        "confidence": 0.9,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "observation_state": observation_state,
        "model_id": "synthetic_tracker",
        "quality_flags": [],
    }


def _attr(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    entity_type: str = "human",
    role_label: str = "player",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "entity_type": entity_type,
        "role_label": role_label,
        "role_confidence": 0.8,
        "quality_flags": [],
    }


def _bundle(
    *,
    run_id: str,
    video_id: str,
    observations: list[dict[str, Any]],
    attributes: list[dict[str, Any]],
    crops: dict[tuple[int, int], np.ndarray],
    entities: dict[int, str] | None = None,
) -> dict[str, Any]:
    tracks = sorted({int(o["track_id"]) for o in observations})
    return {
        "run_id": run_id,
        "video_id": video_id,
        "observations": observations,
        "attributes": attributes,
        "synthetic_crops": crops,
        "frames_bgr": None,
        "entity_by_track": entities or {t: "human" for t in tracks},
        "frame_times_us": {
            int(o["frame_index"]): int(o["frame_index"]) * 40000 for o in observations
        },
    }


def fixture_same_appearance_different_tracklets() -> dict[str, Any]:
    """1. Same kit appearance across two non-overlapping tracklets."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000001", "video_same_app"
    crop = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=1)
    obs, attrs, crops = [], [], {}
    # Track 1: frames 0-3; Track 2: frames 10-13 (no overlap)
    for i, fi in enumerate([0, 1, 2, 3]):
        did = 100 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop.copy()
    for i, fi in enumerate([10, 11, 12, 13]):
        did = 200 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=2,
                detection_id=did,
                bbox=(50, 10, 82, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_different_appearance() -> dict[str, Any]:
    """2. Distinct kit colors → low similarity."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000002", "video_diff_app"
    red = _kit_crop(upper_bgr=(40, 40, 220), lower_bgr=(30, 30, 30), seed=2)
    green = _kit_crop(upper_bgr=(40, 220, 40), lower_bgr=(30, 30, 30), seed=3)
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1, 2]):
        did = 10 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = red.copy()
    for i, fi in enumerate([10, 11, 12]):
        did = 20 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=2,
                detection_id=did,
                bbox=(50, 10, 82, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = green.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_same_kit_hard_negative() -> dict[str, Any]:
    """3. Same team color, different players (hard-negative)."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000003", "video_same_kit"
    # Same upper hue family, slight texture/seed difference.
    a = _kit_crop(upper_bgr=(50, 50, 210), lower_bgr=(25, 25, 25), seed=11)
    b = _kit_crop(upper_bgr=(55, 45, 205), lower_bgr=(28, 22, 30), seed=12)
    return _two_track_bundle(run_id, video_id, a, b)


def fixture_upper_same_lower_different() -> dict[str, Any]:
    """4. Same upper kit, different lower."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000004", "video_upper_same"
    a = _kit_crop(upper_bgr=(30, 30, 200), lower_bgr=(200, 200, 200), seed=21)
    b = _kit_crop(upper_bgr=(30, 30, 200), lower_bgr=(20, 20, 20), seed=22)
    return _two_track_bundle(run_id, video_id, a, b)


def fixture_brightness_shift() -> dict[str, Any]:
    """5. Brightness/contrast shift of same appearance."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000005", "video_bright"
    base = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=31, brightness=1.0)
    bright = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=31, brightness=1.35)
    return _two_track_bundle(run_id, video_id, base, bright)


def fixture_tiny_corrupt_crop() -> dict[str, Any]:
    """6. Tiny/corrupt crops → rejected samples / insufficient evidence."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000006", "video_tiny"
    tiny = _solid_bgr(4, 4, (100, 100, 100), seed=1)
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1]):
        did = 1 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(0, 0, 4, 4),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = tiny
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_partial_occlusion() -> dict[str, Any]:
    """7. Partial occlusion (lower half blacked out on some crops)."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000007", "video_occ"
    full = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 180, 20), seed=41)
    occ = full.copy()
    occ[occ.shape[0] // 2 :, :] = 0
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1, 2]):
        did = 10 + i
        crop = occ if i == 1 else full
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_single_crop_insufficient() -> dict[str, Any]:
    """8. Single observed crop → insufficient_appearance_evidence."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000008", "video_single"
    crop = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=51)
    obs = [
        _obs(
            run_id=run_id,
            video_id=video_id,
            frame_index=0,
            track_id=1,
            detection_id=1,
            bbox=(10, 10, 42, 74),
        )
    ]
    attrs = [_attr(run_id=run_id, video_id=video_id, frame_index=0, detection_id=1)]
    return _bundle(
        run_id=run_id,
        video_id=video_id,
        observations=obs,
        attributes=attrs,
        crops={(0, 1): crop},
    )


def fixture_temporal_overlap() -> dict[str, Any]:
    """9. Temporally overlapping tracks cannot link."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000009", "video_overlap"
    a = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=61)
    b = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=62)
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1, 2, 3]):
        did = 10 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = a.copy()
    for i, fi in enumerate([2, 3, 4, 5]):
        did = 20 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=2,
                detection_id=did,
                bbox=(50, 10, 82, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = b.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_cross_shot_candidate() -> dict[str, Any]:
    """10. Cross-shot gap → candidate evidence only (same video)."""
    return fixture_same_appearance_different_tracklets()


def fixture_cross_video_reject() -> dict[str, Any]:
    """11. Cross-video metadata for reject path (caller uses different video ids)."""
    b = fixture_same_appearance_different_tracklets()
    b["cross_video_probe"] = {
        "source_video_id": b["video_id"],
        "target_video_id": "other_video",
    }
    return b


def fixture_human_ball_reject() -> dict[str, Any]:
    """12. Human-ball link forbidden."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000012", "video_ball"
    human = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=71)
    ball = _solid_bgr(24, 24, (30, 30, 30), noise=5, seed=72)
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1, 2]):
        did = 10 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
                class_id=0,
            )
        )
        attrs.append(
            _attr(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                detection_id=did,
                entity_type="human",
            )
        )
        crops[(fi, did)] = human.copy()
    for i, fi in enumerate([10, 11, 12]):
        did = 30 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=9,
                detection_id=did,
                bbox=(100, 100, 124, 124),
                class_id=32,
            )
        )
        attrs.append(
            _attr(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                detection_id=did,
                entity_type="ball",
                role_label="unknown",
            )
        )
        crops[(fi, did)] = ball.copy()
    return _bundle(
        run_id=run_id,
        video_id=video_id,
        observations=obs,
        attributes=attrs,
        crops=crops,
        entities={1: "human", 9: "ball"},
    )


def fixture_ambiguity_near_scores() -> dict[str, Any]:
    """13. Three similar kits → ambiguity margin / review."""
    run_id, video_id = "run_20260723T190000000000Z_a11100000013", "video_ambig"
    crops_kit = [
        _kit_crop(upper_bgr=(45, 45, 200), lower_bgr=(20, 20, 20), seed=80 + i) for i in range(3)
    ]
    obs, attrs, crops = [], [], {}
    frame_blocks = [(0, 1, 2), (10, 11, 12), (20, 21, 22)]
    for tid, (frames, kit) in enumerate(zip(frame_blocks, crops_kit, strict=True), start=1):
        for j, fi in enumerate(frames):
            did = tid * 100 + j
            obs.append(
                _obs(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=fi,
                    track_id=tid,
                    detection_id=did,
                    bbox=(10.0 * tid, 10, 10.0 * tid + 32, 74),
                )
            )
            attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
            crops[(fi, did)] = kit.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def fixture_role_conflict() -> dict[str, Any]:
    """14. Role conflict warning between player and referee tracks."""
    b = fixture_same_appearance_different_tracklets()
    for a in b["attributes"]:
        if int(a["detection_id"]) >= 200:
            a["role_label"] = "referee"
    b["role_conflict_pairs"] = {(1, 2)}
    return b


def fixture_predicted_rejected() -> dict[str, Any]:
    """Predicted/interpolated observations must not contribute samples."""
    run_id, video_id = "run_20260723T190000000000Z_a111000000aa", "video_pred"
    crop = _kit_crop(upper_bgr=(40, 40, 200), lower_bgr=(20, 20, 20), seed=91)
    obs, attrs, crops = [], [], {}
    states = ["observed", "predicted", "interpolated", "observed"]
    for i, (fi, st) in enumerate(zip([0, 1, 2, 3], states, strict=True)):
        did = 1 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
                observation_state=st,
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def _two_track_bundle(
    run_id: str, video_id: str, crop_a: np.ndarray, crop_b: np.ndarray
) -> dict[str, Any]:
    obs, attrs, crops = [], [], {}
    for i, fi in enumerate([0, 1, 2]):
        did = 10 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=1,
                detection_id=did,
                bbox=(10, 10, 42, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop_a.copy()
    for i, fi in enumerate([10, 11, 12]):
        did = 20 + i
        obs.append(
            _obs(
                run_id=run_id,
                video_id=video_id,
                frame_index=fi,
                track_id=2,
                detection_id=did,
                bbox=(50, 10, 82, 74),
            )
        )
        attrs.append(_attr(run_id=run_id, video_id=video_id, frame_index=fi, detection_id=did))
        crops[(fi, did)] = crop_b.copy()
    return _bundle(
        run_id=run_id, video_id=video_id, observations=obs, attributes=attrs, crops=crops
    )


def all_core_fixtures() -> Mapping[str, Any]:
    return {
        "same_appearance": fixture_same_appearance_different_tracklets,
        "different_appearance": fixture_different_appearance,
        "same_kit": fixture_same_kit_hard_negative,
        "upper_same": fixture_upper_same_lower_different,
        "brightness": fixture_brightness_shift,
        "tiny": fixture_tiny_corrupt_crop,
        "occlusion": fixture_partial_occlusion,
        "single": fixture_single_crop_insufficient,
        "overlap": fixture_temporal_overlap,
        "cross_shot": fixture_cross_shot_candidate,
        "cross_video": fixture_cross_video_reject,
        "human_ball": fixture_human_ball_reject,
        "ambiguity": fixture_ambiguity_near_scores,
        "role_conflict": fixture_role_conflict,
        "predicted": fixture_predicted_rejected,
    }


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "fixture_same_appearance_different_tracklets",
    "fixture_different_appearance",
    "fixture_same_kit_hard_negative",
    "fixture_upper_same_lower_different",
    "fixture_brightness_shift",
    "fixture_tiny_corrupt_crop",
    "fixture_partial_occlusion",
    "fixture_single_crop_insufficient",
    "fixture_temporal_overlap",
    "fixture_cross_shot_candidate",
    "fixture_cross_video_reject",
    "fixture_human_ball_reject",
    "fixture_ambiguity_near_scores",
    "fixture_role_conflict",
    "fixture_predicted_rejected",
    "all_core_fixtures",
]
