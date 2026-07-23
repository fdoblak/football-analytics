"""Synthetic jersey digit crops and observation bundles (Stage 7D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_analytics.identity.jersey_ocr import build_digit_templates

RUNTIME_ROOT = Path("/home/fdoblak/workspace/jersey_ocr_checks")


def _rid(n: int) -> str:
    return f"run_20260723T200000{n:06d}Z_b111{n:08d}"


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not str(RUNTIME_ROOT).startswith("/home/fdoblak/workspace/"):
        raise RuntimeError("jersey OCR runtime root escape")


def render_digit_crop(
    text: str,
    *,
    width: int = 64,
    height: int = 80,
    fg: int = 240,
    bg: int = 40,
    blur_ksize: int = 0,
    noise_std: float = 0.0,
    skew: float = 0.0,
    occlude_frac: float = 0.0,
    low_contrast: bool = False,
) -> np.ndarray:
    """Render synthetic jersey-number crop from programmatic digit templates."""
    if low_contrast:
        fg, bg = 140, 110
    canvas: np.ndarray = np.full((height, width), bg, dtype=np.uint8)
    if not text:
        if noise_std > 0:
            noise = np.random.default_rng(0).normal(0, noise_std, canvas.shape)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    templates = build_digit_templates(width=28, height=40)
    digit_imgs: list[np.ndarray] = []
    for ch in text:
        if not ch.isdigit():
            raise ValueError(f"non-digit in text: {text}")
        templ = (templates[int(ch)] * 255).astype(np.uint8)
        digit_imgs.append(templ)
    gap = 4
    total_w = sum(int(d.shape[1]) for d in digit_imgs) + gap * (len(digit_imgs) - 1)
    total_h = max(int(d.shape[0]) for d in digit_imgs)
    block = np.full((total_h, total_w), 0, dtype=np.uint8)
    x = 0
    for d in digit_imgs:
        block[:, x : x + d.shape[1]] = np.maximum(block[:, x : x + d.shape[1]], d)
        x += int(d.shape[1]) + gap
    # Place centered
    scale = min((width - 8) / total_w, (height - 8) / total_h)
    nw, nh = max(1, int(total_w * scale)), max(1, int(total_h * scale))
    resized = cv2.resize(block, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    mask = resized > 32
    roi = canvas[y0 : y0 + nh, x0 : x0 + nw]
    roi[mask] = fg
    if skew != 0.0:
        m = np.array([[1.0, float(skew), 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        canvas = cv2.warpAffine(
            canvas,
            m,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(bg),
        )
    if occlude_frac > 0:
        oh = max(1, int(height * occlude_frac))
        canvas[height - oh : height, :] = bg
    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    if noise_std > 0:
        noise = np.random.default_rng(1).normal(0, noise_std, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def render_human_frame_with_number(
    text: str,
    *,
    frame_w: int = 160,
    frame_h: int = 220,
    bbox: tuple[float, float, float, float] = (20.0, 20.0, 140.0, 200.0),
    **crop_kwargs: Any,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Full frame with a torso kit region containing the number (or blank)."""
    frame = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    # Body silhouette
    cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 90, 50), thickness=-1)
    crop = render_digit_crop(
        text,
        width=max(16, x1 - x0 - 20),
        height=max(20, int((y1 - y0) * 0.35)),
        **crop_kwargs,
    )
    # Place in upper torso band
    ty0 = y0 + int(0.22 * (y1 - y0))
    tx0 = x0 + 10
    th, tw = crop.shape[:2]
    frame[ty0 : ty0 + th, tx0 : tx0 + tw] = crop
    return frame, bbox


def _sample(
    *,
    run_id: str,
    video_id: str,
    track_id: int,
    frame_index: int,
    detection_id: int,
    role: str = "player",
    observation_state: str = "observed",
    entity_type: str = "human",
    bbox: tuple[float, float, float, float] = (20.0, 20.0, 140.0, 200.0),
    text: str = "10",
    team_id: str | None = None,
    **crop_kwargs: Any,
) -> dict[str, Any]:
    frame, bb = render_human_frame_with_number(text, bbox=bbox, **crop_kwargs)
    return {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": track_id,
        "frame_index": frame_index,
        "detection_id": detection_id,
        "role": role,
        "observation_state": observation_state,
        "entity_type": entity_type,
        "bbox": bb,
        "frame_image": frame,
        "expected_text": text if text else None,
        "team_id": team_id,
        "is_negative": not bool(text),
    }


def fixture_single_digit() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(1),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="7",
        )
        for i in range(3)
    ]
    return _bundle(_rid(1), "v1", samples)


def fixture_two_digit() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(2),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="10",
        )
        for i in range(3)
    ]
    return _bundle(_rid(2), "v1", samples)


def fixture_leading_zero() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(3),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="07",
        )
        for i in range(3)
    ]
    return _bundle(_rid(3), "v1", samples)


def fixture_low_contrast() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(4),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="9",
            low_contrast=True,
        )
        for i in range(2)
    ]
    return _bundle(_rid(4), "v1", samples)


def fixture_blur() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(5),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="4",
            blur_ksize=9,
        )
        for i in range(2)
    ]
    return _bundle(_rid(5), "v1", samples)


def fixture_skew() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(6),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="3",
            skew=0.25,
        )
        for i in range(2)
    ]
    return _bundle(_rid(6), "v1", samples)


def fixture_occlusion() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(7),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="8",
            occlude_frac=0.45,
        )
        for i in range(2)
    ]
    return _bundle(_rid(7), "v1", samples)


def fixture_small_crop() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(8),
            video_id="v1",
            track_id=1,
            frame_index=0,
            detection_id=1,
            text="2",
            bbox=(10.0, 10.0, 28.0, 40.0),
        )
    ]
    return _bundle(_rid(8), "v1", samples)


def fixture_no_number_front() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(9),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="",
        )
        for i in range(3)
    ]
    return _bundle(_rid(9), "v1", samples, negative_ids=[0, 1, 2])


def fixture_sponsor_logo_negative() -> dict[str, Any]:
    """Sponsor-like blobs that must not emit jersey numbers."""
    samples = []
    for i in range(3):
        frame = np.full((220, 160, 3), 30, dtype=np.uint8)
        bbox = (20.0, 20.0, 140.0, 200.0)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 80, 40), thickness=-1)
        # Horizontal sponsor bar (not digit-shaped)
        cv2.rectangle(frame, (x0 + 15, y0 + 50), (x1 - 15, y0 + 70), (220, 220, 220), thickness=-1)
        samples.append(
            {
                "run_id": _rid(10),
                "video_id": "v1",
                "track_id": 1,
                "frame_index": i,
                "detection_id": i + 1,
                "role": "player",
                "observation_state": "observed",
                "entity_type": "human",
                "bbox": bbox,
                "frame_image": frame,
                "expected_text": None,
                "team_id": None,
                "is_negative": True,
            }
        )
    return _bundle(_rid(10), "v1", samples, negative_ids=[0, 1, 2])


def fixture_conflicting_track() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(11), video_id="v1", track_id=1, frame_index=0, detection_id=1, text="10"
        ),
        _sample(
            run_id=_rid(11), video_id="v1", track_id=1, frame_index=1, detection_id=2, text="10"
        ),
        _sample(
            run_id=_rid(11), video_id="v1", track_id=1, frame_index=2, detection_id=3, text="11"
        ),
        _sample(
            run_id=_rid(11), video_id="v1", track_id=1, frame_index=3, detection_id=4, text="11"
        ),
    ]
    return _bundle(_rid(11), "v1", samples)


def fixture_single_weak_observation() -> dict[str, Any]:
    samples = [
        _sample(run_id=_rid(12), video_id="v1", track_id=1, frame_index=0, detection_id=1, text="5")
    ]
    return _bundle(_rid(12), "v1", samples)


def fixture_team_jersey_conflict() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(13),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="10",
            team_id="team_a",
        )
        for i in range(3)
    ]
    b = _bundle(_rid(13), "v1", samples)
    b["team_by_track"] = {1: "team_b"}  # conflicts with sample team_id hint
    b["expected_jersey_team_conflict"] = True
    return b


def fixture_unknown_role() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(14),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="6",
            role="unknown",
        )
        for i in range(2)
    ]
    return _bundle(_rid(14), "v1", samples)


def fixture_predicted_rejected() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(15),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="9",
            observation_state="predicted",
        )
        for i in range(2)
    ]
    return _bundle(_rid(15), "v1", samples)


def fixture_referee_excluded() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(16),
            video_id="v1",
            track_id=1,
            frame_index=i,
            detection_id=i + 1,
            text="1",
            role="referee",
        )
        for i in range(2)
    ]
    return _bundle(_rid(16), "v1", samples)


def fixture_ball_excluded() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(17),
            video_id="v1",
            track_id=1,
            frame_index=0,
            detection_id=1,
            text="8",
            entity_type="ball",
            role="unknown",
        )
    ]
    return _bundle(_rid(17), "v1", samples)


def fixture_color_variants() -> dict[str, Any]:
    samples = [
        _sample(
            run_id=_rid(18),
            video_id="v1",
            track_id=1,
            frame_index=0,
            detection_id=1,
            text="12",
            fg=250,
            bg=20,
        ),
        _sample(
            run_id=_rid(18),
            video_id="v1",
            track_id=1,
            frame_index=1,
            detection_id=2,
            text="12",
            fg=20,
            bg=230,
        ),
        _sample(
            run_id=_rid(18),
            video_id="v1",
            track_id=1,
            frame_index=2,
            detection_id=3,
            text="12",
            fg=240,
            bg=60,
        ),
    ]
    return _bundle(_rid(18), "v1", samples)


def fixture_multi_component() -> dict[str, Any]:
    return fixture_two_digit()


def fixture_leakage_probe() -> dict[str, Any]:
    b = fixture_two_digit()
    b["force_leakage_class"] = "evaluation"
    b["evaluation_label"] = {"track_id": 1, "jersey": 99}
    return b


def _bundle(
    run_id: str,
    video_id: str,
    samples: list[dict[str, Any]],
    *,
    negative_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "samples": samples,
        "negative_observation_indices": list(negative_ids or []),
        "role_by_track": {int(s["track_id"]): str(s["role"]) for s in samples},
        "leakage_class": "synthetic",
    }


FIXTURE_REGISTRY: dict[str, Any] = {
    "single_digit": fixture_single_digit,
    "two_digit": fixture_two_digit,
    "leading_zero": fixture_leading_zero,
    "low_contrast": fixture_low_contrast,
    "blur": fixture_blur,
    "skew": fixture_skew,
    "occlusion": fixture_occlusion,
    "small_crop": fixture_small_crop,
    "no_number": fixture_no_number_front,
    "sponsor_negative": fixture_sponsor_logo_negative,
    "conflict": fixture_conflicting_track,
    "weak_single": fixture_single_weak_observation,
    "team_conflict": fixture_team_jersey_conflict,
    "unknown_role": fixture_unknown_role,
    "predicted": fixture_predicted_rejected,
    "referee": fixture_referee_excluded,
    "ball": fixture_ball_excluded,
    "color_variants": fixture_color_variants,
    "multi_component": fixture_multi_component,
    "leakage": fixture_leakage_probe,
}


__all__ = [
    "RUNTIME_ROOT",
    "FIXTURE_REGISTRY",
    "assert_runtime_root",
    "render_digit_crop",
    "render_human_frame_with_number",
    "fixture_single_digit",
    "fixture_two_digit",
    "fixture_leading_zero",
    "fixture_low_contrast",
    "fixture_blur",
    "fixture_skew",
    "fixture_occlusion",
    "fixture_small_crop",
    "fixture_no_number_front",
    "fixture_sponsor_logo_negative",
    "fixture_conflicting_track",
    "fixture_single_weak_observation",
    "fixture_team_jersey_conflict",
    "fixture_unknown_role",
    "fixture_predicted_rejected",
    "fixture_referee_excluded",
    "fixture_ball_excluded",
    "fixture_color_variants",
    "fixture_multi_component",
    "fixture_leakage_probe",
]
