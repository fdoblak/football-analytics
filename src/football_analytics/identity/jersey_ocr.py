"""OpenCV template/shape digit OCR for jersey numbers (Stage 7D).

Synthetic 0-9 stroke masks are generated programmatically (no font downloads).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

PRODUCER = "jersey_ocr_baseline"
PRODUCER_VERSION = "0.1.0"
TEMPLATE_VERSION = "synthetic_stroke_v1"

# Digits that are easy to confuse under template matching.
SIMILAR_DIGIT_PAIRS = frozenset(
    {frozenset({"1", "7"}), frozenset({"0", "8"}), frozenset({"3", "8"}), frozenset({"5", "6"})}
)


def _stroke(canvas: np.ndarray, pts: Sequence[tuple[int, int]], thickness: int = 3) -> None:
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas, [arr], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA
    )


def _draw_digit(digit: int, *, width: int = 28, height: int = 40) -> np.ndarray:
    """Draw a simple blocky digit mask (white on black) without external fonts."""
    img = np.zeros((height, width), dtype=np.uint8)
    m = 4
    t = max(3, width // 7)
    L, R = m, width - m - 1
    T, B = m, height - m - 1
    MX = (L + R) // 2
    MY = (T + B) // 2
    if digit == 0:
        cv2.rectangle(img, (L, T), (R, B), 255, t)
    elif digit == 1:
        # Thick vertical stem + short serif — keep distinct from 7.
        cv2.line(img, (MX, T), (MX, B), 255, t + 1)
        cv2.line(img, (MX - 6, T + 6), (MX, T), 255, t)
        cv2.line(img, (MX - 5, B), (MX + 5, B), 255, t)
    elif digit == 2:
        _stroke(img, [(L, T), (R, T), (R, MY), (L, MY), (L, B), (R, B)], t)
    elif digit == 3:
        _stroke(img, [(L, T), (R, T), (R, MY), (L + 2, MY), (R, MY), (R, B), (L, B)], t)
    elif digit == 4:
        _stroke(img, [(L, T), (L, MY), (R, MY)], t)
        _stroke(img, [(R - 2, T), (R - 2, B)], t)
    elif digit == 5:
        _stroke(img, [(R, T), (L, T), (L, MY), (R, MY), (R, B), (L, B)], t)
    elif digit == 6:
        _stroke(img, [(R, T), (L, T), (L, B), (R, B), (R, MY), (L, MY)], t)
    elif digit == 7:
        _stroke(img, [(L, T), (R, T), (MX + 2, B)], t)
    elif digit == 8:
        cv2.rectangle(img, (L, T), (R, MY), 255, t)
        cv2.rectangle(img, (L, MY), (R, B), 255, t)
    elif digit == 9:
        _stroke(img, [(L, B), (R, B), (R, T), (L, T), (L, MY), (R, MY)], t)
    else:
        raise ValueError(f"digit out of range: {digit}")
    return img


@lru_cache(maxsize=8)
def build_digit_templates(*, width: int = 28, height: int = 40) -> tuple[np.ndarray, ...]:
    """Return length-10 tuple of float32 templates in [0,1]."""
    out: list[np.ndarray] = []
    for d in range(10):
        mask = _draw_digit(d, width=width, height=height)
        # Light soften only — keep stroke identity for thin digits like 1.
        mask = cv2.GaussianBlur(mask, (3, 3), 0.6)
        templ = mask.astype(np.float32) / 255.0
        out.append(templ)
    return tuple(out)


def clear_digit_template_cache() -> None:
    build_digit_templates.cache_clear()


@dataclass(frozen=True)
class DigitMatch:
    digit: str
    score: float
    second_digit: str
    second_score: float
    margin: float


@dataclass(frozen=True)
class JerseyOcrResult:
    status: str
    raw_text: str | None
    normalized_number: int | None
    digit_count: int | None
    number_score: float | None
    number_margin: float | None
    digit_scores: tuple[float, ...]
    quality_flags: tuple[str, ...]
    reason_codes: tuple[str, ...]
    visibility: str
    readability: str
    source: str


def _preprocess_region(region: np.ndarray, *, config: Mapping[str, Any]) -> np.ndarray:
    prep = config["preprocessing"]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region.copy()
    if prep["clahe"]:
        clahe = cv2.createCLAHE(
            clipLimit=float(prep["clahe_clip"]),
            tileGridSize=(int(prep["clahe_tile"]), int(prep["clahe_tile"])),
        )
        gray = clahe.apply(gray)
    mode = str(prep["threshold_mode"])
    if mode == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Prefer white digits on black; flip if background is bright.
    if float(np.mean(np.asarray(bw, dtype=np.float32))) > 127.0:
        bw = 255 - bw
    k_open = int(prep["morph_open_ksize"])
    k_close = int(prep["morph_close_ksize"])
    if k_open > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)
    if k_close > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)
    return bw


def _component_boxes(
    bw: np.ndarray, *, config: Mapping[str, Any]
) -> list[tuple[int, int, int, int]]:
    prep = config["preprocessing"]
    h, w = bw.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    max_area = int(float(prep["max_component_area_frac"]) * h * w)
    min_area = int(prep["min_component_area"])
    for i in range(1, n):
        x, y, ww, hh, area = (int(stats[i, j]) for j in range(5))
        if area < min_area or area > max_area:
            continue
        if hh <= 0:
            continue
        aspect = ww / float(hh)
        if aspect < float(prep["min_aspect"]) or aspect > float(prep["max_aspect"]):
            continue
        # Reject components hugging the border as likely noise/sponsor blobs.
        if (x <= 0 or y <= 0 or x + ww >= w or y + hh >= h) and area < min_area * 3:
            continue
        boxes.append((x, y, ww, hh, area))
    # Left-to-right, then top.
    boxes.sort(key=lambda b: (b[0], b[1], -b[4]))
    # Keep up to max_digits largest-by-height among left-ordered candidates.
    max_d = int(prep["max_digits"])
    if len(boxes) > max_d:
        # Prefer taller digit-like components while preserving left order.
        ranked = sorted(boxes, key=lambda b: (-b[3], -b[4], b[0]))[:max_d]
        ranked.sort(key=lambda b: (b[0], b[1]))
        boxes = ranked
    return [(x, y, ww, hh) for x, y, ww, hh, _ in boxes]


def _match_digit(
    patch: np.ndarray, templates: Sequence[np.ndarray], *, min_score: float
) -> DigitMatch | None:
    if patch.size == 0:
        return None
    h, w = templates[0].shape[:2]
    resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    scores: list[tuple[float, str]] = []
    for d, templ in enumerate(templates):
        # Normalized correlation via OpenCV matchTemplate (single window).
        res = cv2.matchTemplate(norm, templ, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0]) if res.size else -1.0
        scores.append((score, str(d)))
    scores.sort(key=lambda x: (-x[0], x[1]))
    best_s, best_d = scores[0]
    second_s, second_d = scores[1] if len(scores) > 1 else (-1.0, best_d)
    if best_s < min_score:
        return None
    return DigitMatch(
        digit=best_d,
        score=best_s,
        second_digit=second_d,
        second_score=second_s,
        margin=float(best_s - second_s),
    )


def _normalize_number(raw: str, *, allow_leading_zero: bool) -> tuple[int | None, int, list[str]]:
    flags: list[str] = []
    digits = raw
    digit_count = len(digits)
    if not digits.isdigit() or digit_count not in {1, 2}:
        return None, digit_count, ["INVALID_DIGIT_STRING"]
    has_leading_zero = len(digits) == 2 and digits[0] == "0"
    if has_leading_zero:
        flags.append("leading_zero")
        if not allow_leading_zero:
            return None, digit_count, flags + ["LEADING_ZERO_FORBIDDEN"]
        # Preserve raw_text; numeric optional to satisfy digit_count policy.
        return None, digit_count, flags
    value = int(digits)
    if value < 0 or value > 99:
        return None, digit_count, ["OUT_OF_RANGE"]
    return value, digit_count, flags


def recognize_jersey_number(
    region: np.ndarray | None,
    *,
    config: Mapping[str, Any],
    force_status: str | None = None,
) -> JerseyOcrResult:
    """Run template OCR on a jersey region crop. Abstains rather than inventing."""
    ocr_cfg = config["ocr"]
    source = str(ocr_cfg["source"])
    if force_status == "failed":
        return JerseyOcrResult(
            status="failed",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            number_score=None,
            number_margin=None,
            digit_scores=(),
            quality_flags=("failed",),
            reason_codes=("OCR_EXCEPTION",),
            visibility="unknown",
            readability="none",
            source=source,
        )
    if region is None or region.size == 0:
        return JerseyOcrResult(
            status="no_region",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            number_score=None,
            number_margin=None,
            digit_scores=(),
            quality_flags=("no_region",),
            reason_codes=("NO_REGION",),
            visibility="unknown",
            readability="none",
            source=source,
        )
    try:
        bw = _preprocess_region(region, config=config)
        boxes = _component_boxes(bw, config=config)
        if not boxes:
            return JerseyOcrResult(
                status="no_digits",
                raw_text=None,
                normalized_number=None,
                digit_count=0,
                number_score=None,
                number_margin=None,
                digit_scores=(),
                quality_flags=("no_digits",),
                reason_codes=("NO_DIGIT_COMPONENTS",),
                visibility="visible",
                readability="none",
                source=source,
            )
        tw, th = [int(x) for x in ocr_cfg["digit_match_size"]]
        templates = build_digit_templates(width=tw, height=th)
        matches: list[DigitMatch | None] = []
        for x, y, ww, hh in boxes:
            patch = bw[y : y + hh, x : x + ww]
            m = _match_digit(patch, templates, min_score=float(ocr_cfg["min_digit_score"]))
            matches.append(m)
        matched = [m for m in matches if m is not None]
        # Partial multi-component decode → abstain (critical false-number guard).
        if len(boxes) >= 2 and len(matched) < len(boxes):
            return JerseyOcrResult(
                status="ambiguous",
                raw_text=None,
                normalized_number=None,
                digit_count=len(matched),
                number_score=None,
                number_margin=None,
                digit_scores=tuple(m.score for m in matched),
                quality_flags=("ambiguous", "partial_digit_decode"),
                reason_codes=("PARTIAL_COMPONENT_MATCH",),
                visibility="partial",
                readability="uncertain",
                source=source,
            )
        if not matched:
            return JerseyOcrResult(
                status="no_digits",
                raw_text=None,
                normalized_number=None,
                digit_count=0,
                number_score=None,
                number_margin=None,
                digit_scores=(),
                quality_flags=("no_digits",),
                reason_codes=("DIGIT_SCORE_BELOW_THRESHOLD",),
                visibility="visible",
                readability="none",
                source=source,
            )
        # Cap at 2 digits (left-to-right already ordered via boxes).
        matched = matched[: int(config["preprocessing"]["max_digits"])]
        raw = "".join(m.digit for m in matched)
        digit_scores = tuple(m.score for m in matched)
        number_score = float(sum(digit_scores) / len(digit_scores))
        margins = [m.margin for m in matched]
        number_margin = float(min(margins)) if margins else 0.0
        flags: list[str] = []
        reasons: list[str] = []
        # Similar-digit ambiguity
        for m in matched:
            pair = frozenset({m.digit, m.second_digit})
            if pair in SIMILAR_DIGIT_PAIRS and m.margin < float(ocr_cfg["similar_digit_margin"]):
                flags.append("similar_digits")
                reasons.append(f"SIMILAR_DIGITS:{m.digit}/{m.second_digit}")
        if number_score < float(ocr_cfg["min_number_score"]) or number_margin < float(
            ocr_cfg["ambiguity_margin"]
        ):
            flags.append("ambiguous")
            reasons.append("LOW_SCORE_OR_MARGIN")
            return JerseyOcrResult(
                status="ambiguous",
                raw_text=None,
                normalized_number=None,
                digit_count=len(matched),
                number_score=number_score,
                number_margin=number_margin,
                digit_scores=digit_scores,
                quality_flags=tuple(["ambiguous", *flags, f"score:{number_score:.4f}"]),
                reason_codes=tuple(reasons),
                visibility="partial",
                readability="uncertain",
                source=source,
            )
        norm, digit_count, nflags = _normalize_number(
            raw, allow_leading_zero=bool(ocr_cfg["allow_leading_zero"])
        )
        flags.extend(nflags)
        if "OUT_OF_RANGE" in nflags or "INVALID_DIGIT_STRING" in nflags:
            return JerseyOcrResult(
                status="ambiguous",
                raw_text=None,
                normalized_number=None,
                digit_count=digit_count,
                number_score=number_score,
                number_margin=number_margin,
                digit_scores=digit_scores,
                quality_flags=tuple(["ambiguous", *flags]),
                reason_codes=tuple([*reasons, *nflags]),
                visibility="partial",
                readability="uncertain",
                source=source,
            )
        # Encode raw scores into quality_flags (confidence field stays null).
        flags.append(f"score:{number_score:.4f}")
        flags.append(f"margin:{number_margin:.4f}")
        return JerseyOcrResult(
            status="observed",
            raw_text=raw,
            normalized_number=norm,
            digit_count=digit_count,
            number_score=number_score,
            number_margin=number_margin,
            digit_scores=digit_scores,
            quality_flags=tuple(flags),
            reason_codes=tuple(reasons) if reasons else ("TEMPLATE_MATCH_OK",),
            visibility="visible",
            readability="clear",
            source=source,
        )
    except Exception:  # noqa: BLE001
        return JerseyOcrResult(
            status="failed",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            number_score=None,
            number_margin=None,
            digit_scores=(),
            quality_flags=("failed",),
            reason_codes=("OCR_EXCEPTION",),
            visibility="unknown",
            readability="none",
            source=source,
        )


__all__ = [
    "PRODUCER",
    "PRODUCER_VERSION",
    "TEMPLATE_VERSION",
    "DigitMatch",
    "JerseyOcrResult",
    "clear_digit_template_cache",
    "build_digit_templates",
    "recognize_jersey_number",
]
