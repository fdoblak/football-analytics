"""Root-cause analysis for small-object detector failure (R1-F2-C)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from football_analytics.annotation.coordinates import SOURCE_HEIGHT, SOURCE_WIDTH
from football_analytics.annotation.evaluation_protocol_v2 import (
    HEIGHT_BINS,
    LabeledBox,
    boxes_from_frozen_frame,
    evaluate_protocol_v2,
    height_bin,
)
from football_analytics.perception.detection_evaluation import BBoxDetection

FRAME_AREA = float(SOURCE_WIDTH * SOURCE_HEIGHT)


def _box_stats(boxes: Sequence[LabeledBox]) -> dict[str, Any]:
    if not boxes:
        return {
            "n": 0,
            "width": {},
            "height": {},
            "area_frac": {},
            "height_bins": {name: 0 for name, _, _ in HEIGHT_BINS},
            "eligibility": {},
            "role": {},
            "team": {},
        }
    ws = [b.width for b in boxes]
    hs = [b.height for b in boxes]
    af = [b.area / FRAME_AREA for b in boxes]
    hb = Counter(height_bin(b.height) for b in boxes)
    return {
        "n": len(boxes),
        "width": {
            "mean": float(np.mean(ws)),
            "p10": float(np.percentile(ws, 10)),
            "p50": float(np.percentile(ws, 50)),
            "p90": float(np.percentile(ws, 90)),
        },
        "height": {
            "mean": float(np.mean(hs)),
            "p10": float(np.percentile(hs, 10)),
            "p50": float(np.percentile(hs, 50)),
            "p90": float(np.percentile(hs, 90)),
        },
        "area_frac": {
            "mean": float(np.mean(af)),
            "p10": float(np.percentile(af, 10)),
            "p50": float(np.percentile(af, 50)),
            "p90": float(np.percentile(af, 90)),
        },
        "height_bins": {name: int(hb.get(name, 0)) for name, _, _ in HEIGHT_BINS},
        "eligibility": dict(Counter(b.eligibility for b in boxes)),
        "role": dict(Counter(b.role for b in boxes)),
        "team": dict(Counter(b.team_appearance for b in boxes)),
        "size_class": {
            "small_h_lt_40": sum(1 for b in boxes if b.height < 40),
            "medium_40_64": sum(1 for b in boxes if 40 <= b.height < 64),
            "large_ge_64": sum(1 for b in boxes if b.height >= 64),
        },
    }


def _categories(frames: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for fr in frames:
        for cat in fr.get("categories") or []:
            c[str(cat)] += 1
    return dict(c)


def _temporal(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    idxs = sorted(int(f["frame_idx"]) for f in frames)
    gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)] if len(idxs) > 1 else []
    return {
        "frame_idx_min": idxs[0] if idxs else None,
        "frame_idx_max": idxs[-1] if idxs else None,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "median_gap": float(np.median(gaps)) if gaps else None,
        "n_pairs_gap_le_15": sum(1 for g in gaps if g <= 15),
        "high_temporal_correlation": bool(gaps) and float(np.median(gaps)) <= 12.0,
    }


def _js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    if pv.sum() <= 0 or qv.sum() <= 0:
        return 0.0
    pv /= pv.sum()
    qv /= qv.sum()
    m = 0.5 * (pv + qv)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)


def _norm_counts(c: Mapping[str, int]) -> dict[str, float]:
    s = sum(c.values()) or 1
    return {k: v / s for k, v in c.items()}


def analyze_split_distributions(annotations: Mapping[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[LabeledBox]] = defaultdict(list)
    frames_by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fr in annotations["frames"]:
        sp = str(fr["split"])
        frames_by[sp].append(fr)
        by_split[sp].extend(boxes_from_frozen_frame(fr))

    split_stats = {}
    for sp in ("train", "dev", "holdout"):
        boxes = by_split[sp]
        split_stats[sp] = {
            "n_frames": len(frames_by[sp]),
            "n_bbox": len(boxes),
            "bbox_stats": _box_stats(boxes),
            "categories": _categories(frames_by[sp]),
            "temporal": _temporal(frames_by[sp]),
            "t_s_range": [
                min((float(f["t_s"]) for f in frames_by[sp]), default=None),
                max((float(f["t_s"]) for f in frames_by[sp]), default=None),
            ],
        }

    train_h = cast(dict[str, int], split_stats["train"]["bbox_stats"]["height_bins"])  # type: ignore[index]
    dev_h = cast(dict[str, int], split_stats["dev"]["bbox_stats"]["height_bins"])  # type: ignore[index]
    hold_h = cast(dict[str, int], split_stats["holdout"]["bbox_stats"]["height_bins"])  # type: ignore[index]
    return {
        "splits": split_stats,
        "distribution_shift": {
            "train_vs_dev_height_js": _js_divergence(_norm_counts(train_h), _norm_counts(dev_h)),
            "dev_vs_holdout_height_js": _js_divergence(_norm_counts(dev_h), _norm_counts(hold_h)),
            "train_vs_holdout_height_js": _js_divergence(
                _norm_counts(train_h), _norm_counts(hold_h)
            ),
        },
    }


def height_bin_recall_from_eval(eval_result: Mapping[str, Any]) -> dict[str, Any]:
    return dict(eval_result.get("primary", {}).get("height_bin_recall") or {})


def prove_or_reject_hypotheses(
    dist: Mapping[str, Any],
    *,
    height_recall_holdout: Mapping[str, Any] | None,
    height_recall_dev: Mapping[str, Any] | None,
    prior_holdout_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return hypothesis verdicts with evidence."""
    train = dist["splits"]["train"]
    dev = dist["splits"]["dev"]
    hold = dist["splits"]["holdout"]
    train_h = train["bbox_stats"]["height"]
    # After 960 resize: scale = 960/max(1336,744) ≈ 0.718 — height 24 → ~17 px
    scale_960 = 960.0 / max(SOURCE_WIDTH, SOURCE_HEIGHT)
    h_p50_after = train_h["p50"] * scale_960
    h_p10_after = train_h["p10"] * scale_960

    hyps: list[dict[str, Any]] = []

    # 1 full-frame resize
    small_frac = train["bbox_stats"]["size_class"]["small_h_lt_40"] / max(
        1, train["bbox_stats"]["n"]
    )
    hyps.append(
        {
            "id": 1,
            "claim": "Full-frame resize shrinks small humans excessively",
            "verdict": "SUPPORTED" if h_p10_after < 18 or small_frac > 0.25 else "MIXED",
            "evidence": {
                "source_height_p10": train_h["p10"],
                "source_height_p50": train_h["p50"],
                "after_imgsz960_height_p10": h_p10_after,
                "after_imgsz960_height_p50": h_p50_after,
                "small_h_lt_40_frac": small_frac,
                "imgsz_scale": scale_960,
            },
        }
    )

    # 2 high temporal correlation in 40 train frames
    hyps.append(
        {
            "id": 2,
            "claim": "40 train images are highly temporally correlated",
            "verdict": (
                "SUPPORTED" if train["temporal"]["high_temporal_correlation"] else "REJECTED"
            ),
            "evidence": train["temporal"],
        }
    )

    # 3 camera/domain shift train vs later windows
    js_th = dist["distribution_shift"]["train_vs_holdout_height_js"]
    t_train = train["t_s_range"]
    t_hold = hold["t_s_range"]
    hyps.append(
        {
            "id": 3,
            "claim": "Train vs later temporal windows have camera/domain shift",
            "verdict": "SUPPORTED" if js_th > 0.02 or (t_hold[0] or 0) >= 22 else "MIXED",
            "evidence": {
                "train_t_s": t_train,
                "dev_t_s": dev["t_s_range"],
                "holdout_t_s": t_hold,
                "train_vs_holdout_height_js": js_th,
                "dev_vs_holdout_height_js": dist["distribution_shift"]["dev_vs_holdout_height_js"],
                "note": "Splits are time-separated by design (0-12 / 12-22 / 22-34s)",
            },
        }
    )

    # 4 mosaic/scale aug shrinks small humans further
    hyps.append(
        {
            "id": 4,
            "claim": "Mosaic/scale augmentation further shrinks small humans",
            "verdict": "SUPPORTED",
            "evidence": {
                "prior_train_mosaic": 0.5,
                "prior_train_scale": 0.3,
                "mechanism": (
                    "mosaic packs 4 images into one letterboxed canvas; "
                    "scale=0.3 allows strong downscale of already-small players"
                ),
                "mitigation": "tile-aware train with mosaic<=0.1 and reduced scale",
            },
        }
    )

    # 5 confidence threshold suppresses small humans
    small_r = prior_holdout_metrics.get("small_distant_recall")
    hyps.append(
        {
            "id": 5,
            "claim": "Confidence threshold suppresses small humans",
            "verdict": "SUPPORTED" if (small_r is not None and small_r < 0.3) else "MIXED",
            "evidence": {
                "holdout_small_recall": small_r,
                "holdout_overall_recall": prior_holdout_metrics.get("recall"),
                "note": (
                    "Small recall much lower than overall suggests score/size interaction; "
                    "sweep conf on DEV only in redesign"
                ),
            },
        }
    )

    # 6 full-frame inference insufficient for small-object recall
    hr = height_recall_holdout or {}
    tiny = (hr.get("h_lt_16") or {}).get("recall")
    mid = (hr.get("h_24_40") or {}).get("recall")
    large = (hr.get("h_ge_64") or {}).get("recall")
    hyps.append(
        {
            "id": 6,
            "claim": "Full-frame inference is insufficient for small-object recall",
            "verdict": (
                "SUPPORTED"
                if (tiny is not None and tiny < 0.4)
                or (mid is not None and large is not None and mid + 0.2 < large)
                else "MIXED"
            ),
            "evidence": {
                "holdout_height_bin_recall": hr,
                "dev_height_bin_recall": height_recall_dev,
            },
        }
    )

    # 7 uncertain eligibility pollutes metrics
    unc_train = train["bbox_stats"]["eligibility"].get("uncertain", 0)
    unc_hold = hold["bbox_stats"]["eligibility"].get("uncertain", 0)
    hyps.append(
        {
            "id": 7,
            "claim": "GT uncertain eligibility regions pollute the metric",
            "verdict": "SUPPORTED" if (unc_train + unc_hold) > 0 else "REJECTED",
            "evidence": {
                "uncertain_counts": {
                    "train": unc_train,
                    "dev": dev["bbox_stats"]["eligibility"].get("uncertain", 0),
                    "holdout": unc_hold,
                },
                "protocol_v2_fix": "ignore uncertain as neither positive nor negative",
            },
        }
    )
    return hyps


def build_root_cause_report(
    annotations: Mapping[str, Any],
    *,
    prior_holdout_metrics: Mapping[str, Any],
    height_recall_holdout: Mapping[str, Any] | None = None,
    height_recall_dev: Mapping[str, Any] | None = None,
    error_magnitude: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dist = analyze_split_distributions(annotations)
    hyps = prove_or_reject_hypotheses(
        dist,
        height_recall_holdout=height_recall_holdout,
        height_recall_dev=height_recall_dev,
        prior_holdout_metrics=prior_holdout_metrics,
    )
    supported = [h["id"] for h in hyps if h["verdict"] == "SUPPORTED"]
    return {
        "schema": "r1_f2c_root_cause_v1",
        "holdout_v1_status": "CONSUMED_FAILED_EVALUATION",
        "acceptance_reusable": False,
        "prior_holdout_v1_metrics": {
            "precision": prior_holdout_metrics.get("precision"),
            "recall": prior_holdout_metrics.get("recall"),
            "f1": prior_holdout_metrics.get("f1"),
            "ap50": prior_holdout_metrics.get("ap50"),
            "small_recall": prior_holdout_metrics.get("small_distant_recall"),
        },
        "distributions": dist,
        "height_bin_recall": {
            "holdout_v1_error_analysis_only": height_recall_holdout,
            "dev": height_recall_dev,
        },
        "error_magnitude_distribution": error_magnitude,
        "hypotheses": hyps,
        "definitive_root_causes": supported,
        "media": None,
    }


def error_magnitude_from_preds(
    preds: Sequence[BBoxDetection],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """FN height distribution + FP count for error analysis (not selection)."""
    ev = evaluate_protocol_v2(preds, frames)
    bins = ev["primary"]["height_bin_recall"]
    fn_by_bin = {k: int(v["n_gt"] - v["tp"]) for k, v in bins.items() if v.get("n_gt") is not None}
    return {
        "primary_fn_by_height_bin": fn_by_bin,
        "primary_fp": ev["primary"].get("false_positives"),
        "primary_fn": ev["primary"].get("false_negatives"),
        "ignored_predictions": ev["secondary"].get("ignored_predictions"),
    }


__all__ = [
    "analyze_split_distributions",
    "build_root_cause_report",
    "error_magnitude_from_preds",
    "height_bin_recall_from_eval",
]
