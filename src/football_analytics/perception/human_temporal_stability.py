"""Short-horizon temporal stability for human proposals (R1-F1-R2).

Carried boxes are diagnostic only: max 2 consecutive frames, never counted as
observed detections / GT. Camera discontinuity clears all tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from football_analytics.perception.detection_evaluation import bbox_iou
from football_analytics.perception.human_tiled_detection import HumanProposal

TemporalStatus = Literal["observed", "carried"]


@dataclass
class _Track:
    box: tuple[float, float, float, float]
    score: float
    eligibility: str
    miss: int = 0


@dataclass(frozen=True)
class TemporalProposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    eligibility: str
    temporal_status: TemporalStatus
    source: str

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "score": self.score,
            "eligibility": self.eligibility,
            "temporal_status": self.temporal_status,
            "source": self.source,
        }


class TemporalHumanStabilizer:
    """Greedy IoU association with max-2-frame carry."""

    def __init__(
        self,
        *,
        match_iou: float = 0.35,
        max_carry_frames: int = 2,
        hist_corr_reset: float = 0.55,
    ) -> None:
        if max_carry_frames > 2:
            raise ValueError("max_carry_frames must be <= 2")
        self.match_iou = match_iou
        self.max_carry = max_carry_frames
        self.hist_corr_reset = hist_corr_reset
        self._tracks: list[_Track] = []
        self._prev_hist: np.ndarray | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_hist = None

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def update(self, frame: np.ndarray, proposals: list[HumanProposal]) -> list[TemporalProposal]:
        hist = self._histogram(frame)
        if self._prev_hist is not None:
            corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
            if corr < self.hist_corr_reset:
                self._tracks.clear()
        self._prev_hist = hist

        dets = list(proposals)
        assigned_det: set[int] = set()
        assigned_trk: set[int] = set()
        out: list[TemporalProposal] = []
        next_tracks: list[_Track] = []

        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(dets):
                iou = bbox_iou(tr.box, det.as_xyxy())
                if iou >= self.match_iou:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        for _, ti, di in pairs:
            if ti in assigned_trk or di in assigned_det:
                continue
            det = dets[di]
            assigned_trk.add(ti)
            assigned_det.add(di)
            next_tracks.append(
                _Track(box=det.as_xyxy(), score=det.score, eligibility=det.eligibility, miss=0)
            )
            out.append(
                TemporalProposal(
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    score=det.score,
                    eligibility=det.eligibility,
                    temporal_status="observed",
                    source=det.source,
                )
            )

        for di, det in enumerate(dets):
            if di in assigned_det:
                continue
            next_tracks.append(
                _Track(box=det.as_xyxy(), score=det.score, eligibility=det.eligibility, miss=0)
            )
            out.append(
                TemporalProposal(
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    score=det.score,
                    eligibility=det.eligibility,
                    temporal_status="observed",
                    source=det.source,
                )
            )

        for ti, tr in enumerate(self._tracks):
            if ti in assigned_trk:
                continue
            miss = tr.miss + 1
            if miss <= self.max_carry:
                next_tracks.append(
                    _Track(
                        box=tr.box,
                        score=tr.score * 0.9,
                        eligibility=tr.eligibility,
                        miss=miss,
                    )
                )
                out.append(
                    TemporalProposal(
                        x1=tr.box[0],
                        y1=tr.box[1],
                        x2=tr.box[2],
                        y2=tr.box[3],
                        score=tr.score * 0.9,
                        eligibility=tr.eligibility,
                        temporal_status="carried",
                        source="temporal_carry",
                    )
                )

        self._tracks = next_tracks
        return out


def compute_temporal_diagnostics(
    sequence: list[list[TemporalProposal]],
) -> dict[str, Any]:
    """sequence[t] = proposals at frame t (aligned continuous indices)."""
    empty = {
        "one_frame_disappearance": 0.0,
        "two_frame_disappearance": 0.0,
        "reappearance": 0.0,
        "center_jitter": 0.0,
        "size_jitter": 0.0,
        "short_lived": 0.0,
        "observed_coverage": 0.0,
        "carried_coverage": 0.0,
        "duplicate_rate": 0.0,
        "effective_one_frame_gap": 0.0,
        "diagnostic_not_accuracy": True,
    }
    if len(sequence) < 2:
        return empty

    def centers(
        props: list[TemporalProposal], *, include_carried: bool = False
    ) -> list[tuple[float, float, float, float]]:
        out = []
        for p in props:
            if p.temporal_status == "observed" or (
                include_carried and p.temporal_status == "carried"
            ):
                w = p.x2 - p.x1
                h = p.y2 - p.y1
                out.append(((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2, w, h))
        return out

    one_miss = two_miss = reapp = 0
    jitters: list[float] = []
    size_j: list[float] = []
    short_lived = 0
    obs = car = dup = total = 0
    eff_one = 0

    for props in sequence:
        total += len(props)
        obs += sum(1 for p in props if p.temporal_status == "observed")
        car += sum(1 for p in props if p.temporal_status == "carried")
        for i, a in enumerate(props):
            for b in props[i + 1 :]:
                if bbox_iou(a.as_xyxy(), b.as_xyxy()) > 0.9:
                    dup += 1

    for t in range(1, len(sequence)):
        prev = centers(sequence[t - 1])
        cur = centers(sequence[t])
        matched_prev: set[int] = set()
        matched_cur: set[int] = set()
        pairs: list[tuple[float, float, int, int, float, float, float]] = []
        for i, (cx, cy, w, h) in enumerate(prev):
            for j, (cx2, cy2, w2, h2) in enumerate(cur):
                dist = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
                norm = max(h, 1.0)
                iou = bbox_iou(
                    (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2),
                )
                if iou >= 0.3 or dist / norm < 0.5:
                    pairs.append(
                        (
                            iou,
                            -dist / norm,
                            i,
                            j,
                            dist / norm,
                            abs(w - w2) / max(w, 1),
                            abs(h - h2) / max(h, 1),
                        )
                    )
        pairs.sort(reverse=True)
        for item in pairs:
            _, _, i, j, dn, dw, dh = item
            if i in matched_prev or j in matched_cur:
                continue
            matched_prev.add(i)
            matched_cur.add(j)
            jitters.append(dn)
            size_j.append(0.5 * (dw + dh))
        unmatched_prev = [i for i in range(len(prev)) if i not in matched_prev]
        for i in unmatched_prev:
            cx, cy, w, h = prev[i]
            found1 = False
            if t + 1 < len(sequence):
                for cx2, cy2, w2, h2 in centers(sequence[t + 1]):
                    iou = bbox_iou(
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                        (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2),
                    )
                    if iou >= 0.3:
                        found1 = True
                        break
            if found1:
                one_miss += 1
                reapp += 1
                continue
            found2 = False
            if t + 2 < len(sequence):
                for cx2, cy2, w2, h2 in centers(sequence[t + 2]):
                    iou = bbox_iou(
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                        (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2),
                    )
                    if iou >= 0.3:
                        found2 = True
                        break
            if found2:
                two_miss += 1
                reapp += 1
            else:
                short_lived += 1

        # effective gap with carry included
        prev_e = centers(sequence[t - 1], include_carried=True)
        cur_e = centers(sequence[t], include_carried=True)
        matched_e: set[int] = set()
        used_e: set[int] = set()
        epairs = []
        for i, (cx, cy, w, h) in enumerate(prev_e):
            for j, (cx2, cy2, w2, h2) in enumerate(cur_e):
                iou = bbox_iou(
                    (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2),
                )
                if iou >= 0.3:
                    epairs.append((iou, i, j))
        epairs.sort(reverse=True)
        for _, i, j in epairs:
            if i in matched_e or j in used_e:
                continue
            matched_e.add(i)
            used_e.add(j)
        for i in range(len(prev_e)):
            if i in matched_e:
                continue
            cx, cy, w, h = prev_e[i]
            if t + 1 < len(sequence):
                for cx2, cy2, w2, h2 in centers(sequence[t + 1], include_carried=True):
                    iou = bbox_iou(
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                        (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2),
                    )
                    if iou >= 0.3:
                        eff_one += 1
                        break

    n = max(1, len(sequence))
    return {
        "one_frame_disappearance": float(one_miss),
        "two_frame_disappearance": float(two_miss),
        "reappearance": float(reapp),
        "center_jitter": float(np.mean(jitters)) if jitters else 0.0,
        "size_jitter": float(np.mean(size_j)) if size_j else 0.0,
        "short_lived": float(short_lived),
        "observed_coverage": obs / max(1, total),
        "carried_coverage": car / max(1, total),
        "duplicate_rate": dup / max(1, n),
        "effective_one_frame_gap": float(eff_one),
        "diagnostic_not_accuracy": True,
    }


__all__ = [
    "TemporalHumanStabilizer",
    "TemporalProposal",
    "compute_temporal_diagnostics",
]
