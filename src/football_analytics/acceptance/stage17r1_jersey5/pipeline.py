"""Stage 17-R1 jersey-5 own-video: appearance tracker + visual anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_analytics.acceptance.final_perception_repair.pipeline import _iou

# Human-reviewed visual anchors (agent inspected body crops). OCR not used as proof.
# Team: yellow kit only. Number 5 clearly readable on back.
VISUAL_ANCHORS_JERSEY5: list[dict[str, Any]] = [
    {
        "anchor_id": "j5_a1",
        "frame": 55,
        "t_s": 1.8,
        "box": [948, 446, 66, 87],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view; digit 5 legible",
        "crop_rel": "review/jersey5_anchors/f0055_r1_a5801_body.jpg",
    },
    {
        "anchor_id": "j5_a2",
        "frame": 60,
        "t_s": 1.967,
        "box": [988, 436, 46, 97],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view near ball; digit 5 legible",
        "crop_rel": "review/jersey5_anchors/f0060_r1_a4439_body.jpg",
    },
    {
        "anchor_id": "j5_a3",
        "frame": 290,
        "t_s": 9.633,
        "box": [263, 574, 93, 150],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view; digit 5 legible; black base layer",
        "crop_rel": "review/jersey5_anchors/f0290_r0_a14047_body.jpg",
    },
    {
        "anchor_id": "j5_a4",
        "frame": 295,
        "t_s": 9.8,
        "box": [308, 572, 68, 145],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view; digit 5 legible",
        "crop_rel": "review/jersey5_anchors/f0295_r1_a9908_body.jpg",
    },
    {
        "anchor_id": "j5_a5",
        "frame": 300,
        "t_s": 9.967,
        "box": [346, 556, 68, 135],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view; digit 5 legible",
        "crop_rel": "review/jersey5_anchors/f0300_r1_a9156_body.jpg",
    },
    {
        "anchor_id": "j5_a6",
        "frame": 310,
        "t_s": 10.3,
        "box": [416, 539, 49, 138],
        "team": "team_yellow",
        "kit": "yellow_kit",
        "jersey_number": 5,
        "review_status": "reviewed",
        "evidence": "body crop back view; digit 5 legible",
        "crop_rel": "review/jersey5_anchors/f0310_r3_a6750_body.jpg",
    },
]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _hist_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class AppearanceConfirmedTracker:
    """IoU + appearance histogram tracker (product-safe; not ByteTrack/BoT-SORT)."""

    iou_thresh: float = 0.22
    center_gate_px: float = 90.0
    hist_thresh: float = 0.55
    min_hits: int = 5
    max_age: int = 35
    _next_id: int = 1
    _tracks: dict[int, dict[str, Any]] = field(default_factory=dict)

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], np.ndarray | None]],
    ) -> list[tuple[int, tuple[float, float, float, float], bool]]:
        dets = list(detections)
        assigned: dict[int, int] = {}
        used: set[int] = set()

        # Score matrix: IoU primary, appearance+center secondary
        pairs: list[tuple[float, int, int]] = []
        for tid, st in self._tracks.items():
            box = st["box"]
            hist = st.get("hist")
            cx, cy = _center(box)
            for i, (det, dhist) in enumerate(dets):
                iou = _iou(box, det)
                dx, dy = _center(det)
                dist = float(np.hypot(cx - dx, cy - dy))
                sim = _hist_sim(hist, dhist)
                score = 0.0
                if iou >= self.iou_thresh:
                    score = 2.0 + iou + 0.3 * sim
                elif dist <= self.center_gate_px and sim >= self.hist_thresh:
                    score = 1.0 + sim + (1.0 - dist / self.center_gate_px)
                if score > 0:
                    pairs.append((score, tid, i))
        pairs.sort(reverse=True)
        for _score, tid, i in pairs:
            if tid in assigned or i in used:
                continue
            assigned[tid] = i
            used.add(i)

        for tid, i in assigned.items():
            det, dhist = dets[i]
            st = self._tracks[tid]
            st["hits"] = int(st["hits"]) + 1
            st["age"] = 0
            st["box"] = det
            if dhist is not None:
                prev = st.get("hist")
                st["hist"] = dhist if prev is None else (0.7 * prev + 0.3 * dhist)

        for tid, st in list(self._tracks.items()):
            if tid not in assigned:
                st["age"] = int(st["age"]) + 1

        for i, (det, dhist) in enumerate(dets):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "box": det,
                "hits": 1,
                "age": 0,
                "hist": dhist,
            }
            assigned[tid] = i

        dead = [tid for tid, st in self._tracks.items() if int(st["age"]) > self.max_age]
        for tid in dead:
            del self._tracks[tid]

        out: list[tuple[int, tuple[float, float, float, float], bool]] = []
        for tid, i in assigned.items():
            st_opt = self._tracks.get(tid)
            if st_opt is None:
                continue
            st = st_opt
            det_box: tuple[float, float, float, float] = dets[i][0]
            confirmed = int(st["hits"]) >= self.min_hits
            out.append((tid, det_box, confirmed))
        return out


def match_anchor_to_tracks(
    *,
    anchors: list[dict[str, Any]],
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    iou_thresh: float = 0.25,
) -> dict[str, Any]:
    """Map reviewed jersey-5 anchors to track IDs via IoU."""
    track_votes: dict[int, int] = {}
    per_anchor: list[dict[str, Any]] = []
    for a in anchors:
        fi = int(a["frame"])
        bx = a["box"]
        abox = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))
        best_tid, best_iou = None, 0.0
        for tid, box, _conf in tracks_by_frame.get(fi, []):
            v = _iou(abox, box)
            if v > best_iou:
                best_iou, best_tid = v, tid
        hit = best_tid is not None and best_iou >= iou_thresh
        if hit and best_tid is not None:
            track_votes[int(best_tid)] = track_votes.get(int(best_tid), 0) + 1
        per_anchor.append(
            {
                "anchor_id": a["anchor_id"],
                "frame": fi,
                "matched_track_id": best_tid if hit else None,
                "iou": round(best_iou, 3),
                "matched": hit,
            }
        )
    primary = None
    if track_votes:
        primary = max(track_votes.items(), key=lambda kv: kv[1])[0]
    return {
        "per_anchor": per_anchor,
        "track_votes": {str(k): v for k, v in track_votes.items()},
        "primary_track_id": primary,
        "n_matched_anchors": sum(1 for p in per_anchor if p["matched"]),
        "n_anchors": len(anchors),
        "n_distinct_tracks": len(track_votes),
    }
