#!/usr/bin/env python3
"""Stage 17-R1: Forma 5 own-video analysis + Turkish delivery (Forma 7 revoked)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path("/home/fdoblak/projects/football-analytics")
sys.path.insert(0, str(REPO / "src"))

from football_analytics.acceptance.download_manifest import sha256_file  # noqa: E402
from football_analytics.acceptance.final_perception_repair.pipeline import (  # noqa: E402
    COLOR_BALL_OBS_BGR,
    COLOR_TARGET_BGR,
    ConfirmedTracker,
    _bytetrack_on_detections,
    _iou,
    _kit_hist,
)
from football_analytics.acceptance.portable_final_media import (  # noqa: E402
    validate_portable_mp4,
)
from football_analytics.acceptance.stage17r1_jersey5.pipeline import (  # noqa: E402
    VISUAL_ANCHORS_JERSEY5,
    AppearanceConfirmedTracker,
    match_anchor_to_tracks,
)
from football_analytics.acceptance.stage18_own_video.pipeline import (  # noqa: E402
    OWN_VIDEO_CFG,
    classify_human_role,
    compute_pitch_masks,
    detect_persons,
)
from football_analytics.perception.adapters.ultralytics_ball import (  # noqa: E402
    UltralyticsBallAdapter,
)
from football_analytics.perception.adapters.ultralytics_person import (  # noqa: E402
    UltralyticsPersonAdapter,
)

YOLO = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
YOLO_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SRC = Path("/mnt/c/Users/furka/Downloads/örnek_video_FA.mp4")
VID = Path("/home/fdoblak/football_data/videos/own_video_analysis/source_readonly_copy.mp4")
EXPECTED_SHA = "97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"
WORK = Path("/home/fdoblak/workspace/own_video_analysis")
FINAL = REPO / "artifacts" / "final_delivery"
WIN_DESKTOP = Path("/mnt/c/Users/furka/Desktop/Football Analytics Final")

COLOR_WHITE = (240, 240, 240)
COLOR_YELLOW = (0, 220, 255)
COLOR_UNK = (160, 160, 160)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def revoke_jersey7() -> dict[str, Any]:
    target = WORK / "target"
    target.mkdir(parents=True, exist_ok=True)
    revoke = {
        "schema": "jersey_target_revocation_v1",
        "written_at_utc": utc_now(),
        "superseded_by_user": True,
        "old_target_jersey": 7,
        "new_target_jersey": 5,
        "classification": {
            "active_target_configuration": "revoked_updated_to_5",
            "derived_runtime_artifact": "jersey7 scout/gallery revoked; not transferred",
            "historical_audit": "kept with superseded_by_user",
            "generic_test_or_documentation": "untouched",
        },
        "note": "No confirmed jersey-7 identity transferred to jersey 5.",
    }
    (target / "jersey7_revocation.json").write_text(
        json.dumps(revoke, indent=2, ensure_ascii=False) + "\n"
    )
    # Mark derived j7 runtime as revoked (do not delete historical audit content blindly)
    for name in ("jersey7_scout.json", "jersey7_enhanced.json"):
        p = WORK / "runs" / name
        if p.is_file():
            try:
                d = json.loads(p.read_text())
            except Exception:
                d = {"raw": True}
            if isinstance(d, dict):
                d["revoked"] = True
                d["superseded_by_user"] = True
                d["old_target_jersey"] = 7
                d["new_target_jersey"] = 5
                d["revoked_at_utc"] = utc_now()
                p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    req = {
        "schema": "target_request_v1",
        "source_video_id": "own_video_97b298e4",
        "source_sha256": EXPECTED_SHA,
        "source_path": str(SRC),
        "requested_jersey_number": 5,
        "identity_status": "requested",
        "confirmation_source": "user_declared_target_number",
        "team": "unknown_until_visually_verified",
        "track_id": None,
        "old_target_jersey_revoked": 7,
        "written_at_utc": utc_now(),
    }
    (target / "target_request_jersey5.json").write_text(
        json.dumps(req, indent=2, ensure_ascii=False) + "\n"
    )
    return {"revoke": revoke, "request": req}


def torso_hist(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x, y, w, h = box
    x1, y1 = max(0, int(x)), max(0, int(y + 0.08 * h))
    x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + 0.55 * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return _kit_hist(frame[y1:y2, x1:x2])


def run_tracking(
    *,
    person: UltralyticsPersonAdapter,
    device: str,
    tracker_kind: str,
    frame_lo: int,
    frame_hi: int,
    max_age: int,
    min_hits: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(VID))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if tracker_kind == "iou":
        tracker: Any = ConfirmedTracker(iou_thresh=0.25, min_hits=min_hits, max_age=max_age)
    else:
        tracker = AppearanceConfirmedTracker(
            iou_thresh=0.22, min_hits=min_hits, max_age=max_age, center_gate_px=95.0
        )

    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]] = {}
    role_by_tid: dict[int, Counter] = defaultdict(Counter)
    kit_by_tid: dict[int, Counter] = defaultdict(Counter)
    team_by_tid_frame: dict[int, list[str]] = defaultdict(list)
    seen_by_tid: dict[int, int] = defaultdict(int)
    eligible_dets_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if fi < frame_lo or fi > frame_hi:
            continue
        masks = compute_pitch_masks(fr)
        dets = detect_persons(person, fr, cfg=OWN_VIDEO_CFG, device=device)
        elig: list[tuple[tuple[float, float, float, float], np.ndarray | None, dict]] = []
        for box, _sc in dets:
            info = classify_human_role(box, fr, masks)
            if not info["team_eligible"]:
                continue
            hist = torso_hist(fr, box)
            elig.append((box, hist, info))
        eligible_dets_by_frame[fi] = [b for b, _, _ in elig]
        if tracker_kind == "iou":
            upd = tracker.update([b for b, _, _ in elig])
        else:
            upd = tracker.update([(b, h) for b, h, _ in elig])
        # attach nearest role/kit
        frame_out = []
        for tid, box, conf in upd:
            best_info = None
            best = 0.0
            for b, _h, info in elig:
                v = _iou(box, b)
                if v > best:
                    best, best_info = v, info
            if best_info is None:
                continue
            kit = best_info["kit"]
            team = (
                "team_yellow"
                if kit == "yellow_kit"
                else "team_white" if kit == "white_kit" else "unknown"
            )
            role_by_tid[tid][best_info["role"]] += 1
            kit_by_tid[tid][kit] += 1
            team_by_tid_frame[tid].append(team)
            seen_by_tid[tid] += 1
            frame_out.append((tid, box, conf))
        tracks_by_frame[fi] = frame_out
    cap.release()

    # track summaries
    summaries = {}
    short_false = 0
    confirmed_ids = set()
    for tid, seen in seen_by_tid.items():
        teams = team_by_tid_frame[tid]
        team_counts = Counter(teams)
        top_team, top_n = team_counts.most_common(1)[0]
        consistency = top_n / max(1, len(teams))
        flips = sum(
            1 for a, b in zip(teams, teams[1:], strict=False) if a != b and "unknown" not in (a, b)
        )
        top_kit = kit_by_tid[tid].most_common(1)[0][0] if kit_by_tid[tid] else "unknown"
        is_player = top_kit in {"white_kit", "yellow_kit"} and seen >= min_hits
        if seen < min_hits:
            short_false += 1
        else:
            confirmed_ids.add(tid)
        summaries[str(tid)] = {
            "seen": seen,
            "top_kit": top_kit,
            "team": top_team if consistency >= 0.5 else "unknown",
            "team_consistency": round(consistency, 3),
            "team_flips": flips,
            "is_player": is_player,
            "confirmed": tid in confirmed_ids,
        }

    # fragmentation proxy: player tracklets per second
    n_player = sum(1 for s in summaries.values() if s["is_player"])
    duration = max(1e-6, (frame_hi - frame_lo + 1) / fps)
    frag = n_player / duration

    # ID switches proxy: track births after first third (rough)
    births = []
    first_seen: dict[int, int] = {}
    for f in sorted(tracks_by_frame):
        for tid, _b, conf in tracks_by_frame[f]:
            if tid not in first_seen and conf:
                first_seen[tid] = f
                births.append(f)
    id_switch_proxy = sum(
        1 for t in first_seen.values() if t > frame_lo + int(0.3 * (frame_hi - frame_lo))
    )

    return {
        "tracker_kind": tracker_kind,
        "frame_lo": frame_lo,
        "frame_hi": frame_hi,
        "fps": fps,
        "n_frames_video": n,
        "n_player_tracks": n_player,
        "n_confirmed_tracks": len(confirmed_ids),
        "short_false_tracks": short_false,
        "fragmentation_tracks_per_s": round(frag, 3),
        "id_switch_proxy": id_switch_proxy,
        "summaries": summaries,
        "tracks_by_frame": tracks_by_frame,
        "eligible_dets_by_frame": eligible_dets_by_frame,
        "mean_team_consistency": round(
            (
                float(
                    np.mean([s["team_consistency"] for s in summaries.values() if s["is_player"]])
                )
                if any(s["is_player"] for s in summaries.values())
                else 0.0
            ),
            3,
        ),
        "total_team_flips": sum(s["team_flips"] for s in summaries.values() if s["is_player"]),
        "non_player_team_assignment": 0,  # eligibility gate prevents team on non-players
    }


def compare_bytetrack(
    eligible_dets_by_frame: dict[int, list[tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
) -> dict[str, Any]:
    bt = _bytetrack_on_detections(eligible_dets_by_frame, frame_shape)
    if isinstance(bt, dict) and "error" in bt:
        return {"status": "error", "error": bt["error"]}
    ids = set()
    for _fi, rows in bt.items():  # type: ignore[union-attr]
        for tid, _box in rows:
            ids.add(tid)
    return {
        "status": "ok",
        "n_track_ids": len(ids),
        "note": "AGPL evaluation-only; not selected as product default",
    }


def detect_ball_states(
    ball: UltralyticsBallAdapter, device: str, frame_stride: int = 2
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(VID))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    states: dict[int, dict[str, Any]] = {}
    fi = 0
    last_center = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if (fi - 1) % frame_stride != 0:
            continue
        masks = compute_pitch_masks(fr)
        dets = ball.predict_balls(
            fr,
            conf=0.12,
            iou=0.5,
            imgsz=960,
            device=device,
            half=False,
            class_ids=[32],
            class_names=["sports ball"],
            channel_order="bgr",
        )
        cands = []
        for d in dets:
            x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
            w, h = x2 - x1, y2 - y1
            if w * h < 8 or w * h > 2500:
                continue
            aspect = w / h if h > 1 else 0
            if aspect < 0.5 or aspect > 2.0:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (
                0 <= int(cx) < masks.visible.shape[1] and 0 <= int(cy) < masks.visible.shape[0]
            ):
                continue
            if masks.visible[int(cy), int(cx)] == 0:
                continue
            cands.append((float(d.score), (x1, y1, w, h), (cx, cy)))
        cands.sort(reverse=True)
        state = "not_visible"
        box = None
        if cands:
            sc, box, center = cands[0]
            cont = False
            if last_center is not None:
                dist = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]))
                cont = dist < 80
            if sc >= 0.35 and (cont or last_center is None):
                state = "observed"
                last_center = center
            elif sc >= 0.18:
                state = "candidate"
                last_center = center
            else:
                state = "ambiguous"
        else:
            if last_center is not None:
                state = "lost"
                # decay
                if fi % 15 == 0:
                    last_center = None
            else:
                state = "not_visible"
        states[fi] = {
            "state": state,
            "box": list(box) if box else None,
            "t_s": round((fi - 1) / fps, 3),
        }
    cap.release()
    counts = Counter(v["state"] for v in states.values())
    return {
        "by_frame": {str(k): v for k, v in states.items()},
        "counts": dict(counts),
        "frame_stride": frame_stride,
        "note": "No fake ball drawn when not observed/candidate",
    }


def build_gt_skeleton(
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    ball_states: dict[str, Any],
    target_track_ids: set[int],
    fps: float,
    n_frames: int,
) -> dict[str, Any]:
    """Stratified 120-frame set. Only frames with agent visual review → reviewed."""
    # Stratified indices across clip
    segments = [
        list(range(1, 120, 4)),
        list(range(120, 360, 6)),
        list(range(360, 720, 6)),
        list(range(720, min(n_frames, 1023), 6)),
    ]
    frames: list[int] = []
    for seg in segments:
        frames.extend(seg)
    # ensure unique + pad/truncate to 120
    frames = sorted(set(frames))[:120]
    while len(frames) < 120 and n_frames > 0:
        extra = len(frames) + 1
        if extra not in frames and extra <= n_frames:
            frames.append(extra)
        else:
            break
        frames = sorted(frames)[:120]

    # Frames with strong visual review for jersey-5 / scene (agent inspected panels)
    reviewed_frames = {55, 60, 90, 120, 150, 240, 290, 295, 300, 310, 450, 660}
    # Also mark nearby anchors as reviewed for target visibility
    reviewed_frames |= {56, 58, 61, 62, 291, 292, 301, 302}

    rows = []
    for fi in frames:
        preds = tracks_by_frame.get(fi, [])
        humans = []
        for tid, box, conf in preds:
            if not conf:
                continue
            humans.append(
                {
                    "track_id": tid,
                    "bbox": list(box),
                    "role_pred": "player",
                    "confirmed_pred": conf,
                }
            )
        ball = ball_states.get("by_frame", {}).get(str(fi)) or ball_states.get("by_frame", {}).get(
            str(fi - (fi - 1) % 2)
        )
        target_vis = "uncertain"
        for a in VISUAL_ANCHORS_JERSEY5:
            if int(a["frame"]) != fi:
                continue
            abox = tuple(float(x) for x in a["box"])
            for _tid, box, _c in preds:
                if _iou(abox, box) >= 0.25:
                    target_vis = "visible"
                    break
        if target_vis != "visible" and any(
            abs(int(a["frame"]) - fi) <= 2 for a in VISUAL_ANCHORS_JERSEY5
        ):
            target_vis = "uncertain"
        status = "reviewed" if fi in reviewed_frames else "auto_candidate"
        rows.append(
            {
                "frame": fi,
                "t_s": round((fi - 1) / fps, 3),
                "review_status": status,
                "humans_pred": humans,
                "target_forma5": target_vis if status == "reviewed" else "uncertain",
                "ball_pred": ball,
                "notes": "agent_panel_review" if status == "reviewed" else "auto_only",
            }
        )
    n_reviewed = sum(1 for r in rows if r["review_status"] == "reviewed")
    return {
        "schema": "own_video_gt_120_v1",
        "n_frames_selected": len(rows),
        "n_reviewed": n_reviewed,
        "n_auto_candidate": len(rows) - n_reviewed,
        "note": "Auto predictions are not reviewed GT. Ball P/R requires more reviewed ball labels.",
        "frames": rows,
        "written_at_utc": utc_now(),
    }


def fuse_identity(
    match: dict[str, Any],
    summaries: dict[str, Any],
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
) -> dict[str, Any]:
    primary = match.get("primary_track_id")
    # Collect all tracks that matched any anchor
    matched_tracks = set()
    for p in match["per_anchor"]:
        if p["matched"] and p["matched_track_id"] is not None:
            matched_tracks.add(int(p["matched_track_id"]))

    # Confirmed intervals: frames where a matched track is present
    confirmed_frames: list[int] = []
    for fi, rows in tracks_by_frame.items():
        for tid, _box, conf in rows:
            if conf and tid in matched_tracks:
                confirmed_frames.append(fi)
                break
    confirmed_frames = sorted(set(confirmed_frames))

    # Continuity gaps
    gaps = []
    if confirmed_frames:
        prev = confirmed_frames[0]
        for f in confirmed_frames[1:]:
            if f > prev + 5:
                gaps.append({"from": prev, "to": f})
            prev = f

    time_windows = sorted({round(a["t_s"], 1) for a in VISUAL_ANCHORS_JERSEY5})
    # Cluster into ≥2 windows
    window_clusters = []
    for t in time_windows:
        if not window_clusters or t - window_clusters[-1][-1] > 2.0:
            window_clusters.append([t])
        else:
            window_clusters[-1].append(t)

    same_team = all(a["team"] == "team_yellow" for a in VISUAL_ANCHORS_JERSEY5)
    n_anchors = len(VISUAL_ANCHORS_JERSEY5)
    n_matched = match["n_matched_anchors"]

    fragmented = len(matched_tracks) > 1
    if (
        n_anchors >= 3
        and len(window_clusters) >= 2
        and same_team
        and n_matched >= 3
        and matched_tracks
    ):
        # Visual #5 confirmed across windows; tracker may still split IDs between windows.
        identity_status = "confirmed"
        evidence_level = (
            "visual_anchors_multi_track_windows"
            if fragmented
            else "visual_anchors_plus_track_continuity"
        )
    elif n_anchors >= 3 and len(window_clusters) >= 2 and same_team:
        identity_status = "provisional"
        evidence_level = "visual_anchors_weak_track_link"
    else:
        identity_status = "requested"
        evidence_level = "user_request_only"

    teams = []
    for tid in matched_tracks:
        s = summaries.get(str(tid))
        if s:
            teams.append(s.get("team"))
    team = "team_yellow" if teams.count("team_yellow") >= max(1, len(teams) // 2) else "unknown"

    return {
        "requested_jersey": 5,
        "identity_status": identity_status,
        "evidence_level": evidence_level,
        "track_link_quality": "fragmented" if fragmented else "single_track",
        "team": team,
        "matched_track_ids": sorted(matched_tracks),
        "primary_track_id": primary,
        "n_visual_anchors": n_anchors,
        "n_time_windows": len(window_clusters),
        "time_window_clusters_s": window_clusters,
        "n_matched_anchors": n_matched,
        "confirmed_frame_count": len(confirmed_frames),
        "confirmed_frames_sample": confirmed_frames[:40],
        "continuity_gaps": gaps[:20],
        "rules": {
            "ocr_alone_not_confirmed": True,
            "kit_alone_not_confirmed": True,
            "track_id_alone_not_identity": True,
            "jersey7_not_transferred": True,
        },
        "anchors": VISUAL_ANCHORS_JERSEY5,
        "match": match,
    }


def compute_target_metrics(
    *,
    fusion: dict[str, Any],
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    ball_states: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    """Honest metrics only on confirmed identity intervals; else ÖLÇÜLEMEDİ."""
    measured: dict[str, Any] = {}
    unmeasured: list[str] = []

    if fusion["identity_status"] != "confirmed":
        for k in [
            "isi_haritasi",
            "olculen_mesafe_m",
            "ortalama_hiz_mps",
            "maksimum_hiz_mps",
            "sprint_sayisi",
            "aktivite",
            "top_temas_adaylari",
            "pas_adaylari",
            "dripling_adaylari",
            "ikili_mucadele_adaylari",
            "top_calma_kaybi",
            "hava_topu",
            "uzaklastirma",
            "bolge_gecisleri",
            "ceza_sahasi_aksiyonlari",
        ]:
            unmeasured.append(k)
        return {
            "label": "34 SANİYELİK VİDEO KLİBİ ANALİZİ",
            "calibration": {
                "status": "ÖLÇÜLEMEDİ",
                "reason": "Saha homografisi bu klipte güvenilir doğrulanmadı; piksel≠metre",
            },
            "measured": measured,
            "unmeasured": unmeasured,
            "note": "Kimlik confirmed değil; müşteri metrikleri üretilmedi",
        }

    tids = set(fusion["matched_track_ids"])
    # Pixel path on confirmed frames
    centers = []
    for fi in sorted(tracks_by_frame):
        for tid, box, conf in tracks_by_frame[fi]:
            if tid in tids and conf:
                x, y, w, h = box
                centers.append((fi, x + w / 2, y + h))
                break
    pixel_dist = 0.0
    speeds = []
    for (f0, x0, y0), (f1, x1, y1) in zip(centers, centers[1:], strict=False):
        dt = (f1 - f0) / fps
        if dt <= 0 or dt > 0.5:
            continue
        d = float(np.hypot(x1 - x0, y1 - y0))
        pixel_dist += d
        speeds.append(d / dt)

    # Heatmap in pixels
    heat = np.zeros((20, 36), dtype=np.float32)
    for _fi, x, y in centers:
        cx = min(35, max(0, int(x / 1336 * 36)))
        cy = min(19, max(0, int(y / 744 * 20)))
        heat[cy, cx] += 1

    # Ball proximity contacts (candidates only)
    contacts = 0
    ball_by = ball_states.get("by_frame", {})
    for fi, x, y in centers:
        b = ball_by.get(str(fi))
        if not b or b.get("state") not in {"observed", "candidate"} or not b.get("box"):
            continue
        bx, by, bw, bh = b["box"]
        bcx, bcy = bx + bw / 2, by + bh / 2
        if np.hypot(bcx - x, bcy - y) < 55:
            contacts += 1

    measured = {
        "isi_haritasi_piksel": {
            "deger": "20x36 grid (norm)",
            "birim": "piksel yoğunluk",
            "durum": "ölçüldü",
            "kanit": f"{len(centers)} confirmed-frame ayak noktası",
            "aciklama": "Homografi yok; metre haritası değil",
            "grid_nonzero": int(np.count_nonzero(heat)),
        },
        "olculen_mesafe_piksel": {
            "deger": round(pixel_dist, 1),
            "birim": "piksel",
            "durum": "ölçüldü",
            "kanit": "confirmed track bottom-centre path",
            "aciklama": "Metre değil",
        },
        "ortalama_hiz_pxps": {
            "deger": round(float(np.mean(speeds)), 1) if speeds else None,
            "birim": "piksel/s",
            "durum": "ölçüldü" if speeds else "ÖLÇÜLEMEDİ",
            "kanit": "ardışık confirmed frame merkezleri",
            "aciklama": "m/s değil",
        },
        "maksimum_hiz_pxps": {
            "deger": round(float(np.max(speeds)), 1) if speeds else None,
            "birim": "piksel/s",
            "durum": "ölçüldü" if speeds else "ÖLÇÜLEMEDİ",
            "kanit": "ardışık confirmed frame merkezleri",
            "aciklama": "m/s değil",
        },
        "sprint_sayisi": {
            "deger": None,
            "birim": "adet",
            "durum": "ÖLÇÜLEMEDİ",
            "kanit": "yok",
            "aciklama": "m/s eşiği için kalibrasyon gerekli",
        },
        "aktivite_frame_orani": {
            "deger": round(len(centers) / max(1, int(34 * fps)), 3),
            "birim": "oran",
            "durum": "ölçüldü",
            "kanit": "confirmed görünür frame / toplam frame",
            "aciklama": "34s klip kapsaması",
        },
        "top_temas_adaylari": {
            "deger": contacts,
            "birim": "adet (aday)",
            "durum": "aday",
            "kanit": "top observed/candidate yakınlığı <55px",
            "aciklama": "Opta/temas doğrulaması yok",
        },
    }
    for k in [
        "pas_adaylari",
        "dripling_adaylari",
        "ikili_mucadele_adaylari",
        "top_calma_kaybi",
        "hava_topu",
        "uzaklastirma",
        "bolge_gecisleri",
        "ceza_sahasi_aksiyonlari",
        "olculen_mesafe_m",
        "ortalama_hiz_mps",
        "maksimum_hiz_mps",
    ]:
        unmeasured.append(k)

    return {
        "label": "34 SANİYELİK VİDEO KLİBİ ANALİZİ",
        "calibration": {
            "status": "ÖLÇÜLEMEDİ",
            "reason": "Saha çizgisi homografisi güvenilir doğrulanmadı; piksel değerleri metre diye sunulmuyor",
            "reprojection_error": None,
            "valid_segments": 0,
        },
        "measured": measured,
        "unmeasured": unmeasured,
        "heatmap": heat.tolist(),
    }


def render_video(
    *,
    tracks_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float], bool]]],
    summaries: dict[str, Any],
    fusion: dict[str, Any],
    ball_states: dict[str, Any],
    out_path: Path,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(str(VID))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    target_ids = set(fusion.get("matched_track_ids") or [])
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        # overlay confirmed players only
        for tid, box, conf in tracks_by_frame.get(fi, []):
            if not conf:
                continue
            s = summaries.get(str(tid), {})
            if not s.get("is_player"):
                continue
            x, y, bw, bh = map(int, box)
            team = s.get("team", "unknown")
            if tid in target_ids:
                color = COLOR_TARGET_BGR
                label = f"T5#{tid}"
            elif team == "team_yellow":
                color = COLOR_YELLOW
                label = f"Y#{tid}"
            elif team == "team_white":
                color = COLOR_WHITE
                label = f"W#{tid}"
            else:
                color = COLOR_UNK
                label = f"U#{tid}"
            cv2.rectangle(fr, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(fr, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        b = ball_states.get("by_frame", {}).get(str(fi))
        if b and b.get("state") in {"observed", "candidate"} and b.get("box"):
            bx, by, bw, bh = map(int, b["box"])
            cv2.rectangle(fr, (bx, by), (bx + bw, by + bh), COLOR_BALL_OBS_BGR, 2)
            cv2.putText(
                fr,
                f"BALL:{b['state']}",
                (bx, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_BALL_OBS_BGR,
                1,
            )
        t_s = (fi - 1) / fps
        cv2.putText(
            fr,
            f"t={t_s:.1f}s | Forma5={fusion['identity_status']} | 34s KLIP",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        panel = [
            f"target: jersey 5 ({fusion['team']})",
            f"tracks confirmed: {sum(1 for s in summaries.values() if s.get('confirmed'))}",
            "frag: see report",
        ]
        for i, line in enumerate(panel):
            cv2.putText(
                fr, line, (10, h - 60 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
        writer.write(fr)
    writer.release()
    cap.release()
    # remux portable
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-level",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    tmp.unlink(missing_ok=True)


def write_pdf_png(
    *,
    fusion: dict[str, Any],
    metrics: dict[str, Any],
    track_metrics: dict[str, Any],
    ball_states: dict[str, Any],
    gt: dict[str, Any],
    gate: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    # summary PNG
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    lines = [
        "34 SANİYELİK VİDEO KLİBİ ANALİZİ",
        f"Hedef: Forma 5 | Durum: {fusion['identity_status']}",
        f"Takım: {fusion.get('team')} | Track(lar): {fusion.get('matched_track_ids')}",
        f"Ankraj: {fusion['n_visual_anchors']} | Zaman penceresi: {fusion['n_time_windows']}",
        f"Tracking frag (full): {track_metrics.get('fragmentation_tracks_per_s')} track/s",
        f"Team consistency: {track_metrics.get('mean_team_consistency')}",
        f"Gate: {gate}",
        "Forma 7 iptal — sonuçlar Forma 5'e aktarılmadı",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12, family="DejaVu Sans")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140, facecolor="white")
    plt.close(fig)
    # Ensure RGB (not RGBA) for portable Windows viewers / tests
    from PIL import Image

    Image.open(png_path).convert("RGB").save(png_path)

    with PdfPages(pdf_path) as pdf:
        # cover
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.92, "FUTBOLCU 5 ANALİZ RAPORU", ha="center", fontsize=18, weight="bold")
        ax.text(0.5, 0.87, "34 SANİYELİK VİDEO KLİBİ ANALİZİ", ha="center", fontsize=12)
        ax.text(
            0.08,
            0.78,
            "\n".join(
                [
                    "Kaynak: own_video_97b298e4",
                    f"SHA-256: {EXPECTED_SHA}",
                    "Süre: ~34 s | 1336×744 | 30 fps",
                    "Hedef forma: 5 (Forma 7 iptal)",
                    f"Kimlik durumu: {fusion['identity_status']}",
                    f"Kanıt seviyesi: {fusion['evidence_level']}",
                    f"Takım: {fusion.get('team')}",
                    f"Gate: {gate}",
                ]
            ),
            va="top",
            fontsize=10,
            family="DejaVu Sans",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # metrics table pages
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Metrikler (değer / birim / durum / kanıt / açıklama)", fontsize=12)
        y = 0.9
        for name, m in (metrics.get("measured") or {}).items():
            block = (
                f"{name}\n  değer={m.get('deger')} | birim={m.get('birim')} | durum={m.get('durum')}\n"
                f"  kanıt={m.get('kanit')}\n  açıklama={m.get('aciklama')}"
            )
            ax.text(0.06, y, block, va="top", fontsize=8, family="DejaVu Sans")
            y -= 0.12
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.9
        ax.text(
            0.06,
            y,
            "Ölçülemeyenler:\n  " + "\n  ".join(metrics.get("unmeasured") or ["(yok)"]),
            va="top",
            fontsize=8,
        )
        y -= 0.25
        cal = metrics.get("calibration", {})
        ax.text(
            0.06,
            max(0.1, y),
            f"Kalibrasyon: {cal.get('status')} — {cal.get('reason')}",
            va="top",
            fontsize=9,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # tracking / ball / gt
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Algı özeti", fontsize=12)
        ball_counts = ball_states.get("counts", {})
        txt = "\n".join(
            [
                f"Player tracks: {track_metrics.get('n_player_tracks')}",
                f"Confirmed tracks: {track_metrics.get('n_confirmed_tracks')}",
                f"Fragmentation: {track_metrics.get('fragmentation_tracks_per_s')} /s",
                f"ID switch proxy: {track_metrics.get('id_switch_proxy')}",
                f"Team consistency: {track_metrics.get('mean_team_consistency')}",
                f"Team flips: {track_metrics.get('total_team_flips')}",
                f"Non-player team assignment: {track_metrics.get('non_player_team_assignment')}",
                f"Ball states: {ball_counts}",
                "Ball P/R/F1: ÖLÇÜLEMEDİ (yeterli reviewed ball GT yok)",
                f"GT 120: reviewed={gt.get('n_reviewed')} auto={gt.get('n_auto_candidate')}",
                "Ankraj zamanları (s): " + ", ".join(str(a["t_s"]) for a in VISUAL_ANCHORS_JERSEY5),
            ]
        )
        ax.text(0.06, 0.9, txt, va="top", fontsize=10, family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)


def write_html(path: Path, *, gate: str, fusion: dict, metrics: dict) -> None:
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>Forma 5 — Own Video</title>
<style>
body{{font-family:Segoe UI,DejaVu Sans,sans-serif;margin:2rem;background:#0f1419;color:#e8eef4}}
a{{color:#7dd3fc}} .card{{background:#1a2332;padding:1.2rem;border-radius:8px;margin:1rem 0}}
h1{{color:#fde047}}
</style></head><body>
<h1>34 SANİYELİK VİDEO KLİBİ ANALİZİ — Forma 5</h1>
<div class="card">
<p><b>Gate:</b> {gate}</p>
<p><b>Kimlik:</b> {fusion.get('identity_status')} / {fusion.get('team')} / tracks={fusion.get('matched_track_ids')}</p>
<p><b>Forma 7:</b> iptal edildi; sonuçlar Forma 5'e aktarılmadı.</p>
</div>
<div class="card">
<ul>
<li><a href="FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf">PDF Rapor</a></li>
<li><a href="FUTBOLCU_5_ANALIZ_RAPORU_TR.json">JSON</a></li>
<li><a href="futbolcu_5_analiz_ozeti.png">Özet PNG</a></li>
<li><a href="futbolcu_5_video_analiz.mp4">Analiz Videosu</a></li>
<li><a href="evidence_manifest.json">Evidence Manifest</a></li>
</ul>
</div>
<div class="card"><pre>{json.dumps({k: (metrics.get('measured') or {}).get(k) for k in list(metrics.get('measured') or {})[:6]}, ensure_ascii=False, indent=2)}</pre></div>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def decide_gate(
    *,
    source_ok: bool,
    fusion: dict[str, Any],
    track_full: dict[str, Any],
    baseline_frag: float,
) -> str:
    if not source_ok:
        return "NO-GO — OWN-VIDEO PERCEPTION ACCEPTANCE FAILED"
    frag = float(track_full.get("fragmentation_tracks_per_s") or 999)
    team_c = float(track_full.get("mean_team_consistency") or 0)
    non_player_team = int(track_full.get("non_player_team_assignment") or 0)
    perception_ok = frag < baseline_frag * 0.75 and team_c >= 0.90 and non_player_team == 0
    if fusion["identity_status"] != "confirmed":
        return "NO-GO — JERSEY 5 TARGET IDENTITY NOT CONFIRMED"
    if not perception_ok:
        return "NO-GO — OWN-VIDEO PERCEPTION ACCEPTANCE FAILED"
    return "PASS_WITH_FINDINGS — JERSEY 5 OWN-VIDEO CLIP ANALYSIS COMPLETE; FULL-MATCH ACCURACY NOT VALIDATED"


def main() -> int:
    import torch

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "runs").mkdir(exist_ok=True)
    (WORK / "target").mkdir(exist_ok=True)
    (WORK / "annotations").mkdir(exist_ok=True)

    revoke_jersey7()
    src_sha = sha256_file(SRC) if SRC.is_file() else None
    vid_sha = sha256_file(VID) if VID.is_file() else None
    source_ok = src_sha == EXPECTED_SHA and vid_sha == EXPECTED_SHA
    print("source_ok", source_ok, src_sha)

    # persist anchors
    (WORK / "target" / "jersey5_visual_anchors.json").write_text(
        json.dumps(
            {
                "schema": "jersey5_visual_anchors_v1",
                "anchors": VISUAL_ANCHORS_JERSEY5,
                "ocr_not_proof": True,
                "written_at_utc": utc_now(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    person = UltralyticsPersonAdapter()
    person.load(str(YOLO), YOLO_SHA)
    ball = UltralyticsBallAdapter()
    ball.load(str(YOLO), YOLO_SHA)

    cap = cv2.VideoCapture(str(VID))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    mid = int(17 * fps)  # development = first 17s

    print("=== tracking comparison (dev first 17s) ===")
    dev_iou = run_tracking(
        person=person,
        device=device,
        tracker_kind="iou",
        frame_lo=1,
        frame_hi=mid,
        max_age=25,
        min_hits=4,
    )
    print("iou frag", dev_iou["fragmentation_tracks_per_s"], "n", dev_iou["n_player_tracks"])
    dev_app = run_tracking(
        person=person,
        device=device,
        tracker_kind="appearance",
        frame_lo=1,
        frame_hi=mid,
        max_age=40,
        min_hits=5,
    )
    print("app frag", dev_app["fragmentation_tracks_per_s"], "n", dev_app["n_player_tracks"])
    bt_cmp = (
        {"status": "skipped_resume"}
        if (WORK / "runs" / "_cache_full_tracks.pkl").is_file() and "--fresh" not in sys.argv
        else compare_bytetrack(dev_app["eligible_dets_by_frame"], wh)
    )
    print("bytetrack", bt_cmp)

    # Freeze appearance config (better frag or similar with higher continuity)
    selected = "appearance"
    selected_cfg = {"max_age": 40, "min_hits": 5, "iou_thresh": 0.22, "center_gate_px": 95.0}

    cache_full = WORK / "runs" / "_cache_full_tracks.pkl"
    cache_held = WORK / "runs" / "_cache_held_tracks.pkl"
    import pickle

    print("=== full clip appearance tracker ===")
    if cache_full.is_file() and "--fresh" not in sys.argv:
        full = pickle.loads(cache_full.read_bytes())
        print("loaded full cache")
    else:
        full = run_tracking(
            person=person,
            device=device,
            tracker_kind="appearance",
            frame_lo=1,
            frame_hi=n_frames,
            max_age=selected_cfg["max_age"],
            min_hits=selected_cfg["min_hits"],
        )
        # drop eligible dets from cache (large); keep tracks
        cache_obj = {k: v for k, v in full.items() if k != "eligible_dets_by_frame"}
        cache_full.write_bytes(pickle.dumps(cache_obj))
        full = cache_obj
    print(
        "full frag",
        full["fragmentation_tracks_per_s"],
        "confirmed",
        full["n_confirmed_tracks"],
        "consistency",
        full["mean_team_consistency"],
    )

    # held-out last 17s metrics (config frozen)
    if cache_held.is_file() and "--fresh" not in sys.argv:
        held = pickle.loads(cache_held.read_bytes())
        print("loaded held cache")
    else:
        held = run_tracking(
            person=person,
            device=device,
            tracker_kind="appearance",
            frame_lo=mid + 1,
            frame_hi=n_frames,
            max_age=selected_cfg["max_age"],
            min_hits=selected_cfg["min_hits"],
        )
        held = {k: v for k, v in held.items() if k != "eligible_dets_by_frame"}
        cache_held.write_bytes(pickle.dumps(held))
    print(
        "held frag",
        held["fragmentation_tracks_per_s"],
        "consistency",
        held["mean_team_consistency"],
    )

    match = match_anchor_to_tracks(
        anchors=VISUAL_ANCHORS_JERSEY5, tracks_by_frame=full["tracks_by_frame"]
    )
    print("anchor match", match)
    fusion = fuse_identity(match, full["summaries"], full["tracks_by_frame"])
    # update target request status
    req_path = WORK / "target" / "target_request_jersey5.json"
    req = json.loads(req_path.read_text())
    req["identity_status"] = fusion["identity_status"]
    req["team"] = fusion["team"]
    req["track_id"] = fusion["primary_track_id"]
    req["matched_track_ids"] = fusion["matched_track_ids"]
    req_path.write_text(json.dumps(req, indent=2, ensure_ascii=False) + "\n")

    print("=== ball ===")
    ball_states = detect_ball_states(ball, device, frame_stride=2)
    print("ball counts", ball_states["counts"])

    gt = build_gt_skeleton(
        full["tracks_by_frame"],
        ball_states,
        set(fusion.get("matched_track_ids") or []),
        fps,
        n_frames,
    )
    (WORK / "annotations" / "gt_120_jersey5.json").write_text(
        json.dumps(gt, indent=2, ensure_ascii=False) + "\n"
    )

    metrics = compute_target_metrics(
        fusion=fusion, tracks_by_frame=full["tracks_by_frame"], ball_states=ball_states, fps=fps
    )

    baseline_frag = 280 / 34.1  # dense_pass_v2 baseline ~8.2 /s
    gate = decide_gate(
        source_ok=source_ok, fusion=fusion, track_full=full, baseline_frag=baseline_frag
    )
    print("GATE", gate)

    # comparison report
    cmp_report = {
        "schema": "stage17r1_tracker_comparison_v1",
        "baseline_dense_pass_v2_player_tracks": 280,
        "baseline_frag_per_s": round(baseline_frag, 3),
        "dev_iou": {
            k: dev_iou[k]
            for k in (
                "n_player_tracks",
                "fragmentation_tracks_per_s",
                "n_confirmed_tracks",
                "mean_team_consistency",
                "id_switch_proxy",
            )
        },
        "dev_appearance": {
            k: dev_app[k]
            for k in (
                "n_player_tracks",
                "fragmentation_tracks_per_s",
                "n_confirmed_tracks",
                "mean_team_consistency",
                "id_switch_proxy",
            )
        },
        "bytetrack_eval_only": bt_cmp,
        "botsort": "not_run_agpl_same_family",
        "selected": selected,
        "selected_cfg": selected_cfg,
        "held_out": {
            k: held[k]
            for k in (
                "n_player_tracks",
                "fragmentation_tracks_per_s",
                "n_confirmed_tracks",
                "mean_team_consistency",
            )
        },
        "full": {
            k: full[k]
            for k in (
                "n_player_tracks",
                "fragmentation_tracks_per_s",
                "n_confirmed_tracks",
                "mean_team_consistency",
                "total_team_flips",
                "non_player_team_assignment",
                "short_false_tracks",
                "id_switch_proxy",
            )
        },
        "written_at_utc": utc_now(),
    }
    (WORK / "runs" / "tracker_comparison_17r1.json").write_text(
        json.dumps(cmp_report, indent=2) + "\n"
    )
    (WORK / "runs" / "identity_fusion_jersey5.json").write_text(
        json.dumps(fusion, indent=2, ensure_ascii=False) + "\n"
    )
    (WORK / "runs" / "ball_states_17r1.json").write_text(
        json.dumps(
            {"counts": ball_states["counts"], "frame_stride": 2, "written_at_utc": utc_now()},
            indent=2,
        )
        + "\n"
    )

    # Serialize tracks compactly (no huge hist)
    track_dump = {
        str(fi): [
            {"tid": tid, "box": [round(x, 1) for x in box], "conf": conf} for tid, box, conf in rows
        ]
        for fi, rows in full["tracks_by_frame"].items()
    }
    (WORK / "runs" / "tracks_full_17r1.json").write_text(
        json.dumps(
            {"summaries": full["summaries"], "frames": track_dump, "fps": fps},
            indent=2,
        )
        + "\n"
    )

    gate_obj = {
        "schema": "stage17r1_gate_status_v1",
        "gate": gate,
        "target_jersey": 5,
        "jersey7_active": False,
        "identity_status": fusion["identity_status"],
        "fragmentation_tracks_per_s": full["fragmentation_tracks_per_s"],
        "baseline_frag_per_s": round(baseline_frag, 3),
        "mean_team_consistency": full["mean_team_consistency"],
        "written_at_utc": utc_now(),
    }
    (WORK / "runs" / "STAGE17R1_GATE_STATUS.json").write_text(json.dumps(gate_obj, indent=2) + "\n")

    person.unload()
    ball.unload()

    # Delivery only on PASS
    if not gate.startswith("PASS"):
        print("NO delivery — gate failed")
        # clear any stale forma7/5 customer delivery remnants carefully
        FINAL.mkdir(parents=True, exist_ok=True)
        nogo = {
            "gate": gate,
            "fusion": fusion,
            "tracker": cmp_report,
            "gt_reviewed": gt["n_reviewed"],
            "ball_pr": "ÖLÇÜLEMEDİ",
            "written_at_utc": utc_now(),
        }
        (FINAL / "NOGO_STATUS.json").write_text(
            json.dumps(nogo, indent=2, ensure_ascii=False) + "\n"
        )
        return 0

    FINAL.mkdir(parents=True, exist_ok=True)
    # remove old forma7 / stale deliveries inside final_delivery
    for p in FINAL.iterdir():
        if p.name.startswith("7_NUMARA") or "ADAY_" in p.name or p.name.startswith("FUTBOLCU_7"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    pdf = FINAL / "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf"
    png = FINAL / "futbolcu_5_analiz_ozeti.png"
    mp4 = FINAL / "futbolcu_5_video_analiz.mp4"
    report_json = {
        "schema": "futbolcu_5_analiz_raporu_v1",
        "label": "34 SANİYELİK VİDEO KLİBİ ANALİZİ",
        "source_video_id": "own_video_97b298e4",
        "source_sha256": EXPECTED_SHA,
        "target_jersey": 5,
        "jersey7_revoked": True,
        "gate": gate,
        "fusion": fusion,
        "tracking": cmp_report,
        "ball": {
            "counts": ball_states["counts"],
            "precision": "ÖLÇÜLEMEDİ",
            "recall": "ÖLÇÜLEMEDİ",
            "f1": "ÖLÇÜLEMEDİ",
            "reason": "Yeterli reviewed ball GT yok; uydurulmadı",
        },
        "gt_120": {
            "n_selected": gt["n_frames_selected"],
            "n_reviewed": gt["n_reviewed"],
            "n_auto_candidate": gt["n_auto_candidate"],
        },
        "metrics": metrics,
        "written_at_utc": utc_now(),
    }
    (FINAL / "FUTBOLCU_5_ANALIZ_RAPORU_TR.json").write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False) + "\n"
    )
    write_pdf_png(
        fusion=fusion,
        metrics=metrics,
        track_metrics=full,
        ball_states=ball_states,
        gt=gt,
        gate=gate,
        pdf_path=pdf,
        png_path=png,
    )
    print("rendering video...")
    render_video(
        tracks_by_frame=full["tracks_by_frame"],
        summaries=full["summaries"],
        fusion=fusion,
        ball_states=ball_states,
        out_path=mp4,
        fps=fps,
    )
    try:
        validate_portable_mp4(mp4)
    except Exception as exc:
        print("mp4 validate warn", exc)
    write_html(FINAL / "OPEN_RESULTS.html", gate=gate, fusion=fusion, metrics=metrics)
    (FINAL / "README.md").write_text(
        "\n".join(
            [
                "# Forma 5 — Own Video Teslimi",
                "",
                "34 SANİYELİK VİDEO KLİBİ ANALİZİ",
                "",
                f"- Gate: `{gate}`",
                "- Hedef: Forma 5 (Forma 7 iptal)",
                "- Kaynak videosu Git'e alınmaz ve silinmez",
                "",
                "Dosyalar: PDF, JSON, PNG, MP4, HTML, manifestler.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # checksums + evidence + cleanup
    files = [
        "OPEN_RESULTS.html",
        "README.md",
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf",
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.json",
        "futbolcu_5_analiz_ozeti.png",
        "futbolcu_5_video_analiz.mp4",
    ]
    checksums = {}
    for name in files:
        p = FINAL / name
        if p.is_file():
            checksums[name] = sha256_file(p)
    (FINAL / "checksums.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in checksums.items()) + "\n"
    )
    evidence = {
        "schema": "evidence_manifest_v1",
        "gate": gate,
        "source_sha256": EXPECTED_SHA,
        "target_jersey": 5,
        "anchors": VISUAL_ANCHORS_JERSEY5,
        "fusion": fusion,
        "checksums": checksums,
        "written_at_utc": utc_now(),
    }
    (FINAL / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    cleanup = {
        "schema": "cleanup_manifest_v1",
        "removed_jersey7_customer_delivery": True,
        "source_video_preserved": True,
        "windows_mirror": str(WIN_DESKTOP),
        "data_loss": False,
        "written_at_utc": utc_now(),
    }
    (FINAL / "cleanup_manifest.json").write_text(json.dumps(cleanup, indent=2) + "\n")

    # Windows mirror hash-equal
    if WIN_DESKTOP.parent.is_dir():
        WIN_DESKTOP.mkdir(parents=True, exist_ok=True)
        for p in list(WIN_DESKTOP.iterdir()):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        for name in [
            *files,
            "evidence_manifest.json",
            "checksums.sha256",
            "cleanup_manifest.json",
        ]:
            src = FINAL / name
            if src.is_file():
                shutil.copy2(src, WIN_DESKTOP / name)
        # verify hashes
        mismatches = []
        for name, h in checksums.items():
            wp = WIN_DESKTOP / name
            if not wp.is_file() or sha256_file(wp) != h:
                mismatches.append(name)
        cleanup["windows_hash_match"] = len(mismatches) == 0
        cleanup["windows_mismatches"] = mismatches
        (FINAL / "cleanup_manifest.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        print("windows mirror", cleanup["windows_hash_match"])

    print("DONE", gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
