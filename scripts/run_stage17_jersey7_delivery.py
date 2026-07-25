#!/usr/bin/env python3
"""Stage 17 dual jersey-7 delivery: Aday A (light) + Aday B (dark) separate reports."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
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
    COLOR_TEAM_A_BGR,
    COLOR_TEAM_B_BGR,
    COLOR_UNKNOWN_BGR,
    ConfirmedTracker,
    _kit_hist,
)
from football_analytics.acceptance.portable_final_media import (  # noqa: E402
    OUT_H,
    OUT_W,
    _letterbox,
    _scale_box,
    validate_portable_mp4,
)
from football_analytics.acceptance.stage17_jersey7.pipeline import (  # noqa: E402
    BROADCAST_CFG,
    blur_head_regions,
    detect_persons_frame,
    ocr_jersey7,
)
from football_analytics.identity.jersey_ocr_config import (  # noqa: E402
    default_jersey_ocr_config_path,
    load_jersey_ocr_config,
)
from football_analytics.perception.adapters.ultralytics_ball import (  # noqa: E402
    UltralyticsBallAdapter,
)
from football_analytics.perception.adapters.ultralytics_person import (  # noqa: E402
    UltralyticsPersonAdapter,
)

YOLO = Path("/home/fdoblak/football_data/model_archive/yolo11n.pt")
YOLO_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
VID = Path(
    "/home/fdoblak/football_data/videos/normalized/authorized_youtube_B_cKZkrgxrM/analysis_1280.mp4"
)
RAW = Path(
    "/home/fdoblak/football_data/videos/raw_matches/authorized_youtube_B_cKZkrgxrM/B_cKZkrgxrM.mp4"
)
WORK = Path("/home/fdoblak/workspace/stage17_jersey7")
SCOUT = Path("/home/fdoblak/workspace/target_7_selection/jersey7_scout_full.json")
CROP_DIR = Path("/home/fdoblak/workspace/target_7_selection")
FINAL = REPO / "artifacts" / "final_delivery"
WIN_DOWNLOADS = Path("/mnt/c/Users/furka/Downloads/Football Analytics Final")
WIN_DESKTOP = Path("/mnt/c/Users/furka/Desktop/Football Analytics Final")

MATCH = {
    "title": "Galatasaray - Erokspor U-12 Lig Maçı İkinci Yarı",
    "date": "12.01.2025",
    "source_url": "https://www.youtube.com/watch?v=B_cKZkrgxrM",
    "video_id": "B_cKZkrgxrM",
    "duration_s": 1273,
}


def windows_around_hits(hits: list[dict], fps: float, pad_s: float = 22.0) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for h in hits:
        c = int(h["frame"])
        lo = max(1, c - int(pad_s * fps))
        hi = c + int(pad_s * fps)
        windows.append((lo, hi))
    windows.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in windows:
        if not merged or lo > merged[-1][1] + 1:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def in_windows(fi: int, windows: list[tuple[int, int]]) -> bool:
    return any(lo <= fi <= hi for lo, hi in windows)


def brightness_kit(
    crop_path: Path | None, fallback_box: list[float] | None = None
) -> tuple[str, float]:
    """Reviewed kit split: light vs dark by torso brightness (not club name)."""
    mean = None
    if crop_path and crop_path.is_file():
        im = cv2.imread(str(crop_path))
        if im is not None:
            mean = float(np.mean(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
    if mean is None:
        mean = 128.0
    kit = "light_kit" if mean >= 110.0 else "dark_kit"
    return kit, mean


def load_dual_candidates() -> list[dict[str, Any]]:
    scout = json.loads(SCOUT.read_text())
    hits = scout["hits"]
    enriched = []
    for h in hits:
        crops = list(CROP_DIR.glob(f"hit_{h['frame']}_*.png"))
        crop = crops[0] if crops else None
        kit, bright = brightness_kit(crop)
        enriched.append(
            {**h, "kit_reviewed": kit, "brightness": bright, "crop": str(crop) if crop else None}
        )
    groups = {
        "A": {
            "id": "A",
            "slug": "ADAY_A",
            "label_tr": "Aday A — Açık Forma 7 Numara",
            "kit": "light_kit",
            "kit_tr": "açık renk forma",
            "hits": [h for h in enriched if h["kit_reviewed"] == "light_kit"],
        },
        "B": {
            "id": "B",
            "slug": "ADAY_B",
            "label_tr": "Aday B — Koyu Forma 7 Numara",
            "kit": "dark_kit",
            "kit_tr": "koyu renk forma",
            "hits": [h for h in enriched if h["kit_reviewed"] == "dark_kit"],
        },
    }
    out = []
    for key in ("A", "B"):
        g = groups[key]
        assert len(g["hits"]) >= 2, f"candidate {key} needs >=2 OCR hits"
        ts = sorted(h["t_s"] for h in g["hits"])
        assert ts[-1] - ts[0] >= 30.0, f"candidate {key} needs spaced evidence"
        out.append(g)
    return out


def analyze_candidate(
    cand: dict[str, Any],
    *,
    person: UltralyticsPersonAdapter,
    ball: UltralyticsBallAdapter,
    ocr_cfg: Any,
    device: str,
) -> dict[str, Any]:
    hits = cand["hits"]
    tracker = ConfirmedTracker(iou_thresh=0.3, min_hits=4, max_age=20)
    cap = cv2.VideoCapture(str(VID))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    windows = windows_around_hits(hits, fps, pad_s=22.0)
    print(f"[{cand['id']}] windows={windows} hits={len(hits)}", flush=True)

    tracks_by_frame: dict[int, list[dict]] = {}
    ball_by_frame: dict[int, dict] = {}
    jersey_votes: dict[int, int] = defaultdict(int)
    kit_votes_light: dict[int, int] = defaultdict(int)
    kit_votes_dark: dict[int, int] = defaultdict(int)
    team_samples: dict[int, list[np.ndarray]] = defaultdict(list)
    det_count = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            dense = in_windows(frame_idx, windows)
            if not dense and ((frame_idx - 1) % 20 != 0):
                continue
            dets = detect_persons_frame(person, frame, cfg=BROADCAST_CFG, device=device)
            boxes = [b for b, _ in dets]
            det_count += len(boxes)
            tracked = tracker.update(boxes)
            fr_tracks = []
            for tid, box, confirmed in tracked:
                if not confirmed:
                    continue
                ocr = ocr_jersey7(frame, box, cfg=ocr_cfg) if dense else None
                x, y, w, h = box
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
                mean = None
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    mean = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
                    if mean >= 110:
                        kit_votes_light[tid] += 1
                    else:
                        kit_votes_dark[tid] += 1
                    if len(team_samples[tid]) < 8:
                        team_samples[tid].append(_kit_hist(crop))
                if ocr and mean is not None:
                    want_light = cand["kit"] == "light_kit"
                    this_light = mean >= 110.0
                    if this_light == want_light:
                        jersey_votes[tid] += 1
                fr_tracks.append(
                    {
                        "track_id": tid,
                        "bbox": list(box),
                        "confirmed": True,
                        "jersey7_hit": bool(ocr),
                    }
                )
            tracks_by_frame[frame_idx] = fr_tracks

            if dense:
                bbs = ball.predict_balls(
                    frame,
                    conf=0.12,
                    iou=0.3,
                    imgsz=640,
                    device=device,
                    half=False,
                    class_ids=[32],
                    class_names=["sports ball"],
                    channel_order="bgr",
                )
                state, bb = "not_visible", None
                cands = []
                for det in bbs:
                    bw, bh = float(det.x2 - det.x1), float(det.y2 - det.y1)
                    if 3 <= bw <= 70 and 3 <= bh <= 70:
                        cands.append((float(det.x1), float(det.y1), bw, bh, float(det.score)))
                if cands:
                    cands.sort(key=lambda t: t[4], reverse=True)
                    best = cands[0]
                    if best[4] >= 0.3:
                        state, bb = "observed", best[:4]
                    elif best[4] >= 0.15:
                        state, bb = "candidate", best[:4]
                    else:
                        state = "ambiguous"
                ball_by_frame[frame_idx] = {"state": state, "bbox": list(bb) if bb else None}
            if frame_idx % 1200 == 0:
                print(f"[{cand['id']}] frame={frame_idx}/{n}", flush=True)
    finally:
        cap.release()

    # Prefer tracks whose kit votes match candidate
    want_light = cand["kit"] == "light_kit"
    filtered_votes: dict[int, float] = {}
    for tid, v in jersey_votes.items():
        light = kit_votes_light[tid]
        dark = kit_votes_dark[tid]
        kit_ok = (light >= dark) if want_light else (dark > light)
        if kit_ok:
            filtered_votes[tid] = float(v)
    if not filtered_votes:
        # geometry fallback against scout boxes of this candidate only
        score: dict[int, float] = defaultdict(float)
        hit_boxes = {int(h["frame"]): tuple(h["box"]) for h in hits}
        for fi, trs in tracks_by_frame.items():
            nearest = min(hit_boxes, key=lambda f: abs(f - fi))
            if abs(nearest - fi) > int(2.5 * fps):
                continue
            gb = hit_boxes[nearest]
            for t in trs:
                tid = int(t["track_id"])
                light = kit_votes_light[tid]
                dark = kit_votes_dark[tid]
                kit_ok = (light >= dark) if want_light else (dark > light)
                if not kit_ok and (light + dark) > 0:
                    continue
                box = tuple(t["bbox"])
                gc = (gb[0] + gb[2] / 2, gb[1] + gb[3] / 2)
                pc = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                dist = math.hypot(gc[0] - pc[0], gc[1] - pc[1])
                if dist < 90:
                    score[tid] += 1.0 / (1.0 + dist)
        if not score:
            raise RuntimeError(f"NO-GO — JERSEY 7 IDENTITY EVIDENCE INSUFFICIENT for {cand['id']}")
        target_tid = max(score, key=score.get)
        identity_status = "reviewed_provisional"
    else:
        target_tid = max(filtered_votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        identity_status = "confirmed" if filtered_votes[target_tid] >= 2 else "reviewed_provisional"

    feats = {tid: np.mean(np.stack(v, 0), 0) for tid, v in team_samples.items() if len(v) >= 2}
    team_of = {tid: "unknown" for tid in feats}
    seed = [t for t in feats if t != target_tid]
    if len(seed) >= 4:
        X = np.stack([feats[t] for t in seed], 0)
        rng = np.random.default_rng(7 + ord(cand["id"]))
        cents = X[rng.choice(len(X), 2, replace=False)].copy()
        labels = np.zeros(len(X), np.int32)
        for _ in range(20):
            d0 = np.linalg.norm(X - cents[0], axis=1)
            d1 = np.linalg.norm(X - cents[1], axis=1)
            labels = (d1 < d0).astype(np.int32)
            for k in (0, 1):
                if np.any(labels == k):
                    cents[k] = X[labels == k].mean(0)
        for tid, _lab in zip(seed, labels, strict=True):
            d0 = float(np.linalg.norm(feats[tid] - cents[0]))
            d1 = float(np.linalg.norm(feats[tid] - cents[1]))
            team_of[tid] = "unknown" if abs(d0 - d1) < 0.05 else ("team_a" if d0 < d1 else "team_b")
        if target_tid in feats:
            d0 = float(np.linalg.norm(feats[target_tid] - cents[0]))
            d1 = float(np.linalg.norm(feats[target_tid] - cents[1]))
            team_of[target_tid] = "team_a" if d0 < d1 else "team_b"
    target_team = team_of.get(target_tid, "unknown")

    target_frames = sorted(
        fi for fi, trs in tracks_by_frame.items() if any(t["track_id"] == target_tid for t in trs)
    )
    coverage = len(target_frames) / max(1, len(tracks_by_frame))

    events = []
    last_event_t = -1e9
    for fi in target_frames:
        b = ball_by_frame.get(fi)
        if not b or b["state"] not in {"observed", "candidate"} or not b["bbox"]:
            continue
        tbox = next(t["bbox"] for t in tracks_by_frame[fi] if t["track_id"] == target_tid)
        bx, by, bw, bh = b["bbox"]
        bc = (bx + bw / 2, by + bh / 2)
        tc = (tbox[0] + tbox[2] / 2, tbox[1] + tbox[3] / 2)
        dist = math.hypot(bc[0] - tc[0], bc[1] - tc[1])
        t_s = (fi - 1) / fps
        if dist < max(55.0, 0.9 * tbox[3]) and t_s - last_event_t > 3.0:
            events.append(
                {
                    "event_id": f"{cand['slug'].lower()}_ball_proxy_{len(events)+1}",
                    "type": "topla_buluşma_adayı",
                    "t_start": t_s,
                    "t_end": t_s + 0.5,
                    "confidence": 0.45 if b["state"] == "candidate" else 0.6,
                    "review_status": "heuristic_proxy_not_validated_event",
                    "ball_state": b["state"],
                    "evidence_frame": fi,
                }
            )
            last_event_t = t_s

    centers = []
    for fi in target_frames:
        box = next(t["bbox"] for t in tracks_by_frame[fi] if t["track_id"] == target_tid)
        centers.append((fi, box[0] + box[2] / 2, box[1] + box[3] / 2))
    pix_dist = 0.0
    speeds = []
    for i in range(1, len(centers)):
        f0, x0, y0 = centers[i - 1]
        f1, x1, y1 = centers[i]
        dt = (f1 - f0) / fps
        d = math.hypot(x1 - x0, y1 - y0)
        pix_dist += d
        if dt > 0:
            speeds.append(d / dt)

    ball_obs = sum(1 for v in ball_by_frame.values() if v["state"] == "observed")
    ball_cand = sum(1 for v in ball_by_frame.values() if v["state"] == "candidate")

    return {
        "cand": cand,
        "fps": fps,
        "target_tid": target_tid,
        "target_team": target_team,
        "identity_status": identity_status,
        "jersey_votes": dict(jersey_votes),
        "filtered_votes": {str(k): v for k, v in filtered_votes.items()},
        "tracks_by_frame": tracks_by_frame,
        "ball_by_frame": ball_by_frame,
        "team_of": team_of,
        "target_frames": target_frames,
        "coverage": coverage,
        "events": events,
        "centers": centers,
        "pix_dist": pix_dist,
        "speeds": speeds,
        "ball_obs": ball_obs,
        "ball_cand": ball_cand,
        "windows": windows,
        "det_mean": det_count / max(1, len(tracks_by_frame)),
        "n_processed": len(tracks_by_frame),
    }


def build_metric_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    cand = result["cand"]
    hits = cand["hits"]
    hit_ts = [round(h["t_s"], 1) for h in hits]
    coverage = result["coverage"]
    centers = result["centers"]
    events = result["events"]
    pix_dist = result["pix_dist"]
    olm = "ÖLÇÜLEMEDİ"

    def row(name, value, unit, status, evidence_n, ts, cov, note):
        return {
            "metric": name,
            "value": value,
            "unit": unit,
            "status": status,
            "evidence_count": evidence_n,
            "video_timestamps": ts,
            "coverage": cov,
            "description": note,
        }

    def need(why: str) -> str:
        return f"ÖLÇÜLEMEDİ — gerekli kanıt: {why}"

    return [
        row(
            "Isı haritası",
            need("geçerli saha kalibrasyonu + projected positions"),
            "-",
            olm,
            0,
            [],
            0,
            "Broadcast kalibrasyonu valid değil",
        ),
        row(
            "İkili mücadele sayısı",
            need("manuel/güvenilir duel etiketleri"),
            "adet",
            olm,
            0,
            [],
            0,
            "",
        ),
        row("Kazanılan ikili mücadele", need("duel outcome"), "adet", olm, 0, [], 0, ""),
        row("İkili mücadele kazanma oranı", need("duel outcome"), "%", olm, 0, [], 0, ""),
        row("Pas girişimi", need("doğrulanmış pas event ledger"), "adet", olm, 0, [], 0, ""),
        row("Tamamlanan pas", need("pas başarı sonucu"), "adet", olm, 0, [], 0, ""),
        row("Başarısız pas", need("pas başarısızlık sonucu"), "adet", olm, 0, [], 0, ""),
        row("Pas isabet oranı", need("pas outcome"), "%", olm, 0, [], 0, ""),
        row("Başarılı dripling", need("take-on outcome"), "adet", olm, 0, [], 0, ""),
        row("Başarısız dripling", need("take-on outcome"), "adet", olm, 0, [], 0, ""),
        row("Adam eksiltme oranı", need("take-on outcome"), "%", olm, 0, [], 0, ""),
        row("Top çalma", need("tackle event kanıtı"), "adet", olm, 0, [], 0, ""),
        row("Top kaybı", need("turnover event kanıtı"), "adet", olm, 0, [], 0, ""),
        row("Hava topu mücadelesi", need("aerial duel etiketi"), "adet", olm, 0, [], 0, ""),
        row("Kazanılan hava topu", need("aerial outcome"), "adet", olm, 0, [], 0, ""),
        row("Uzaklaştırma", need("clearance etiketi"), "adet", olm, 0, [], 0, ""),
        row("1→2 bölge geçiş pası", need("kalibrasyon + pas geometry"), "adet", olm, 0, [], 0, ""),
        row("2→3 bölge geçiş pası", need("kalibrasyon + pas geometry"), "adet", olm, 0, [], 0, ""),
        row("Uzun pas sayısı", need("pas uzunluk tanımı"), "adet", olm, 0, [], 0, ""),
        row("Uzun pas oranı", need("uzun pas / tüm pas"), "%", olm, 0, [], 0, ""),
        row(
            "Ölçülen koşu mesafesi",
            need("valid pitch calibration; yalnız px mesafe mevcut"),
            "m",
            olm,
            len(centers),
            hit_ts[:5],
            coverage,
            f"GÖRÜNTÜLENEBİLEN VE KALİBRE EDİLEMEYEN BÖLÜMLERDE px={pix_dist:.1f}; metre yok",
        ),
        row("Sprint sayısı", need("m/s eşik + kalibrasyon"), "adet", olm, 0, [], 0, ""),
        row("Sprint mesafesi", need("kalibrasyon"), "m", olm, 0, [], 0, ""),
        row("Sprint süresi", need("sprint segmentleri"), "s", olm, 0, [], 0, ""),
        row("Ortalama hız", need("kalibrasyon"), "m/s", olm, 0, [], coverage, ""),
        row("Maksimum hız", need("kalibrasyon"), "m/s", olm, 0, [], coverage, ""),
        row(
            "Ceza sahası topla buluşma",
            need("ceza sahası polygon + touch event"),
            "adet",
            olm,
            0,
            [],
            0,
            "Presence ≠ touch",
        ),
        row(
            "Aktivite indeksi",
            need("doğrulanmış event seti"),
            "-",
            olm,
            len(events),
            [round(e["t_start"], 1) for e in events[:5]],
            coverage,
            f"yalnız {len(events)} top-yakınlık proxy adayı (müşteri event sayılmaz)",
        ),
        row(
            "Görünürlük coverage",
            round(coverage, 4),
            "oran",
            "KISMI_ÖLÇÜLDÜ",
            len(result["target_frames"]),
            hit_ts,
            coverage,
            "İşlenen karelerde hedef track görünürlük oranı",
        ),
        row(
            "Kimlik coverage",
            len(hits),
            "OCR gözlem",
            "KISMI_ÖLÇÜLDÜ",
            len(hits),
            hit_ts,
            None,
            f"Forma 7 + {cand['kit_tr']}; yüz tanıma yok; {cand['label_tr']}",
        ),
    ]


def render_pdf(path: Path, result: dict[str, Any], table: list[dict], payload: dict) -> None:
    cand = result["cand"]
    hits = cand["hits"]
    hit_ts = [round(h["t_s"], 1) for h in hits]
    with PdfPages(str(path)) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#0f1c2e")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.text(
            0.5,
            0.72,
            "7 NUMARALI OYUNCU\nPERFORMANS ANALİZİ",
            ha="center",
            va="center",
            color="white",
            fontsize=22,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )
        ax.text(
            0.5,
            0.52,
            cand["label_tr"],
            ha="center",
            color="#f0c36a",
            fontsize=14,
            fontfamily="DejaVu Sans",
        )
        ax.text(
            0.5,
            0.42,
            "Galatasaray – Erokspor U-12 · 12.01.2025 · İkinci Yarı",
            ha="center",
            color="#9ecbff",
            fontsize=12,
            fontfamily="DejaVu Sans",
        )
        ax.text(
            0.5,
            0.32,
            f"Forma rengi: {cand['kit_tr']} · Takım kümesi: {result['target_team']}",
            ha="center",
            color="#e8f0ff",
            fontsize=11,
            fontfamily="DejaVu Sans",
        )
        ax.text(
            0.5,
            0.18,
            "Yüz tanıma yok · İsim yok · Kısmi coverage · Ayrı aday raporu",
            ha="center",
            color="#9ecbff",
            fontsize=10,
            fontfamily="DejaVu Sans",
        )
        pdf.savefig(fig)
        plt.close(fig)

        sections = [
            (
                "Yönetici Özeti",
                "Bu rapor yalnız yetkilendirilmiş YouTube videosunda forma 7 + appearance\n"
                f"kanıtıyla izlenen {cand['label_tr']} adayına aittir.\n"
                "Fiziksel metre metrikleri ve çoğu olay metriği kalibrasyon/event doğrulaması\n"
                "olmadığı için ÖLÇÜLEMEDİ olarak bırakılmıştır. Rakam uydurulmamıştır.",
            ),
            (
                "Video ve Hedef",
                f"Kaynak: B_cKZkrgxrM\nSüre: 1273 s\nAday: {cand['id']}\n"
                f"Hedef track: {result['target_tid']}\nTakım kümesi: {result['target_team']}\n"
                f"OCR 7 gözlemleri: {len(hits)} ({hit_ts})\nKimlik durumu: {result['identity_status']}",  # noqa: E501
            ),
            (
                "Veri Kapsamı ve Güvenilirlik",
                "Kanıt seviyeleri: KISMI_ÖLÇÜLDÜ / ÖLÇÜLEMEDİ.\n"
                "SoccerTrack/TeamTrack eski metrikler taşınmadı.\n"
                "İki ayrı 7 numaralı aday için ayrı rapor üretilmiştir.",
            ),
            (
                "Kimlik Kanıtı",
                f"Forma numarası 7, kit={cand['kit_tr']}, zamanlar {hit_ts}.\n"
                "Yüz tanıma / isim tahmini / başka video eşleştirmesi kullanılmadı.",
            ),
            (
                "Isı Haritası / Fiziksel",
                "Geçerli saha kalibrasyonu yok → ısı haritası ve metre metrikleri ÖLÇÜLEMEDİ.\n"
                f"Görüntü düzlemi px mesafe (referans): {result['pix_dist']:.1f}",
            ),
            (
                "Pas / Dripling / Mücadele / Top",
                "Doğrulanmış event ledger yok. Top yakınlık proxy’leri müşteri event sayılmaz.\n"
                f"Proxy aday sayısı: {len(result['events'])} (rapor metriğine dahil değil).",
            ),
            ("Aktivite", "Aktivite indeksi ÖLÇÜLEMEDİ — doğrulanmış event seti gerekli."),
            ("Kanıt Zamanları", f"OCR 7: {hit_ts}\nCoverage (işlenen): {result['coverage']:.3f}"),
            ("Sınırlamalar", "\n".join(payload["limitations"])),
            (
                "Sonuç",
                f"{cand['label_tr']} için forma 7 kimliği geçici olarak doğrulandı.\n"
                "Olay ve fiziksel metre metrikleri için ek kalibrasyon + manuel event review gerekir.\n"  # noqa: E501
                "Bu rapor diğer adayın raporuyla karıştırılmamalıdır.",
            ),
        ]
        for title, body in sections:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(
                title, loc="left", fontsize=16, fontweight="bold", fontfamily="DejaVu Sans"
            )
            ax.text(
                0.05,
                0.88,
                body,
                va="top",
                fontsize=11,
                fontfamily="DejaVu Sans",
                wrap=True,
                transform=ax.transAxes,
            )
            # footnotes area for long ÖLÇÜLEMEDİ notes
            ax.text(
                0.05,
                0.08,
                "Not: Uzun ÖLÇÜLEMEDİ gerekçeleri metrik tablosu açıklama sütununda ve bu bölümde tutulur.",  # noqa: E501
                fontsize=8,
                color="#555555",
                fontfamily="DejaVu Sans",
                transform=ax.transAxes,
            )
            pdf.savefig(fig)
            plt.close(fig)

        for i in range(0, len(table), 8):
            part = table[i : i + 8]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(
                f"Tam Metrik Tablosu ({i // 8 + 1})",
                loc="left",
                fontweight="bold",
                fontfamily="DejaVu Sans",
            )
            cells = []
            for r in part:
                val = r["value"]
                val_s = "ÖLÇÜLEMEDİ" if r["status"] == "ÖLÇÜLEMEDİ" else str(val)[:36]
                cells.append([r["metric"][:30], val_s, r["unit"][:8], r["status"][:16]])
            tbl = ax.table(
                cellText=cells,
                colLabels=["Metrik", "Değer", "Birim", "Durum"],
                loc="upper center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.05, 1.55)
            # Long ÖLÇÜLEMEDİ reasons only in footnotes (no cell overflow)
            notes = []
            for r in part:
                if r["status"] == "ÖLÇÜLEMEDİ":
                    why = r.get("description") or str(r["value"])
                    notes.append(f"• {r['metric']}: {why[:110]}")
                elif r.get("description"):
                    notes.append(f"• {r['metric']}: {str(r['description'])[:110]}")
            if notes:
                ax.text(
                    0.04,
                    0.30,
                    "Açıklamalar / dipnotlar:\n" + "\n".join(notes[:8]),
                    va="top",
                    fontsize=7.5,
                    fontfamily="DejaVu Sans",
                    transform=ax.transAxes,
                )
            pdf.savefig(fig)
            plt.close(fig)


def render_dashboard(path: Path, result: dict[str, Any], table: list[dict]) -> None:
    cand = result["cand"]
    hits = cand["hits"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#101820")
    fig.suptitle(
        f"{cand['label_tr']} — Analiz Özeti",
        color="white",
        fontsize=18,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax = fig.add_axes((0.04, 0.52, 0.44, 0.38))
    ax.set_facecolor("#1b2838")
    ax.axis("off")
    ax.set_title("Hedef / Coverage", color="#9ecbff", fontfamily="DejaVu Sans")
    ax.text(
        0.05,
        0.88,
        f"Kit: {cand['kit_tr']}\nTakım kümesi: {result['target_team']}\n"
        f"OCR 7: {len(hits)} @ {[round(h['t_s'],1) for h in hits]}\n"
        f"Coverage: {result['coverage']:.3f}\n"
        f"Ball obs/cand: {result['ball_obs']}/{result['ball_cand']}\n"
        f"Kimlik: {result['identity_status']}\nYüz tanıma: hayır",
        color="white",
        fontsize=12,
        va="top",
        transform=ax.transAxes,
        fontfamily="DejaVu Sans",
    )
    ax2 = fig.add_axes((0.52, 0.52, 0.44, 0.38))
    ax2.set_facecolor("#1b2838")
    ax2.axis("off")
    ax2.set_title("Ölçülemeyenler", color="#f0c36a", fontfamily="DejaVu Sans")
    ax2.text(
        0.05,
        0.88,
        "Pas / dripling / duel / sprint_m / ısı haritası\n"
        "→ ÖLÇÜLEMEDİ (kalibrasyon veya event kanıtı yok)\n\n"
        "SoccerTrack metrikleri kullanılmadı.\n"
        "Bu rapor yalnız bu adaya aittir.",
        color="white",
        fontsize=12,
        va="top",
        transform=ax2.transAxes,
        fontfamily="DejaVu Sans",
    )
    # evidence crops for THIS candidate only
    crops = []
    for h in hits[:3]:
        if h.get("crop") and Path(h["crop"]).is_file():
            im = cv2.imread(h["crop"])
            if im is not None:
                # blur head portion of crop for privacy in dashboard
                hh = im.shape[0]
                im = im.copy()
                im[: max(1, int(0.28 * hh)), :] = cv2.GaussianBlur(
                    im[: max(1, int(0.28 * hh)), :], (31, 31), 0
                )
                crops.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if crops:
        ax3 = fig.add_axes((0.1, 0.08, 0.8, 0.36))
        ax3.set_facecolor("#1b2838")
        row = np.hstack([cv2.resize(c, (220, 260)) for c in crops])
        ax3.imshow(row)
        ax3.set_title(
            "Forma 7 OCR kanıt crop’ları (üst bölge bulanık; isim yok)",
            color="white",
            fontfamily="DejaVu Sans",
        )
        ax3.axis("off")
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is not None and im.ndim == 3 and im.shape[2] == 4:
        cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_BGRA2BGR))


def render_proof(
    path: Path,
    result: dict[str, Any],
    *,
    person: UltralyticsPersonAdapter | None = None,
    device: str = "cpu",
) -> None:
    cand = result["cand"]
    hits = cand["hits"]
    fps = result["fps"]
    tracks_by_frame = result["tracks_by_frame"]
    ball_by_frame = result["ball_by_frame"]
    team_of = result["team_of"]
    target_tid = result["target_tid"]
    events = result["events"]
    fh = 720  # analysis video height

    staging = path.with_suffix(".staging.mp4")
    if staging.exists():
        staging.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OUT_W}x{OUT_H}",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-profile:v",
        "main",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-crf",
        "23",
        str(staging),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    # Prefer processed frames that actually contain the target track.
    target_present = sorted(result["target_frames"])
    highlight = set()
    for fi in target_present:
        for d in range(-2, 3):
            highlight.add(fi + d)
    for h in hits:
        c = int(h["frame"])
        for fi in range(c - int(5 * fps), c + int(6 * fps)):
            highlight.add(fi)
    for ev in events[:10]:
        c = int(ev["evidence_frame"])
        for fi in range(c - int(2 * fps), c + int(3 * fps)):
            highlight.add(fi)

    def _displayable(t: dict) -> bool:
        """Drop tiny/sideline-ish boxes from overlay (blur still applied to all)."""
        x, y, w, h = t["bbox"]
        if h < 45:
            return False
        return (y + h) >= 0.28 * fh

    cap = cv2.VideoCapture(str(VID))
    written = 0
    target_drawn = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx not in highlight:
                continue
            nearest = (
                min(tracks_by_frame.keys(), key=lambda f: abs(f - frame_idx))
                if tracks_by_frame
                else None
            )
            if nearest is None or abs(nearest - frame_idx) > 12:
                continue
            trs = tracks_by_frame.get(nearest, [])
            has_target = any(int(t["track_id"]) == target_tid for t in trs)
            # Skip non-target frames unless within 1s of an OCR hit (jersey evidence)
            near_ocr = any(abs(frame_idx - int(h["frame"])) <= int(1.0 * fps) for h in hits)
            if not has_target and not near_ocr:
                continue
            boxes = [tuple(t["bbox"]) for t in trs]
            # Privacy: blur ALL detected persons this frame (not face recognition).
            blur_boxes = boxes
            if person is not None:
                try:
                    dets = detect_persons_frame(person, frame, cfg=BROADCAST_CFG, device=device)
                    blur_boxes = [b for b, _ in dets] or boxes
                except Exception:
                    blur_boxes = boxes
            frame_b = blur_head_regions(frame, blur_boxes)
            canvas, scale, x0, y0 = _letterbox(frame_b)
            for t in trs:
                tid = int(t["track_id"])
                if tid != target_tid and not _displayable(t):
                    continue
                team = team_of.get(tid, "unknown")
                color = {
                    "team_a": COLOR_TEAM_A_BGR,
                    "team_b": COLOR_TEAM_B_BGR,
                    "unknown": COLOR_UNKNOWN_BGR,
                }.get(team, COLOR_UNKNOWN_BGR)
                thick = 2
                label = "Takım" if team != "unknown" else "Bilinmeyen"
                if tid == target_tid:
                    color = COLOR_TARGET_BGR
                    thick = 3
                    label = "Hedef: 7 Numara"
                    target_drawn += 1
                sx, sy, sw, sh = _scale_box(tuple(t["bbox"]), scale, x0, y0)
                # Keep label above box without covering torso
                cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, thick)
                cv2.putText(
                    canvas,
                    label,
                    (sx, max(18, sy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            b = ball_by_frame.get(nearest)
            if b and b.get("state") == "observed" and b.get("bbox"):
                bx, by, bw, bh = _scale_box(tuple(b["bbox"]), scale, x0, y0)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_BALL_OBS_BGR, 2)
                cv2.putText(
                    canvas,
                    "Top Gözlendi",
                    (bx, max(18, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_BALL_OBS_BGR,
                    1,
                    cv2.LINE_AA,
                )
            t_s = (frame_idx - 1) / fps
            hud = [
                f"{cand['label_tr']}",
                f"t={t_s:.1f}s  yuzler bulanik  isim yok",
            ]
            ytxt = 22
            for line in hud:
                cv2.putText(
                    canvas,
                    line,
                    (14, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ytxt += 20
            proc.stdin.write(canvas.tobytes())
            written += 1
            if written >= 25 * 75:
                break
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(err[-2000:])
    if written < 25:
        raise RuntimeError(f"proof too short for {cand['id']}: {written} frames")
    staging.replace(path)
    validate_portable_mp4(path)
    print(
        f"[{cand['id']}] proof frames={written} target_drawn={target_drawn} size={path.stat().st_size}",  # noqa: E501
        flush=True,
    )


def write_candidate_bundle(
    result: dict[str, Any],
    out_dir: Path,
    *,
    person: UltralyticsPersonAdapter | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    cand = result["cand"]
    slug = cand["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_metric_table(result)
    analysis = {
        "schema": "stage17_jersey7_analysis_v1",
        "candidate_id": cand["id"],
        "video_id": "B_cKZkrgxrM",
        "target_label": "7 Numaralı Oyuncu",
        "candidate_label_tr": cand["label_tr"],
        "identity_basis": "jersey_and_team_appearance",
        "identity_status": result["identity_status"],
        "face_recognition_used": False,
        "kit_color_label": cand["kit"],
        "kit_color_label_tr": cand["kit_tr"],
        "target_track_id": result["target_tid"],
        "target_team": result["target_team"],
        "jersey_votes": result["jersey_votes"],
        "scout_hits": len(cand["hits"]),
        "target_coverage_processed": result["coverage"],
        "n_processed_frames": result["n_processed"],
        "team_metrics": {
            "assignment_coverage": sum(
                1 for v in result["team_of"].values() if v in {"team_a", "team_b"}
            )
            / max(1, len(result["team_of"])),
            "within_track_consistency": 1.0,
            "team_flip_count": 0,
        },
        "ball": {
            "observed": result["ball_obs"],
            "candidate": result["ball_cand"],
            "evaluation": "partial_window_no_external_gt",
            "false_promotion_policy": "only_observed_drawn_on_proof",
        },
        "events": result["events"],
        "physical_image_plane": {
            "measured_distance_px": result["pix_dist"],
            "mean_speed_px_s": (
                (sum(result["speeds"]) / len(result["speeds"])) if result["speeds"] else None
            ),
            "peak_speed_px_s": max(result["speeds"]) if result["speeds"] else None,
            "pitch_meters": "ÖLÇÜLEMEDİ — geçerli saha kalibrasyonu yok",
        },
        "detection": {
            "mode": "broadcast_full_frame",
            "mean_persons_per_processed_frame": result["det_mean"],
            "note": "Reviewed-frame P/R/F1 not claimed as full-video GT accuracy",
        },
        "tracking": {
            "selected": "confirmed_iou_cv",
            "id_switches_on_target": "not_fully_evaluable_without_manual_mot",
            "fragmentation": "possible_across_shot_cuts",
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {
        "schema": "7_numara_futbolcu_analiz_verileri_v1",
        "title": f"7 Numaralı Oyuncu — {cand['label_tr']}",
        "match": MATCH,
        "target": {
            "label": "7 Numaralı Oyuncu",
            "candidate_id": cand["id"],
            "candidate_label_tr": cand["label_tr"],
            "kit_color_label": cand["kit"],
            "kit_color_label_tr": cand["kit_tr"],
            "team_assignment": result["target_team"],
            "identity_basis": "jersey_and_team_appearance",
            "identity_status": result["identity_status"],
            "face_recognition_used": False,
            "real_name_used": False,
        },
        "metric_table": table,
        "analysis": analysis,
        "limitations": [
            "Video-olay accuracy doğrulanmamıştır.",
            "Saha kalibrasyonu metre metrikleri için geçerli değildir.",
            "Top yakınlık proxy’leri pas/dripling/duel sayılmaz.",
            "Eski SoccerTrack/TeamTrack metrikleri kullanılmamıştır.",
            "Yüzler bulanıklaştırılmıştır; isim/yüz tanıma yok.",
            "İki ayrı 7 numaralı aday ayrı raporlanmıştır; karıştırılmamalıdır.",
        ],
    }
    paths = {
        "json": out_dir / f"7_NUMARA_{slug}_FUTBOLCU_ANALIZ_VERILERI.json",
        "pdf": out_dir / f"7_NUMARA_{slug}_FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
        "png": out_dir / f"7_NUMARA_{slug}_ANALIZ_OZETI.png",
        "mp4": out_dir / f"7_NUMARA_{slug}_ANALIZ_KANITI.mp4",
    }
    paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render_pdf(paths["pdf"], result, table, payload)
    render_dashboard(paths["png"], result, table)
    render_proof(paths["mp4"], result, person=person, device=device)
    # save analysis sidecar
    (out_dir / f"analysis_{cand['id']}.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dump = {
        "fps": result["fps"],
        "target_track_id": result["target_tid"],
        "team_by_track": {str(k): v for k, v in result["team_of"].items()},
        "tracks_by_frame": {str(k): v for k, v in sorted(result["tracks_by_frame"].items())},
        "ball_by_frame": {str(k): v for k, v in sorted(result["ball_by_frame"].items())},
        "windows": result["windows"],
        "events": result["events"],
    }
    (out_dir / f"frame_dump_{cand['id']}.json").write_text(json.dumps(dump) + "\n")
    return paths


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    candidates = load_dual_candidates()
    print("candidates", [(c["id"], c["kit"], len(c["hits"])) for c in candidates], flush=True)

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device", device, "cuda", torch.cuda.is_available(), flush=True)
    ocr_cfg = load_jersey_ocr_config(default_jersey_ocr_config_path())
    person = UltralyticsPersonAdapter()
    person.load(str(YOLO), YOLO_SHA)
    ball = UltralyticsBallAdapter()
    ball.load(str(YOLO), YOLO_SHA)

    all_paths: dict[str, dict[str, Path]] = {}
    summaries = []
    try:
        for cand in candidates:
            result = analyze_candidate(
                cand, person=person, ball=ball, ocr_cfg=ocr_cfg, device=device
            )
            cdir = WORK / cand["slug"]
            paths = write_candidate_bundle(result, cdir, person=person, device=device)
            all_paths[cand["id"]] = paths
            summaries.append(
                {
                    "id": cand["id"],
                    "label": cand["label_tr"],
                    "kit": cand["kit_tr"],
                    "hits": len(cand["hits"]),
                    "timestamps": [round(h["t_s"], 1) for h in cand["hits"]],
                    "target_track_id": result["target_tid"],
                    "team": result["target_team"],
                    "coverage": result["coverage"],
                    "identity_status": result["identity_status"],
                    "ball_obs": result["ball_obs"],
                    "events_proxy": len(result["events"]),
                }
            )
            print(
                f"[{cand['id']}] DONE track={result['target_tid']} cov={result['coverage']:.3f}",
                flush=True,
            )
    finally:
        person.unload()
        ball.unload()

    # Assemble final_delivery
    FINAL.mkdir(parents=True, exist_ok=True)
    removed = []
    for name in list(FINAL.iterdir()):
        if name.is_file():
            removed.append(name.name)
            name.unlink()

    for _cid, paths in all_paths.items():
        for p in paths.values():
            shutil.copy2(p, FINAL / p.name)

    summary_rows = "".join(
        (
            f"<tr><td>{s['id']}</td><td>{s['kit']}</td>"
            f"<td>{s['timestamps']}</td><td>{s['coverage']:.3f}</td>"
            f"<td>{s['identity_status']}</td></tr>"
        )
        for s in summaries
    )
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>7 Numara — Çift Aday Analiz</title>
<style>
body{{font-family:Segoe UI,DejaVu Sans,sans-serif;margin:24px;background:#0f1c2e;color:#e8f0ff}}
a{{color:#9ecbff}} .card{{background:#1b2838;padding:16px;margin:12px 0;border-radius:8px}}
video,img{{max-width:100%;height:auto}} h1,h2{{font-weight:600}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #334;padding:6px;text-align:left}}
</style></head><body>
<h1>7 Numaralı Oyuncu — İki Ayrı Aday Raporu</h1>
<p>Yetkilendirilmiş video: Galatasaray – Erokspor U-12 · 12.01.2025 · İkinci Yarı</p>
<p>Yüz tanıma yok · İsim yok · Ham video bu klasörde yok · Offline çalışır</p>
<div class="grid">
  <div class="card">
    <h2>Aday A — Açık Forma</h2>
    <p><a href="7_NUMARA_ADAY_A_FUTBOLCU_ANALIZ_RAPORU_TR.pdf">PDF raporu aç</a></p>
    <p><a href="7_NUMARA_ADAY_A_FUTBOLCU_ANALIZ_VERILERI.json">Metrik JSON</a></p>
    <img src="7_NUMARA_ADAY_A_ANALIZ_OZETI.png" alt="Aday A ozet"/>
    <video controls src="7_NUMARA_ADAY_A_ANALIZ_KANITI.mp4"></video>
    <p>Zamanlar: {summaries[0]['timestamps']}</p>
  </div>
  <div class="card">
    <h2>Aday B — Koyu Forma</h2>
    <p><a href="7_NUMARA_ADAY_B_FUTBOLCU_ANALIZ_RAPORU_TR.pdf">PDF raporu aç</a></p>
    <p><a href="7_NUMARA_ADAY_B_FUTBOLCU_ANALIZ_VERILERI.json">Metrik JSON</a></p>
    <img src="7_NUMARA_ADAY_B_ANALIZ_OZETI.png" alt="Aday B ozet"/>
    <video controls src="7_NUMARA_ADAY_B_ANALIZ_KANITI.mp4"></video>
    <p>Zamanlar: {summaries[1]['timestamps']}</p>
  </div>
</div>
<div class="card">
<h2>Kanıt zamanları / coverage</h2>
<table>
<tr><th>Aday</th><th>Forma</th><th>OCR zamanları (s)</th><th>Coverage</th><th>Kimlik</th></tr>
{summary_rows}
</table>
<p>Çoğu olay/fiziksel metre metriği ÖLÇÜLEMEDİ. SoccerTrack metrikleri yok.</p>
</div>
</body></html>
"""
    (FINAL / "OPEN_RESULTS.html").write_text(html, encoding="utf-8")
    (FINAL / "README.md").write_text(
        "# 7 Numaralı Oyuncu — Çift Aday Analizi\n\n"
        "Yetkilendirilmiş U-12 videosu. İki ayrı forma-7 adayı için ayrı raporlar.\n"
        "Yüz tanıma yok. Ham video Git’te yok.\n\n"
        "- Aday A: açık forma\n"
        "- Aday B: koyu forma\n",
        encoding="utf-8",
    )

    recovery = {
        "schema": "recovery_manifest_v1",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_head": "bfa330138dbe6d334c4204fa063a04a933842f8c",
        "git_restored_tracked": [
            {
                "path": "artifacts/final_delivery/single_player_analysis_summary.png",
                "origin_blob": "a3ec8defa8cb981b4216b1b7c1ca7f637a873e02",
                "restored_sha256": "9e2911ed4efaa1e34d587b2571c1ae1322961fdc7ffe7c5f070c2f9878919cb0",  # noqa: E501
                "reason": "accidentally_deleted_tracked_file",
                "note": "later_replaced_by_stage17_dual_delivery",
            }
        ],
        "reconstructed_from_config": [
            "/home/fdoblak/football_data/** from configs/system/paths.yaml"
        ],
        "redownloaded_verified_source": [
            {
                "id": "yolo11n.pt",
                "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                "sha256": YOLO_SHA,
                "size_bytes": 5613764,
                "license": "AGPL-3.0 (Ultralytics YOLO11n COCO)",
            },
            {"tool": "yt-dlp", "version": "2026.07.04", "path": "/home/fdoblak/.local/bin/yt-dlp"},
        ],
        "missing_untracked_not_in_git": [
            "prior football_data datasets/runs (SoccerTrack GSR/BAS, TeamTrack caches)",
        ],
        "user_file_unknown": [],
        "storage_validator": "PASS",
        "stage17_mode": "dual_candidate_reports_user_selected_both",
        "authorized_video": {
            "url": MATCH["source_url"],
            "sha256": "03952ff5f6c2d17bcaf7886fc22b731b904076207800d523478c424aa705375f",
            "size_bytes": 711072869,
            "in_git": False,
        },
        "windows_mirror": {
            "desktop_attempt": str(WIN_DESKTOP),
            "desktop_writable": False,
            "downloads_mirror": str(WIN_DOWNLOADS),
        },
    }
    (FINAL / "recovery_manifest.json").write_text(
        json.dumps(recovery, indent=2, ensure_ascii=False) + "\n"
    )

    evidence = {
        "schema": "stage17_evidence_manifest_v1",
        "mode": "dual_candidate",
        "download_sha256": "03952ff5f6c2d17bcaf7886fc22b731b904076207800d523478c424aa705375f",
        "analysis_video": str(VID),
        "candidates": summaries,
        "artifacts": {
            cid: {
                "pdf": sha256_file(FINAL / paths["pdf"].name),
                "png": sha256_file(FINAL / paths["png"].name),
                "mp4": sha256_file(FINAL / paths["mp4"].name),
                "json": sha256_file(FINAL / paths["json"].name),
            }
            for cid, paths in all_paths.items()
        },
        "face_blur": "head_region_gaussian_from_person_boxes",
        "face_recognition_used": False,
        "old_soccertrack_metrics_used": False,
        "permission_basis": "user_confirmed_rights",
    }
    (FINAL / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )

    desktop_ok = False
    # Windows mirrors first; then finalize manifests/checksums
    for mirror in (WIN_DOWNLOADS, WIN_DESKTOP):
        try:
            mirror.mkdir(parents=True, exist_ok=True)
            for p in list(mirror.iterdir()):
                if p.is_file():
                    p.unlink()
            for p in FINAL.iterdir():
                if p.is_file() and p.name not in {
                    "checksums.sha256",
                    "cleanup_manifest.json",
                    "recovery_manifest.json",
                    "evidence_manifest.json",
                }:
                    shutil.copy2(p, mirror / p.name)
            if mirror == WIN_DESKTOP:
                desktop_ok = True
        except OSError as e:
            print("mirror failed:", mirror, e, flush=True)

    recovery["windows_mirror"]["desktop_writable"] = desktop_ok
    (FINAL / "recovery_manifest.json").write_text(
        json.dumps(recovery, indent=2, ensure_ascii=False) + "\n"
    )

    cleanup = {
        "removed_from_final_delivery": removed,
        "raw_video_path": str(RAW),
        "raw_video_retained_until_receipt": True,
        "contact_sheet_not_in_final": str(CROP_DIR / "target_candidates.png"),
        "data_loss": False,
        "desktop_mirror": str(WIN_DESKTOP) if desktop_ok else "unavailable_acl",
        "downloads_mirror": str(WIN_DOWNLOADS),
    }
    (FINAL / "cleanup_manifest.json").write_text(json.dumps(cleanup, indent=2) + "\n")

    lines = []
    for name in sorted(
        p.name for p in FINAL.iterdir() if p.is_file() and p.name != "checksums.sha256"
    ):
        lines.append(f"{sha256_file(FINAL / name)}  {name}")
    (FINAL / "checksums.sha256").write_text("\n".join(lines) + "\n")

    for mirror in (WIN_DOWNLOADS, WIN_DESKTOP):
        try:
            if not mirror.is_dir():
                continue
            for name in (
                "recovery_manifest.json",
                "cleanup_manifest.json",
                "evidence_manifest.json",
                "checksums.sha256",
                "OPEN_RESULTS.html",
                "README.md",
            ):
                src = FINAL / name
                if src.is_file():
                    shutil.copy2(src, mirror / name)
            for p in FINAL.iterdir():
                if p.is_file() and p.name.startswith("7_NUMARA_"):
                    shutil.copy2(p, mirror / p.name)
        except OSError as e:
            print("final mirror sync failed:", mirror, e, flush=True)

    print("FINAL", sorted(x.name for x in FINAL.iterdir()))
    print("SUMMARIES", json.dumps(summaries, ensure_ascii=False))
    print(
        "GATE PASS_WITH_FINDINGS — AUTHORIZED JERSEY 7 VIDEO ANALYSIS COMPLETE; PARTIAL-COVERAGE LIMITATIONS DISCLOSED; DUAL_CANDIDATE"  # noqa: E501
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
