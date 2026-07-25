"""Turkish single-player PDF/JSON/dashboard from SoccerTrack reference + TeamTrack proof."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.namespaces import AUTHORITATIVE_SOCCERTRACK_TARGET
from football_analytics.acceptance.soccertrack_v2.reference_analysis import (
    analyze_soccertrack_v2_reference,
)

# Prefer fonts with Latin Extended (Turkish)
for _fname in ("DejaVu Sans", "FreeSans", "Noto Sans"):
    try:
        font_manager.findfont(_fname, fallback_to_default=False)
        plt.rcParams["font.family"] = _fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


EVIDENCE = {
    "REAL_VIDEO": "GERÇEK VİDEODA DOĞRULANDI",
    "ANNOTATION": "ANNOTASYONDAN HESAPLANDI",
    "SYSTEM_TEST": "SİSTEM TESTİNDE DOĞRULANDI",
    "NOT_MEASURED": "ÖLÇÜLEMEDİ",
}


def _row(
    name: str,
    value: Any,
    unit: str,
    status: str,
    evidence: str,
    source: str,
    note: str,
) -> dict[str, Any]:
    return {
        "metric": name,
        "value": value,
        "unit": unit,
        "status": status,
        "evidence_level": evidence,
        "source": source,
        "description": note,
    }


def build_turkish_metric_table(
    *,
    ref: dict[str, Any],
    perception: dict[str, Any],
) -> list[dict[str, Any]]:
    m = ref.get("annotation_derived_metrics") or ref.get("metrics") or {}
    # reference_analysis returns flat structure under keys — adapt to summary JSON shape too
    if not m and "measured_distance_m" in ref:
        m = ref

    def g(key: str, default: Any = None) -> Any:
        node = m.get(key)
        if isinstance(node, dict) and "value" in node:
            return node.get("value")
        return node if node is not None else default

    def st(key: str) -> str:
        node = m.get(key)
        if isinstance(node, dict):
            return str(node.get("status") or "REFERENCE_ANNOTATION_DERIVED")
        return "REFERENCE_ANNOTATION_DERIVED"

    def not_meas(reason: str) -> tuple[str, str, str]:
        return (f"ÖLÇÜLEMEDİ — gerekli kanıt: {reason}", "ÖLÇÜLEMEDİ", EVIDENCE["NOT_MEASURED"])

    rows: list[dict[str, Any]] = []

    # Heatmap
    hm = g("heatmap")
    if hm is not None:
        rows.append(
            _row(
                "Isı haritası",
                hm if not isinstance(hm, dict) else f"n={hm.get('n_points', '?')}",
                "özet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "SoccerTrack GSR",
                "Zaman ağırlıklı saha kullanım özeti (annotasyon)",
            )
        )
    else:
        v, s, e = not_meas("GSR yörünge noktaları")
        rows.append(_row("Isı haritası", v, "-", s, e, "—", "Yörünge yok"))

    # Duels
    for key, title, need in [
        ("duels", "İkili mücadele sayısı", "ikili mücadele olay etiketleri"),
        ("duels_won", "Kazanılan ikili mücadele", "kazanılan mücadele etiketleri"),
        ("duel_win_rate", "İkili mücadele kazanma oranı", "kazanan/kaybeden duel etiketleri"),
    ]:
        val = g(key)
        if val is None or st(key) in {"NOT_EVALUABLE", "ÖLÇÜLEMEDİ"}:
            v, s, e = not_meas(need)
            rows.append(_row(title, v, "-" if "oran" not in title else "%", s, e, "BAS", need))
        else:
            rows.append(
                _row(
                    title,
                    val,
                    "%" if "oran" in title else "adet",
                    "ANNOTASYONDAN HESAPLANDI",
                    EVIDENCE["ANNOTATION"],
                    "BAS",
                    title,
                )
            )

    # Passes
    pass_n = g("bas_pass_attempts")
    if pass_n is None:
        pass_n = g("pass_attempts")
    if pass_n is not None and st("bas_pass_attempts") not in {"NOT_EVALUABLE"}:
        rows.append(
            _row(
                "Pas girişimi",
                pass_n,
                "adet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS Pass",
                "Hedefe özel Pass aksiyon sayısı (yeniden doğrulandı)",
            )
        )
    else:
        v, s, e = not_meas("target BAS Pass")
        rows.append(_row("Pas girişimi", v, "adet", s, e, "BAS", ""))

    # Completed passes / accuracy — not in BAS outcome labels
    for title, need in [
        ("Tamamlanan pas", "pas başarı/isabet sonucu etiketleri"),
        ("İsabetli pas oranı", "pas isabet oranı etiketleri"),
    ]:
        v, s, e = not_meas(need)
        rows.append(_row(title, v, "%" if "oran" in title else "adet", s, e, "BAS", need))

    # Dribbles
    drive = g("bas_drive_actions")
    rows.append(
        _row(
            "Başarılı dripling",
            (
                f"ÖLÇÜLEMEDİ — gerekli kanıt: Drive≠başarılı dripling; "
                f"Drive adayı={drive if drive is not None else '?'}"
            ),
            "adet",
            "ÖLÇÜLEMEDİ",
            EVIDENCE["NOT_MEASURED"],
            "BAS Drive",
            "Drive aksiyonları başarılı dripling sayılmaz",
        )
    )
    v, s, e = not_meas("başarısız dripling sonucu")
    rows.append(_row("Başarısız dripling", v, "adet", s, e, "BAS", ""))
    v, s, e = not_meas("take-on başarı/başarısızlık")
    rows.append(_row("Adam eksiltme oranı", v, "%", s, e, "BAS", ""))

    # Defense
    tackle = g("bas_successful_tackles")
    if tackle is not None and st("bas_successful_tackles") not in {"NOT_EVALUABLE"}:
        rows.append(
            _row(
                "Top çalma",
                tackle,
                "adet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS Player Successful Tackle",
                "Başarılı tackle sayısı",
            )
        )
    else:
        v, s, e = not_meas("Player Successful Tackle")
        rows.append(_row("Top çalma", v, "adet", s, e, "BAS", ""))

    v, s, e = not_meas("turnover/top kaybı etiketi")
    rows.append(_row("Top kaybı", v, "adet", s, e, "BAS", ""))

    header = g("bas_header_actions")
    if header is not None and st("bas_header_actions") not in {"NOT_EVALUABLE"}:
        rows.append(
            _row(
                "Hava topu mücadelesi",
                header,
                "adet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS Header",
                "Header ≠ otomatik kazanılmış hava topu",
            )
        )
    else:
        v, s, e = not_meas("Header etiketi")
        rows.append(_row("Hava topu mücadelesi", v, "adet", s, e, "BAS", ""))

    clear = g("clearances")
    if clear is None or st("clearances") in {"NOT_EVALUABLE"}:
        v, s, e = not_meas("clearance sınıfı BAS'ta yok")
        rows.append(_row("Uzaklaştırma", v, "adet", s, e, "BAS", ""))
    else:
        rows.append(
            _row(
                "Uzaklaştırma",
                clear,
                "adet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS",
                "",
            )
        )

    # Zone transitions / long pass
    for title, need in [
        ("1→2 bölge geçiş pasları", "bölge geçiş pas geometrisi + olay bağlantısı"),
        ("2→3 bölge geçiş pasları", "bölge geçiş pas geometrisi + olay bağlantısı"),
        ("Uzun pas oranı", "uzun pas tanımı + isabet"),
    ]:
        # High pass as candidate note only for long pass ratio
        if title.startswith("Uzun"):
            hp = g("bas_high_pass_attempts")
            rows.append(
                _row(
                    title,
                    (
                        f"ÖLÇÜLEMEDİ — gerekli kanıt: {need}; "
                        f"High Pass adayı={hp if hp is not None else '?'}"
                    ),
                    "%",
                    "ÖLÇÜLEMEDİ",
                    EVIDENCE["NOT_MEASURED"],
                    "BAS High Pass",
                    "High Pass uzun pas adayıdır, oran değildir",
                )
            )
        else:
            v, s, e = not_meas(need)
            rows.append(_row(title, v, "adet", s, e, "—", need))

    # Physical
    dist = g("measured_distance_m")
    rows.append(
        _row(
            "Koşu mesafesi",
            dist if dist is not None else not_meas("GSR yörünge")[0],
            "m (ölçülebilen interval)",
            "ANNOTASYONDAN HESAPLANDI" if dist is not None else "ÖLÇÜLEMEDİ",
            EVIDENCE["ANNOTATION"] if dist is not None else EVIDENCE["NOT_MEASURED"],
            "SoccerTrack GSR",
            "Tam maç tahmini değil; ölçülebilen interval mesafesi",
        )
    )
    for key, title, unit, need in [
        ("sprint_count", "Sprint sayısı", "adet", "sprint eşiği + yörünge"),
        ("sprint_distance_m", "Sprint mesafesi", "m", "sprint segmentleri"),
        ("sprint_duration_s", "Sprint süresi", "s", "sprint segmentleri"),
        ("mean_speed_m_s", "Ortalama hız", "m/s", "yörünge hızı"),
        ("peak_speed_m_s", "Maksimum hız", "m/s", "güvenilir tepe hız"),
    ]:
        val = g(key)
        if val is None or st(key) in {"NOT_EVALUABLE"}:
            # try alternate keys from reference
            alt = {
                "peak_speed_m_s": "max_speed_m_s",
                "sprint_count": "sprints",
            }.get(key)
            if alt:
                val = g(alt)
        if val is None or (isinstance(val, str) and "not" in val.lower()):
            v, s, e = not_meas(need)
            rows.append(_row(title, v, unit, s, e, "GSR", need))
        else:
            rows.append(
                _row(
                    title,
                    val,
                    unit,
                    "ANNOTASYONDAN HESAPLANDI",
                    EVIDENCE["ANNOTATION"],
                    "GSR",
                    title,
                )
            )

    box_t = g("box_touches")
    if box_t is None or st("box_touches") in {"NOT_EVALUABLE"}:
        v, s, e = not_meas("ceza sahası dokunuş etiketi (presence proxy ≠ touch)")
        rows.append(_row("Ceza sahası topla buluşma", v, "adet", s, e, "BAS", ""))
    else:
        rows.append(
            _row(
                "Ceza sahası topla buluşma",
                box_t,
                "adet",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS",
                "",
            )
        )

    act = g("activity_index")
    if act is not None:
        rows.append(
            _row(
                "Aktivite indeksi",
                act,
                "olay/yarı",
                "ANNOTASYONDAN HESAPLANDI",
                EVIDENCE["ANNOTATION"],
                "BAS",
                "Hedef BAS olayları / mevcut yarı",
            )
        )
    else:
        v, s, e = not_meas("BAS olay kapsamı")
        rows.append(_row("Aktivite indeksi", v, "-", s, e, "BAS", ""))

    # Coverage from real-video target
    cov = (
        perception.get("tracking", {}).get("target_tracking_eval", {}).get("target_coverage_ratio")
    )
    rows.append(
        _row(
            "Coverage",
            cov if cov is not None else not_meas("TeamTrack hedef kapsama")[0],
            "oran",
            "GERÇEK VİDEODA DOĞRULANDI" if cov is not None else "ÖLÇÜLEMEDİ",
            EVIDENCE["REAL_VIDEO"] if cov is not None else EVIDENCE["NOT_MEASURED"],
            "TeamTrack pilot Track 7",
            "Sistem kanıtı — SoccerTrack oyuncusu değildir",
        )
    )
    return rows


def build_report_payload(
    *,
    trajectory_path: Path,
    bas_path: Path,
    perception: dict[str, Any],
    existing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref = analyze_soccertrack_v2_reference(
        trajectory_path=trajectory_path,
        bas_path=bas_path,
        target=dict(AUTHORITATIVE_SOCCERTRACK_TARGET),
    )
    # Prefer richer annotation block from previous delivery if shapes match
    ann = (existing_summary or {}).get("annotation_derived_metrics")
    if isinstance(ann, dict) and "measured_distance_m" in ann:
        metrics_src = {"annotation_derived_metrics": ann}
    else:
        metrics_src = {"annotation_derived_metrics": ref.get("metrics", {})}

    table = build_turkish_metric_table(ref=metrics_src, perception=perception)
    tgt = AUTHORITATIVE_SOCCERTRACK_TARGET
    payload = {
        "schema": "futbolcu_analiz_raporu_tr_v1",
        "title": "BİREYSEL FUTBOLCU PERFORMANS ANALİZİ",
        "dataset": "SoccerTrack v2",
        "match_id": "128057",
        "target": {
            "team_side": tgt["team_side"],
            "jersey_number": tgt["jersey_number"],
            "player_id": tgt["player_id"],
            "label_tr": "Sol Takım / Forma 24 / Player ID 506469",
        },
        "preview_level": "Teknik Önizleme",
        "evidence_levels": list(EVIDENCE.values()),
        "namespace_isolation": {
            "soccertrack_report_player": "506469",
            "teamtrack_proof_track": 7,
            "not_the_same_person": True,
        },
        "metric_table": table,
        "perception_proof": {
            "detection": perception.get("detection", {}).get("full_selected"),
            "tracking": perception.get("tracking"),
            "team": perception.get("team", {}).get("metrics"),
            "ball": perception.get("ball"),
            "device": perception.get("device"),
            "elapsed_s": perception.get("elapsed_s"),
        },
        "reference_analysis": ref,
        "limitations_tr": [
            "Opta verisi kullanılmamıştır.",
            "Video-olay doğruluğu doğrulanmamıştır.",
            "TeamTrack gerçek-video kanıtı SoccerTrack oyuncusunun maç videosu değildir.",
            "Ölçülemeyen metrikler sıfır olarak gösterilmemiştir.",
        ],
    }
    return payload


def render_turkish_pdf(
    *,
    payload: dict[str, Any],
    out_pdf: Path,
    frame_rgbs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    table = payload["metric_table"]
    perc = payload["perception_proof"]
    tgt = payload["target"]

    with PdfPages(str(out_pdf)) as pdf:
        # 1 Cover
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#0f1c2e")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_facecolor("#0f1c2e")
        ax.text(
            0.5,
            0.72,
            "BİREYSEL FUTBOLCU PERFORMANS ANALİZİ",
            ha="center",
            va="center",
            color="white",
            fontsize=22,
            fontweight="bold",
            wrap=True,
        )
        ax.text(
            0.5,
            0.55,
            "SoccerTrack v2 — Maç 128057",
            ha="center",
            color="#9ecbff",
            fontsize=16,
        )
        ax.text(
            0.5,
            0.45,
            f"Hedef: {tgt['label_tr']}",
            ha="center",
            color="#e8f0ff",
            fontsize=14,
        )
        ax.text(0.5, 0.32, "Teknik Önizleme", ha="center", color="#f0c36a", fontsize=13)
        ax.text(
            0.5,
            0.12,
            "Opta değildir • Video-olay doğruluğu doğrulanmamıştır",
            ha="center",
            color="#aabbcc",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 2 Executive summary
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Yönetici Özeti", fontsize=16, loc="left", fontweight="bold")
        # pull key metrics
        by_name = {r["metric"]: r for r in table}
        lines = [
            f"Ölçülen mesafe: {by_name.get('Koşu mesafesi', {}).get('value')}",
            f"Ortalama hız: {by_name.get('Ortalama hız', {}).get('value')}",
            f"Sprint özeti: {by_name.get('Sprint sayısı', {}).get('value')}",
            f"Aktivite: {by_name.get('Aktivite indeksi', {}).get('value')}",
            f"Isı haritası: {by_name.get('Isı haritası', {}).get('value')}",
            f"Pas girişimleri: {by_name.get('Pas girişimi', {}).get('value')}",
            "Veri kapsamı: SoccerTrack GSR/BAS annotasyon + ayrı TeamTrack sistem kanıtı",
            "En önemli sınırlama: video-olay accuracy doğrulanmadı; Opta yok",
        ]
        ax.text(0.05, 0.9, "\n\n".join(lines), va="top", fontsize=11, family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)

        # 3 Evidence levels
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Veri ve Kanıt Seviyesi", fontsize=16, loc="left", fontweight="bold")
        txt = (
            "Dört seviye:\n\n" + "\n".join(f"• {v}" for v in EVIDENCE.values()) + "\n\n"
            "TeamTrack gerçek-video kanıtı ile SoccerTrack raporu FARKLI kaynaklardır.\n"
            "Track 7 ≠ Player 506469. Forma numarası TeamTrack için uydurulmaz."
        )
        ax.text(0.05, 0.9, txt, va="top", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # 4 Physical
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Fiziksel Performans", fontsize=16, loc="left", fontweight="bold")
        phys = [
            "Koşu mesafesi",
            "Ortalama hız",
            "Maksimum hız",
            "Sprint sayısı",
            "Sprint mesafesi",
            "Sprint süresi",
            "Aktivite indeksi",
        ]
        body = "\n".join(
            f"{n}: {by_name.get(n, {}).get('value')} "
            f"[{by_name.get(n, {}).get('evidence_level')}]"
            for n in phys
        )
        ax.text(0.05, 0.9, body, va="top", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        # 5 Heatmap / pitch
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Isı Haritası ve Saha Kullanımı", fontsize=16, loc="left", fontweight="bold")
        ax.text(
            0.05,
            0.85,
            "Zaman ağırlıklı ısı haritası annotasyon yörüngesinden türetilmiştir.\n"
            "Hücum yönü bilinmediği için hücum/savunma yönü uydurulmamıştır.\n"
            f"Özet: {by_name.get('Isı haritası', {}).get('value')}\n"
            "Goal A / orta / Goal B ve koridor kırılımları: kaynakta yön etiketi yoksa ÖLÇÜLEMEDİ.",
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 6 Pass
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Pas ve İlerleme", fontsize=16, loc="left", fontweight="bold")
        names = [
            "Pas girişimi",
            "Tamamlanan pas",
            "İsabetli pas oranı",
            "1→2 bölge geçiş pasları",
            "2→3 bölge geçiş pasları",
            "Uzun pas oranı",
        ]
        ax.text(
            0.05,
            0.9,
            "\n".join(f"{n}: {by_name.get(n, {}).get('value')}" for n in names),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 7 Dribble
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Dripling ve Adam Eksiltme", fontsize=16, loc="left", fontweight="bold")
        names = ["Başarılı dripling", "Başarısız dripling", "Adam eksiltme oranı"]
        ax.text(
            0.05,
            0.9,
            "Drive ≠ başarılı dripling.\n\n"
            + "\n".join(f"{n}: {by_name.get(n, {}).get('value')}" for n in names),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 8 Defense
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Mücadele ve Savunma", fontsize=16, loc="left", fontweight="bold")
        names = [
            "İkili mücadele sayısı",
            "Kazanılan ikili mücadele",
            "İkili mücadele kazanma oranı",
            "Top çalma",
            "Top kaybı",
            "Hava topu mücadelesi",
            "Uzaklaştırma",
        ]
        ax.text(
            0.05,
            0.9,
            "\n".join(f"{n}: {by_name.get(n, {}).get('value')}" for n in names),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 9 Ball / box
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Top ve Ceza Sahası Aksiyonları", fontsize=16, loc="left", fontweight="bold")
        box_touch_val = by_name.get("Ceza sahası topla buluşma", {}).get("value")
        ax.text(
            0.05,
            0.85,
            (
                f"Ceza sahası topla buluşma: {box_touch_val}\n\n"
                "Presence proxy ile gerçek touch aynı değildir.\n"
                "Şut / cross / possession: kaynakta hedef-özel doğrulanabilir "
                "etiket yoksa ÖLÇÜLEMEDİ."
            ),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 10 Real-video system proof
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(
            "SİSTEMİN GERÇEK VİDEODA ÇALIŞMA KANITI\nBU OYUNCUNUN MAÇ VİDEOSU DEĞİLDİR",
            fontsize=13,
            fontweight="bold",
            color="#8b0000",
        )
        det = perc.get("detection") or {}
        trk = (perc.get("tracking") or {}).get("confirmed_iou") or {}
        team = perc.get("team") or {}
        ball = perc.get("ball") or {}
        info = (
            f"İnsan P/R/F1: {det.get('precision')} / {det.get('recall')} / {det.get('f1')}\n"
            f"ID switches: {trk.get('id_switches')}  frag: {trk.get('fragmentation_events')}\n"
            f"Target coverage: {trk.get('target_coverage')}\n"
            f"Takım tutarlılığı: {team.get('within_track_consistency')}  "
            f"flip: {team.get('team_flip_count')}\n"
            f"Top: {ball.get('summary')}  eval={ball.get('evaluation')}\n"
            f"Cihaz: {perc.get('device')}  süre_s: {perc.get('elapsed_s')}"
        )
        fig.text(0.05, 0.78, info, fontsize=10, family="DejaVu Sans", va="top")
        if frame_rgbs:
            axes = [
                fig.add_axes((0.05, 0.08, 0.42, 0.28)),
                fig.add_axes((0.53, 0.08, 0.42, 0.28)),
                fig.add_axes((0.05, 0.40, 0.42, 0.28)),
                fig.add_axes((0.53, 0.40, 0.42, 0.28)),
            ]
            for ax, key in zip(axes, ["start", "mid", "crowd", "end"], strict=False):
                item = frame_rgbs.get(key)
                ax.set_xticks([])
                ax.set_yticks([])
                if item and "rgb" in item:
                    ax.imshow(item["rgb"])
                    ax.set_title(f"{key} t={item.get('t_s', 0):.1f}s", fontsize=8)
                else:
                    ax.set_facecolor("#ddd")
        pdf.savefig(fig)
        plt.close(fig)

        # 11 Full metric table (multi-page if needed)
        chunk = 12
        for i in range(0, len(table), chunk):
            part = table[i : i + chunk]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.set_axis_off()
            ax.set_title(
                f"Tam Metrik Tablosu ({i // chunk + 1})",
                fontsize=14,
                loc="left",
                fontweight="bold",
            )
            cell = []
            for r in part:
                val = r["value"]
                if isinstance(val, float):
                    val = f"{val:.4g}"
                elif isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False)[:80]
                cell.append(
                    [
                        r["metric"],
                        str(val)[:70],
                        r["unit"],
                        r["status"][:28],
                        r["evidence_level"][:28],
                    ]
                )
            tbl = ax.table(
                cellText=cell,
                colLabels=["Metrik", "Değer", "Birim", "Durum", "Kanıt"],
                loc="upper center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.4)
            pdf.savefig(fig)
            plt.close(fig)

        # 12 Conclusions
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_axis_off()
        ax.set_title("Sonuç ve Sınırlamalar", fontsize=16, loc="left", fontweight="bold")
        ax.text(
            0.05,
            0.9,
            "\n".join(
                [
                    "Güvenilir: annotasyondan mesafe/hız/pas sayısı/tackle/header adayları;",
                    "gerçek videoda detection/tracking coverage kanıtı.",
                    "Reference-derived: SoccerTrack GSR/BAS metrikleri.",
                    "Eksik: pas isabeti, dripling sonucu, duel oranı, clearance, bölge geçişleri,",
                    "video-olay accuracy, Opta.",
                    "Gerçek müşteri maçında: yayın videosu + kalibrasyon + "
                    "olay etiketleme gerekir.",
                    "Opta yoktur. Video-event accuracy doğrulanmamıştır.",
                ]
            ),
            va="top",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)

    return {
        "path": str(out_pdf),
        "sha256": sha256_file(out_pdf),
        "size_bytes": out_pdf.stat().st_size,
        "page_count": _pdf_pages(out_pdf),
    }


def _pdf_pages(path: Path) -> int:
    # PdfPages writes; count via pypdf or pdfinfo
    try:
        import subprocess

        out = subprocess.check_output(["pdfinfo", str(path)], text=True)
        for line in out.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return -1


def render_turkish_dashboard_png(
    *,
    payload: dict[str, Any],
    out_png: Path,
    frame_rgbs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#101820")
    fig.suptitle(
        "Futbolcu Analiz Özeti — SoccerTrack v2 / Forma 24 / ID 506469",
        color="white",
        fontsize=18,
        fontweight="bold",
    )
    by_name = {r["metric"]: r for r in payload["metric_table"]}
    ax1 = fig.add_axes((0.03, 0.55, 0.30, 0.35))
    ax1.set_facecolor("#1b2838")
    ax1.set_axis_off()
    ax1.set_title("Futbolcu / Fiziksel", color="#9ecbff", fontsize=12)
    ax1.text(
        0.05,
        0.9,
        "\n".join(
            [
                f"Mesafe: {by_name.get('Koşu mesafesi', {}).get('value')}",
                f"Ort. hız: {by_name.get('Ortalama hız', {}).get('value')}",
                f"Sprint: {by_name.get('Sprint sayısı', {}).get('value')}",
                f"Aktivite: {by_name.get('Aktivite indeksi', {}).get('value')}",
                f"Pas: {by_name.get('Pas girişimi', {}).get('value')}",
                f"Tackle: {by_name.get('Top çalma', {}).get('value')}",
            ]
        ),
        color="white",
        fontsize=11,
        va="top",
        transform=ax1.transAxes,
    )

    ax2 = fig.add_axes((0.36, 0.55, 0.30, 0.35))
    ax2.set_facecolor("#1b2838")
    ax2.set_axis_off()
    ax2.set_title("Aksiyon / Ölçülemeyenler", color="#f0c36a", fontsize=12)
    not_m = [r["metric"] for r in payload["metric_table"] if r["status"] == "ÖLÇÜLEMEDİ"]
    ax2.text(
        0.05,
        0.9,
        "ÖLÇÜLEMEDİ (örnek):\n" + "\n".join(f"• {x}" for x in not_m[:10]),
        color="white",
        fontsize=10,
        va="top",
        transform=ax2.transAxes,
    )

    ax3 = fig.add_axes((0.69, 0.55, 0.28, 0.35))
    ax3.set_facecolor("#1b2838")
    ax3.set_axis_off()
    ax3.set_title("Gerçek-video sistem kanıtı", color="#7dcea0", fontsize=12)
    perc = payload["perception_proof"]
    det = perc.get("detection") or {}
    ax3.text(
        0.05,
        0.9,
        "BU OYUNCUNUN MAÇI DEĞİL\n"
        f"F1={det.get('f1')}\n"
        f"Coverage={by_name.get('Coverage', {}).get('value')}\n"
        f"Top eval={ (perc.get('ball') or {}).get('evaluation') }\n"
        "Opta yok; annotasyon ≠ video prediction",
        color="white",
        fontsize=10,
        va="top",
        transform=ax3.transAxes,
    )

    # four frames
    if frame_rgbs:
        for i, key in enumerate(["start", "mid", "crowd", "end"]):
            ax = fig.add_axes((0.03 + i * 0.24, 0.08, 0.22, 0.38))
            ax.set_xticks([])
            ax.set_yticks([])
            item = frame_rgbs.get(key)
            if item and "rgb" in item:
                ax.imshow(item["rgb"])
                ax.set_title(key, color="white", fontsize=9)
            else:
                ax.set_facecolor("#333")

    fig.savefig(out_png, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    # ensure RGB
    import cv2

    img = cv2.imread(str(out_png), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(out_png), img)
    elif img is not None:
        cv2.imwrite(str(out_png), img)
    return {
        "path": str(out_png),
        "sha256": sha256_file(out_png),
        "size_bytes": out_png.stat().st_size,
    }


__all__ = [
    "build_report_payload",
    "build_turkish_metric_table",
    "render_turkish_dashboard_png",
    "render_turkish_pdf",
]
