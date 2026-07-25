"""Assemble final_delivery bundle + Windows mirror for Turkish release."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.portable_final_media import validate_portable_mp4

FINAL_NAMES = [
    "OPEN_RESULTS.html",
    "README.md",
    "FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
    "FUTBOLCU_ANALIZ_RAPORU_TR.json",
    "single_player_analysis_summary.png",
    "real_video_analysis_proof.mp4",
    "evidence_manifest.json",
    "checksums.sha256",
    "cleanup_manifest.json",
]

OLD_REMOVE = [
    "real_video_tracking_proof.mp4",
    "single_player_analysis_summary.json",
]


def write_open_results_html(out: Path) -> None:
    html = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<title>Futbol Analitiği — Nihai Sonuçlar</title>
<style>
body{font-family:Segoe UI,DejaVu Sans,sans-serif;margin:24px;background:#0f1c2e;color:#e8f0ff}
a{color:#9ecbff} .card{background:#1b2838;padding:16px;margin:12px 0;border-radius:8px}
h1{color:#f0c36a} video,img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%} td,th{border:1px solid #345;padding:6px;text-align:left}
</style>
</head>
<body>
<h1>Nihai Türkçe Futbolcu Analiz Teslimi</h1>
<div class="card">
<h2>1. Nihai Türkçe PDF raporu</h2>
<p><a href="FUTBOLCU_ANALIZ_RAPORU_TR.pdf">FUTBOLCU_ANALIZ_RAPORU_TR.pdf</a></p>
</div>
<div class="card">
<h2>2. Türkçe dashboard PNG</h2>
<p><a href="single_player_analysis_summary.png">single_player_analysis_summary.png</a></p>
<p><img src="single_player_analysis_summary.png" alt="dashboard"/></p>
</div>
<div class="card">
<h2>3. Yeni gerçek-video analysis proof MP4</h2>
<p><a href="real_video_analysis_proof.mp4">real_video_analysis_proof.mp4</a></p>
<video controls src="real_video_analysis_proof.mp4"></video>
</div>
<div class="card">
<h2>4. Temel metrik tablosu</h2>
<p>Tam tablo PDF ve <a href="FUTBOLCU_ANALIZ_RAPORU_TR.json">JSON</a> içindedir.</p>
</div>
<div class="card">
<h2>5. Kanıt seviyesi açıklaması</h2>
<ul>
<li>GERÇEK VİDEODA DOĞRULANDI</li>
<li>ANNOTASYONDAN HESAPLANDI</li>
<li>SİSTEM TESTİNDE DOĞRULANDI</li>
<li>ÖLÇÜLEMEDİ</li>
</ul>
<p>TeamTrack sistem kanıtı, SoccerTrack oyuncu raporundan ayrıdır. Opta yoktur.
Video-olay doğruluğu doğrulanmamıştır.</p>
</div>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")


def write_readme(out: Path) -> None:
    out.write_text(
        """# Football Analytics — Final Delivery (TR)

- `FUTBOLCU_ANALIZ_RAPORU_TR.pdf` — Nihai Türkçe futbolcu raporu
- `FUTBOLCU_ANALIZ_RAPORU_TR.json` — Canonical metrik tablosu
- `single_player_analysis_summary.png` — Türkçe dashboard
- `real_video_analysis_proof.mp4` — TeamTrack gerçek-video sistem kanıtı
- `OPEN_RESULTS.html` — Çevrimdışı görüntüleyici

TeamTrack Track 7 ≠ SoccerTrack Player 506469.
Opta yoktur. Video-event accuracy doğrulanmamıştır.
""",
        encoding="utf-8",
    )


def write_checksums(folder: Path) -> Path:
    lines = []
    for name in FINAL_NAMES:
        if name in {"checksums.sha256", "cleanup_manifest.json", "evidence_manifest.json"}:
            continue
        p = folder / name
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {name}")
    out = folder / "checksums.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def assemble_final_delivery(
    *,
    final_dir: Path,
    pdf: Path,
    report_json: Path,
    png: Path,
    mp4: Path,
    perception_report: dict[str, Any],
    removed: list[str],
    windows_mirror: Path | None,
) -> dict[str, Any]:
    final_dir.mkdir(parents=True, exist_ok=True)
    # remove old artifacts after new ones validated
    for name in OLD_REMOVE:
        p = final_dir / name
        if p.exists():
            p.unlink()
            removed.append(str(p))

    shutil.copy2(pdf, final_dir / "FUTBOLCU_ANALIZ_RAPORU_TR.pdf")
    shutil.copy2(report_json, final_dir / "FUTBOLCU_ANALIZ_RAPORU_TR.json")
    shutil.copy2(png, final_dir / "single_player_analysis_summary.png")
    shutil.copy2(mp4, final_dir / "real_video_analysis_proof.mp4")
    write_open_results_html(final_dir / "OPEN_RESULTS.html")
    write_readme(final_dir / "README.md")

    # validate mp4 portable profile
    v = validate_portable_mp4(final_dir / "real_video_analysis_proof.mp4")

    evidence = {
        "schema": "final_delivery_evidence_manifest_v2",
        "perception": {
            "detection_f1": perception_report.get("detection", {})
            .get("full_selected", {})
            .get("f1"),
            "selected_tracker": perception_report.get("tracking", {}).get("selected"),
            "team_flip_count": perception_report.get("team", {})
            .get("metrics", {})
            .get("team_flip_count"),
            "ball_evaluation": perception_report.get("ball", {}).get("evaluation"),
        },
        "portable_mp4_validation": v,
        "isolation": perception_report.get("isolation"),
    }
    (final_dir / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    write_checksums(final_dir)
    # include checksums hash in cleanup
    cleanup = {
        "removed_count": len(removed),
        "removed_kinds": ["legacy_proof_mp4", "english_summary_json", "non_final_mirror_residue"],
        "retained": FINAL_NAMES,
        "note": "legacy proof/media removed after real_video_analysis_proof.mp4 validated",
    }
    (final_dir / "cleanup_manifest.json").write_text(
        json.dumps(cleanup, indent=2, sort_keys=True) + "\n"
    )
    # refresh checksums including manifests optionally — keep as content files only

    mirror_info = None
    if windows_mirror is not None:
        windows_mirror.mkdir(parents=True, exist_ok=True)
        # clear non-final
        for p in windows_mirror.iterdir():
            if p.name not in FINAL_NAMES and p.is_file():
                p.unlink()
                removed.append(str(p))
        for name in FINAL_NAMES:
            src = final_dir / name
            if src.is_file():
                shutil.copy2(src, windows_mirror / name)
        # hash equality
        mismatches = []
        for name in FINAL_NAMES:
            a = final_dir / name
            b = windows_mirror / name
            if a.is_file() and b.is_file():
                if sha256_file(a) != sha256_file(b):
                    mismatches.append(name)
            elif a.is_file() != b.is_file():
                mismatches.append(name)
        mirror_info = {"path": str(windows_mirror), "mismatches": mismatches}

    present = sorted(p.name for p in final_dir.iterdir() if p.is_file())
    return {
        "final_dir": str(final_dir),
        "present": present,
        "mp4_validation": v,
        "windows_mirror": mirror_info,
        "removed": removed,
    }


__all__ = ["assemble_final_delivery", "write_open_results_html"]
