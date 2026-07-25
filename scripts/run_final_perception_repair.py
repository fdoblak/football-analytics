#!/usr/bin/env python3
"""Run final perception repair → Turkish report → final_delivery assembly."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path("/home/fdoblak/projects/football-analytics")
sys.path.insert(0, str(REPO / "src"))

from football_analytics.acceptance.final_perception_repair.delivery import (  # noqa: E402
    assemble_final_delivery,
)
from football_analytics.acceptance.final_perception_repair.pipeline import (  # noqa: E402
    run_perception_repair,
)
from football_analytics.acceptance.final_perception_repair.proof_video import (  # noqa: E402
    build_analysis_proof_mp4,
)
from football_analytics.acceptance.final_perception_repair.turkish_report import (  # noqa: E402
    build_report_payload,
    render_turkish_dashboard_png,
    render_turkish_pdf,
)
from football_analytics.acceptance.portable_final_media import download_source  # noqa: E402

WORK = Path("/home/fdoblak/workspace/final_perception_repair")
SEQ_ROOT = Path("/home/fdoblak/football_data/datasets/teamtrack")
FINAL = REPO / "artifacts" / "final_delivery"
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Final")
TRAJ = Path(
    "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
    "reference_ground_truth/target_trajectory_reference.json"
)
BAS = Path(
    "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
    "reference_ground_truth/bas_reference_events.json"
)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    art = WORK / "artifacts"
    art.mkdir(exist_ok=True)

    # source integrity
    mp4 = download_source(WORK)
    dest = SEQ_ROOT / "source/soccer_side/F_20200220_1_0330_0360/img1.mp4"
    if not dest.exists() or dest.stat().st_size != mp4.stat().st_size:
        shutil.copy2(mp4, dest)

    print("=== perception repair ===", flush=True)
    perception = run_perception_repair(
        sequence_root=SEQ_ROOT / "source",
        work_dir=art,
        device="cuda:0",
    )
    print(
        json.dumps(
            {
                k: perception[k]
                for k in ("detection", "tracking", "team", "ball")
                if k in perception
            },
            indent=2,
        )[:2000],
        flush=True,
    )

    print("=== proof video ===", flush=True)
    proof = build_analysis_proof_mp4(
        sequence_root=SEQ_ROOT / "source",
        frame_dump_json=art / "frame_dump.json",
        perception_report_json=art / "perception_report.json",
        final_mp4=art / "real_video_analysis_proof.mp4",
    )
    # drop heavy rgb before saving receipt
    receipt = {k: v for k, v in proof.items() if k != "selected_rgb"}
    (art / "proof_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    print("=== turkish report ===", flush=True)
    existing = None
    old_sum = FINAL / "single_player_analysis_summary.json"
    if old_sum.is_file():
        existing = json.loads(old_sum.read_text())
    payload = build_report_payload(
        trajectory_path=TRAJ,
        bas_path=BAS,
        perception=perception,
        existing_summary=existing,
    )
    (art / "FUTBOLCU_ANALIZ_RAPORU_TR.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pdf_info = render_turkish_pdf(
        payload=payload,
        out_pdf=art / "FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
        frame_rgbs=proof.get("selected_rgb"),
    )
    png_info = render_turkish_dashboard_png(
        payload=payload,
        out_png=art / "single_player_analysis_summary.png",
        frame_rgbs=proof.get("selected_rgb"),
    )
    print("pdf", pdf_info, flush=True)
    print("png", png_info, flush=True)

    print("=== assemble final_delivery ===", flush=True)
    removed: list[str] = []
    assembly = assemble_final_delivery(
        final_dir=FINAL,
        pdf=art / "FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
        report_json=art / "FUTBOLCU_ANALIZ_RAPORU_TR.json",
        png=art / "single_player_analysis_summary.png",
        mp4=art / "real_video_analysis_proof.mp4",
        perception_report=perception,
        removed=removed,
        windows_mirror=WIN if WIN.parent.is_dir() else None,
    )
    (art / "assembly_receipt.json").write_text(json.dumps(assembly, indent=2) + "\n")
    print(json.dumps(assembly, indent=2), flush=True)

    # cleanup raw source after use
    for p in (dest, WORK / "img1.mp4"):
        if p.is_file():
            p.unlink()
            print("cleaned", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
