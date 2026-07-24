"""Offline Stage 16-R4 acceptance CLI handlers (no network/token)."""

from __future__ import annotations

import json
from pathlib import Path

from football_analytics.acceptance.isolation import assert_namespaces_isolated
from football_analytics.acceptance.namespaces import (
    AUTHORITATIVE_SOCCERTRACK_TARGET,
    GATE_TECHNICAL_PREVIEW,
)
from football_analytics.acceptance.self_contained import (
    ScenarioConfig,
    build_technical_preview_report,
    render_technical_preview_png,
    run_self_contained_acceptance,
    validate_two_run_determinism,
    write_report,
)
from football_analytics.acceptance.soccertrack_v2.reference_analysis import (
    analyze_soccertrack_v2_reference,
)

DEFAULT_TRAJ = Path(
    "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
    "reference_ground_truth/target_trajectory_reference.json"
)
DEFAULT_BAS = Path(
    "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
    "reference_ground_truth/bas_reference_events.json"
)
DEFAULT_TEAMTRACK_REPORT = Path(
    "artifacts/evidence/stage_16_real_video_pilot/real_video_pilot_report.json"
)
FINAL_JSON = Path("artifacts/final/single_player_analysis_summary.json")
FINAL_PNG_GH = Path("artifacts/final/single_player_analysis_summary.png")
FINAL_PNG_LOCAL = Path(
    "/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png"
)


def cmd_acceptance_generate(*, output_dir: Path, seed: int = 16040) -> int:
    cfg = ScenarioConfig(seed=seed)
    result = run_self_contained_acceptance(output_dir=output_dir, config=cfg)
    print(json.dumps({"status": "ok", "paths": result["paths"]}, indent=2))
    return 0


def cmd_acceptance_run(*, output_dir: Path, seed: int = 16040) -> int:
    return cmd_acceptance_generate(output_dir=output_dir, seed=seed)


def cmd_acceptance_validate(*, dir_a: Path, dir_b: Path) -> int:
    check = validate_two_run_determinism(dir_a, dir_b)
    print(json.dumps(check, indent=2, sort_keys=True))
    return 0 if check["equal"] else 1


def cmd_reference_soccertrack_v2(
    *,
    output: Path,
    trajectory: Path = DEFAULT_TRAJ,
    bas: Path = DEFAULT_BAS,
) -> int:
    assert_namespaces_isolated(
        soccertrack_player_id=str(AUTHORITATIVE_SOCCERTRACK_TARGET["player_id"])
    )
    report = analyze_soccertrack_v2_reference(trajectory_path=trajectory, bas_path=bas)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference_json: {output}")
    print(f"target: {report['target']}")
    return 0


def cmd_report_render_final(
    *,
    self_contained_dir: Path,
    reference_json: Path,
    teamtrack_report: Path = DEFAULT_TEAMTRACK_REPORT,
    project_root: Path | None = None,
    trajectory_path: Path = DEFAULT_TRAJ,
) -> int:
    root = project_root or Path.cwd()
    receipt = json.loads((self_contained_dir / "acceptance_receipt.json").read_text())
    reference = json.loads(reference_json.read_text())
    tt_path = teamtrack_report if teamtrack_report.is_absolute() else root / teamtrack_report
    tt_raw = json.loads(tt_path.read_text())
    tt_summary = {
        "detection": (tt_raw.get("evaluation") or {}).get("detection"),
        "target_tracking": (tt_raw.get("evaluation") or {}).get("target_tracking"),
        "pilot": tt_raw.get("pilot"),
        "sequence": (tt_raw.get("dataset") or {}).get("sequence_id"),
    }
    report = build_technical_preview_report(
        reference=reference,
        self_contained_receipt=receipt,
        teamtrack_summary=tt_summary,
        deterministic_ts="2026-07-25T00:00:00Z",
    )
    json_path = root / FINAL_JSON
    sha = write_report(json_path, report)

    heatmap_points: list[tuple[float, float]] = []
    if trajectory_path.is_file():
        traj = json.loads(trajectory_path.read_text())
        pid = str(AUTHORITATIVE_SOCCERTRACK_TARGET["player_id"])
        for i, p in enumerate(traj.get("points") or []):
            if str(p.get("player_id")) != pid:
                continue
            if i % 8 != 0:
                continue
            heatmap_points.append((float(p["x_m"]), float(p["y_m"])))

    png = render_technical_preview_png(
        report=report,
        output_local=FINAL_PNG_LOCAL,
        output_github=root / FINAL_PNG_GH,
        heatmap_points=heatmap_points,
    )
    # ensure only one final customer visual under artifacts/final
    keep = {"single_player_analysis_summary.png", "single_player_analysis_summary.json"}
    for p in (root / "artifacts/final").glob("*"):
        if p.name not in keep and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            p.unlink()
    print(GATE_TECHNICAL_PREVIEW)
    print(f"final_json: {json_path} sha256={sha}")
    print(f"final_png: {png}")
    return 0


__all__ = [
    "cmd_acceptance_generate",
    "cmd_acceptance_run",
    "cmd_acceptance_validate",
    "cmd_reference_soccertrack_v2",
    "cmd_report_render_final",
]
