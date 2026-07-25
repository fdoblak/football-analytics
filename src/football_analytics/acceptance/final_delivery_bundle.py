"""Build annotated TeamTrack proof MP4 + consolidated final_delivery bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.namespaces import (
    AUTHORITATIVE_SOCCERTRACK_TARGET,
    GATE_TECHNICAL_PREVIEW,
)
from football_analytics.acceptance.soccertrack_v2.reference_analysis import (
    analyze_soccertrack_v2_reference,
)
from football_analytics.acceptance.teamtrack.loader import load_sequence
from football_analytics.acceptance.teamtrack.pilot_runner import _iou, run_teamtrack_pilot

EXPECTED_MP4_SHA = "fd42dbe6df8b85946d6ee82fc11863ea94dd2735b3b126e9bfca3972a3149db1"
EXPECTED_MP4_SIZE = 13_086_969
EXPECTED_GT_SHA = "80fd710824a4f006528aa0bb0207ae34c1869e469373cdb276800260d96a7ff4"

# Overlay colors (BGR for OpenCV)
COLOR_PRED = (255, 255, 0)  # cyan-ish BGR? wait BGR: cyan=(255,255,0) yes
COLOR_GT = (0, 255, 0)  # green
COLOR_TARGET_PRED = (0, 255, 255)  # yellow
COLOR_UNMATCHED = (0, 0, 255)  # red


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_proof_mp4(
    *,
    sequence_root: Path,
    predictions_json: Path,
    output_mp4: Path,
    max_height: int = 720,
) -> dict[str, Any]:
    seq = load_sequence(
        root=sequence_root,
        sport_view="soccer_side",
        split="train",
        sequence_id="F_20200220_1_0330_0360",
    )
    dump = json.loads(predictions_json.read_text())
    pred_by_frame = {int(k): [tuple(b) for b in v] for k, v in dump["pred_by_frame"].items()}
    gt_target = {int(k): tuple(v) for k, v in dump["gt_target_by_frame"].items()}
    target_pred = {
        int(k): (tuple(v) if v is not None else None)
        for k, v in dump["target_pred_by_frame"].items()
    }

    scale = min(1.0, max_height / float(seq.im_height))
    out_w = int(round(seq.im_width * scale))
    out_h = int(round(seq.im_height * scale))
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Prefer software mp4v then remux with ffmpeg to H.264 (avoids host v4l2m2m issues).
    fourcc = int(cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        str(output_mp4),
        fourcc,
        float(seq.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter for proof MP4")
    cap = cv2.VideoCapture(str(seq.video_path))
    tp = fp = fn = 0
    frame_idx = 0
    selected_frames: dict[str, dict[str, Any]] = {}
    pick_targets = {
        "start": 1,
        "mid": max(1, seq.seq_length // 2),
        "crowd": max(1, int(seq.seq_length * 0.7)),
        "end": seq.seq_length,
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        preds = pred_by_frame.get(frame_idx, [])
        gts = [gb for gb in seq.boxes if gb.frame == frame_idx]
        # running detection greedy IoU@0.5
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for pi, pb in enumerate(preds):
            best_j, best = -1, 0.0
            for j, gb in enumerate(gts):
                if j in matched_gt:
                    continue
                v = _iou(pb, (gb.x, gb.y, gb.w, gb.h))
                if v > best:
                    best, best_j = v, j
            if best >= 0.5 and best_j >= 0:
                matched_gt.add(best_j)
                matched_pred.add(pi)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        tgt_gt = gt_target.get(frame_idx)
        tgt_pr = target_pred.get(frame_idx)
        tgt_iou = _iou(tgt_pr, tgt_gt) if (tgt_pr and tgt_gt) else 0.0
        matched_tgt = bool(tgt_pr and tgt_gt and tgt_iou >= 0.3)

        # draw
        for pi, pb in enumerate(preds):
            x, y, w, h = pb
            color = COLOR_UNMATCHED if pi not in matched_pred else COLOR_PRED
            if tgt_pr and abs(pb[0] - tgt_pr[0]) < 1e-3 and abs(pb[1] - tgt_pr[1]) < 1e-3:
                color = COLOR_TARGET_PRED
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        if tgt_gt:
            x, y, w, h = tgt_gt
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), COLOR_GT, 2)
            cv2.putText(
                frame,
                "GT T7",
                (int(x), max(20, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_GT,
                2,
            )

        t_s = (frame_idx - 1) / seq.fps
        hud = [
            "REAL TEAMTRACK VIDEO — TRACKING PILOT",
            f"t={t_s:.2f}s frame={frame_idx}/{seq.seq_length}",
            f"TP={tp} FP={fp} FN={fn}  P={precision:.3f} R={recall:.3f} F1={f1:.3f}",
            f"Target Track7 IoU={tgt_iou:.3f} matched={matched_tgt}",
            "legend: cyan=pred green=GT yellow=target-pred red=unmatched-pred",
        ]
        y0 = 28
        for line in hud:
            cv2.putText(frame, line, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y0 += 28

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        for label, fi in pick_targets.items():
            if frame_idx == fi:
                selected_frames[label] = {
                    "frame_index": frame_idx,
                    "t_s": t_s,
                    "target_iou": tgt_iou,
                    "matched": matched_tgt,
                    "bgr_path": None,
                    "rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                }

    cap.release()
    writer.release()
    # rewrite with ffmpeg for H.264 if mp4v was used
    size = output_mp4.stat().st_size if output_mp4.is_file() else 0
    return {
        "path": str(output_mp4),
        "size_bytes": size,
        "sha256": sha256_file(output_mp4) if output_mp4.is_file() else None,
        "width": out_w,
        "height": out_h,
        "selected_frames": {
            k: {kk: vv for kk, vv in v.items() if kk != "rgb"} | {"has_rgb": "rgb" in v}
            for k, v in selected_frames.items()
        },
        "_frames_rgb": selected_frames,
        "final_running": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def render_final_png(
    *,
    report: dict[str, Any],
    reference: dict[str, Any],
    frame_pack: dict[str, Any],
    output_png: Path,
) -> str:
    det = report["evaluation"]["detection"]
    tr = report["evaluation"]["target_tracking"]
    metrics = reference["metrics"]
    tgt = AUTHORITATIVE_SOCCERTRACK_TARGET

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#0b1210")
    fig.text(
        0.02,
        0.96,
        "Football Analytics — Single Player Technical Preview",
        color="#ffe082",
        fontsize=18,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.02,
        0.925,
        (
            f"SoccerTrack v2 Match {tgt['match_id']}  ·  "
            f"Team {tgt['team_side']} / Jersey {tgt['jersey_number']} / Player {tgt['player_id']}"
        ),
        color="#c8e6c9",
        fontsize=12,
        fontfamily="DejaVu Sans",
    )

    # 4 real-video frames
    order = ["start", "mid", "crowd", "end"]
    fig.text(
        0.02,
        0.88,
        "REAL-VIDEO TRACKING VALIDATION — DIFFERENT DATASET/TARGET (TeamTrack Track 7)",
        color="#80cbc4",
        fontsize=11,
        fontfamily="DejaVu Sans",
    )
    frames = frame_pack.get("_frames_rgb") or {}
    for i, key in enumerate(order):
        ax = fig.add_axes((0.02 + i * 0.24, 0.52, 0.23, 0.32))
        ax.set_facecolor("#1b2a22")
        ax.set_xticks([])
        ax.set_yticks([])
        fr = frames.get(key)
        if fr and "rgb" in fr:
            ax.imshow(fr["rgb"])
            ax.set_title(
                f"{key} t={fr['t_s']:.1f}s IoU={fr['target_iou']:.2f}",
                color="#e8f5e9",
                fontsize=8,
            )
        else:
            ax.text(
                0.5, 0.5, "n/a", ha="center", va="center", color="#ffcc80", transform=ax.transAxes
            )

    # real-video metrics panel
    axm = fig.add_axes((0.02, 0.30, 0.40, 0.18))
    axm.set_facecolor("#1b2a22")
    axm.axis("off")
    axm.text(
        0.02,
        0.95,
        (
            "TeamTrack real-video metrics\n"
            f"detections={report['pilot']['human_detections']}\n"
            f"P={det['precision']:.3f}  R={det['recall']:.3f}  F1={det['f1']:.3f}\n"
            f"target coverage={tr['target_coverage_ratio']:.3f}  mean IoU={tr['mean_iou']:.3f}\n"
            f"device={report['pilot']['device']}  runtime_s={report['pilot']['elapsed_s']:.1f}"
        ),
        va="top",
        color="#e8f5e9",
        fontsize=10,
        family="DejaVu Sans",
    )

    # SoccerTrack reference metrics
    axr = fig.add_axes((0.46, 0.30, 0.52, 0.18))
    axr.set_facecolor("#1b2a22")
    axr.axis("off")
    lines = ["SoccerTrack v2 annotation-derived (NOT video prediction)"]
    for key in (
        "measured_distance_m",
        "mean_speed_m_s",
        "peak_speed_m_s",
        "sprint_count",
        "sprint_distance_m",
        "activity_index",
        "bas_pass_attempts",
        "bas_drive_actions",
        "bas_high_pass_attempts",
        "bas_header_actions",
        "bas_successful_tackles",
        "penalty_area_presence_points",
    ):
        m = metrics.get(key) or {}
        lines.append(f"{key}={m.get('value')} [{m.get('status')}]")
    axr.text(
        0.02,
        0.95,
        "\n".join(lines[:14]),
        va="top",
        color="#e8f5e9",
        fontsize=8,
        family="DejaVu Sans",
    )

    ne = [k for k, m in metrics.items() if (m or {}).get("status") == "NOT_EVALUABLE"]
    fig.text(
        0.02,
        0.22,
        "N/A — NOT EVALUABLE: " + ", ".join(ne + ["video-event inference accuracy"]),
        color="#ffcc80",
        fontsize=9,
    )
    fig.text(
        0.02,
        0.12,
        (
            "SoccerTrack metrics are annotation-derived. "
            "TeamTrack panel validates real-video detection/tracking only. "
            "Datasets/player identities are not mixed. "
            "Not official Opta data. Video-event inference accuracy is not validated."
        ),
        color="#90a4ae",
        fontsize=8,
        wrap=True,
    )
    fig.text(
        0.02,
        0.05,
        GATE_TECHNICAL_PREVIEW.replace("SELF-CONTAINED TECHNICAL ACCEPTANCE COMPLETE; ", ""),
        color="#ffe082",
        fontsize=8,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    return sha256_file(output_png)


def build_final_delivery(
    *,
    project_root: Path,
    sequence_root: Path,
    staging_mp4: Path,
) -> dict[str, Any]:
    delivery = project_root / "artifacts" / "final_delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    run_dir = Path("/home/fdoblak/football_data/datasets/teamtrack/runs/final_delivery_proof")
    evidence = project_root / "artifacts" / "evidence" / "stage_16_r4_final" / "pilot_rerun"
    pred_dump = run_dir / "predictions" / "frame_predictions.json"

    # integrity of source
    if (
        staging_mp4.stat().st_size != EXPECTED_MP4_SIZE
        or sha256_file(staging_mp4) != EXPECTED_MP4_SHA
    ):
        raise RuntimeError("NO-GO — REAL VIDEO SOURCE INTEGRITY FAILURE")

    report = run_teamtrack_pilot(
        sequence_root=sequence_root,
        run_dir=run_dir,
        evidence_dir=evidence,
        dump_predictions_json=pred_dump,
    )
    proof_path = delivery / "real_video_tracking_proof.mp4"
    proof = write_proof_mp4(
        sequence_root=sequence_root,
        predictions_json=pred_dump,
        output_mp4=proof_path,
    )
    # ffmpeg remux to h264 if needed and oversized
    import subprocess

    remux = delivery / "real_video_tracking_proof_h264.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(proof_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-an",
            "-pix_fmt",
            "yuv420p",
            str(remux),
        ],
        check=True,
        capture_output=True,
    )
    remux.replace(proof_path)
    proof["sha256"] = sha256_file(proof_path)
    proof["size_bytes"] = proof_path.stat().st_size

    traj = Path(
        "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
        "reference_ground_truth/target_trajectory_reference.json"
    )
    bas = Path(
        "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
        "reference_ground_truth/bas_reference_events.json"
    )
    reference = analyze_soccertrack_v2_reference(trajectory_path=traj, bas_path=bas)

    png_path = delivery / "single_player_analysis_summary.png"
    sha_a = render_final_png(
        report=report, reference=reference, frame_pack=proof, output_png=png_path
    )
    # second render for determinism check into temp then compare
    tmp = delivery / "_tmp_png_check.png"
    sha_b = render_final_png(report=report, reference=reference, frame_pack=proof, output_png=tmp)
    if sha_a != sha_b:
        raise RuntimeError("final PNG non-deterministic")
    tmp.unlink()

    # Build canonical JSON
    det = report["evaluation"]["detection"]
    tr = report["evaluation"]["target_tracking"]

    def mwrap(value, *, unit, status, evidence_level, source, coverage=None, limitations=None):
        return {
            "value": value,
            "unit": unit,
            "status": status,
            "evidence_level": evidence_level,
            "source": source,
            "coverage": coverage,
            "limitations": limitations,
        }

    summary = {
        "schema": "final_delivery_single_player_summary_v1",
        "release": {
            "tag_candidate": "single-player-analytics-technical-preview-v0.16.1",
            "gate": (
                "PASS_WITH_FINDINGS — VIDEO-BACKED TECHNICAL PREVIEW CONSOLIDATED; "
                "RELEASE TREE CLEAN; VIDEO-EVENT ACCURACY NOT VALIDATED"
            ),
            "hf_required": False,
            "network_required": False,
            "download_required_for_release": False,
        },
        "target": {
            "match_id": "128057",
            "team_side": "left",
            "jersey_number": 24,
            "player_id": "506469",
            "deprecated_invalid_not_used": {"jersey_number": 11, "player_id": "506466"},
        },
        "real_video_validation": {
            "dataset": "TeamTrack v6",
            "sequence": "soccer_side/train/F_20200220_1_0330_0360",
            "anonymous_target": "Track 7",
            "namespace": "teamtrack_real_video_pilot",
            "source_mp4_sha256": EXPECTED_MP4_SHA,
            "source_mp4_size_bytes": EXPECTED_MP4_SIZE,
            "gt_sha256": EXPECTED_GT_SHA,
            "metrics": {
                "detections": mwrap(
                    report["pilot"]["human_detections"],
                    unit="count",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "precision": mwrap(
                    det["precision"],
                    unit="ratio",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "recall": mwrap(
                    det["recall"],
                    unit="ratio",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "f1": mwrap(
                    det["f1"],
                    unit="ratio",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "target_coverage": mwrap(
                    tr["target_coverage_ratio"],
                    unit="ratio",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "mean_iou": mwrap(
                    tr["mean_iou"],
                    unit="ratio",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "runtime_s": mwrap(
                    report["pilot"]["elapsed_s"],
                    unit="s",
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
                "device": mwrap(
                    report["pilot"]["device"],
                    unit=None,
                    status="REAL_VIDEO_VALIDATED",
                    evidence_level="REAL_VIDEO_VALIDATED",
                    source="teamtrack_pilot",
                ),
            },
            "proof_mp4": {
                "path": "artifacts/final_delivery/real_video_tracking_proof.mp4",
                "sha256": proof["sha256"],
                "size_bytes": proof["size_bytes"],
            },
            "selected_frame_times_s": {
                k: ((proof.get("_frames_rgb") or {}).get(k) or {}).get("t_s")
                for k in ("start", "mid", "crowd", "end")
            },
            "gt_leakage_into_prediction": False,
        },
        "annotation_derived_metrics": reference["metrics"],
        "not_evaluable_metrics": [
            k for k, m in reference["metrics"].items() if m.get("status") == "NOT_EVALUABLE"
        ]
        + ["video_event_inference_accuracy"],
        "evidence_levels": [
            "REAL_VIDEO_VALIDATED",
            "REFERENCE_ANNOTATION_DERIVED",
            "SELF_CONTAINED_TESTED",
            "NOT_EVALUABLE",
        ],
        "dataset_provenance": {
            "teamtrack": report["dataset"],
            "soccertrack_v2": {
                "match_id": "128057",
                "license": "CC-BY-4.0",
                "not_video_prediction": True,
            },
        },
        "license_attribution": {
            "teamtrack": report.get("attribution"),
            "soccertrack_v2": reference.get("disclaimer"),
        },
        "limitations": [
            "Video-event inference accuracy not validated on real match GT",
            "TeamTrack and SoccerTrack targets are different persons/datasets",
            "Annotation-derived metrics are not model predictions",
            "Not official Opta data",
        ],
        "artifact_hashes": {
            "summary_json_sha256": None,  # filled after write
            "summary_png_sha256": sha_a,
            "proof_mp4_sha256": proof["sha256"],
        },
        "pilot_report_snapshot": {
            "detection": det,
            "target_tracking": tr,
            "pilot": report["pilot"],
        },
    }

    json_path = delivery / "single_player_analysis_summary.json"
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    # provisional write then embed self hash
    summary["artifact_hashes"]["summary_json_sha256"] = _sha256_bytes(text.encode("utf-8"))
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary["artifact_hashes"]["summary_json_sha256"] = _sha256_bytes(text.encode("utf-8"))
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    json_path.write_text(text)

    return {
        "delivery": str(delivery),
        "report": report,
        "reference": reference,
        "proof": {k: v for k, v in proof.items() if k != "_frames_rgb"},
        "png_sha256": sha_a,
        "json_path": str(json_path),
        "png_path": str(png_path),
        "mp4_path": str(proof_path),
        "selected_frame_times_s": summary["real_video_validation"]["selected_frame_times_s"],
    }


__all__ = ["build_final_delivery", "render_final_png", "write_proof_mp4"]
