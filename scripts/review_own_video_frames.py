#!/usr/bin/env python3
"""Local lightweight frame-review tool for Stage 17-R2 own-video GT.

Does NOT invent reviewed=true. Opens prelabels + overlays; writes append-only
review decisions. Use for human/ball/calibration labeling sessions.

Examples:
  python scripts/review_own_video_frames.py list --split holdout --kind human
  python scripts/review_own_video_frames.py show --frame 709 --kind human
  python scripts/review_own_video_frames.py apply-decision --kind ball --frame 661 \\
      --visible visible --centre 640,400 --reviewed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORK = Path("/home/fdoblak/workspace/own_video_analysis/stage17r2")
DECISIONS = WORK / "review_decisions"
GT_OUT = Path(
    "/home/fdoblak/projects/football-analytics/artifacts/diagnostics/own_video_recovery/gt"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prelabels(kind: str) -> list[dict[str, Any]]:
    path = WORK / f"prelabels_{kind}.json"
    if not path.is_file():
        raise SystemExit(f"missing prelabels: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def decisions_path(kind: str) -> Path:
    DECISIONS.mkdir(parents=True, exist_ok=True)
    return DECISIONS / f"decisions_{kind}.jsonl"


def append_decision(kind: str, payload: dict[str, Any]) -> None:
    path = decisions_path(kind)
    payload = {**payload, "written_at_utc": utc_now(), "schema": f"review_decision_{kind}_v1"}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def cmd_list(args: argparse.Namespace) -> int:
    rows = load_prelabels(args.kind)
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    print(f"kind={args.kind} n={len(rows)} split={args.split or 'all'}")
    for r in rows[: args.limit]:
        extra = ""
        if args.kind == "human":
            extra = f" humans={len(r.get('humans', []))}"
        else:
            extra = f" visible_auto={r.get('visible_auto')}"
        print(f"  frame={r['frame']:4d} t={r['t_s']:6.2f}s split={r['split']}{extra}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.kind == "human":
        overlay = WORK / "overlays_human" / f"h_{args.frame:04d}.jpg"
    else:
        overlay = WORK / "overlays_ball" / f"b_{args.frame:04d}.jpg"
    frame = WORK / "frames_review" / f"f{args.frame:04d}.jpg"
    for p in (overlay, frame):
        print(p, "OK" if p.is_file() else "MISSING")
    if args.open and overlay.is_file():
        # best-effort local open (WSL)
        for cmd in (("wslview", str(overlay)), ("xdg-open", str(overlay))):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except OSError:
                continue
    rows = [r for r in load_prelabels(args.kind) if int(r["frame"]) == int(args.frame)]
    if rows:
        print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:4000])
    return 0


def cmd_apply_ball(args: argparse.Namespace) -> int:
    if not args.reviewed:
        raise SystemExit("refusing to write ball decision without --reviewed")
    centre = None
    if args.centre:
        x_s, y_s = args.centre.split(",")
        centre = [float(x_s), float(y_s)]
    bbox = None
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("bbox must be x,y,w,h")
        bbox = parts
    append_decision(
        "ball",
        {
            "frame": int(args.frame),
            "split": args.split,
            "reviewed": True,
            "review_status": "reviewed",
            "visible": args.visible,
            "centre": centre,
            "bbox": bbox,
            "occlusion": args.occlusion,
            "reviewer": "cursor_agent_visual",
            "provenance": args.note or "direct_visual_review",
        },
    )
    print("appended ball decision", args.frame)
    return 0


def cmd_apply_human_frame(args: argparse.Namespace) -> int:
    """Apply a whole-frame human review JSON file (list of corrected humans)."""
    if not args.reviewed:
        raise SystemExit("refusing without --reviewed")
    humans = json.loads(Path(args.humans_json).read_text(encoding="utf-8"))
    # enforce team=null for referee/staff
    for h in humans:
        if h.get("role") in {"referee", "staff"}:
            h["team"] = None
    append_decision(
        "human",
        {
            "frame": int(args.frame),
            "split": args.split,
            "reviewed": True,
            "review_status": "reviewed",
            "humans": humans,
            "reviewer": "cursor_agent_visual",
            "provenance": args.note or "direct_visual_review",
        },
    )
    print("appended human decision", args.frame, "n=", len(humans))
    return 0


def cmd_export_gt(args: argparse.Namespace) -> int:
    """Merge decisions into GT JSON under diagnostics (never mark auto as reviewed)."""
    GT_OUT.mkdir(parents=True, exist_ok=True)
    for kind in ("human", "ball"):
        pre = {int(r["frame"]): r for r in load_prelabels(kind)}
        reviewed: dict[int, dict[str, Any]] = {}
        path = decisions_path(kind)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if not d.get("reviewed"):
                    continue
                reviewed[int(d["frame"])] = d
        out_rows = []
        for fi, row in sorted(pre.items()):
            if fi in reviewed:
                out_rows.append(
                    {**row, **reviewed[fi], "reviewed": True, "review_status": "reviewed"}
                )
            else:
                out_rows.append({**row, "reviewed": False, "review_status": "auto_candidate"})
        out = {
            "schema": f"own_video_gt_{kind}_v1",
            "n_frames": len(out_rows),
            "n_reviewed": sum(1 for r in out_rows if r.get("reviewed")),
            "n_auto_candidate": sum(1 for r in out_rows if not r.get("reviewed")),
            "splits": {
                s: {
                    "n": sum(1 for r in out_rows if r.get("split") == s),
                    "reviewed": sum(
                        1 for r in out_rows if r.get("split") == s and r.get("reviewed")
                    ),
                }
                for s in ("train", "dev", "holdout")
            },
            "note": "reviewed=true only from review_decisions; auto never promoted",
            "frames": out_rows,
            "written_at_utc": utc_now(),
        }
        (GT_OUT / f"gt_{kind}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(kind, "reviewed", out["n_reviewed"], "/", out["n_frames"])
    # copy contact sheets summary (small) into diagnostics if present
    sheets = WORK / "contact_sheets"
    dest = GT_OUT / "contact_sheets_sample"
    if sheets.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(sheets.glob("*.jpg"))[:8]:
            shutil.copy2(p, dest / p.name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--kind", choices=["human", "ball"], required=True)
    p_list.add_argument("--split", choices=["train", "dev", "holdout"], default=None)
    p_list.add_argument("--limit", type=int, default=40)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show")
    p_show.add_argument("--kind", choices=["human", "ball"], required=True)
    p_show.add_argument("--frame", type=int, required=True)
    p_show.add_argument("--open", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_ball = sub.add_parser("apply-ball")
    p_ball.add_argument("--frame", type=int, required=True)
    p_ball.add_argument("--split", required=True)
    p_ball.add_argument("--visible", choices=["visible", "not_visible", "ambiguous"], required=True)
    p_ball.add_argument("--centre", default=None, help="x,y")
    p_ball.add_argument("--bbox", default=None, help="x,y,w,h")
    p_ball.add_argument("--occlusion", default="none")
    p_ball.add_argument("--note", default="")
    p_ball.add_argument("--reviewed", action="store_true")
    p_ball.set_defaults(func=cmd_apply_ball)

    p_hum = sub.add_parser("apply-human-frame")
    p_hum.add_argument("--frame", type=int, required=True)
    p_hum.add_argument("--split", required=True)
    p_hum.add_argument("--humans-json", required=True)
    p_hum.add_argument("--note", default="")
    p_hum.add_argument("--reviewed", action="store_true")
    p_hum.set_defaults(func=cmd_apply_human_frame)

    p_exp = sub.add_parser("export-gt")
    p_exp.set_defaults(func=cmd_export_gt)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
