"""Stage 16 acceptance gate helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            b = handle.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_png_hash_equality(local: Path, github: Path) -> dict:
    if not local.is_file() or not github.is_file():
        return {"status": "MISSING", "local": str(local), "github": str(github)}
    a = _sha256(local)
    b = _sha256(github)
    return {
        "status": "PASS" if a == b else "FAIL",
        "local_sha256": a,
        "github_sha256": b,
        "equal": a == b,
    }


def check_attribution(project_root: Path) -> dict:
    notices = (project_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    ok = "SoccerTrack v2" in notices and "CC BY 4.0" in notices
    return {"status": "PASS" if ok else "FAIL", "has_soccertrack_cc_by": ok}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--local-png",
        type=Path,
        default=Path(
            "/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png"
        ),
    )
    parser.add_argument(
        "--github-png",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    root = args.project_root
    github_png = args.github_png or (root / "artifacts/final/single_player_analysis_summary.png")
    report = {
        "attribution": check_attribution(root),
        "png_hash": check_png_hash_equality(args.local_png, github_png),
        "registry": {
            "status": "PENDING",
            "note": "dataset_registry soccertrack_v2_single_match_acceptance checked separately",
        },
    }
    print(json.dumps(report, indent=2))
    fails = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
