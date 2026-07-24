#!/usr/bin/env python3
"""Local deep validation (CI-equivalent) for Stage 15D parity.

Runs secret scan, CI workflow YAML checks, and project check (local quick).
Does not call GitHub API; remote CI remains UNVERIFIABLE_AGENT_API_CONTEXT.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EXIT_PASS = 0
EXIT_FINDING = 1


def _run(cmd: list[str], *, timeout: int = 300) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-project", action="store_true")
    args = parser.parse_args(argv)

    steps = [
        [PYTHON, str(REPO_ROOT / "scripts" / "check_secrets.py"), "--root", str(REPO_ROOT)],
        [PYTHON, str(REPO_ROOT / "scripts" / "check_ci_workflow.py")],
    ]
    if not args.skip_project:
        steps.append(
            [
                PYTHON,
                str(REPO_ROOT / "scripts" / "check_project.py"),
                "--profile",
                "local",
                "--quick",
            ]
        )

    results: list[dict[str, Any]] = []
    failed = False
    for cmd in steps:
        item = _run(cmd)
        results.append(item)
        if item["returncode"] != 0:
            failed = True

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "remote_ci_status": "UNVERIFIABLE_AGENT_API_CONTEXT",
        "invented_green_remote_ci": False,
        "failed": failed,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"local_deep_validation failed={failed}")
        for r in results:
            print(f"  rc={r['returncode']} cmd={' '.join(r['cmd'])}")
        print("FINDING: remote CI unverifiable (RISK-042); local parity executed")
    return EXIT_FINDING if failed else EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
