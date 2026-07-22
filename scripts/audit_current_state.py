#!/usr/bin/env python3
"""Read-only Stage 0 style audit helper for football-analytics.

Safe by design:
- No package installs, downloads, git fetch/pull/clone, mounts, or deletes.
- Subprocess calls use timeouts; failures become structured status entries.
- Secrets in remotes are sanitized before printing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("FOOTBALL_ANALYTICS_ROOT", "/home/fdoblak/projects/football-analytics")
)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except OSError as exc:
        return {"ok": False, "returncode": -2, "stdout": "", "stderr": str(exc)}


def sanitize_remote(url: str | None) -> str | None:
    if not url:
        return None
    url = re.sub(r"https://[^@\s]+@", "https://***@", url)
    url = re.sub(r"ghp_[A-Za-z0-9]+", "ghp_***", url)
    url = re.sub(r"github_pat_[A-Za-z0-9_]+", "github_pat_***", url)
    return url


def load_paths(project_root: Path) -> dict[str, Any]:
    paths_file = project_root / "configs" / "system" / "paths.yaml"
    result: dict[str, Any] = {"paths_file": str(paths_file), "loaded": False}
    if not paths_file.exists():
        result["status"] = "MISSING"
        return result
    text = paths_file.read_text(encoding="utf-8", errors="replace")
    # Minimal YAML subset parser avoided; return raw + common keys via regex.
    result["loaded"] = True
    result["status"] = "PRESENT_NOT_VERIFIED"
    result["raw_bytes"] = paths_file.stat().st_size
    for key in ("project_root", "soccernet_root", "third_party_root", "workspace", "ssd_root"):
        m = re.search(rf"{key}:\s*(\S+)", text)
        if m:
            result[key] = m.group(1)
    return result


def audit_main_repo(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(project_root), "exists": project_root.is_dir()}
    if not out["exists"]:
        out["status"] = "MISSING"
        return out
    out["git_toplevel"] = run(["git", "rev-parse", "--show-toplevel"], cwd=project_root)
    out["branch"] = run(["git", "branch", "--show-current"], cwd=project_root)
    head = run(["git", "rev-parse", "HEAD"], cwd=project_root)
    out["head"] = head
    remotes = run(["git", "remote", "-v"], cwd=project_root)
    if remotes["ok"]:
        remotes["stdout"] = "\n".join(
            sanitize_remote(line) or "" for line in remotes["stdout"].splitlines()
        )
    out["remotes"] = remotes
    out["status_porcelain"] = run(["git", "status", "--porcelain=v1"], cwd=project_root)
    out["log1"] = run(["git", "log", "-1", "--decorate", "--oneline"], cwd=project_root)
    required = [
        "pyproject.toml",
        "environment.yml",
        ".gitignore",
        "README.md",
        "external_repos.lock.yaml",
        "model_registry.yaml",
        "dataset_registry.yaml",
        "configs/system/paths.yaml",
    ]
    out["files"] = {
        rel: ("VERIFIED" if (project_root / rel).is_file() else "MISSING") for rel in required
    }
    return out


def audit_storage(paths: dict[str, Any]) -> dict[str, Any]:
    ssd = Path(paths.get("ssd_root", "/mnt/d/football_data"))
    mnt_d = Path("/mnt/d")
    workspace = Path(paths.get("workspace", "/home/fdoblak/workspace"))
    return {
        "mnt_d_exists": mnt_d.exists(),
        "mnt_d_is_dir": mnt_d.is_dir(),
        "football_data_exists": ssd.exists(),
        "workspace_exists": workspace.exists(),
        "workspace_runs": (workspace / "runs").is_dir(),
        "workspace_staging": (workspace / "staging").is_dir(),
        "workspace_cache": (workspace / "cache").is_dir(),
        "workspace_current": (
            str((workspace / "current").resolve())
            if (workspace / "current").exists()
            else None
        ),
    }


def audit_ai_dev(python_bin: Path) -> dict[str, Any]:
    if not python_bin.exists():
        return {"status": "MISSING", "python_bin": str(python_bin)}
    code = (
        "import json,sys;\n"
        "out={'executable':sys.executable,'version':sys.version.split()[0]};\n"
        "pkgs={};\n"
        "for n,m in [('torch','torch'),('numpy','numpy'),('cv2','cv2'),"
        "('ultralytics','ultralytics'),('pyarrow','pyarrow')]:\n"
        "  try:\n"
        "    mod=__import__(m); pkgs[n]={'ok':True,'version':getattr(mod,'__version__','?')}\n"
        "  except Exception as e:\n"
        "    pkgs[n]={'ok':False,'error':type(e).__name__}\n"
        "out['packages']=pkgs\n"
        "try:\n"
        "  import torch\n"
        "  out['cuda']={'available':bool(torch.cuda.is_available()),"
        "'runtime':getattr(torch.version,'cuda',None),"
        "'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\n"
        "except Exception as e:\n"
        "  out['cuda']={'error':str(e)}\n"
        "print(json.dumps(out))\n"
    )
    res = run([str(python_bin), "-c", code], timeout=60)
    parsed: dict[str, Any]
    if res["ok"] and res["stdout"]:
        try:
            parsed = json.loads(res["stdout"].splitlines()[-1])
        except json.JSONDecodeError:
            parsed = {"parse_error": True, "raw": res["stdout"][:500]}
    else:
        parsed = {"status": "BLOCKED", "stderr": res["stderr"][:500]}
    parsed["probe"] = {"ok": res["ok"], "returncode": res["returncode"]}
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="football-analytics root (or set FOOTBALL_ANALYTICS_ROOT)",
    )
    parser.add_argument(
        "--ai-dev-python",
        type=Path,
        default=Path(
            os.environ.get(
                "AI_DEV_PYTHON",
                "/home/fdoblak/miniconda3/envs/ai-dev/bin/python",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path inside the project (not required)",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "read_only",
        "project_root": str(args.project_root),
        "paths": load_paths(args.project_root),
        "main_repo": audit_main_repo(args.project_root),
        "storage": audit_storage(load_paths(args.project_root)),
        "ai_dev": audit_ai_dev(args.ai_dev_python),
        "ffmpeg": run(["ffmpeg", "-version"], timeout=10),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            timeout=15,
        ),
        "notes": [
            "This script does not clone, fetch, install, download, or mutate git state.",
            "Prefer configs/environment/current_ai_dev.json from Stage 0 for frozen evidence.",
            "Missing /dev/nvidia* alone is not proof of WSL CUDA failure; prefer /dev/dxg and /usr/lib/wsl/lib.",
            "Agent-context CUDA False should be classified carefully; see Stage 0 gpu_validation.",
        ],
    }
    # Reduce ffmpeg noise
    if report["ffmpeg"]["ok"]:
        report["ffmpeg"]["stdout"] = report["ffmpeg"]["stdout"].splitlines()[0]

    text = json.dumps(report, indent=2)
    if args.output:
        out = args.output
        if not out.is_absolute():
            out = args.project_root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
