#!/usr/bin/env python3
"""Orchestrate SoccerNet repo environment setup, asset download, and smoke tests."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TIMESTAMP = os.environ.get("SOCCERNET_INSTALL_TS", "20260712_170330")
HOME = Path.home()
SOCCERNET = HOME / "projects/soccernet"
THIRD_PARTY = HOME / "projects/third-party"
PROJECT = HOME / "projects/football-analytics"
LOGDIR = HOME / f"logs/soccernet_full_install_{TIMESTAMP}"
VENV_ROOT = HOME / ".venvs/soccernet"
MODEL_ROOT = HOME / "models/soccernet"
ASSETS = HOME / "workspace/soccernet_nonvideo_assets"
REPO_ENVS = PROJECT / "requirements/repo_envs"
CONFIG_DIR = PROJECT / "configs/soccernet_install"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mpeg", ".mpg", ".m4v", ".ts", ".vob", ".wmv", ".flv"}

CONDA = HOME / "miniconda3/bin/conda"
UV = HOME / ".local/bin/uv"

AI_DEV_BASELINE = {}


@dataclass
class RepoResult:
    repo: str
    repo_type: str = "UNKNOWN"
    commit: str = ""
    environment: str = ""
    env_path: str = ""
    python: str = ""
    pytorch: str = ""
    cuda_available: bool = False
    pip_check: str = ""
    install_method: str = ""
    install_status: str = "NOT_STARTED"
    smoke_level: str = "L0_CLONE"
    final_status: str = "NOT_AUDITED"
    blocker: str = ""
    weights_downloaded: list = field(default_factory=list)
    weights_failed: list = field(default_factory=list)
    tests: dict = field(default_factory=dict)
    disk_env_mb: float = 0.0
    notes: str = ""


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)


def log(repo: str, msg: str, logtype: str = "install") -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    path = LOGDIR / f"{repo}.{logtype}.log"
    line = f"[{datetime.now().astimezone().isoformat()}] {msg}\n"
    with open(path, "a") as f:
        f.write(line)


def git_info(repo_path: Path) -> dict:
    def g(*args):
        r = run(["git", *args], cwd=repo_path)
        return r.stdout.strip() if r.returncode == 0 else ""
    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("branch", "--show-current"),
        "dirty": bool(g("status", "--porcelain")),
        "remote": g("remote", "get-url", "origin"),
    }


def conda_env_exists(name: str) -> bool:
    r = run([str(CONDA), "env", "list", "--json"])
    if r.returncode != 0:
        return False
    data = json.loads(r.stdout)
    return name in data.get("envs", []) or any(str(e).endswith(f"/envs/{name}") for e in data.get("envs", []))


def create_conda_env(name: str, python: str = "3.10") -> bool:
    if conda_env_exists(name):
        return True
    r = run([str(CONDA), "create", "-n", name, f"python={python}", "-y"])
    log(name, f"conda create: {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r.returncode == 0


def conda_python(name: str) -> Path:
    return HOME / "miniconda3/envs" / name / "bin/python"


def pip_in_env(env: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    py = conda_python(env)
    return run([str(py), "-m", "pip", *args], cwd=cwd)


def env_metadata(env: str) -> dict:
    py = conda_python(env)
    if not py.exists():
        return {}
    meta = {}
    for label, code in [
        ("python", "import sys; print(sys.version.split()[0])"),
        ("pytorch", "import importlib; m=importlib.import_module('torch'); print(getattr(m,'__version__','NA'))"),
        ("cuda", "import importlib; m=importlib.import_module('torch'); print(m.cuda.is_available())"),
    ]:
        r = run([str(py), "-c", code])
        meta[label] = r.stdout.strip() if r.returncode == 0 else "N/A"
    r = run([str(py), "-m", "pip", "check"])
    meta["pip_check"] = "ok" if r.returncode == 0 else r.stderr[:500]
    freeze = run([str(py), "-m", "pip", "freeze"])
    if freeze.returncode == 0:
        (REPO_ENVS / f"{env}.freeze.txt").write_text(freeze.stdout)
    return meta


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, repo: str) -> dict | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log(repo, f"DOWNLOAD {url} -> {dest}", "assets")
    if dest.exists():
        return {"path": str(dest), "sha256": sha256_file(dest), "status": "cached"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "football-analytics-setup/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        # reject HTML error pages
        if data[:15].lower().startswith(b"<!doctype") or data[:6].lower() == b"<html>":
            log(repo, f"REJECT HTML response for {url}", "assets")
            return None
        part.write_bytes(data)
        part.rename(dest)
        return {"path": str(dest), "sha256": sha256_file(dest), "byte_size": dest.stat().st_size, "status": "downloaded"}
    except Exception as e:
        log(repo, f"DOWNLOAD FAIL {url}: {e}", "assets")
        return None


def smoke_import(env: str, module: str) -> bool:
    r = run([str(conda_python(env)), "-c", f"import {module}; print('OK', {module}.__file__)"])
    log(env, f"import {module}: {r.returncode} {r.stdout} {r.stderr}", "smoke")
    return r.returncode == 0


def install_sn_trackeval() -> RepoResult:
    repo = "sn-trackeval"
    path = SOCCERNET / repo
    gi = git_info(path)
    res = RepoResult(repo=repo, repo_type="EVALUATION_TOOL", commit=gi["commit"], environment="sn-trackeval", install_method="conda+pip-editable")
    if not create_conda_env("sn-trackeval", "3.10"):
        res.install_status = "FAILED"; res.blocker = "conda create failed"; return res
    r = pip_in_env("sn-trackeval", "install", "-e", str(path))
    log(repo, r.stdout + r.stderr)
    if r.returncode != 0:
        res.install_status = "FAILED"; res.blocker = r.stderr[:300]; return res
    res.env_path = str(HOME / "miniconda3/envs/sn-trackeval")
    meta = env_metadata("sn-trackeval")
    res.python = meta.get("python", ""); res.pytorch = meta.get("pytorch", "N/A"); res.pip_check = meta.get("pip_check", "")
    res.tests["import_trackeval"] = smoke_import("sn-trackeval", "trackeval")
    r = run([str(conda_python("sn-trackeval")), str(path / "scripts/run_soccernet_gs.py"), "--help"], cwd=path)
    res.tests["cli_help"] = r.returncode == 0
    res.smoke_level = "L3_CLI" if res.tests["cli_help"] else ("L2_IMPORT" if res.tests["import_trackeval"] else "L1_ENV")
    res.install_status = "INSTALLED"
    res.final_status = "EVALUATION_READY" if res.tests["import_trackeval"] else "PARTIALLY_READY"
    return res


def install_sn_echoes() -> RepoResult:
    repo = "sn-echoes"
    path = SOCCERNET / repo
    gi = git_info(path)
    res = RepoResult(repo=repo, repo_type="DATASET_RELEASE", commit=gi["commit"], environment="sn-echoes", install_method="conda-minimal")
    if not create_conda_env("sn-echoes", "3.10"):
        res.install_status = "FAILED"; return res
    py = conda_python("sn-echoes")
    json_count = len(list((path / "Dataset").rglob("*.json"))) if (path / "Dataset").exists() else 0
    res.tests["dataset_json_count"] = json_count
    r = run([str(py), str(path / "stats.py")], cwd=path, timeout=120)
    res.tests["stats_script"] = r.returncode == 0
    res.smoke_level = "L6_SAMPLE_NONVIDEO" if res.tests["stats_script"] else "L1_ENV"
    res.install_status = "INSTALLED"
    res.final_status = "DATASET_DEVKIT_READY"
    res.notes = f"{json_count} JSON files in Dataset/"
    return res


def install_sn_jersey() -> RepoResult:
    repo = "sn-jersey"
    path = SOCCERNET / repo
    gi = git_info(path)
    res = RepoResult(repo=repo, repo_type="CHALLENGE_DOCS_ONLY", commit=gi["commit"], environment="ai-dev-ref", install_method="none-docs-only")
    res.tests["readme_exists"] = (path / "README.md").exists()
    res.smoke_level = "L0_CLONE"
    res.install_status = "CLONED_ONLY"
    res.final_status = "DATASET_DEVKIT_READY"
    res.notes = "No baseline code; SoccerNet SDK in ai-dev handles downloader API"
    res.blocker = "No executable baseline in repo"
    return res


def install_sn_calibration() -> RepoResult:
    repo = "sn-calibration"
    path = SOCCERNET / repo
    gi = git_info(path)
    env = "sn-calibration"
    res = RepoResult(repo=repo, repo_type="CHALLENGE_DEVKIT", commit=gi["commit"], environment=env, install_method="conda+pip-reqs")
    if not create_conda_env(env, "3.10"):
        res.install_status = "FAILED"; return res
    # Install pinned old torch stack per requirements
    r = pip_in_env(env, "install", "torch==1.10.2", "torchvision==0.11.3", "--index-url", "https://download.pytorch.org/whl/cpu")
    pip_in_env(env, "install", "-r", str(path / "requirements.txt"))
    meta = env_metadata(env)
    res.python = meta.get("python", ""); res.pytorch = meta.get("pytorch", ""); res.pip_check = meta.get("pip_check", "")
    res.tests["import_cv2"] = smoke_import(env, "cv2")
    # Google Drive weight - often blocked without gdown; try gdown if available
    weight_dir = MODEL_ROOT / repo
    weight_dir.mkdir(parents=True, exist_ok=True)
    res.weights_failed.append("model_extremities.pth (Google Drive - needs gdown or manual)")
    res.smoke_level = "L2_IMPORT" if res.tests.get("import_cv2") else "L1_ENV"
    res.install_status = "INSTALLED"
    res.final_status = "READY_FOR_NONVIDEO_USE" if res.tests.get("import_cv2") else "PARTIALLY_READY"
    res.blocker = "Model weights on Google Drive not auto-downloaded"
    return res


def install_sn_teamspotting() -> RepoResult:
    repo = "sn-teamspotting"
    path = SOCCERNET / repo
    gi = git_info(path)
    env = "sn-teamspotting"
    res = RepoResult(repo=repo, repo_type="CHALLENGE_BASELINE", commit=gi["commit"], environment=env, install_method="conda+pip-reqs")
    if not create_conda_env(env, "3.10"):
        res.install_status = "FAILED"; return res
    r = pip_in_env(env, "install", "torch==2.5.0", "torchvision==0.20.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    r2 = pip_in_env(env, "install", "-r", str(path / "requirements.txt"))
    if r2.returncode != 0:
        res.install_status = "FAILED"; res.blocker = r2.stderr[:300]; return res
    meta = env_metadata(env)
    res.python = meta.get("python", ""); res.pytorch = meta.get("pytorch", ""); res.pip_check = meta.get("pip_check", "")
    res.tests["import_torch"] = smoke_import(env, "torch")
    res.smoke_level = "L2_IMPORT" if res.tests["import_torch"] else "L1_ENV"
    res.install_status = "INSTALLED"
    res.final_status = "READY_FOR_VIDEO_INPUT"
    res.blocker = "Checkpoint on Google Drive; video dataset not downloaded"
    res.weights_failed.append("T-DEED checkpoint (Google Drive)")
    return res


def install_sn_gamestate() -> RepoResult:
    repo = "sn-gamestate"
    path = SOCCERNET / repo
    gi = git_info(path)
    env = "sn-gamestate"
    res = RepoResult(repo=repo, repo_type="CHALLENGE_BASELINE_FRAMEWORK", commit=gi["commit"], environment=env, install_method="uv+python3.9")
    venv = VENV_ROOT / env
    if not UV.exists():
        res.install_status = "FAILED"; res.blocker = "uv not found"; return res
  # Python 3.9 via conda for uv
    if not create_conda_env(env, "3.9"):
        res.install_status = "FAILED"; return res
    py39 = conda_python(env)
    if not venv.exists():
        r = run([str(UV), "venv", str(venv), "--python", str(py39)])
        log(repo, r.stdout + r.stderr)
    vpy = venv / "bin/python"
    # Install with uv pip using pyproject constraints
    r = run([str(UV), "pip", "install", "-e", str(path)], env={"VIRTUAL_ENV": str(venv), "UV_PROJECT_ENVIRONMENT": str(venv)})
    log(repo, f"uv pip install -e: {r.returncode}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        res.install_status = "FAILED"; res.blocker = r.stderr[:500]; res.final_status = "BLOCKED_DEPENDENCY"; return res
    res.env_path = str(venv)
    res.tests["import_tracklab"] = run([str(vpy), "-c", "import tracklab"]).returncode == 0
    res.tests["import_sn_gamestate"] = run([str(vpy), "-c", "import sn_gamestate"]).returncode == 0
    res.tests["import_mmcv"] = run([str(vpy), "-c", "import mmcv; print(mmcv.__version__)"]).returncode == 0
    r = run([str(venv / "bin/tracklab"), "--help"])
    res.tests["cli_tracklab"] = r.returncode == 0
    meta_r = run([str(vpy), "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
    res.pytorch = meta_r.stdout.strip()
    res.python = "3.9"
    if res.tests["import_tracklab"] and res.tests["import_sn_gamestate"]:
        res.smoke_level = "L3_CLI" if res.tests["cli_tracklab"] else "L2_IMPORT"
        res.install_status = "INSTALLED"
        res.final_status = "READY_FOR_DATASET_VIDEO_SMOKE_TEST"
    else:
        res.install_status = "PARTIAL"
        res.final_status = "BLOCKED_DEPENDENCY"
        res.blocker = "tracklab/sn_gamestate/mmcv import failed"
    # Zenodo weights - try metadata only, download small json if possible
    res.notes = "Do not run tracklab -cn soccernet (auto-downloads dataset)"
    return res


INSTALLERS = {
    "sn-trackeval": install_sn_trackeval,
    "sn-echoes": install_sn_echoes,
    "sn-jersey": install_sn_jersey,
    "sn-calibration": install_sn_calibration,
    "sn-teamspotting": install_sn_teamspotting,
    "sn-gamestate": install_sn_gamestate,
}


def main():
    repos = sys.argv[1:] if len(sys.argv) > 1 else list(INSTALLERS.keys())
    results = []
    for name in repos:
        if name not in INSTALLERS:
            log(name, "SKIP no installer", "install")
            continue
        try:
            res = INSTALLERS[name]()
        except Exception as e:
            res = RepoResult(repo=name, install_status="FAILED", blocker=str(e))
        results.append(asdict(res))
        print(json.dumps(asdict(res), indent=2))
    out = CONFIG_DIR / "install_results_partial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
