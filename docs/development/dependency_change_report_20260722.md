# Dependency change report — Stage 2A (2026-07-22)

## Scope

Install **only** into Conda env `ai-dev` via
`/home/fdoblak/miniconda3/envs/ai-dev/bin/python -m pip`.

## Dry-run

Report: `/home/fdoblak/workspace/environment_checks/dependency_dry_run_stage_02a_20260722T145819Z.json`

Planned installs (no protected package uninstall/upgrade):

- pyarrow, pytest, pytest-cov, ruff, black, isort, mypy, build
- transitive: coverage, iniconfig, pluggy, pathspec, mypy-extensions, pytokens, librt, ast-serialize, pyproject_hooks

`wheel` already satisfied.

## Protected before/after (unchanged)

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| opencv-python | 5.0.0.93 |
| opencv-python-headless | 5.0.0.93 |
| ultralytics | 8.4.91 |
| SoccerNet | 0.1.62 |

## Newly installed exact versions

| Package | Version |
|---|---|
| pyarrow | 25.0.0 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.15.22 |
| black | 26.5.1 |
| isort | 8.0.1 |
| mypy | 2.3.0 |
| build | 1.5.0 |

## Evidence (Git-excluded)

`/home/fdoblak/workspace/environment_checks/environment_*_stage_02a_*.json`
`pip_freeze_*`, `pip_check_*`, dry-run JSON/log
