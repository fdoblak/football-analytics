# Stage 2A — Python package, tooling, GitHub sync

**Date:** 2026-07-22
**Start HEAD:** `015debd77b5936274bdf8d65d3113242e05506e4`
**Target commit:** `Establish Python package and development tooling`

## Delivered

- `pyproject.toml` (setuptools, version `0.1.0.dev0`, console script)
- Package CLI: `--version`, `info` (side-effect free)
- `requirements/{base,dev,constraints-ai-dev}.txt`, `environment.yml`
- README, LICENSE (proprietary), `.editorconfig`, `.gitignore` build ignores
- Package smoke tests (`tests/package/`)
- Dev docs + dependency change report
- Editable install + ruff/black/isort/mypy/build

## Environment

Installed into `ai-dev` only (after safe dry-run): pyarrow 25.0.0, pytest 9.1.1, pytest-cov 7.1.0, ruff 0.15.22, black 26.5.1, isort 8.0.1, mypy 2.3.0, build 1.5.0.

Protected Torch/NumPy/OpenCV/Ultralytics/SoccerNet pins unchanged.

## GitHub

Private `football-analytics` sync attempted after local commit (see final report for auth/push outcome).

## Next

Aşama 2B — Run Kimliği, Config, Logging, Hash ve Environment Kayıtları
