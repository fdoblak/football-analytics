# Foundation CI (Stage 2D)

Workflow: `.github/workflows/ci.yml`

Safety validator: `scripts/check_ci_workflow.py` (YAML-only; **does not** call
the GitHub API).

## Triggers

- `push` → `main`
- `pull_request` → `main`
- `workflow_dispatch`

**Not used:** `pull_request_target`.

## Permissions

```yaml
permissions:
  contents: read
```

No repository write, no secrets interpolation, no deploy/publish, no sudo,
no Docker, no GPU, no external repo/model/dataset downloads.

Concurrency: `ci-${{ github.workflow }}-${{ github.ref }}` with
`cancel-in-progress: true`. Job timeouts: quality 20m, tests 30m.

## SHA-pinned actions

Mutable tags (`@v6`) are rejected by the safety validator. Official actions only:

| Action | Pin | Comment |
|--------|-----|---------|
| `actions/checkout` | `d23441a48e516b6c34aea4fa41551a30e30af803` | `# v6` |
| `actions/setup-python` | `5fda3b95a4ea91299a34e894583c3862153e4b97` | `# v7` |

Checkout: `persist-credentials: false`, `fetch-depth: 1`.
Python: exact `3.10.20`. Pip cache keyed on `requirements/ci.txt`,
`requirements/base.txt`, `pyproject.toml`.

## Dependencies

`requirements/ci.txt` is **lightweight**: base runtime + pytest/ruff/black/
isort/mypy/build — **no** torch, ultralytics, SoccerNet, or CUDA wheels.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/ci.txt
python -m pip install -e . --no-deps
```

## Jobs

**quality:** ruff, black, isort, mypy, `python -m build`, sdist content smoke
(pipeline + cache_policy + schemas present; no parquet),
`check_ci_workflow.py --strict`.

**tests:** pytest; runtime foundation; data-contract synthetic; stage-cache
synthetic; `check_project.py --profile ci --quick --strict`; secrets scan.

## Local equivalent commands

```bash
python scripts/check_ci_workflow.py --workflow .github/workflows/ci.yml --strict
python -m pip install -r requirements/ci.txt
python -m pip install -e . --no-deps
ruff check .
black --check .
isort --check-only .
mypy src/football_analytics
python -m build
python -m pytest
python scripts/check_runtime_foundation.py --config configs/project/defaults.yaml
python scripts/check_data_contracts.py --registry configs/data/schema_registry.yaml \
  --synthetic-roundtrip --migration-smoke
python scripts/check_stage_cache.py --config configs/system/cache_policy.yaml --synthetic
python scripts/check_project.py --profile ci --quick --strict
python scripts/check_secrets.py --root .
```

## Remote CI visibility

Cursor Agent access to `api.github.com` returns **proxy 403**. Therefore
`gh run list` / API-based remote run inspection is unavailable in this agent
context.

```text
remote_ci_status = UNVERIFIABLE_AGENT_API_CONTEXT
```

Local-equivalent commands + workflow safety validator are the authoritative
evidence for foundation closure. Do not invent remote green checks.
