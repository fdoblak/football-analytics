# football-analytics

Proprietary broadcast football video analytics pipeline (detection → tracking →
identity → calibration → game state → events → reports/API).

**Current stage:** Stage 2D — foundation complete
(`PASS_WITH_FINDINGS — STAGE 2 FOUNDATION COMPLETE`).
Stages 0–2D closed. Milestone tag: **`foundation-v0.1.0`**.

## Principles

- Isolated external SoccerNet/third-party repos (locked SHAs); prefer adapters.
- Active storage is **WSL-local** (`/home/fdoblak/football_data`).
- Planned D: archive (`/mnt/d/football_data`) remains **unverified**.
- Local archive shares the same WSL VHDX failure domain — **not** an independent backup.
- Cleanup uses **quarantine**, not permanent delete.
- Secrets never belong in registries, commits, or logs.
- RTX 3050 Laptop **4 GB VRAM** — keep models small; GPU inference gated before Stage 5.

## Layout

```text
src/football_analytics/   # installable package
scripts/                  # validators & archive tools
configs/                  # paths, archive, security, cache policies
schemas/                  # JSON schemas / registries / pipeline / cache
tests/                    # unittest + pytest
docs/                     # audits, ADRs, stages, development
requirements/             # base / dev / ci / ai-dev constraints
.github/workflows/        # least-privilege SHA-pinned CI
```

## Development environment

Primary env: Conda **`ai-dev`** (Python 3.10).

```bash
source /home/fdoblak/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev
cd /home/fdoblak/projects/football-analytics
python -m pip install -e . --no-deps
```

Protected CUDA/vision stack pins live in `requirements/constraints-ai-dev.txt`
(`torch==2.11.0+cu128`, NumPy, OpenCV, Ultralytics, SoccerNet, …). Do not casually upgrade them.

See [docs/development/environment.md](docs/development/environment.md).

## Working CLI (Stage 2D)

```bash
football-analytics --version
football-analytics info
football-analytics run-id
football-analytics config validate --config configs/project/defaults.yaml
football-analytics config fingerprint --config configs/project/defaults.yaml
football-analytics environment show
football-analytics contracts list
football-analytics contracts show detections --version 1
football-analytics project check --profile local --quick
football-analytics cache inspect <64_hex_cache_key>
football-analytics cache verify <64_hex_cache_key>
```

No `run` / `ingest` / `detect` / `track` / `evaluate` commands yet.

Foundation docs:

- [Stage interface](docs/development/stage_interface.md)
- [Cache design](docs/development/cache_design.md)
- [Project validation](docs/development/project_validation.md)
- [CI](docs/development/ci.md)
- [Runtime foundation](docs/development/runtime_foundation.md)
- [Canonical contracts](docs/data/canonical_contracts.md)

## Tests and quality

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest
ruff check .
black --check .
isort --check-only .
mypy src/football_analytics
```

Validators:

```bash
python scripts/check_storage.py --config configs/system/paths.yaml
python scripts/check_registries.py --model-registry model_registry.yaml \
  --dataset-registry dataset_registry.yaml --external-lock external_repos.lock.yaml \
  --verify-files --verify-repos
python scripts/check_secrets.py --root /home/fdoblak/projects/football-analytics
python scripts/check_runtime_foundation.py --config configs/project/defaults.yaml
python scripts/check_data_contracts.py --registry configs/data/schema_registry.yaml
python scripts/check_stage_cache.py --config configs/system/cache_policy.yaml --synthetic
python scripts/check_ci_workflow.py --workflow .github/workflows/ci.yml --strict
python scripts/check_project.py --profile local --quick
# or: football-analytics project check --profile local --quick
```

## Data / models / licenses

- Broadcast datasets are **not** downloaded in Stages 0–2D.
- Demo MP4s inside external clones are not project datasets.
- Model weights (e.g. `SV_*.pth`) may exist locally; redistribution license is `review_required`.
- See `docs/legal/license_inventory.md`, `docs/data/data_access_matrix.md`, `THIRD_PARTY_NOTICES.md`.

## Archive safety

Documented in `docs/operations/archive_and_cleanup.md`. Never archive unknown runs;
cleanup requires exact confirmation and re-verification.

## Secrets

Policy: `docs/security/secrets_policy.md`. Use `.env.example` names only; never commit `.env`.

## Not done yet

- Video ingest / probe / normalize / frame timebase (Stage 3)
- Detection / tracking pipelines
- Real match evaluation
- Independent off-device backup
- Cache GC / automatic purge

## Roadmap / GitHub

- Development workflow: [docs/development/git_github_workflow.md](docs/development/git_github_workflow.md)
- CI notes: [docs/development/ci.md](docs/development/ci.md)
- Stage docs under `docs/stages/` (see [Stage 2 completion](docs/stages/stage_02_completion.md))

## License

Proprietary — see `LICENSE`. Third-party code remains under its own terms.
