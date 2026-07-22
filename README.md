# football-analytics

Proprietary broadcast football video analytics pipeline (detection → tracking →
identity → calibration → game state → events → reports/API).

**Current stage:** Stage 2C — Canonical PyArrow data contracts and schema migrations.
Stages 0–2B (audit, storage, registries, archive, package, GitHub sync, runtime foundation)
are closed.

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
configs/                  # paths, archive, security policies
schemas/                  # JSON schemas / registries schemas
tests/                    # unittest + pytest
docs/                     # audits, ADRs, stages, development
requirements/             # base / dev / ai-dev constraints
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

## Working CLI (Stage 2B)

```bash
football-analytics --version
football-analytics info
football-analytics run-id
football-analytics config validate --config configs/project/defaults.yaml
football-analytics config fingerprint --config configs/project/defaults.yaml
football-analytics environment show
football-analytics contracts list
football-analytics contracts show detections --version 1
```

No `run` / `ingest` / `detect` / `track` / `evaluate` commands yet.

See [docs/development/runtime_foundation.md](docs/development/runtime_foundation.md)
and [docs/data/canonical_contracts.md](docs/data/canonical_contracts.md).

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
```

## Data / models / licenses

- Broadcast datasets are **not** downloaded in Stages 0–2A.
- Demo MP4s inside external clones are not project datasets.
- Model weights (e.g. `SV_*.pth`) may exist locally; redistribution license is `review_required`.
- See `docs/legal/license_inventory.md`, `docs/data/data_access_matrix.md`, `THIRD_PARTY_NOTICES.md`.

## Archive safety

Documented in `docs/operations/archive_and_cleanup.md`. Never archive unknown runs;
cleanup requires exact confirmation and re-verification.

## Secrets

Policy: `docs/security/secrets_policy.md`. Use `.env.example` names only; never commit `.env`.

## Not done yet

- Canonical schemas / foundation contracts (Stage 2B+)
- Ingest / detection / tracking pipelines
- Real match evaluation
- Independent off-device backup
- `v0.1.0-foundation` tag (only after Stage 2 completes)

## Roadmap / GitHub

- Development workflow: [docs/development/git_github_workflow.md](docs/development/git_github_workflow.md)
- Stage docs under `docs/stages/`

## License

Proprietary — see `LICENSE`. Third-party code remains under its own terms.
