# Development environment (`ai-dev`)

## Interpreter

```text
/home/fdoblak/miniconda3/envs/ai-dev/bin/python
Python 3.10.20
```

Activate:

```bash
source /home/fdoblak/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev
```

## Protected packages (do not casual-upgrade)

See `requirements/constraints-ai-dev.txt` for exact pins including:

- `torch==2.11.0+cu128` (+ matching torchvision/torchaudio)
- `numpy==2.2.6`, `pandas==2.3.3`
- `opencv-python==5.0.0.93`, `opencv-python-headless==5.0.0.93`
- `ultralytics==8.4.91`, `SoccerNet==0.1.62`

## Stage 2A tooling (installed into `ai-dev` only)

| Package | Exact version (Stage 2A) |
|---|---|
| pyarrow | 25.0.0 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.15.22 |
| black | 26.5.1 |
| isort | 8.0.1 |
| mypy | 2.3.0 |
| build | 1.5.0 |
| wheel | 0.47.0 (pre-existing) |

## Editable install

```bash
python -m pip install -e . --no-deps
```

## GPU note

Agent sessions may report `cuda_available=False`. Classification remains
`AGENT_CONTEXT_GPU_UNVERIFIABLE` until a host GPU inference smoke before Stage 5.

## Isolation

Do not merge SoccerNet/third-party Conda envs into `ai-dev`. Do not install into
system Python or Conda `base`.
