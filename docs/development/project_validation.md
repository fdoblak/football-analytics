# Project validation (Stage 2D)

Unified health runner: `scripts/check_project.py` and CLI
`football-analytics project check`.

Core: `football_analytics.pipeline.project_check.run_project_checks`.

## Profiles

| Profile | Role |
|---------|------|
| `local` (default) | Host-aware: git dirty/remote, storage paths, deep synthetic smokes |
| `ci` | Pure/repo checks; host-only probes explicitly `SKIP` |

## Modes

| Mode | Role |
|------|------|
| `quick` (default) | Fast, mostly read-only package/config/schema/validator subset |
| `deep` | Quick + synthetic roundtrips (data contracts, stage/cache, etc.) under Git-outside roots |

Flags: `--profile {local,ci}`, `--quick` / `--deep`, `--strict`, `--quiet`,
`--json-out <path>`.

## Check record

Each check: `id`, `category`, `status`, `severity`, `message`, `evidence`,
`duration_ms`.

Statuses: **`PASS` | `WARN` | `FAIL` | `SKIP`**.

Top-level report: `schema_version`, `profile`, `mode`, timestamps,
`duration_ms`, `overall_status`, `exit_code`, `checks`, `summary`,
`environment_classification` (`LOCAL_HOST_PROFILE` or `CI_PROFILE`).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | `PASS`, or non-strict `PASS_WITH_WARNINGS` |
| `1` | Findings under `--strict` (WARN promoted) |
| `2` | Usage / config error |
| `3` | `FAIL` / integrity / security |

One check exception becomes an isolated `FAIL` record; it does not abort the
whole report. Subprocesses use `shell=False`, 120s timeout, bounded capture,
and secret redaction.

## CI vs local

**CI profile explicit SKIPs** (must not look like PASS):

- Host WSL GPU
- `/home/fdoblak` absolute machine paths
- Local SoccerNet clones / model weights
- Real storage backend probes
- Git credential materialization
- NDA dataset presence

**Local** runs storage path checks and optional deep synthetic smokes.
**CI** still requires package metadata, cache policy, pipeline schemas, stage
import/key determinism, secrets, runtime foundation, data contracts, and
workflow safety. Full pytest is **not** nested inside project check (CI runs
pytest as a separate job) — recorded as intentional `SKIP`.

Protected CUDA/vision package pins are enforced on `local`; skipped on `ci`
(lightweight `requirements/ci.txt` has no torch stack).

## Runtime report paths

JSON reports belong **outside Git**:

```text
/home/fdoblak/workspace/project_checks/project_validation_<timestamp>.json
```

Deep mode must only mutate synthetic fixtures under workspace check roots and
clean up in `finally`. Do not rewrite real user runs, archives, or production
cache entries.
