# Runtime foundation (Stage 2B)

## Run ID

Canonical format:

```text
run_YYYYMMDDTHHMMSSffffffZ_<12_lowercase_hex>
```

Example: `run_20260722T171530123456Z_a1b2c3d4e5f6`

- UTC, lexicographically sortable
- Suffix from `secrets.token_hex(6)` (no global RNG)
- No path separators, spaces, `..`, or shell metacharacters
- Max length 48
- Clock/suffix injectable for tests

Stage 1D archive policy accepts **both** the older
`run_YYYYMMDD_HHMMSS_<6-12 alnum>` fixture format and this Stage 2B format.

## Config

Layers (low → high precedence):

1. `configs/project/defaults.yaml`
2. Explicit user YAML (`yaml.safe_load`, max 256 KiB, mapping root)
3. Allowlisted env overrides (`FOOTBALL_ANALYTICS_LOG_LEVEL`, `FOOTBALL_ANALYTICS_LOG_FORMAT`)
4. Programmatic overrides

Rules: unknown top-level keys rejected; secret-shaped keys rejected; NaN/Infinity rejected;
no `eval`/`exec`; no uncontrolled `~`/`$HOME` expansion; resolved config is immutable
(`MappingProxyType`).

Schema: `schemas/resolved_config.schema.json` (`schema_version: 1`).

## Fingerprint

SHA-256 over canonical UTF-8 JSON (`sort_keys=True`, compact separators, `allow_nan=False`).
Record includes `algorithm`, `canonicalization_version`, `digest`.

## Hashing

Streaming file SHA-256 with bounded chunk size, regular-file only, symlink/special rejection,
before/after stat mutation detection. Directory manifests use relative POSIX paths, sorted files,
canonical JSON digest.

## Redaction

Marker: `[REDACTED]`. Case-insensitive sensitive keys; nested structures; bearer/token patterns;
URL userinfo sanitization; no input mutation.

## Structured logging

Stdlib logger `football_analytics` (module: `structured_logging.py` to avoid shadowing `logging`).
Human console + JSONL file (`RotatingFileHandler`). Required JSONL fields include
`schema_version`, `timestamp_utc`, `level`, `logger`, `event`, `message`, `run_id`, `stage`,
`context`, `exception`. Idempotent setup; root logger untouched.

## Environment record

`schemas/environment_record.schema.json`. Allowlisted package versions via
`importlib.metadata` only (no torch import). GPU field fixed to
`AGENT_CONTEXT_GPU_UNVERIFIABLE`. Git metadata via timed subprocess; remotes sanitized.
No full environment dump.

## Run context vs run manifest

| Contract | Role |
|----------|------|
| `run_context` | Identity + provenance at initialization (`status=initialized`) |
| `run_manifest` | Stage 1D archive artifact inventory (`pending`/`completed`/…) |

## Atomic records

`write_json_record`: same-dir temp, fsync, `os.replace`, default no-overwrite, mode `0600`,
containment, secret-key rejection.

## CLI

```bash
football-analytics --version
football-analytics info
football-analytics run-id
football-analytics config validate --config configs/project/defaults.yaml
football-analytics config fingerprint --config configs/project/defaults.yaml [--json]
football-analytics environment show [--json]
```

No `run` / `ingest` / `detect` / `track` commands in Stage 2B.

## Validator

```bash
python scripts/check_runtime_foundation.py --config configs/project/defaults.yaml
python scripts/check_runtime_foundation.py --config configs/project/defaults.yaml \
  --synthetic-run --json-out /home/fdoblak/workspace/foundation_checks/report.json
```

Synthetic runs only under `/home/fdoblak/workspace/foundation_checks/` (Git-ignored workspace).

Exit codes: `0` PASS, `1` finding, `2` usage/config, `3` integrity/security.

## Limitations

- Not a pipeline orchestrator or cache engine (Stage 2D)
- Not PyArrow table schemas (Stage 2C)
- Does not create real user runs under `football_data`
- Does not verify host GPU
