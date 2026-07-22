# Stage interface (Stage 2D)

Single-stage execution contract for foundation pipelines. **No DAG scheduler** —
stages are registered and run one at a time via `execute_stage`.

Package: `football_analytics.pipeline` (importing it does not load torch or PyArrow).

## Stage identity

`StageIdentity` (frozen):

| Field | Role |
|-------|------|
| `name` | Safe id: `[a-z][a-z0-9_]{1,63}` |
| `version` | Positive int |
| `code_fingerprint` | 64-hex SHA-256 of stage code identity |
| `input_contracts` / `output_contracts` | `ContractRef(name, version)` tuples |
| `deterministic` | Must be true to participate in cache |
| `cacheable` | Explicit opt-in for content-addressed cache |

Helper: `make_stage_identity(...)`.

## Protocol

```text
Stage
  identity -> StageIdentity
  execute(StageRequest) -> StageExecutionOutput
```

`StageRegistry` is an in-process map keyed by `(name, version)`. Duplicate
registration fails. No dynamic import, entry points, or eval.

## Request / result

**`StageRequest`:** `run_id`, `stage_identity`, `config_fingerprint`,
`compatibility_fingerprint`, frozen `inputs` (`logical_name` → `ArtifactRef`),
distinct `working_directory` / `output_directory`, `requested_at_utc`,
`cache_policy_enabled`.

**`StageExecutionOutput`:** `outputs`, `metrics`, `warnings` (from `execute`).

**`StageResult`:** full execution record (`schema_version=1`) including
`status`, `cache_key`, `cache_hit`, timing, redacted inputs/outputs, metrics,
warnings, optional `{class, message}` error (no traceback), and
`execution_fingerprint`.

Secret-bearing keys are rejected; metrics must be finite JSON scalars.

## Artifact refs

`ArtifactRef` is content-addressed and path-relative:

- POSIX `relative_path` (no `..`, absolute, or backslash)
- `sha256` (64 hex) + `size_bytes`
- Parquet / `*.parquet` requires `contract_name`, `contract_version`,
  `schema_fingerprint`
- Build/verify/copy via `artifacts.build_artifact_ref`,
  `verify_artifact_on_disk`, `copy_artifact_file` (copy only; symlink/special
  rejected; hardlinks rejected when policy says so)

## Determinism and cacheability

Cache lookup/publish runs only when **all** are true:

1. `identity.cacheable` and `identity.deterministic`
2. system `CachePolicyConfig.enabled`
3. `request.cache_policy_enabled`
4. not `force_miss`

Otherwise the stage executes without publish.

## Statuses

Allowlist: `succeeded` | `failed` | `cache_hit` | `skipped` | `cancelled`.

Invariants:

- `cache_hit` status ⇒ `cache_hit=True`
- `failed` ⇒ `error` present
- `succeeded` / `cache_hit` ⇒ `error` is `None`

## Lifecycle (`execute_stage`)

1. Validate request identity matches stage
2. Compute cache key (or zero-key on key error)
3. If cacheable: verify + restore on hit (skip `stage.execute`)
4. On corrupt hit: optional quarantine, treat as miss
5. Execute stage; failures → `failed` result, **no publish**
6. Verify outputs on disk under `output_directory`
7. Publish if cacheable
8. Optional atomic `stage_execution_receipt.json`

## Synthetic fixture

`SyntheticEchoStage` (`synthetic_echo` v1) is a **test fixture**, not a product
stage. Writes deterministic `echo.bin`; tracks `_executions` for cache-hit proofs.

## Out of scope

DAG orchestration, multi-stage graphs, remote/distributed cache, automatic
purge/GC, real video ingest/detect/track stages.
