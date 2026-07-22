# Stage 2D — Stage interface, cache, project validator, CI, foundation closure

**Date:** 2026-07-22
**Start HEAD:** `2b5954387e53639612902c2271f5b58a4ac17294`
**Commit message:** `Complete foundation with stage contracts cache and CI`

## 1. Purpose

Deliver the Stage protocol, content-addressed local cache, unified project
validator, least-privilege SHA-pinned CI, validators/CLI, and Stage 2 foundation
closure — without DAG orchestration, real video ingest, or GPU inference.

## 2. Starting SHA

`2b5954387e53639612902c2271f5b58a4ac17294` on clean `main` with
`local main = origin/main = ls-remote main` (Stage 2C closed).

## 3. Baseline

pytest **311** passed; ruff/black/isort/mypy PASS; runtime foundation PASS;
data contracts PASS; storage PASS; registry PASS_WITH_WARNINGS (license/access);
secrets 0; pip check clean.

## 4. Prior review

Reused Stage 2B `hash_canonical_json` / `sha256_file` / `write_json_record` /
`validate_run_id` / redaction, and Stage 2C contract fingerprints. Pipeline
package import remains free of torch/PyArrow. No new pip packages for the
pipeline core; CI uses lightweight `requirements/ci.txt`.

## 5. Stage interface

`Stage` protocol + `StageRegistry` + `StageIdentity` under
`football_analytics.pipeline`. Explicit register/get/list only — no dynamic
import/eval. See `docs/development/stage_interface.md`.

## 6. Request / result

Frozen `StageRequest` / `StageResult` / `StageExecutionOutput` with secret-safe
`to_dict()`, status allowlist, and execution fingerprint. Working and output
directories must differ.

## 7. Artifact reference

`ArtifactRef` with safe relative paths, SHA-256 + size, Parquet contract
metadata requirements, and copy/verify helpers rejecting symlink/special/
policy-hardlink cases.

## 8. Cache key

Canonical JSON + SHA-256 (`CACHE_KEY_VERSION=1`) over stage identity, config/
compatibility fingerprints, ordered input identities, and output contracts.
Path-independent: no absolute paths, hostnames, timestamps, secrets, or run IDs.

## 9. Cache layout

`<cache_root>/v1/sha256/<ab>/<rest>/` with `cache_manifest.json`,
`stage_result.json`, and `artifacts/`. Locks under `v1/locks/`.

## 10. Lock / concurrency

Exclusive `fcntl.flock` with timeout. Concurrent publish: first atomic rename
wins; loser cleans temp and treats existing entry as winner (re-verify).

## 11. Publish / read / restore

Atomic temp-dir publish (copy-only), verify-on-publish/read, restore into
`output_directory` without overwrite. Failed stages never publish.

## 12. Corruption / quarantine

Corrupt hits quarantine (move + receipt, `permanent_delete_performed=false`)
then miss-and-re-execute. No permanent delete; `automatic_purge` forced false.

## 13. Execution lifecycle

`execute_stage`: validate → key → hit/restore or miss/execute → verify outputs
→ optional publish → optional receipt. Cache hit skips `stage.execute`.

## 14. Schemas / config

- `schemas/pipeline/{artifact_ref,stage_request,stage_result,stage_execution_receipt}.schema.json`
- `schemas/cache/cache_manifest.schema.json`
- `configs/system/cache_policy.yaml` (`layout_version: 1`, `algorithm: sha256`)

## 15. Stage / cache validator

`scripts/check_stage_cache.py` — policy, schemas, key determinism, synthetic
miss→hit, corruption/concurrency smokes. Runtime root:
`/home/fdoblak/workspace/stage_cache_checks/`.

## 16. Project validator

`scripts/check_project.py` / `football-analytics project check` — profiles
`local`/`ci`, modes `quick`/`deep`, statuses PASS/WARN/FAIL/SKIP, exit codes
0/1/2/3. Reports under `/home/fdoblak/workspace/project_checks/`.

## 17. CI workflow

`.github/workflows/ci.yml` — push/PR/`workflow_dispatch` on `main`;
`permissions: contents: read`; quality + tests jobs; no GPU/external clones.

## 18. Action SHA pins

- `actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803` `# v6`
- `actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97` `# v7`

## 19. CI security validator

`scripts/check_ci_workflow.py` — triggers, least privilege, SHA pins, official
action allowlist, no `pull_request_target`/sudo/curl|bash/deploy/secrets.
Does not call GitHub API.

## 20. New test count

**154** new tests (`tests/pipeline/` + scripts/CLI package tests).

## 21. Total tests

**465** pytest passed (baseline 311 + 154).

## 22. Synthetic E2E

Stage/cache synthetic + corruption + concurrency smokes **PASS** under
`/home/fdoblak/workspace/stage_cache_checks/`.

## 23. Deep project validation

`check_project.py --profile local --deep` → `PASS_WITH_WARNINGS` (dirty tree
pre-commit only); synthetic stages PASS; cleanup PASS.

## 24. Runtime report paths

```text
/home/fdoblak/workspace/stage_cache_checks/stage_cache_validation_20260722T201012Z.json
/home/fdoblak/workspace/project_checks/project_validation_20260722T201106Z.json
```

(Git-outside.)

## 25. Fixture cleanup

Synthetic fixtures/temp cache/locks cleaned after smokes; permanent delete
false; deep `cleanup_verification` PASS.

## 26. Quality tools

ruff / black / isort / mypy / pip check **PASS**.

## 27. Build

`python -m build` **PASS**. Sdist includes pipeline modules, `cache_policy.yaml`,
and pipeline/cache schemas. Excludes `.github`, runtime JSON, cache, quarantine.

## 28. Protected packages

Unchanged: pyarrow 25.0.0, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
torchaudio 2.11.0+cu128, numpy 2.2.6, pandas 2.3.3, opencv-python 5.0.0.93,
opencv-python-headless 5.0.0.93, ultralytics 8.4.91, SoccerNet 0.1.62.

## 29. Changed files (summary)

`src/football_analytics/pipeline/**`, schemas/pipeline|cache, cache_policy,
CLI project/cache helpers, scripts/check_{stage_cache,project,ci_workflow}.py,
`.github/workflows/ci.yml`, `requirements/ci.txt`, tests, docs
(development/stages/decisions/risks), README.

## 30. Findings (open; not closed)

- Cursor `api.github.com` proxy 403 (carry-over)
- Remote CI run status `UNVERIFIABLE_AGENT_API_CONTEXT`
- Registry license/access warnings (carry-over)
- GPU `AGENT_CONTEXT_GPU_UNVERIFIABLE` / Stage 5 host gate (carry-over)
- Same-VHDX archive is not an independent backup (carry-over)
- RISK-029 validation memory (pylist) carry-over
- Cache GC / automatic purge absent by design (RISK-041)

## 31. Acceptance

All Stage 2D acceptance criteria met locally. Remote CI API visibility is
finding `UNVERIFIABLE_AGENT_API_CONTEXT`, not alone NO-GO.

## 32. Gate

`PASS_WITH_FINDINGS — STAGE 2 FOUNDATION COMPLETE`

## 33. Foundation tag

Annotated tag target: `foundation-v0.1.0` (`Stage 2 foundation complete`).
Not a production release; package version remains `0.1.0.dev0`.
Tag create/push: performed after successful main push (see close report).

## 34. Next stage (name only)

Aşama 3 — Güvenli Video Ingest, Probe, Normalize ve Frame Zaman Tabanı
