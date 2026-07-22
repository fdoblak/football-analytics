# Stage cache design (Stage 2D)

Local content-addressed cache for deterministic, cacheable stages.
Policy: `configs/system/cache_policy.yaml`. Root: `system.cache` from
`configs/system/paths.yaml` (default `/home/fdoblak/workspace/cache`).

## Key composition

`compute_cache_key` → SHA-256 of canonical JSON (`CACHE_KEY_VERSION = 1`).

**Included:**

- `cache_key_version`
- stage `name` / `version` / `code_fingerprint` / `deterministic` / `cacheable`
- stage `output_contracts`
- `config_fingerprint`, `compatibility_fingerprint`
- inputs sorted by logical name: `sha256`, `size_bytes`, contract fields,
  `media_type`, `relative_path`

**Excluded:** absolute paths, username/hostname, timestamps, secrets,
`working_directory`, `output_directory`, `run_id`, `requested_at_utc`.

Key format: 64 lowercase hex. Same semantic inputs ⇒ same key; any byte/schema/
config/fingerprint change ⇒ different key.

## Layout (v1)

```text
<cache_root>/v1/sha256/<ab>/<remaining62>/
  cache_manifest.json
  stage_result.json
  artifacts/…
<cache_root>/v1/locks/<ab>/<remaining62>.lock
```

- Fan-out: first two hex chars of key, then remaining 62
- Entries are immutable; never overwrite an existing final entry
- Partial publishes live under `.tmp_publish_*` until atomic rename
- No absolute source paths stored in manifests
- Dir mode `0700`, file mode `0600`

## Atomic publish

`publish_cache_entry`:

1. Refuse if policy disabled
2. If final exists → return it (caller re-verifies); no overwrite
3. Stage artifacts into temp dir (copy, never hardlink)
4. Re-verify hashes when `verify_on_publish`
5. Enforce `max_entry_files` / `max_entry_bytes` / `max_manifest_bytes`
6. Write `cache_manifest.json` + `stage_result.json` via atomic JSON writer
7. Under lock: rename temp → final (same filesystem)

Failed publish cleans the temp tree.

## Locking

`acquire_key_lock` uses exclusive non-blocking `fcntl.flock` with bounded
`lock_timeout_seconds` (default 30). Lock files live under `v1/locks/`.

## Integrity

On read (`verify_on_read`):

- Manifest/result presence and size bounds
- Stage name/version + config fingerprint match
- Recomputed cache key must equal entry key
- Every listed artifact hash/size verified; unexpected files rejected
- Symlinks, special files, and hardlinks rejected per policy

## Restore

`restore_cache_entry` verifies then copies artifacts into `output_directory`
with **no overwrite**. Partial restore cleans files it created.

## Corruption / quarantine

Corrupt hits during `execute_stage` may call `quarantine_cache_entry` when
`quarantine_corrupt_entries` is true and a `quarantine_root` is provided.

- Move entry to quarantine (never permanent delete)
- Write `quarantine_receipt.json` with `permanent_delete_performed: false`
- Treat as cache miss and re-execute

## No automatic purge

`automatic_purge` **must** be `false` in Stage 2D. Policy load fails otherwise.
There is no GC; disk growth is an acknowledged limitation (RISK-041).

## Trust model

- Local filesystem trust only (not a shared remote CAS)
- Integrity rests on SHA-256 of artifact bytes + key recomputation
- Cache does not authenticate writers beyond process OS permissions
- Symlink/hardlink/special-file rejection reduces escape/poisoning surface
- Quarantine preserves evidence; operators decide retention

## Limitations

- No cross-host sharing or signed manifests
- No automatic eviction / size quotas beyond publish-time caps
- Concurrent publishers: first rename wins; loser discards temp
- Non-deterministic or non-cacheable stages never publish
- Failed executions never publish
