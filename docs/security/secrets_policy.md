# Secrets policy (Stage 1C)

Technical policy for credentials and sensitive material in `football-analytics`.
**Do not store real secrets in this repository.**

## In scope

| Class | Examples | Allowed storage | Forbidden |
|---|---|---|---|
| SoccerNet credential | account password / download auth | env / gitignored `.env` / OS store | YAML registries, docs, commits |
| NDA access password | broadcast portal password | env / OS store | registries, logs, screenshots |
| Google Drive token/cookie | OAuth token, cookie jar | path via env to **gitignored** file | tracked JSON, URL query |
| Hugging Face token | `HUGGINGFACE_TOKEN` | env | model/dataset registries |
| GitHub token | `ghp_*`, `github_pat_*` | env / credential helper | remotes with embedded token |
| API key | third-party APIs | env | manifests, reports |
| Private key | `*.pem`, `*.key` | OS store / encrypted volume | git, chat logs |
| `.env` | local overrides | gitignored only | tracked except `.env.example` |
| Personal path/identity | home paths in public paste | minimize; redact in shared reports | unnecessary PII in public docs |

## Runtime approach

1. Prefer **environment variables**.
2. Optional **gitignored** `.env` (never commit).
3. Prefer **OS credential store** when available.
4. **No secrets** in `model_registry.yaml`, `dataset_registry.yaml`, `external_repos.lock.yaml`, logs, or manifests.
5. Use `.env.example` for **names only** (empty values).

## Redaction

- Logs: redact tokens/passwords before write.
- Screenshots/reports: blur or omit credential UI; scrub URL query secrets.
- Secret scanner evidence: show prefix/suffix only (`scripts/check_secrets.py`).

## Rotation

- Rotate on suspected exposure, staff change, or owner schedule.
- After rotation, revoke old credentials at the provider.

## Accidental commit procedure

1. **Stop** further pushes of the affected commit.
2. **Rotate** the exposed credential immediately.
3. Remove from working tree; ensure `.gitignore` covers the path.
4. History purge only with **owner approval** (not performed by Stage 1C automation).
5. Re-scan with `scripts/check_secrets.py`.

## Scanner

```bash
python scripts/check_secrets.py --root /home/fdoblak/projects/football-analytics
```

Staged mode before commit:

```bash
python scripts/check_secrets.py --root /home/fdoblak/projects/football-analytics --staged
```

Index: `configs/security/secret_policy.yaml`.
