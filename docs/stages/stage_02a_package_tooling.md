# Stage 2A — Python package, tooling, GitHub sync

**Date:** 2026-07-22
**Start HEAD:** `015debd77b5936274bdf8d65d3113242e05506e4`
**Target commit:** `Establish Python package and development tooling`

## Delivered

- `pyproject.toml` (setuptools, version `0.1.0.dev0`, console script)
- Package CLI: `--version`, `info` (side-effect free)
- `requirements/{base,dev,constraints-ai-dev}.txt`, `environment.yml`
- README, LICENSE (proprietary), `.editorconfig`, `.gitignore` build ignores
- Package smoke tests (`tests/package/`)
- Dev docs + dependency change report
- Editable install + ruff/black/isort/mypy/build

## Environment

Installed into `ai-dev` only (after safe dry-run): pyarrow 25.0.0, pytest 9.1.1, pytest-cov 7.1.0, ruff 0.15.22, black 26.5.1, isort 8.0.1, mypy 2.3.0, build 1.5.0.

Protected Torch/NumPy/OpenCV/Ultralytics/SoccerNet pins unchanged.

## GitHub

Private `football-analytics` synchronized via Git smart HTTPS after Stage 2A package commit. See resolution below.

## GitHub Synchronization Resolution

**Date:** 2026-07-22 (UTC)

### Blockers resolved and remaining

1. **Authentication blocker (resolved):** Secure GitHub authentication was previously missing for Agent-side sync. The user completed GitHub web authentication in their own environment for account `fdoblak`. Credential helper was configured for Git smart HTTPS without exposing credential contents.
2. **Cursor API proxy blocker (still present):** Cursor Agent access to `api.github.com` remains blocked (HTTP 403 via proxy). Therefore `gh api` / `gh repo create` / `gh repo view` and API-based visibility or metadata verification were not used.
3. **Repository creation (user):** The user created `fdoblak/football-analytics` in the GitHub web UI as **Private**, empty (no README / `.gitignore` / LICENSE initialization).
4. **Transport used:** Git smart HTTPS against `github.com` only (no GitHub REST/GraphQL API).

### Pre-push finding — sn-depth cache

- Registry `--verify-repos` initially failed integrity because external lock entry `sn_depth` (`dirty: false`) had an untracked Python cache directory:
  - `/home/fdoblak/projects/soccernet/sn-depth/baseline/ZoeDepth/zoedepth/data/__pycache__`
- Only `*.pyc` files were present (no symlink/special/non-cache data). The directory was moved with `mv` into a recoverable quarantine under `/home/fdoblak/workspace/quarantine/external_repo_cache/` (permanent delete not performed). Evidence JSON kept outside Git under `/home/fdoblak/workspace/registry_checks/`.
- After quarantine, `sn-depth` working tree was clean; HEAD remained `9f6636fafb11447a5bada765e197928ee9efc467` (lock-aligned).
- Registry revalidation: `PASS_WITH_WARNINGS` (exit 0) with only previously documented license/access review warnings — no dirty-repo / HEAD / checksum integrity errors.

### First push

- Remote added (credential-free URL): `https://github.com/fdoblak/football-analytics.git`
- Remote heads/tags were empty before the first push (`git ls-remote`).
- Command: `git push -u origin main` (no force / force-with-lease / mirror).
- First pushed SHA: `af6b93afadf8208bac501346092f5ef52db7ea12`
- Local `main`, `origin/main`, and `git ls-remote --heads origin refs/heads/main` matched that SHA.
- Stage 0–2A ancestor checkpoint chain was present on `origin/main`.
- No tags or releases created. No dataset/video/model/checkpoint or runtime/build artifacts uploaded. Secret/binary history gate was clean before push.

### Visibility evidence level

- API-based private-visibility verification was **not** possible due to the Cursor `api.github.com` proxy 403.
- Private visibility is recorded from the **user’s confirmation** at web UI creation time, plus successful authenticated Git smart HTTPS push to the private empty repository.

### Stage gate

Stage 2A GitHub synchronization gate is **technically closed** for this repository: history synced, SHA triple-match verified, Stage 2A documentation closure commit follows this section.

## Next

Aşama 2B — Run Kimliği, Config, Logging, Hash ve Environment Kayıtları
