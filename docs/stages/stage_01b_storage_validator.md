# Stage 1B — Folder Standard, paths.yaml, and Storage Validator

| Field | Value |
|-------|-------|
| Stage | 1B |
| Date | 2026-07-22 |
| Gate decision | **PASS — STORAGE CONTRACT VALIDATED** |
| Next stage started? | **No** |

## 1. Amaç

Make `paths.yaml` an enforceable storage contract, ship a read-only-by-default validator with opt-in probe, add offline unit tests, and prove the live WSL local backend.

## 2. Başlangıç checkpoint’i

| Field | Value |
|-------|-------|
| Branch | `main` |
| HEAD | `6167cb2283723c37d439b0c497bf3b210fca6f01` |
| Message | Resolve Stage 1A storage backend |
| Working tree | clean |
| Git gate | PASS |

No contradiction among `paths.yaml`, Stage 1A status JSON, ADR-0002, and `/home/fdoblak/football_data`.

## 3. Config değişiklikleri

`configs/system/paths.yaml`:

- Kept `storage.active_backend: wsl_local` and active leaf paths under `/home/fdoblak/football_data`
- Kept `planned_archive_root` / `planned_archive_status: unverified`
- Added `storage_validation` with integer thresholds:
  - `minimum_free_bytes: 21474836480`
  - `warning_free_bytes: 107374182400`
  - absolute / under-root / symlink-escape flags

## 4. Validator tasarımı

`scripts/check_storage.py`:

- CLI: `--config`, `--json-out`, `--probe`, `--strict`, `--quiet`
- Default: fully read-only
- Validates backend allow-list, required keys, absolute paths, forbidden broad roots, traversal, symlink escape, duplicates, directory existence
- Capacity gate PASS/WARNING/FAIL
- Probe: `O_EXCL`, fsync, SHA-256, exact-path cleanup in `finally`
- Atomic JSON write (temp + fsync + replace); refuses overwrite
- Exit codes 0/1/2/3
- Reports numeric UID/GID without auto “fixing” Agent NS `root` appearance

## 5. Güvenlik invariantları

- No mounts, chmod/chown, recursive deletes, package installs
- Probe only under active root; no overwrite of existing files
- Planned archive not treated as active
- Runtime reports live under workspace, not Git

## 6. Test listesi

`tests/storage/test_check_storage.py` (unittest, 22 cases):

1. Valid tree PASS
2. Relative active_root rejected
3. `/` rejected
4. `/home/fdoblak` rejected
5. Missing required key rejected
6. Outside-root path rejected
7. `..` escape rejected
8. Symlink escape rejected
9. Duplicate canonical path rejected
10. Unknown backend rejected
11. String threshold rejected
12. minimum ≥ warning rejected
13. Missing directory FAIL
14. Read-only default creates no files
15. Probe write/read/hash/cleanup PASS
16. Probe refuses overwrite
17. No probe residue
18. Missing unverified archive OK
19. JSON output valid
20. Exit code contract
21. World-writable warning
22. Broken YAML → config error

## 7. Test sonuçları

```text
python -m unittest discover -s tests -p "test_*.py" -v
Ran 22 tests ... OK
```

## 8. Gerçek salt-okunur validation

```text
status=PASS exit_code=0
active_backend=wsl_local
active_root=/home/fdoblak/football_data
capacity=PASS free_bytes≈993517658112
```

All required path statuses: PASS.

## 9. Gerçek probe validation

```text
status=PASS exit_code=0
probe_passed=True cleanup_verified=True
sha256=e4104e0eab8c37f6a458c9e0092c3cf15e6a42c30c09a2beb63ecf37ba1a60bc
```

## 10. Runtime JSON rapor yolu

```text
/home/fdoblak/workspace/storage_checks/storage_validation_20260722T140104Z.json
```

Not staged / not committed.

## 11. Probe cleanup kanıtı

`find ... -name '.storage_probe_*'` count = **0** under active root after probe run.

## 12. Bulgular / warnings

None on live validation (`warnings=[]`, `errors=[]`).

## 13. Değişen / eklenen dosyalar

- `configs/system/paths.yaml`
- `scripts/check_storage.py`
- `tests/__init__.py`
- `tests/storage/__init__.py`
- `tests/storage/test_check_storage.py`
- `docs/storage/storage_layout.md`
- `docs/stages/stage_01b_storage_validator.md`

## 14. Kabul kriterleri

All Stage 1B acceptance checks met (see final report table).

## 15. Gate kararı

**PASS — STORAGE CONTRACT VALIDATED**

## 16. Sonraki aşama — yalnız isim

**Aşama 1C — Dataset/Model Registry, Lisans, NDA ve Secret Politikası**
