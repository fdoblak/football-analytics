# Stage 1C — Dataset/Model Registry, License, NDA & Secret Policy

**Gate:** `PASS_WITH_FINDINGS — GOVERNANCE ACTIVE`
**Date:** 2026-07-22
**Commit message (target):** `Add registries and data governance controls`

## 1. Amaç

Sürümlü model/dataset registry, external lock genişletmesi (22 repo), lisans/erişim envanteri, secret politikası, registry validator ve secret scanner — **indirme veya secret üretmeden**.

## 2. Başlangıç checkpoint’i

| Alan | Değer |
|---|---|
| Branch | `main` |
| HEAD | `080839241666e45919d98fe00b4bb9cefcf99a26` |
| Message | Add storage validation and path safety checks |
| Working tree | clean (precondition) |
| Storage | `wsl_local` → `/home/fdoblak/football_data`; archive `/mnt/d/football_data` unverified |

## 3. Model registry özeti

- Dosya: `model_registry.yaml` (`schema_version: 1`)
- Schema: `schemas/registries/model_registry.schema.json`
- Kayıt: 2 model (`sn_banner_sv_kp`, `sn_banner_sv_lines`)
- Durum: `available` (dosya bütünlüğü); **inference edilmedi** (`tested: false`, `test_scope: file_integrity_only`)
- `license_status: review_required` (ağırlık lisansı kod lisansından türetilmedi)

## 4. İki modelin full hash/size doğrulaması

| ID | Path | size_bytes | sha256 |
|---|---|---|---|
| sn_banner_sv_kp | `/home/fdoblak/models/soccernet/sn-banner/SV_kp.pth` | 264964645 | `7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113` |
| sn_banner_sv_lines | `/home/fdoblak/models/soccernet/sn-banner/SV_lines.pth` | 264857893 | `2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1` |

Kaynak: `no_bells_just_whistles` @ `bd993b31c2917096c23bb8aadf148314d17f8345` (lock eşleşmesi doğrulandı). Dosyalar taşınmadı/kopyalanmadı.

## 5. Dataset registry özeti

- Dosya: `dataset_registry.yaml` (`schema_version: 1`)
- Schema: `schemas/registries/dataset_registry.schema.json`
- 8 kayıt: `golden_clips` + 7 SoccerNet task ailesi
- Hepsi `planned` / `not_downloaded`; `local_path: null`; `checksum: null`

## 6. Datasetlerin indirilmediği doğrulaması

Broadcast/match corpus yok. Demo MP4’ler registry’de yok. Validator `available`/`verified` varlık iddiası üretmedi.

## 7. External lock genişletmesi

`external_repos.lock.yaml` içine `third_party_repositories:` eklendi. Mevcut 19 SoccerNet `repositories:` kaydı korunmuştur (minimal migration: yeni grup + `updated_at` / `update_notes`).

## 8. 22 repo lock sonucu

| Grup | Sayı | HEAD↔lock |
|---|---|---|
| SoccerNet | 19 | 19/19 match, dirty=false |
| Third-party | 3 | 3/3 match, dirty=false |
| **Toplam** | **22** | **22/22** |

Third-party:

- tracklab `5767e86c…` tag `v1.3.24`
- pnlcalib `8c87391d…`
- no_bells_just_whistles `bd993b31…`

Fetch/pull yapılmadı.

## 9. Lisans envanteri

`docs/legal/license_inventory.md` — kod LICENSE kanıtı; eksik LICENSE → `review_required`. Dataset/model lisansları karıştırılmadı.

## 10. Data access matrix

`docs/data/data_access_matrix.md` — NDA yalnız broadcast için ayrı satır; tüm SoccerNet task’lara genellenmedi.

## 11. Secret policy

- `docs/security/secrets_policy.md`
- `configs/security/secret_policy.yaml`
- Runtime: env / gitignored `.env` / OS store; registry’ye secret yok

## 12. `.gitignore` ve `.env.example`

- `.gitignore`: secrets, caches, `*.pth`/`*.mp4`/… + `!tests/fixtures/**` allowlist (doğrulandı)
- `.env.example`: yalnız boş placeholder değişken adları

## 13. Registry validator

`scripts/check_registries.py` — `--verify-files`, `--verify-repos`, exit 0/1/2/3, atomik JSON.

## 14. Secret scanner

`scripts/check_secrets.py` — redaction, binary/large/symlink skip, staged mode.

## 15. Test listesi ve sonuçları

| Suite | Count | Result |
|---|---|---|
| `tests/storage/test_check_storage.py` | 22 | PASS |
| `tests/registries/test_check_registries.py` | 22 | PASS |
| `tests/security/test_check_secrets.py` | 16 | PASS |
| **Total** | **60** | **PASS** |

## 16. Gerçek runtime validation

| Check | Result |
|---|---|
| `check_storage.py` | PASS |
| `check_registries.py --verify-files --verify-repos` | PASS_WITH_WARNINGS (license/access review) |
| External repos verified | 22/22 |
| Model hash/size | exact match |
| `check_secrets.py` | PASS, findings=0 |

## 17. Runtime JSON yolları (Git dışı)

- `/home/fdoblak/workspace/registry_checks/registry_validation_20260722T142701Z.json`
- `/home/fdoblak/workspace/security_checks/secret_scan_20260722T142701Z.json`

(Önceki deneme: `…T142354Z.json` — secret scanner FP düzeltmesi sonrası güncel olan `…T142701Z`.)

## 18. Findings / review-required

- Model weight redistribution license (SV_*.pth)
- SoccerNet dataset access/license per task (`access_level: unknown`)
- Repos without local LICENSE: sn-calibration, sn-caption, sn-depth, sn-echoes, sn-jersey, sn-nvs, sn-tracking
- PTS-baseline BSD-style SPDX confirmation
- GPL isolation for pnlcalib / NBJW / several sn-* GPL trees
- Archive `/mnt/d` still unverified (Stage 1A-R1 carry-over; not Stage 1C blocker)

## 19. Değişen dosyalar

Registries, schemas, validators, tests, legal/security docs, `.gitignore`, `.env.example`, lock third-party group, `audit_current_state.py` dual-group lock audit, this stage doc, `THIRD_PARTY_NOTICES.md`.

## 20. Kabul kriterleri

Hepsi geçti (lisans review maddeleri bilerek açık). Paket/dataset/model indirme yok; Stage 1D başlatılmadı.

## 21. Gate kararı

**`PASS_WITH_FINDINGS — GOVERNANCE ACTIVE`**

Teknik kontroller geçti; kritik olmayan lisans/access review maddeleri belgelendi.

## 22. Sonraki aşama

**Aşama 1D — Arşivleme, Checksum, Cleanup ve Aşama 1 Kapanışı**
