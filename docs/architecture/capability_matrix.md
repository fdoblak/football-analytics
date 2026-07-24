# Capability matrix — Single-player product (through Stage 15)

| Capability | Stage | Status | Accuracy claim |
|------------|-------|--------|----------------|
| Storage / archive / cleanup | 1 | CLOSED | N/A |
| Runtime / cache / CI foundation | 2 | CLOSED | N/A |
| Video ingest / normalize / timeline | 3 | CLOSED | Operational only |
| Broadcast shots / camera / routing | 4 | CLOSED | Synthetic only |
| Human / ball / role detection | 5 | CLOSED | NOT_EVALUATED real GT |
| Human / ball tracking | 6 | CLOSED | NOT_EVALUATED real GT |
| Identity / ReID / jersey / target | 7 | CLOSED | Manual confirm required when uncertain |
| Pitch features / homography / projection | 8 | CLOSED | NOT_EVALUATED real GT |
| Physical distance / speed / sprint / heatmap | 9 | CLOSED | NOT_EVALUATED real GT |
| Human–ball proximity / possession | 10 | CLOSED | NOT_EVALUATED real GT |
| Passing / reception / progression | 11 | CLOSED | NOT_EVALUATED real GT |
| Duels / take-on / tackle / aerial / clearance | 12 | CLOSED | NOT_EVALUATED real GT |
| Target event ledger / metrics | 13 | CLOSED | NOT_EVALUATED real GT |
| E2E orchestration / review / report / render | 14 | CLOSED | Synthetic E2E only |
| Pre-release hardening (15A–15G) | 15 | CLOSED | Machine-local gates only |
| Real-match acceptance / final visual | 16 | **OPEN** | Required before production claims |

## Hardening capabilities (Stage 15)

| Capability | Module | Default |
|------------|--------|---------|
| Bounded pylist / streaming | `hardening.materialize` | max 50k rows |
| Cache GC | `hardening.cache_gc` | dry-run; quarantine optional |
| Disk gates | `hardening.disk_gate` | 2 GiB pipeline / 1 GiB cache |
| GPU profile | `hardening.gpu_profile` | RTX 3050 batch=1; Agent GPU unverifiable |
| Network | `hardening.network` | downloads denied |
| Licensing gates | `hardening.licensing` | `production_approved=false` |
| Storage readiness | `hardening.storage_readiness` | `/mnt/d` not claimed |
| CI parity | `hardening.ci_parity` | local deep validation |

See also: `docs/architecture/single_player_pipeline.md`, `docs/operations/production_readiness_checklist.md`.
