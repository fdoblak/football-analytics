# Production readiness checklist (post Stage 15)

Machine-local implementation is complete through Stage 15. This checklist separates
**ready now** from **Stage 16 / external**.

## Ready (Stage 15)

- [x] Single-player orchestration / review / report / renderer (synthetic)
- [x] Bounded materialization policy (RISK-029)
- [x] Cache GC dry-run / quarantine (RISK-041); no automatic permanent delete
- [x] Disk-space gates, concurrency locks, failure receipts, interrupted-run recovery
- [x] RTX 3050 4GB bounded batch profile + CPU fallback
- [x] Agent GPU marked `AGENT_CONTEXT_GPU_UNVERIFIABLE`
- [x] No-network download defaults
- [x] Secret/log redaction + large artifact prevention for evidence
- [x] AGPL/GPL adapters gated `evaluation_only` / `production_approved=false`
- [x] License inventory + third-party notices (technical only)
- [x] Storage readiness without claiming `/mnt/d`
- [x] Local CI-equivalent + workflow SHA pin validator
- [x] Synthetic acceptance (positive/negative/ambiguous/low-coverage/not_evaluable)

## Not ready — Stage 16 / external

- [ ] Reviewed real-match ground truth and accuracy claims
- [ ] Legal clearance for AGPL/GPL redistribution or network service
- [ ] Model weight redistribution clearance (`review_required` rows)
- [ ] Independent off-device backup (verified `/mnt/d` or equivalent)
- [ ] Remote GitHub Actions green status verified outside agent API proxy
- [ ] Final customer visual written to reserved Stage 16 paths
- [ ] Manual identity workflow exercised on real video
- [ ] Opta official import path (license-only; no scraping)

## Do not claim

- Real football accuracy from synthetic fixtures
- Production approval for evaluation-only models
- Independent backup while `independent_backup: false`
- Green remote CI when status is `UNVERIFIABLE_AGENT_API_CONTEXT`
