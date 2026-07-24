# Stage 16 / 16-R1 Completion

| Field | Value |
|-------|-------|
| Stage | 16-R1 (resume from Stage 16 NO-GO) |
| Date | 2026-07-24 |
| Start HEAD | `8d8370f3881d307990d92c7b34c58fd77dd9d75a` |
| Gate | **`NO-GO — OFFICIAL SOCCERTRACK VIDEO UNAVAILABLE`** |
| Tag `single-player-analytics-v1.0.0` | **Not created** |

## Resume baseline

- Branch `main`, clean tree, local = origin = ls-remote at start SHA
- External SoccerTrack-v2 lock: `3ee38e481aab9de0f1d099c1cdde15302eb63f49` (clean)
- Match `128057` GSR/BAS/raw/license checksums verified against `source_manifest.json` (PASS)
- Target retained: Team left / Jersey 24 / `player_id` 506469
- Acceptance adapter tests: 13 passed
- GSR/BAS **not** re-downloaded or deleted

## Video recovery attempts (official sources only)

### Primary — Hugging Face `atomscott/soccertrack-v2`

- Anonymous `dataset_info` / API / resolve URLs → **HTTP 401** (`Invalid username or password` / page title 404)
- Repo not listable; author/search counts 0
- Could **not** pin dataset revision SHA or enumerate exact LFS/Xet video paths
- No HF token requested or used; TLS verification left on
- Tools: `huggingface_hub` 1.23.0 present; `hf` / `huggingface-cli` not on PATH

### Secondary — Official Google Drive mirror

- Root folder listable; `videos/` folder id `1Wy1LgQPm9hW4aOQLw2Z3newzuzlKXcNv`
- Match `128057` video folder id `1HJ5an5M45BjzFwJMR5y8N0KoyrSTLnKh`
- Exact files discovered:
  - `videos/128057/128057_panorama_1st_half.mp4` → Drive id `1A2y5s0xgxU7yedVgqTW3bgLTZWfJO05C`
  - `videos/128057/128057_panorama_2nd_half.mp4` → Drive id `1Vah7favjAsm5yYCjU4ic8Vlcyi0whnQ4`
- Virus-scan confirm form obtainable, but download body → **Quota exceeded**
- One-shot `gdown` → **Too many users have viewed or downloaded this file recently**
- Stopped further Drive retries (no quota bypass)

### Match switch

- 128057 panoramic MP4s **are published** on the official Drive mirror but **not downloadable** under quota
- HF tree unavailable → cannot verify alternate match video availability on canonical HF
- No unauthorized third-party video sources used

## Not completed (blocked on official video)

- Pilot / full-match pipeline
- Held-out prediction evaluation against live predictions
- Final report JSON / dual-path PNG
- Release tag
- Post-success video cleanup (nothing downloaded)

## Preserved for future resume

- Local GSR/BAS/raw/license under `/home/fdoblak/football_data/datasets/soccertrack_v2/source/`
- Run namespaces under `.../runs/128057/`
- Adapter, leakage tests, provenance, prior Stage 16 NO-GO git history
- Receipt: `artifacts/evidence/stage_16/video_download_recovery_r1.json`
- Manifest revision history retains original Drive-quota NO-GO entry

## Explicit non-claims

- No fake pipeline accuracy
- No synthetic final PNG
- No Opta claim
- No unauthorized video mirror
