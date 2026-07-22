# Data access matrix (Stage 1C)

Planned / not-downloaded datasets only. **No local corpus claimed.**
Broadcast video NDA is **not** generalized to every SoccerNet task.

| Dataset/task | Use in project | Current status | Access level | NDA | Credentials | License status | Redistribution | Local path | Planned stage | Evidence/notes |
|---|---|---|---|---|---|---|---|---|---|---|
| golden_clips | Smoke / regression fixtures | planned | restricted | no | no | review_required | no | null | MVP-1 | Owner-selected short clips; not present |
| soccernet_tracking | MOT / TrackLab path | not_downloaded | unknown | unknown (not assumed) | unknown | review_required | unknown | null | MVP-1 | sn-tracking code locked; data via SoccerNet download path — confirm per-asset terms |
| soccernet_field_localization_calibration | Camera calibration | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | MVP-2 | sn-calibration reference; no LICENSE in code repo |
| soccernet_game_state | Game state reconstruction | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | MVP-2 | sn-gamestate / TrackLab; Zenodo weights noted historically — not downloaded here |
| soccernet_reid_thumbnails | Player re-ID | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | identity | sn-reid; within-action IDs only |
| soccernet_jersey_number | Jersey OCR / classification | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | identity | sn-jersey mostly challenge kit |
| soccernet_ball_action_spotting | Ball action spotting | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | MVP-3 | sn-spotting / PTS-baseline ecosystem |
| soccernet_team_ball_action_spotting | Team-conditioned spotting | not_downloaded | unknown | unknown | unknown | review_required | unknown | null | MVP-3 | sn-teamspotting |
| SoccerNet broadcast full matches *(not registered as available)* | Optional broadcast research | not_downloaded | nda_required *(when used)* | **yes if broadcast corpus** | typically registration | review_required | no | null | later / optional | Do **not** treat as downloaded; NDA applies to broadcast access — **not** auto-applied to all task packs |

**Explicit non-claims**

- Repo-internal demo MP4s are **not** project datasets.
- `available` / `verified` not used for any row above.
- No credentials stored in registries.
