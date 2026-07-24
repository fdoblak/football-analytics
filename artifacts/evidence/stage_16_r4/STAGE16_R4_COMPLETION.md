# Stage 16-R4 — Self-contained technical acceptance

| Field | Value |
|-------|--------|
| Stage | 16-R4 |
| Gate | **`PASS_WITH_FINDINGS — SELF-CONTAINED TECHNICAL ACCEPTANCE COMPLETE; REAL-VIDEO TRACKING VALIDATED; VIDEO-EVENT ACCURACY NOT VALIDATED`** |
| Tag | `single-player-analytics-technical-preview-v0.16.0` (**not** `single-player-analytics-v1.0.0`) |
| Baseline HEAD | `8bd7b78d1f3cd20ac1eeb1da859f0ad8728e4486` |

## Decision

Gated/private SoccerTrack v2 video is **not** an active product or release dependency.
Missing inputs are not waited on forever. Three independent evidence tracks:

| Track | Namespace | Role |
|-------|-----------|------|
| A | `teamtrack_real_video_pilot` | Real-video CV proof (ingest/decode/GPU/detection/tracking) |
| B | `soccertrack_v2_reference_analysis` | Annotation-derived reference from licensed GSR/BAS (not video prediction) |
| C | `self_contained_deterministic_acceptance` | Offline deterministic contract/metric/report E2E |

Cross-namespace mixing hard-fails. Annotation-derived values are never presented as model predictions.

## Authoritative SoccerTrack target (unchanged)

```text
match_id: 128057
team_side: left
jersey_number: 24
player_id: 506469
```

Deprecated invalid target `jersey 11 / player_id 506466` remains refused.

TeamTrack pilot target (`F_20200220_1_0330_0360` / track 7) is a **separate person** and must not be merged.

## HF / Drive

- Hugging Face account/token **not required** for CLI, tests, build, or release gates.
- Google Drive quota retry loops are not part of the release path.
- SoccerTrack v2 panoramic MP4 = **optional external validation source** only.

## Final deliverables

- `artifacts/final/single_player_analysis_summary.json`
- `artifacts/final/single_player_analysis_summary.png` (SHA-equal twin under `/home/fdoblak/football_data/rendered_outputs/final/`)

Disclaimers on report/PNG:

```text
TECHNICAL PREVIEW
REFERENCE-ANNOTATION-DERIVED
VIDEO EVENT-INFERENCE ACCURACY NOT VALIDATED
NOT OFFICIAL OPTA DATA
```

## Offline CLI

```text
football-analytics acceptance generate
football-analytics acceptance run
football-analytics acceptance validate
football-analytics acceptance reference soccertrack-v2
football-analytics report render-final
```

Default: no network, no download, no token, no purchase, no gated source requirement.
