# Final delivery — Single Player Technical Preview

This folder is the **only** current customer-facing output set for the technical preview.

## Files

| File | Purpose |
|------|---------|
| `single_player_analysis_summary.png` | One consolidated technical preview visual |
| `single_player_analysis_summary.json` | Canonical metrics + evidence levels |
| `real_video_tracking_proof.mp4` | Annotated TeamTrack real-video tracking proof (≤30s) |
| `evidence_manifest.json` | Claim → artifact binding |
| `checksums.sha256` | SHA-256 of delivery binaries/JSON |
| `cleanup_manifest.json` | Safe cleanup receipt (`data_loss=false`) |

## How to read the PNG

1. **Header** — SoccerTrack v2 Match 128057 / left / jersey 24 / player 506469 (annotation-derived report target).
2. **Four frames** — TeamTrack real-video tracking proof (Track 7). Different dataset/person than SoccerTrack.
3. **Left metrics panel** — detection P/R/F1, target coverage, mean IoU, device/runtime from TeamTrack.
4. **Right metrics panel** — SoccerTrack GSR/BAS annotation-derived values (not video predictions).
5. **NOT EVALUABLE** list — do not treat as zero.

## MP4 color legend

- cyan — matched prediction
- green — ground-truth Track 7
- yellow — target prediction associated to Track 7
- red — unmatched prediction

Banner: `REAL TEAMTRACK VIDEO — TRACKING PILOT` (no SoccerTrack jersey labels).

## What is real-video validated

TeamTrack `F_20200220_1_0330_0360`: ingest, GPU detection/tracking, target coverage/IoU, detection F1.

## What is annotation-derived

SoccerTrack v2 match 128057 GSR/BAS for player 506469 (distance/speed/sprint/BAS counts, etc.).

## What is not validated

Pass accuracy, duel win rate, failed dribbles, clearances, possession, box touches, and **video-event inference accuracy**. Not official Opta data.

## License / attribution

- TeamTrack — MIT (Atom Scott et al.)
- SoccerTrack v2 annotations — CC BY 4.0 (Atom Scott et al.)
- Project technical preview; not Opta.

## SHA check

```bash
sha256sum -c checksums.sha256
```
