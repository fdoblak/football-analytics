# Stage 16-R4-FIX2 — Portable media repair

## Root cause (old MP4)

| Issue | Detail |
|-------|--------|
| Atom order | `mdat` before `moov` (no faststart) |
| Profile/level | High @ Level 5.0 |
| Resolution | 4680×720 ultra-wide |
| Effect | Windows / some players fail seek/play despite Linux ffmpeg decode |

## Root cause (old PNG)

8-bit **RGBA** (alpha). Rebuilt as 8-bit **RGB**.

## New profile

H.264/AVC `avc1`, `yuv420p`, Main@L4.0, CFR 25, `+faststart`, 1280×720 letterboxed, ~30s, ~0.86 MiB.

Gate: `PASS_WITH_FINDINGS — PORTABLE FINAL DELIVERY VALIDATED; VIDEO-EVENT ACCURACY NOT VALIDATED`
