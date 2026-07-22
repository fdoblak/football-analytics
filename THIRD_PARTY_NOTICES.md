# Third-party notices (technical inventory)

This document is a **technical inventory** of external repositories referenced by
`football-analytics`. It is **not** a warranty, license grant, or legal advice.

## Policy

- Prefer **adapter / subprocess / isolated environment** integration over vendoring.
- **No project-internal copy** of third-party application source was performed in Stage 1C.
- Exact pins live in `external_repos.lock.yaml`.
- Code license ≠ dataset license ≠ model weight license.

## Used / runtime candidates

| Name | Source URL | Locked commit | License evidence | Notes |
|---|---|---|---|---|
| TrackLab | https://github.com/TrackingLaboratory/tracklab.git | `5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2` (tag `v1.3.24`) | `LICENSE` → MIT | `integration: runtime_and_reference` |
| PnLCalib | https://github.com/mguti97/PnLCalib.git | `8c87391d6f4ea40c5e4d65e61529916c7a49ce62` | `LICENSE` → GPL-2.0 | `integration: calibration_candidate` |
| No-Bells-Just-Whistles | https://github.com/mguti97/No-Bells-Just-Whistles.git | `bd993b31c2917096c23bb8aadf148314d17f8345` | `LICENSE` → GPL-2.0 | Source of local SV_*.pth weights; **weight license review_required** |

## SoccerNet ecosystem (reference / baseline)

Nineteen SoccerNet organization repositories are locked under `repositories:` in
`external_repos.lock.yaml`. See `docs/legal/license_inventory.md` for per-repo
LICENSE paths and `review_required` gaps (missing LICENSE files).

## Ambiguities (explicit)

- Repos **without** a local LICENSE file: sn-calibration, sn-caption, sn-depth,
  sn-echoes, sn-jersey, sn-nvs, sn-tracking → `review_required`.
- PTS-baseline LICENSE is BSD-style text without a confirmed SPDX label in this inventory.
- Model checkpoints `SV_kp.pth` / `SV_lines.pth` are **not** cleared for redistribution
  solely because the NBJW repository is GPL-2.0.

## Adapter / isolation

Until compliance review completes for copyleft components, treat GPL-licensed trees as
**isolated reference or subprocess** candidates; do not casually merge their source into
this project's distribution boundary.
