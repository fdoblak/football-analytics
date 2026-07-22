# License inventory (Stage 1C)

Technical inventory of **local repository code licenses** for locked external sources.
This is **not** a legal opinion and **not** a dataset/model redistribution clearance.

**Rules applied**

- Evidence = local `LICENSE` / `LICENSE.txt` / `COPYING` / README pointers only.
- No SPDX invented when a LICENSE file is missing.
- Repo code license ≠ SoccerNet challenge dataset license ≠ model weight license.
- Broadcast/NDA access is **not** inferred from code LICENSE files.

## SoccerNet ecosystem (19)

| Repo (lock id) | Full commit | Remote | License file | Detected license/SPDX | License status | Integration role | Redistribution note | Review requirement |
|---|---|---|---|---|---|---|---|---|
| active_spotting | `33a81cb834978ee474ecec0c5a76b6f3f99b4bf4` | https://github.com/SoccerNet/ActiveSpotting.git | `…/ActiveSpotting/LICENSE` | MIT | present | action_spotting_baseline | Code MIT; datasets separate | Confirm dataset terms before publish |
| pts_baseline | `af2ea8234e0c887758ef674071da2a17e2bc6c61` | https://github.com/SoccerNet/PTS-baseline.git | `…/PTS-baseline/LICENSE` | BSD-style (LICENSE text) | review_required | ball_action_spotting_baseline | Confirm exact SPDX wording | SPDX mapping review |
| soccernet_sdk_source | `74461027ac2095ce2f8d4ee991eccb5dd5f42459` | https://github.com/SoccerNet/SoccerNet.git | `…/SoccerNet/LICENSE` | MIT | present | sdk_source_reference | SDK ≠ data license | Dataset access review |
| soccernet_v3 | `7d483a85ad62b5e98f59427eabee8cb87c710d7b` | https://github.com/SoccerNet/SoccerNet-v3.git | `…/SoccerNet-v3/LICENSE` | MIT | present | annotation_tool_reference | Tool license ≠ annotations | — |
| sn_banner | `f6d50b24a33d6705d4c04dc4d4d93ecd12b08e74` | https://github.com/SoccerNet/sn-banner.git | `…/sn-banner/LICENSE.txt` | GPL-3.0 | present | banner_detection_reference | GPL for code; weights separate | Weight redistribution review |
| sn_calibration | `ab38f461bec729fead86b6986839de1bb826f16d` | https://github.com/SoccerNet/sn-calibration.git | *(none found)* | unknown | review_required | camera_calibration_reference | No local LICENSE file | Obtain license evidence |
| sn_caption | `c05973d4f00853e208d54965f4d6fa47364b8d66` | https://github.com/SoccerNet/sn-caption.git | *(none found)* | unknown | review_required | dense_captioning_reference | No local LICENSE file | Obtain license evidence |
| sn_depth | `9f6636fafb11447a5bada765e197928ee9efc467` | https://github.com/SoccerNet/sn-depth.git | *(none found)* | unknown | review_required | depth_estimation_reference | No local LICENSE file | Obtain license evidence |
| sn_echoes | `7105a85b7a8c1c000a31a30d0c29c388105c3de5` | https://github.com/SoccerNet/sn-echoes.git | *(none found)* | unknown | review_required | audio_commentary_reference | No local LICENSE file | Obtain license evidence |
| sn_gamestate | `1c958345067218297d221e45e1a6405f975f83e0` | https://github.com/SoccerNet/sn-gamestate.git | `…/sn-gamestate/LICENSE` | GPL-3.0 | present | game_state_reference | Code GPL-3; dataset separate | Dataset + TrackLab coupling review |
| sn_grounding | `910bf859ac6d7aff2b80a6d66155956254f24c6b` | https://github.com/SoccerNet/sn-grounding.git | `…/sn-grounding/LICENSE` | MIT | present | grounding_reference | — | — |
| sn_jersey | `2f43b48c59eefe0bb5d948888db07f55f51208ad` | https://github.com/SoccerNet/sn-jersey.git | *(none found)* | unknown | review_required | jersey_devkit | No local LICENSE file | Obtain license evidence |
| sn_mvfoul | `502fb44a76c254e332394f095d54abc830131a44` | https://github.com/SoccerNet/sn-mvfoul.git | `…/sn-mvfoul/LICENSE` | GPL-3.0 | present | mvfoul_reference | — | — |
| sn_nvs | `1655ab19b3bd78f624a96d0f0c27ec2c9f550f61` | https://github.com/SoccerNet/sn-nvs.git | *(none found)* | unknown | review_required | novel_view_synthesis_reference | No local LICENSE file | Obtain license evidence |
| sn_reid | `621e2b0f2d2a7a3e207b8dd747542b6608bf72db` | https://github.com/SoccerNet/sn-reid.git | `…/sn-reid/LICENSE` | MIT (Kaiyang Zhou) | present | reid_reference | Torchreid heritage | Dataset terms separate |
| sn_spotting | `9842826f94e1419580a9d17219c11aca7225f7ce` | https://github.com/SoccerNet/sn-spotting.git | `…/sn-spotting/LICENSE` | MIT | present | action_spotting_reference | — | — |
| sn_teamspotting | `091fed2fc35c33f7489f3596958a2fe385e37d65` | https://github.com/SoccerNet/sn-teamspotting.git | `…/sn-teamspotting/LICENSE` | GPL-3.0 | present | team_action_spotting_devkit | — | — |
| sn_trackeval | `9c25232f6f2b56c9f203f1eb55784ff1e97df683` | https://github.com/SoccerNet/sn-trackeval.git | `…/sn-trackeval/LICENSE` | MIT | present | evaluation_adapter_and_cli | — | — |
| sn_tracking | `b0bbba35e07ff58010b6313ef8aa59ef663ad392` | https://github.com/SoccerNet/sn-tracking.git | *(none found)* | unknown | review_required | dataset_and_reference | No local LICENSE file | Code + tracking data terms |

Exact commits for every row are authoritative in `external_repos.lock.yaml` (`repositories`).

## Third-party (3)

| Repo (lock id) | Full commit | Remote | License file | Detected license/SPDX | License status | Integration role | Redistribution note | Review requirement |
|---|---|---|---|---|---|---|---|---|
| tracklab | `5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2` | https://github.com/TrackingLaboratory/tracklab.git | `…/tracklab/LICENSE` | MIT | present | runtime_and_reference | Tag `v1.3.24` locked | Adapter/isolation policy |
| pnlcalib | `8c87391d6f4ea40c5e4d65e61529916c7a49ce62` | https://github.com/mguti97/PnLCalib.git | `…/pnlcalib/LICENSE` | GPL-2.0 | present | calibration_candidate | Copyleft implications if linked | Integration boundary review |
| no_bells_just_whistles | `bd993b31c2917096c23bb8aadf148314d17f8345` | https://github.com/mguti97/No-Bells-Just-Whistles.git | `…/no-bells-just-whistles/LICENSE` | GPL-2.0 | present | model_source_and_reference | **Code** GPL-2.0; **weights** not cleared by this file | Model weight license review_required |

## Model weights (not code)

| Artifact | Path | Code source license | Weight license status |
|---|---|---|---|
| `SV_kp.pth` | `/home/fdoblak/models/soccernet/sn-banner/SV_kp.pth` | NBJW repo GPL-2.0 | `review_required` (not derived) |
| `SV_lines.pth` | `/home/fdoblak/models/soccernet/sn-banner/SV_lines.pth` | NBJW repo GPL-2.0 | `review_required` (not derived) |

## Dataset licenses

SoccerNet task datasets are **not** covered by the code LICENSE rows above. See `docs/data/data_access_matrix.md`.
