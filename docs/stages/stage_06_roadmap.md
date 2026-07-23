# Stage 6 — Multi-object tracking (contracts → baselines)

Stage 5 (`detection-baseline-v0.5.0`) closed fused detection bundles. Stage 6
adds multi-object tracking contracts and human/ball tracker baselines for the
single-`target_player` product.

Stage 6 does **not** claim player identity, ReID, team/jersey, events, or
physical metrics.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **6A** | Multi-object tracking & track lifecycle contracts | Arrow sidecars, lifecycle transitions, ID/time/bbox rules, request/receipt JSON, evaluator stubs, synthetic validators. **No tracker algorithm.** | **IN TREE** |
| **6B** | Human multi-object tracking baseline & evaluation | In-repo IoU + constant-velocity greedy MOT; association eval; no ReID identity claims. | **IN TREE** |
| **6C** | Ball tracking baseline / lost-ball handling | Motion-first association; primary/ambiguity sidecar; short-gap prediction; long-gap new track; no ReID. | **IN TREE** (this stage) |
| **6D** | Human+ball track fusion, quality gates, Stage 6 close | Named only; not started. | **NOT STARTED** |

## Product link

Tracking consumes Stage 5 detection bundles and Stage 4 analysis windows.
Track IDs are **not** player/ball identities. Camera exit/re-entry sameness is
unproven until later ReID stages. Predicted/interpolated points are not
physical measurements by default. Primary ball candidate is not a true-ball
guarantee.

## SoccerNet note

`sn-tracking` / `sn-trackeval` / `sn-gamestate` are read-only references.
`sn-trackeval` remains a **future adapter candidate only**.

## Next stage (name only)

`Aşama 6D — İnsan ve Top Takip Birleştirme, Kalite Kapıları ve Aşama 6 Kapanışı`
