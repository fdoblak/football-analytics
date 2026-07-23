# Stage 4 completion — Broadcast understanding baseline

Stage 4 closes as a **technical broadcast-structure baseline** for single-target
evidence routing. It does **not** claim player analysis, Opta parity, or
real-match accuracy.

## Sub-stages

| ID | Deliverable | Status |
|----|-------------|--------|
| **4A** | Shot / camera Arrow contracts, typed models, suitability semantics, validators | CLOSED |
| **4B** | Rule-based shot-boundary baseline + evaluation (OpenCV streaming) | CLOSED |
| **4C** | Rule-based camera-view baseline (view/framing/graphics/motion/playability) | CLOSED |
| **4D** | Segment fusion, playability routing, `analysis_windows`, review queue | CLOSED |

## Supported vs unsupported camera axes (4C honesty)

**Supported (heuristic, abstain-capable):** view_family subset
(`main_broadcast`, `player_isolation`, `graphics`, `unknown`), framing_scale
subset, camera_motion subset, graphics_status, playability.

**Left unknown / not invented:** `camera_position`, `replay_status`, and finer
view families without reliable evidence.

## Replay limitation

No replay detector ships in Stage 4. Confirmed-replay windows are tested only
via synthetic camera metadata. **Replay unknown never auto-counts as live
events.** Confirmed replay blocks live-event and physical-metric eligibility.

## Downstream safety

`analysis_windows` eligibility axes gate tracking, calibration, identity,
ball analysis, live events, and physical metrics. Unsafe eligibility false
positives are held at zero on frozen synthetic evaluation. Manual review is
required for unknown, conflict, gap, low coverage, and replay-unknown cases.

## Synthetic evaluation

Shot F1 / camera macro-F1 / routing safety metrics on synthetic fixtures prove
**pipeline integrity**, not stadium performance. Do not cite them as match
accuracy.

## Not validated on real match video

- Boundary recall under dissolves/wipes
- Camera classification under real lighting / grass variance
- Replay / camera-position inference
- Target-player presence inside “eligible” windows

## Stage 5 entry conditions

Stage 5 may start only with an explicit user prompt. Suggested first
sub-stage: player / goalkeeper / referee / ball detection **contracts** (no
model claims until contracts + validators exist).

## Open risks (honest)

- RISK-029 pylist pressure in broadcast bundle validation remains open for
  large tables.
- RISK-043 (new): routing policy may over-abstain on real footage; calibrate
  only with reviewed labels — never invent live eligibility.
- No real-match broadcast corpus in-tree; dataset absence (RISK-012) continues.
