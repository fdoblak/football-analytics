# Single-Player Metric Dictionary — Version 1

**dictionary_version:** 1
**schema companion:** `schemas/metrics/single_player_report.schema.json`
**registry companion:** `configs/metrics/single_player_metrics.yaml`
**product scope:** `docs/scope/single_player_product_v1.md`

All metrics are computed for **one** `target_player` only. Team/opponent context is intermediate.

## Global null / status policy

**Never invent a successful zero.** Prefer an explicit status:

| Status / reason | When to use |
|-----------------|-------------|
| `not_evaluable` | Preconditions for the metric are not met |
| `insufficient_coverage` | Too little tracked / visible time or samples |
| `identity_uncertain` | Target identity not reliable; may require `manual_identity_confirmation_required` |
| `ball_tracking_insufficient` | Ball track quality too low for the event family |
| `calibration_insufficient` | Pitch calibration too weak for distance/zone/speed metrics |
| `event_uncertain` | Candidate event exists but success/fail cannot be decided |

Shared fields for every metric below:

- **version:** 1
- **confidence / coverage:** Required on every emitted metric row; may be null only if status is terminal failure before measurement
- **source:** `video_derived` | `official_opta` | `opta_style_project` (see provenance labels)
- **validation:** Unit tests on fixtures + coverage gates + identity gate before attribution

---

## 1. `opta_style_events`

| Field | Value |
|-------|-------|
| metric_id | `opta_style_events` |
| TR name | Opta-tarzı / olay verisi özeti |
| EN name | Opta-style / event data summary |
| definition | Versioned event bundle for the target player: project-generated Opta-style events and/or licensed official Opta imports. Not a single scalar; report references the event table. |
| numerator | Count of attributed events in scope (by event type when rolled up) |
| denominator | N/A for the bundle; per-type rates use type-specific denominators |
| unit | events (table) / count when rolled up |
| min observations | ≥1 attributed event **or** explicit empty-with-coverage when match was fully processed with zero events |
| success | Events attributed only under reliable identity; provenance set correctly |
| fail | Events attributed under identity failure |
| null / not_evaluable | `identity_uncertain`; missing licensed Opta when only official Opta was requested; `insufficient_coverage` |
| confidence / coverage | Event detection confidence; fraction of match time with usable tracks |
| source | `official_opta` only if licensed import; else `opta_style_project` / `video_derived` |
| validation | Schema + provenance labels; no scraping; no Opta brand mislabeling |
| version | 1 |

---

## 2. `heatmap`

| Field | Value |
|-------|-------|
| metric_id | `heatmap` |
| TR name | Isı haritası |
| EN name | Heatmap |
| definition | Spatial occupancy density of the target player on a calibrated pitch plane when possible; attack direction normalized per half. |
| numerator | Weighted dwell / sample density on pitch cells |
| denominator | Track samples in evaluable time (coverage-normalized) |
| unit | density grid + coverage % |
| min observations | Config `min_track_samples` (placeholder in registry) |
| success | Calibrated pitch heatmap with coverage reported |
| fail | Pixel-only motion treated as pitch truth without disclaimer |
| null / not_evaluable | `calibration_insufficient`; `identity_uncertain`; `insufficient_coverage` |
| confidence / coverage | Calibration confidence; % of match with verified target track |
| source | `video_derived` |
| validation | Half attack-direction flip check; coverage must appear in report |
| version | 1 |

---

## 3. `duels_won`

| Field | Value |
|-------|-------|
| metric_id | `duels_won` |
| TR name | İkili mücadele kazanma sayısı |
| EN name | Duels won |
| definition | Count of ground (and optionally total) duels won by the target player. Aerials reported separately in aerial family; do not double-count the same event. |
| numerator | Evaluable duels with status `won` |
| denominator | N/A (count) |
| unit | count |
| min observations | ≥1 evaluable duel attempt **or** not_evaluable if duel detection unavailable |
| success | `won` only when both involvement and outcome are reliable |
| fail | Counting uncertain outcomes as won |
| null / not_evaluable | `event_uncertain`; `identity_uncertain`; `insufficient_coverage` |
| confidence / coverage | Mean event confidence; fraction of candidate duels evaluable |
| source | `video_derived` or `official_opta` |
| validation | Duplicate event ID suppression; ground vs aerial split |
| version | 1 |

---

## 4. `duel_win_rate`

| Field | Value |
|-------|-------|
| metric_id | `duel_win_rate` |
| TR name | İkili mücadele kazanma oranı |
| EN name | Duel win rate |
| definition | Won / (won + lost) over **evaluable** duels only. Uncertain duels excluded from denominator. |
| numerator | Duels won (evaluable) |
| denominator | Duels won + lost (evaluable); uncertain excluded |
| unit | ratio [0, 1] |
| min observations | `min_evaluable_duels` (registry placeholder) |
| success | Rate computed only from evaluable outcomes |
| fail | Including uncertain in denominator as losses/wins |
| null / not_evaluable | Below min observations; `identity_uncertain`; `event_uncertain` dominant |
| confidence / coverage | Based on evaluable share of candidate duels |
| source | `video_derived` or `official_opta` |
| validation | Rate null when denominator is below min |
| version | 1 |

---

## 5. `pass_attempts`

| Field | Value |
|-------|-------|
| metric_id | `pass_attempts` |
| TR name | Pas denemesi |
| EN name | Pass attempts |
| definition | Passes initiated by the target player with reliable start attribution. |
| numerator | Pass attempts attributed to target |
| denominator | N/A (count) |
| unit | count |
| min observations | Pipeline ran with ball+player interaction capability; else not_evaluable |
| success | Start on target track with ball possession evidence |
| fail | Team passes attributed without target initiation proof |
| null / not_evaluable | `ball_tracking_insufficient`; `identity_uncertain`; off-camera termination may still count as attempt if start is certain (completion separate) |
| confidence / coverage | Pass-start confidence; visible initiation coverage |
| source | `video_derived` or `official_opta` |
| validation | Attempt ≠ completed; off-camera ends may be `uncertain` for completion |
| version | 1 |

---

## 6. `pass_completion_rate`

| Field | Value |
|-------|-------|
| metric_id | `pass_completion_rate` |
| TR name | İsabetli / başarılı pas oranı |
| EN name | Pass completion rate |
| definition | Completed passes / evaluable pass attempts. Failed and completed only; uncertain excluded from denominator. |
| numerator | Completed passes |
| denominator | Completed + failed (evaluable attempts) |
| unit | ratio [0, 1] |
| min observations | `min_evaluable_passes` |
| success | Clear receipt by teammate / retained possession per event rules |
| fail | Counting off-camera uncertain as completed |
| null / not_evaluable | Below min; `event_uncertain` majority; `ball_tracking_insufficient` |
| confidence / coverage | Evaluable pass share |
| source | `video_derived` or `official_opta` |
| validation | Denominator excludes uncertain |
| version | 1 |

---

## 7. `dribbles_successful`

| Field | Value |
|-------|-------|
| metric_id | `dribbles_successful` |
| TR name | Başarılı dribbling |
| EN name | Successful dribbles |
| definition | Successful take-ons / dribbles with ball control and defender interaction; not mere running with the ball in open space without a duel/take-on context. |
| numerator | Successful dribble events |
| denominator | N/A (count) |
| unit | count |
| min observations | Capability available; else not_evaluable |
| success | Retain possession past engaged defender per status enum |
| fail | Counting pure progressive carries without defender engagement as dribbles |
| null / not_evaluable | `event_uncertain`; `ball_tracking_insufficient`; `identity_uncertain` |
| confidence / coverage | Event confidence; engagement evidence coverage |
| source | `video_derived` or `official_opta` |
| validation | No double-count of same sequence |
| version | 1 |

---

## 8. `dribbles_failed`

| Field | Value |
|-------|-------|
| metric_id | `dribbles_failed` |
| TR name | Başarısız dribbling |
| EN name | Failed dribbles |
| definition | Failed take-on / dribble attempts by the target player. |
| numerator | Failed dribble events |
| denominator | N/A (count) |
| unit | count |
| min observations | Same family as successful dribbles |
| success | Clear loss after engaged take-on attempt |
| fail | Counting dispossessed without take-on intent as failed dribble without taxonomy note |
| null / not_evaluable | `event_uncertain`; `ball_tracking_insufficient`; `identity_uncertain` |
| confidence / coverage | As dribbles_successful |
| source | `video_derived` or `official_opta` |
| validation | Mutual exclusivity with successful for same event_id |
| version | 1 |

---

## 9. `take_on_success_rate`

| Field | Value |
|-------|-------|
| metric_id | `take_on_success_rate` |
| TR name | Adam eksiltme başarı oranı |
| EN name | Take-on success rate |
| definition | Successful / (successful + failed) evaluable take-ons. Uncertain excluded. |
| numerator | Successful take-ons |
| denominator | Successful + failed |
| unit | ratio [0, 1] |
| min observations | `min_evaluable_take_ons` |
| success | Rate only on evaluable outcomes |
| fail | Inflating rate with open-space carries |
| null / not_evaluable | Below min; `event_uncertain`; `identity_uncertain` |
| confidence / coverage | Evaluable take-on share |
| source | `video_derived` or `official_opta` |
| validation | Linked to dribble status enums |
| version | 1 |

---

## 10. `tackles_interceptions_recoveries`

| Field | Value |
|-------|-------|
| metric_id | `tackles_interceptions_recoveries` |
| TR name | Top çalma (tackle / interception / recovery) |
| EN name | Tackles, interceptions, and recoveries |
| definition | Defensive ball-wins by the target. Keep **tackle**, **interception**, and **recovery** as separate subcounts; TR report may show a combined “top çalma” rollup without dropping subtypes. |
| numerator | Per-subtype counts; optional rollup sum |
| denominator | N/A (counts) |
| unit | count (by subtype + optional total) |
| min observations | Defensive event capability present |
| success | Subtype labels preserved in machine output |
| fail | Collapsing subtypes irreversibly |
| null / not_evaluable | `ball_tracking_insufficient`; `event_uncertain`; `identity_uncertain` |
| confidence / coverage | Per-subtype confidence |
| source | `video_derived` or `official_opta` |
| validation | Subtype schema required in report extras |
| version | 1 |

---

## 11. `ball_losses`

| Field | Value |
|-------|-------|
| metric_id | `ball_losses` |
| TR name | Top kaybı |
| EN name | Ball losses / turnovers |
| definition | Target-attributed losses: failed pass, failed dribble, dispossessed, miscontrol (taxonomy open to versioned subtypes). Do **not** attribute every team loss to the target. |
| numerator | Target-attributed loss events |
| denominator | N/A (count); optional rate uses target possessions as denominator in later versions |
| unit | count |
| min observations | Possession attribution capability |
| success | Loss linked to target possession end |
| fail | Team-wide turnovers dumped onto target |
| null / not_evaluable | `identity_uncertain`; `ball_tracking_insufficient`; `event_uncertain` |
| confidence / coverage | Possession-link confidence |
| source | `video_derived` or `official_opta` |
| validation | Reason subtype required when known |
| version | 1 |

---

## 12. `aerial_duels`

| Field | Value |
|-------|-------|
| metric_id | `aerial_duels` |
| TR name | Hava topu mücadelesi |
| EN name | Aerial duel attempts |
| definition | Aerial duel attempts where the target is actually involved (not merely near a header by a teammate). |
| numerator | Aerial attempts involving target |
| denominator | N/A (count) |
| unit | count |
| min observations | Aerial detection capability |
| success | Involvement evidence (jump/contest + ball contest) |
| fail | Counting nearby aerials without involvement |
| null / not_evaluable | `event_uncertain`; `identity_uncertain`; `insufficient_coverage` |
| confidence / coverage | Involvement confidence |
| source | `video_derived` or `official_opta` |
| validation | Distinct from ground `duels_*` |
| version | 1 |

---

## 13. `aerial_win_rate`

| Field | Value |
|-------|-------|
| metric_id | `aerial_win_rate` |
| TR name | Hava topu kazanma oranı |
| EN name | Aerial win rate |
| definition | Aerial won / (won + lost) over evaluable aerials. Uncertain excluded. |
| numerator | Aerials won |
| denominator | Aerials won + lost |
| unit | ratio [0, 1] |
| min observations | `min_evaluable_aerials` |
| success | Outcome reliable |
| fail | Including non-involved aerials |
| null / not_evaluable | Below min; `event_uncertain`; `identity_uncertain` |
| confidence / coverage | Evaluable aerial share |
| source | `video_derived` or `official_opta` |
| validation | Same exclusion rules as duel_win_rate |
| version | 1 |

---

## 14. `clearances`

| Field | Value |
|-------|-------|
| metric_id | `clearances` |
| TR name | Uzaklaştırma |
| EN name | Clearances |
| definition | Defensive actions where the target deliberately clears the ball from a danger area. Distinct from a routine completed pass. |
| numerator | Clearance events |
| denominator | N/A (count) |
| unit | count |
| min observations | Event taxonomy available |
| success | Defensive clear intent + danger-zone context |
| fail | Labeling ordinary successful passes as clearances |
| null / not_evaluable | `event_uncertain`; `calibration_insufficient` for zone context; `identity_uncertain` |
| confidence / coverage | Zone + intent confidence |
| source | `video_derived` or `official_opta` |
| validation | Mutually exclusive labeling vs normal pass when clearance criteria met |
| version | 1 |

---

## 15. `progressive_passes_def_to_mid`

| Field | Value |
|-------|-------|
| metric_id | `progressive_passes_def_to_mid` |
| TR name | Birinci bölgeden ikinci bölgeye geçiş pasları |
| EN name | Progressive passes defensive third → middle third |
| definition | Completed (or evaluable) passes by the target that start in the defensive third and end in the middle third on a half-normalized pitch. Zone thirds from metric config. |
| numerator | Qualifying passes |
| denominator | N/A (count); optional rate vs pass attempts in later versions |
| unit | count |
| min observations | Calibration + pass events |
| success | Start/end thirds reliable after attack-direction normalize |
| fail | Using raw broadcast pixels without calibration |
| null / not_evaluable | `calibration_insufficient`; `event_uncertain`; `identity_uncertain` |
| confidence / coverage | Calibration coverage × pass completion confidence |
| source | `video_derived` |
| validation | Third boundaries from versioned config |
| version | 1 |

---

## 16. `progressive_passes_mid_to_att`

| Field | Value |
|-------|-------|
| metric_id | `progressive_passes_mid_to_att` |
| TR name | İkinci bölgeden üçüncü bölgeye geçiş pasları |
| EN name | Progressive passes middle third → attacking third |
| definition | Same as above for middle → attacking third. |
| numerator | Qualifying passes |
| denominator | N/A (count) |
| unit | count |
| min observations | Calibration + pass events |
| success | Reliable thirds after half normalize |
| fail | Ignoring half-time attack direction flip |
| null / not_evaluable | `calibration_insufficient`; `event_uncertain`; `identity_uncertain` |
| confidence / coverage | As progressive_passes_def_to_mid |
| source | `video_derived` |
| validation | Shared zone config with metric 15 |
| version | 1 |

---

## 17. `long_pass_attempts`

| Field | Value |
|-------|-------|
| metric_id | `long_pass_attempts` |
| TR name | Uzun pas denemesi |
| EN name | Long pass attempts |
| definition | Pass attempts whose calibrated pitch distance ≥ versioned `long_pass_distance_m` threshold. Threshold requires calibration; do not silently assume a universal constant. |
| numerator | Long pass attempts |
| denominator | N/A (count) |
| unit | count |
| min observations | Calibration sufficient for distance |
| success | Distance on pitch plane ≥ threshold |
| fail | Using uncalibrated pixel length as meters |
| null / not_evaluable | `calibration_insufficient`; `ball_tracking_insufficient`; `identity_uncertain` |
| confidence / coverage | Calibration confidence for distance |
| source | `video_derived` or `official_opta` (if licensed distances) |
| validation | Threshold marked `requires_calibration` in registry |
| version | 1 |

---

## 18. `long_pass_completion_rate`

| Field | Value |
|-------|-------|
| metric_id | `long_pass_completion_rate` |
| TR name | Uzun pas başarı oranı |
| EN name | Long pass completion rate |
| definition | Completed long passes / evaluable long pass attempts. |
| numerator | Completed long passes |
| denominator | Completed + failed long passes (evaluable) |
| unit | ratio [0, 1] |
| min observations | `min_evaluable_long_passes` |
| success | Same completion rules as passes, restricted to long subset |
| fail | Mixing short passes into denominator |
| null / not_evaluable | `calibration_insufficient`; below min; `event_uncertain` |
| confidence / coverage | Long-pass evaluable share |
| source | `video_derived` or `official_opta` |
| validation | Depends on long_pass_attempts definition |
| version | 1 |

---

## 19. `distance_covered_m`

| Field | Value |
|-------|-------|
| metric_id | `distance_covered_m` |
| TR name | Koşu mesafesi |
| EN name | Distance covered |
| definition | Path length of the target on the calibrated pitch plane. Camera motion must not inflate distance. Long invisible gaps must not be fabricated via unconstrained interpolation. Report measured distance **with** coverage. |
| numerator | Sum of evaluable pitch-plane segments (meters) |
| denominator | N/A; coverage % is co-reported |
| unit | meters |
| min observations | `min_track_samples` + calibration |
| success | Calibrated segments only; interpolation limits documented |
| fail | Pixel-track length as meters; inventing path across long gaps |
| null / not_evaluable | `calibration_insufficient`; `identity_uncertain`; `insufficient_coverage` |
| confidence / coverage | Mandatory coverage fraction of match time |
| source | `video_derived` |
| validation | Gap policy from registry; co-report coverage |
| version | 1 |

---

## 20. `sprint_count`

| Field | Value |
|-------|-------|
| metric_id | `sprint_count` |
| TR name | Sprint adedi |
| EN name | Sprint count |
| definition | Number of sprint events meeting versioned speed threshold, min duration, min consecutive samples, gap tolerance, and smoothing rules. |
| numerator | Sprint events |
| denominator | N/A (count) |
| unit | count |
| min observations | Speed series with calibration |
| success | All sprint config gates satisfied |
| fail | Silent universal threshold without config version |
| null / not_evaluable | `calibration_insufficient`; `insufficient_coverage`; `identity_uncertain` |
| confidence / coverage | Speed-series coverage |
| source | `video_derived` |
| validation | Registry sprint block versioned |
| version | 1 |

---

## 21. `sprint_max_speed_mps`

| Field | Value |
|-------|-------|
| metric_id | `sprint_max_speed_mps` |
| TR name | Maksimum sprint hızı |
| EN name | Maximum sprint speed |
| definition | Peak speed during evaluable sprint (or evaluable high-speed) windows after outlier filtering; always with confidence. |
| numerator | Max filtered speed sample / window peak |
| denominator | N/A |
| unit | m/s |
| min observations | `min_samples` inside sprint/speed windows |
| success | Outlier-filtered peak with confidence |
| fail | Reporting single noisy spike as max without filter |
| null / not_evaluable | `calibration_insufficient`; `insufficient_coverage`; `identity_uncertain` |
| confidence / coverage | Required on value |
| source | `video_derived` |
| validation | Outlier policy versioned in registry |
| version | 1 |

---

## 22. `speed_avg_mps`

| Field | Value |
|-------|-------|
| metric_id | `speed_avg_mps` |
| TR name | Ortalama / dağılımsal koşu hızı |
| EN name | Average / distributional running speed |
| definition | Mean (and optional distributional summaries) of evaluable on-ball/off-ball locomotion speed when coverage is reliable. Do not report a marketing “average speed” without coverage. |
| numerator | Sum of evaluable speed samples |
| denominator | Count of evaluable speed samples |
| unit | m/s |
| min observations | Registry `min_samples` |
| success | Coverage above threshold |
| fail | Averaging across huge invisible gaps |
| null / not_evaluable | `insufficient_coverage`; `calibration_insufficient`; `identity_uncertain` |
| confidence / coverage | Mandatory |
| source | `video_derived` |
| validation | Distributional extras optional but versioned |
| version | 1 |

---

## 23. `penalty_area_ball_touches`

| Field | Value |
|-------|-------|
| metric_id | `penalty_area_ball_touches` |
| TR name | Rakip ceza sahasında topla buluşma |
| EN name | Ball touches in opponent penalty area |
| definition | Touches / controls by the target inside the opponent penalty area on a normalized pitch. Presence in the box without ball contact does **not** count. |
| numerator | Qualifying touches |
| denominator | N/A (count) |
| unit | count |
| min observations | Calibration + ball interaction |
| success | Pitch-in-box + ball contact evidence |
| fail | Counting box occupancy alone |
| null / not_evaluable | `calibration_insufficient`; `ball_tracking_insufficient`; `identity_uncertain` |
| confidence / coverage | Box geometry confidence × touch confidence |
| source | `video_derived` or `official_opta` |
| validation | Opponent box after attack-direction normalize |
| version | 1 |

---

## 24. `activation`

| Field | Value |
|-------|-------|
| metric_id | `activation` |
| TR name | Oyun içi aktivasyon / aktiflik |
| EN name | In-game activation / involvement |
| definition | Explainable composite from versioned components (examples): mobility in evaluable time; on-ball action rate; off-ball high-intensity runs; possessions involved; press/duel involvement; visibility/track coverage. If a single score is emitted, components and formula **must** appear in the report. Not a vague marketing KPI. |
| numerator | Weighted sum of normalized components (formula in registry) |
| denominator | Component normalization bases |
| unit | score (unitless) + component breakdown |
| min observations | Enough coverage to evaluate majority of components; else not_evaluable |
| success | Formula + components published with values |
| fail | Opaque single number without breakdown |
| null / not_evaluable | `insufficient_coverage`; `identity_uncertain`; undefined formula version |
| confidence / coverage | Propagated from components |
| source | `video_derived` |
| validation | Formula version pinned in registry |
| version | 1 |

---

## 25. `coverage_reliability`

| Field | Value |
|-------|-------|
| metric_id | `coverage_reliability` |
| TR name | Analiz coverage ve güvenilirlik |
| EN name | Analysis coverage and reliability |
| definition | Meta-metric: track coverage, identity confidence, calibration coverage, ball-tracking coverage, and fraction of metrics evaluable. Always present even when other metrics are null. |
| numerator | Aggregated coverage measures (see registry) |
| denominator | Match / analysis duration bases |
| unit | ratios / percentages + ordinal reliability |
| min observations | Always computable after a run attempt |
| success | Honest reporting of gaps |
| fail | Omitting low coverage to imply completeness |
| null / not_evaluable | Only if run never started; otherwise always emit |
| confidence / coverage | Self-describing |
| source | `project-generated` |
| validation | Required top-level fields in report schema |
| version | 1 |

---

## Provenance labels

| Label | Meaning |
|-------|---------|
| `official_opta` | Licensed Opta import only |
| `opta_style_project` | Project event taxonomy inspired by industry Opta-style schemas |
| `video_derived` | Estimated from broadcast/video pipeline |
| `project-generated` | Project meta or composite (e.g. coverage) |

Never present `video_derived` / `opta_style_project` as official Opta.
