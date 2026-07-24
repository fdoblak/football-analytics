"""Deterministic anonymous target selection from SoccerTrack v2 reference GT."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from football_analytics.acceptance.contracts import (
    EXTERNAL_REFERENCE_CONFIRMATION,
    RoleName,
)
from football_analytics.acceptance.soccertrack_v2.loader import (
    bas_path,
    gsr_path,
    iter_gsr_player_observations,
    load_bas_events,
)


@dataclass
class PlayerCoverageStats:
    player_id: str
    role_counts: dict[str, int] = field(default_factory=dict)
    jersey_counts: dict[str, int] = field(default_factory=dict)
    team_counts: dict[str, int] = field(default_factory=dict)
    frames_half1: int = 0
    frames_half2: int = 0
    valid_coords: int = 0
    bas_events: int = 0

    @property
    def total_frames(self) -> int:
        return self.frames_half1 + self.frames_half2

    @property
    def dominant_role(self) -> str:
        if not self.role_counts:
            return RoleName.OTHER.value
        return max(self.role_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def dominant_jersey(self) -> Optional[int]:
        usable = {k: v for k, v in self.jersey_counts.items() if k not in ("None", "")}
        if not usable:
            return None
        key = max(usable.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return int(key)

    @property
    def dominant_team(self) -> Optional[str]:
        usable = {k: v for k, v in self.team_counts.items() if k not in ("None", "")}
        if not usable:
            return None
        return max(usable.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def coord_validity(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return self.valid_coords / float(self.total_frames)


@dataclass
class TargetSelectionReceipt:
    match_id: str
    selected_player_id: str
    team_side: str
    jersey_number: int
    display_name: str
    confirmation_source: str
    reason: str
    candidates: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    partition_note: str = "half1=tuning; half2=held_out"


def _accumulate_gsr(
    root: Path,
    match_id: str,
    *,
    sample_stride: int = 1,
) -> dict[str, PlayerCoverageStats]:
    """Accumulate coverage stats; sample_stride>1 speeds large COCO GSR files."""
    stats: dict[str, PlayerCoverageStats] = {}
    for half in (1, 2):
        path = gsr_path(root, match_id, half)
        if not path.is_file():
            continue
        for obs in iter_gsr_player_observations(
            path, half=half, sample_stride=sample_stride
        ):
            st = stats.setdefault(obs.player_id, PlayerCoverageStats(player_id=obs.player_id))
            st.role_counts[obs.role] = st.role_counts.get(obs.role, 0) + 1
            jkey = str(obs.jersey_number)
            st.jersey_counts[jkey] = st.jersey_counts.get(jkey, 0) + 1
            tkey = str(obs.team_side)
            st.team_counts[tkey] = st.team_counts.get(tkey, 0) + 1
            if half == 1:
                st.frames_half1 += 1
            else:
                st.frames_half2 += 1
            if abs(obs.x_m) <= 60.0 and abs(obs.y_m) <= 40.0:
                st.valid_coords += 1
    return stats


def select_target_player(
    *,
    root: Path,
    match_id: str,
    min_frames: int = 1000,
    sample_stride: int = 1,
) -> TargetSelectionReceipt:
    """Select an outfield player with jersey, coverage, and BAS activity."""
    root = Path(root)
    stats = _accumulate_gsr(root, match_id, sample_stride=sample_stride)
    bas = load_bas_events(bas_path(root, match_id))
    bas_counts: dict[str, int] = defaultdict(int)
    for ev in bas:
        if ev.player_id:
            bas_counts[ev.player_id] += 1
    for pid, n in bas_counts.items():
        if pid in stats:
            stats[pid].bas_events = n
        else:
            stats[pid] = PlayerCoverageStats(player_id=pid, bas_events=n)

    candidates: list[PlayerCoverageStats] = []
    excluded: list[dict[str, Any]] = []
    for st in stats.values():
        role = st.dominant_role
        jersey = st.dominant_jersey
        team = st.dominant_team
        reasons: list[str] = []
        if role != RoleName.PLAYER.value:
            reasons.append(f"role={role}")
        if jersey is None:
            reasons.append("jersey_null")
        if team is None:
            reasons.append("team_null")
        if st.total_frames < min_frames:
            reasons.append("low_coverage")
        if st.frames_half1 == 0 or st.frames_half2 == 0:
            reasons.append("missing_half")
        if st.coord_validity < 0.5:
            reasons.append("low_coord_validity")
        if reasons:
            excluded.append(
                {
                    "player_id": st.player_id,
                    "reasons": reasons,
                    "frames": st.total_frames,
                    "bas_events": st.bas_events,
                }
            )
            continue
        candidates.append(st)

    if not candidates:
        raise RuntimeError("No eligible outfield target candidates")

    def sort_key(st: PlayerCoverageStats) -> tuple:
        # Prefer high coverage, BAS activity, validity; deterministic tie-break by player_id
        return (
            -(st.frames_half1 + st.frames_half2),
            -st.bas_events,
            -st.coord_validity,
            str(st.dominant_jersey),
            st.player_id,
        )

    ranked = sorted(candidates, key=sort_key)
    best = ranked[0]
    team = str(best.dominant_team)
    jersey = int(best.dominant_jersey)  # type: ignore[arg-type]
    display = f"SoccerTrack v2 Match {match_id} / Team {team} / Jersey {jersey}"
    reason = (
        "Deterministic outfield selection: role=player, non-null jersey, "
        "both halves, high coverage/coord validity, BAS activity; "
        "tie-break by lexicographic player_id."
    )
    cand_rows = [
        {
            "player_id": s.player_id,
            "team": s.dominant_team,
            "jersey": s.dominant_jersey,
            "frames_half1": s.frames_half1,
            "frames_half2": s.frames_half2,
            "bas_events": s.bas_events,
            "coord_validity": round(s.coord_validity, 4),
        }
        for s in ranked[:25]
    ]
    return TargetSelectionReceipt(
        match_id=str(match_id),
        selected_player_id=best.player_id,
        team_side=team,
        jersey_number=jersey,
        display_name=display,
        confirmation_source=EXTERNAL_REFERENCE_CONFIRMATION,
        reason=reason,
        candidates=cand_rows,
        excluded=excluded[:200],
    )


def write_target_receipt(receipt: TargetSelectionReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
