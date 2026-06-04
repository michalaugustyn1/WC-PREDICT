"""
2026 FIFA World Cup bracket definition.

Format: 48 teams, 12 groups (A–L) of 4 teams.
  - Top 2 from each group advance (24 teams)
  - 8 best 3rd-place teams advance (8 teams)
  - → Round of 32, then standard knockout

UPDATE GROUPS below with the actual draw if not already correct.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Groups — UPDATE with the actual 2026 draw
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# ---------------------------------------------------------------------------
# Current team stats for simulation (ELO + recent form)
# These are filled in by main.py after computing ELO from historical data.
# You can also override manually here.
# ---------------------------------------------------------------------------
TEAM_STATS: dict[str, dict] = {}


@dataclass
class GroupResult:
    team: str
    points: int = 0
    gd: int = 0       # goal difference
    gf: int = 0       # goals for


def simulate_group_stage(
    group: list[str],
    match_fn,        # callable(team_a, team_b) → (goals_a, goals_b)
) -> list[GroupResult]:
    """Round-robin within a group, return standings sorted by points/GD/GF."""
    standings = {t: GroupResult(team=t) for t in group}

    for i, home in enumerate(group):
        for away in group[i + 1:]:
            ga, gb = match_fn(home, away)
            if ga > gb:
                standings[home].points += 3
            elif ga < gb:
                standings[away].points += 3
            else:
                standings[home].points += 1
                standings[away].points += 1

            standings[home].gd += ga - gb
            standings[away].gd += gb - ga
            standings[home].gf += ga
            standings[away].gf += gb

    return sorted(
        standings.values(),
        key=lambda r: (r.points, r.gd, r.gf),
        reverse=True,
    )


def get_r32_teams(group_results: dict[str, list[GroupResult]]) -> tuple[list[str], list[str], list[str]]:
    """Return (firsts, seconds, best8_thirds) — the 32 teams that qualify."""
    groups = list(group_results.keys())
    firsts = [group_results[g][0].team for g in groups]   # 12
    seconds = [group_results[g][1].team for g in groups]  # 12
    thirds_ranked = sorted(
        [group_results[g][2] for g in groups],
        key=lambda r: (r.points, r.gd, r.gf),
        reverse=True,
    )
    best8_thirds = [r.team for r in thirds_ranked[:8]]    # 8
    return firsts, seconds, best8_thirds


def bracket_r32(group_results: dict[str, list[GroupResult]]) -> list[tuple[str, str]]:
    """
    Build exactly 16 Round-of-32 matchups from 32 qualified teams.
    Simplified seeding (replace with official bracket once published):
      - firsts[0..7]  vs best8_thirds[0..7]   (8 matchups)
      - firsts[8..11] vs seconds[0..3]         (4 matchups)
      - seconds[4..7] vs seconds[8..11]        (4 matchups)
    """
    firsts, seconds, best8_thirds = get_r32_teams(group_results)

    matchups: list[tuple[str, str]] = []
    matchups += [(firsts[i], best8_thirds[i]) for i in range(8)]
    matchups += [(firsts[8 + i], seconds[i]) for i in range(4)]
    matchups += [(seconds[4 + i], seconds[8 + i]) for i in range(4)]

    assert len(matchups) == 16, f"Expected 16 matchups, got {len(matchups)}"
    return matchups
