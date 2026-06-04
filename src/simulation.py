"""
Monte Carlo simulation of the 2026 FIFA World Cup.
Tracks per-team probabilities for every round and group position.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src import model as mdl
from src.tournament import GROUPS, TEAM_STATS, simulate_group_stage, bracket_r32, get_r32_teams

_DEFAULT_STATS = {"elo": 1500, "goals_scored": 1.2, "goals_conceded": 1.2, "win_rate": 0.33}

ROUNDS = ["qualified", "r16", "qf", "sf", "final", "winner"]
GROUP_POSITIONS = ["p1st", "p2nd", "p3rd", "p4th"]


def _stats(team: str) -> dict:
    return TEAM_STATS.get(team, _DEFAULT_STATS)


def _sample_goals(p_a: float, p_b: float) -> tuple[int, int]:
    lam_a = 0.5 + 1.5 * p_a
    lam_b = 0.5 + 1.5 * p_b
    return int(np.random.poisson(lam_a)), int(np.random.poisson(lam_b))


def _group_match(team_a: str, team_b: str, clf) -> tuple[int, int]:
    proba = mdl.predict_proba(clf, _stats(team_a), _stats(team_b))
    return _sample_goals(proba[0], proba[2])


def _knockout_match(team_a: str, team_b: str, clf) -> str:
    proba = mdl.predict_proba(clf, _stats(team_a), _stats(team_b))
    p_a, _, p_b = proba
    ga, gb = _sample_goals(p_a, p_b)
    if ga == gb:
        return team_a if np.random.random() < 0.5 + 0.1 * (p_a - p_b) else team_b
    return team_a if ga > gb else team_b


def _play_knockout_round(survivors: list[str], clf) -> list[str]:
    return [_knockout_match(survivors[i], survivors[i + 1], clf) for i in range(0, len(survivors), 2)]


def simulate_once(clf) -> dict:
    """
    Simulate one full tournament.
    Returns a dict with group positions and which teams reached each round.
    """
    group_results = {}
    group_positions: dict[str, int] = {}   # team → 1/2/3/4

    for group_name, teams in GROUPS.items():
        standings = simulate_group_stage(
            teams,
            match_fn=lambda a, b, clf=clf: _group_match(a, b, clf),
        )
        group_results[group_name] = standings
        for pos, result in enumerate(standings, 1):
            group_positions[result.team] = pos

    # Round of 32: 16 matchups → 16 survivors (R16)
    matchups = bracket_r32(group_results)
    qualified = {t for pair in matchups for t in pair}   # 32 teams
    r16_teams = [_knockout_match(a, b, clf) for a, b in matchups]
    qf_teams  = _play_knockout_round(r16_teams, clf)
    sf_teams  = _play_knockout_round(qf_teams, clf)
    final_teams = _play_knockout_round(sf_teams, clf)
    winner = _knockout_match(final_teams[0], final_teams[1], clf)

    return {
        "group_positions": group_positions,
        "qualified": qualified,
        "r16": set(r16_teams),
        "qf": set(qf_teams),
        "sf": set(sf_teams),
        "final": set(final_teams),
        "winner": winner,
    }


def run(clf, n: int = 50_000) -> pd.DataFrame:
    """
    Run n simulations. Returns a DataFrame with per-team probabilities for
    each group position (1st–4th) and each knockout round.
    Sorted by winner probability descending.
    """
    all_teams = [t for teams in GROUPS.values() for t in teams]
    team_to_group = {t: g for g, teams in GROUPS.items() for t in teams}

    counts: dict[str, dict] = {
        team: defaultdict(int) for team in all_teams
    }

    for _ in tqdm(range(n), desc="Simulating"):
        result = simulate_once(clf)

        for team, pos in result["group_positions"].items():
            counts[team][f"p{pos}st" if pos == 1 else f"p{pos}nd" if pos == 2 else f"p{pos}rd" if pos == 3 else "p4th"] += 1

        for round_key in ROUNDS:
            val = result[round_key]
            members = {val} if isinstance(val, str) else val
            for team in members:
                counts[team][round_key] += 1

    rows = []
    for team in all_teams:
        c = counts[team]
        rows.append({
            "team": team,
            "group": team_to_group[team],
            "p_1st":      round(c.get("p1st", 0) / n * 100, 1),
            "p_2nd":      round(c.get("p2nd", 0) / n * 100, 1),
            "p_3rd":      round(c.get("p3rd", 0) / n * 100, 1),
            "p_4th":      round(c.get("p4th", 0) / n * 100, 1),
            "p_qualified": round(c.get("qualified", 0) / n * 100, 1),
            "p_r16":      round(c.get("r16", 0) / n * 100, 1),
            "p_qf":       round(c.get("qf", 0) / n * 100, 1),
            "p_sf":       round(c.get("sf", 0) / n * 100, 1),
            "p_final":    round(c.get("final", 0) / n * 100, 1),
            "p_winner":   round(c.get("winner", 0) / n * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("p_winner", ascending=False).reset_index(drop=True)
