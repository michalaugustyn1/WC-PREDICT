"""
ELO rating system for national teams.

K-factors by tournament type (higher = more important):
  World Cup final / SF   : 60
  World Cup group / QF   : 50
  Continental final      : 40
  Continental group      : 35
  World Cup qualifier    : 30
  Friendly               : 20
"""

import pandas as pd
from collections import defaultdict

BASE_RATING = 1500.0

K_MAP = {
    "FIFA World Cup": 50,
    "UEFA Euro": 35,
    "Copa America": 35,
    "Africa Cup of Nations": 35,
    "AFC Asian Cup": 35,
    "Gold Cup": 30,
    "World Cup qualification": 30,
    "UEFA Nations League": 25,
    "Friendly": 20,
}

HOME_ADVANTAGE = 100  # ELO points added for home team


def _k_factor(tournament: str) -> float:
    for key, k in K_MAP.items():
        if key.lower() in tournament.lower():
            return k
    return 25.0


def _expected(ra: float, rb: float, home_advantage: float = 0) -> float:
    return 1 / (1 + 10 ** ((rb - ra - home_advantage) / 400))


def _result(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def compute_elo(df: pd.DataFrame) -> dict[str, float]:
    """
    Iterate over matches in chronological order and compute final ELO ratings.
    Returns a dict of {team_name: elo_rating}.
    """
    ratings: dict[str, float] = defaultdict(lambda: BASE_RATING)
    df = df.sort_values("date")

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        neutral = row.neutral if hasattr(row, "neutral") else False
        adv = 0 if neutral else HOME_ADVANTAGE

        ra, rb = ratings[home], ratings[away]
        ea = _expected(ra, rb, adv)
        eb = 1 - ea

        sa, sb = _result(row.home_score, row.away_score)
        k = _k_factor(row.tournament)

        ratings[home] = ra + k * (sa - ea)
        ratings[away] = rb + k * (sb - eb)

    return dict(ratings)


def build_elo_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns [date, team, elo] representing the ELO
    rating of each team just BEFORE each match they play (useful for features).
    """
    ratings: dict[str, float] = defaultdict(lambda: BASE_RATING)
    records = []
    df = df.sort_values("date")

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        neutral = row.neutral if hasattr(row, "neutral") else False
        adv = 0 if neutral else HOME_ADVANTAGE

        ra, rb = ratings[home], ratings[away]
        records.append({"date": row.date, "match_id": row.index, "team": home, "elo": ra, "role": "home"})
        records.append({"date": row.date, "match_id": row.index, "team": away, "elo": rb, "role": "away"})

        ea = _expected(ra, rb, adv)
        eb = 1 - ea
        sa, sb = _result(row.home_score, row.away_score)
        k = _k_factor(row.tournament)

        ratings[home] = ra + k * (sa - ea)
        ratings[away] = rb + k * (sb - eb)

    return pd.DataFrame(records)
