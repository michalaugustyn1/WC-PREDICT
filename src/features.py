"""
Feature engineering for match outcome prediction.

Each row in the training set represents one match with features describing
the two teams at the time the match was played.
"""

import pandas as pd
import numpy as np


def _recent_stats(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 10) -> dict:
    """Goals scored/conceded and win rate in last N matches before a date."""
    played = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < before_date)
    ].sort_values("date").tail(n)

    if played.empty:
        return {"goals_scored": 1.2, "goals_conceded": 1.2, "win_rate": 0.33}

    scored, conceded, wins = [], [], []
    for row in played.itertuples(index=False):
        if row.home_team == team:
            scored.append(row.home_score)
            conceded.append(row.away_score)
            wins.append(1 if row.home_score > row.away_score else 0)
        else:
            scored.append(row.away_score)
            conceded.append(row.home_score)
            wins.append(1 if row.away_score > row.home_score else 0)

    return {
        "goals_scored": np.mean(scored),
        "goals_conceded": np.mean(conceded),
        "win_rate": np.mean(wins),
    }


def build_features(df: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix from historical match data.
    Target: 0 = home win, 1 = draw, 2 = away win.
    """
    elo_pivot = elo_history.pivot_table(
        index=["date", "team"], values="elo", aggfunc="first"
    )

    rows = []
    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        date = row.date

        try:
            elo_home = elo_pivot.loc[(date, home), "elo"]
            elo_away = elo_pivot.loc[(date, away), "elo"]
        except KeyError:
            continue

        neutral = row.neutral if hasattr(row, "neutral") else False
        hs = _recent_stats(df, home, date)
        aws = _recent_stats(df, away, date)

        if row.home_score > row.away_score:
            outcome = 0
        elif row.home_score == row.away_score:
            outcome = 1
        else:
            outcome = 2

        rows.append({
            "elo_diff": elo_home - elo_away,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "neutral": int(neutral),
            "home_goals_scored": hs["goals_scored"],
            "home_goals_conceded": hs["goals_conceded"],
            "home_win_rate": hs["win_rate"],
            "away_goals_scored": aws["goals_scored"],
            "away_goals_conceded": aws["goals_conceded"],
            "away_win_rate": aws["win_rate"],
            "outcome": outcome,
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "elo_diff",
    "elo_home",
    "elo_away",
    "neutral",
    "home_goals_scored",
    "home_goals_conceded",
    "home_win_rate",
    "away_goals_scored",
    "away_goals_conceded",
    "away_win_rate",
]
