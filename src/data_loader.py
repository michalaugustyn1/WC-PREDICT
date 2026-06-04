"""
Loads historical international football results.
Source: https://github.com/martj42/international_results
"""

import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def load_results(use_cache: bool = True) -> pd.DataFrame:
    """Load all historical international match results."""
    cache_path = DATA_DIR / "results.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading match results...")
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return pd.read_csv(cache_path, parse_dates=["date"])


def filter_competitive(df: pd.DataFrame, from_year: int = 1990) -> pd.DataFrame:
    """Keep only competitive matches with recorded scores from a given year onward."""
    friendly_keywords = ["Friendly", "friendly"]
    mask = ~df["tournament"].str.contains("|".join(friendly_keywords), na=False)
    df = df[mask & (df["date"].dt.year >= from_year)].copy()
    return df.dropna(subset=["home_score", "away_score"])
