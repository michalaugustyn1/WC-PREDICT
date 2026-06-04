"""Extended diagnostic to find the inversion bug."""
from src import data_loader, elo, features, model
import pandas as pd
import numpy as np

df = data_loader.load_results()
df_comp = data_loader.filter_competitive(df, from_year=1993)

# ---- 1. Check for NaN scores in the raw data ----
print("=== NaN scores in df_comp ===")
print(f"  NaN home_score: {df_comp['home_score'].isna().sum()}")
print(f"  NaN away_score: {df_comp['away_score'].isna().sum()}")
print()

# Drop NaN scores before proceeding
df_clean = df_comp.dropna(subset=["home_score", "away_score"]).copy()
print(f"  Rows after dropping NaN scores: {len(df_clean)} (was {len(df_comp)})\n")

# ---- 2. Check training feature correlations ----
print("=== Building ELO history and features ===")
elo_history = elo.build_elo_history(df_clean)
feature_df = features.build_features(df_clean, elo_history)
print(f"  Training samples: {len(feature_df)}")
print(f"  NaN counts:\n{feature_df.isnull().sum().to_string()}\n")

# Correlation between elo_diff and outcome
print("=== elo_diff by outcome ===")
for outcome, label in [(0, "home wins"), (1, "draw"), (2, "away wins")]:
    subset = feature_df[feature_df["outcome"] == outcome]["elo_diff"]
    print(f"  outcome={outcome} ({label}): mean elo_diff = {subset.mean():.1f}  n={len(subset)}")
print()

# ---- 3. Verify reversed prediction ----
clf = model.load()
final_elo = elo.compute_elo(df_clean)

stats_spain = {"elo": final_elo.get("Spain", 1500), "goals_scored": 1.5, "goals_conceded": 0.8, "win_rate": 0.6}
stats_cabo  = {"elo": 1500, "goals_scored": 1.2, "goals_conceded": 1.2, "win_rate": 0.33}

p1 = model.predict_proba(clf, stats_spain, stats_cabo)
p2 = model.predict_proba(clf, stats_cabo, stats_spain)
print("=== Prediction check ===")
print(f"  Spain(1960) vs CaboVerde(1500): {p1.round(3)}  (expect: A>>B)")
print(f"  CaboVerde(1500) vs Spain(1960): {p2.round(3)}  (expect: B>>A)")
print()

# ---- 4. Sample training rows to see ELO values ----
print("=== Sample training feature rows (top 5 by elo_diff) ===")
sample = feature_df.nlargest(5, "elo_diff")[["elo_diff", "elo_home", "elo_away", "outcome"]]
print(sample.to_string())
print()
print("=== Sample training feature rows (bottom 5 by elo_diff) ===")
sample2 = feature_df.nsmallest(5, "elo_diff")[["elo_diff", "elo_home", "elo_away", "outcome"]]
print(sample2.to_string())
