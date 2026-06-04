"""
2026 FIFA World Cup Winner Predictor

Usage:
  python main.py train       # download data, train model
  python main.py predict     # run Monte Carlo simulation
  python main.py all         # train then predict
"""

import sys
import pandas as pd
from src import data_loader, elo, features, model, simulation, tournament


def run_train():
    print("=== STEP 1: Load historical match data ===")
    df = data_loader.load_results()
    df_comp = data_loader.filter_competitive(df, from_year=1993)
    print(f"  Loaded {len(df_comp):,} competitive matches")

    print("\n=== STEP 2: Compute ELO ratings ===")
    elo_history = elo.build_elo_history(df_comp)
    final_elo = elo.compute_elo(df_comp)
    top10 = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 teams by ELO:")
    for rank, (team, rating) in enumerate(top10, 1):
        print(f"    {rank:2}. {team:<25} {rating:.0f}")

    print("\n=== STEP 3: Build feature matrix ===")
    feature_df = features.build_features(df_comp, elo_history)
    print(f"  {len(feature_df):,} training samples")
    print(f"  Outcome distribution:\n{feature_df['outcome'].value_counts().sort_index().to_string()}")

    print("\n=== STEP 4: Train XGBoost model ===")
    clf = model.train(feature_df)

    # Persist ELO ratings for simulation
    _save_team_stats(df_comp, final_elo)
    return clf, final_elo


def _save_team_stats(df: pd.DataFrame, final_elo: dict):
    """Populate TEAM_STATS with ELO + recent form for every WC team."""
    all_teams = [t for teams in tournament.GROUPS.values() for t in teams]

    for team in all_teams:
        stats = features._recent_stats(df, team, df["date"].max() + pd.Timedelta(days=1), n=15)
        tournament.TEAM_STATS[team] = {
            "elo": final_elo.get(team, elo.BASE_RATING),
            **stats,
        }

    print("\n  Team ELO ratings for simulation:")
    for team in sorted(all_teams, key=lambda t: tournament.TEAM_STATS[t]["elo"], reverse=True):
        e = tournament.TEAM_STATS[team]["elo"]
        print(f"    {team:<30} ELO: {e:.0f}")


def run_predict(clf=None, n_sims: int = 50_000):
    if clf is None:
        clf = model.load()
        print("Loading data to rebuild team stats...")
        df = data_loader.load_results()
        df_comp = data_loader.filter_competitive(df, from_year=1993)
        final_elo = elo.compute_elo(df_comp)
        _save_team_stats(df_comp, final_elo)

    print(f"\n=== Running {n_sims:,} tournament simulations ===")
    results = simulation.run(clf, n=n_sims)

    # --- Per-group breakdown ---
    print("\n=== Group Stage Probabilities ===")
    for group_name in sorted(results["group"].unique()):
        grp = results[results["group"] == group_name].sort_values("p_1st", ascending=False)
        print(f"\n  Group {group_name}")
        print(f"  {'Team':<28} {'1st':>5} {'2nd':>5} {'3rd':>5} {'4th':>5} {'LL':>5} {'Qual':>6}")
        print(f"  {'-'*64}")
        for _, row in grp.iterrows():
            print(
                f"  {row['team']:<28} "
                f"{row['p_1st']:>4.1f}% "
                f"{row['p_2nd']:>4.1f}% "
                f"{row['p_3rd']:>4.1f}% "
                f"{row['p_4th']:>4.1f}% "
                f"{row['p_lucky_loser']:>4.1f}% "
                f"{row['p_qualified']:>5.1f}%"
            )

    # --- Overall knockout progression ---
    print("\n=== Knockout Round Probabilities (sorted by win %) ===")
    print(f"  {'Team':<28} {'Grp':>4} {'R32':>5} {'R16':>5} {'QF':>5} {'SF':>5} {'F':>5} {'Win':>6}")
    print(f"  {'-'*75}")
    for _, row in results.iterrows():
        if row["p_qualified"] < 0.5:
            continue
        print(
            f"  {row['team']:<28} "
            f"  {row['group']}  "
            f"{row['p_qualified']:>4.1f}% "
            f"{row['p_r16']:>4.1f}% "
            f"{row['p_qf']:>4.1f}% "
            f"{row['p_sf']:>4.1f}% "
            f"{row['p_final']:>4.1f}% "
            f"{row['p_winner']:>5.2f}%"
        )

    results.to_csv("data/predictions.csv", index=False)
    print("\n  Full results saved → data/predictions.csv")
    return results


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000

    if cmd == "train":
        run_train()
    elif cmd == "predict":
        run_predict(n_sims=n_sims)
    elif cmd == "all":
        clf, _ = run_train()
        run_predict(clf, n_sims=n_sims)
    else:
        print(__doc__)
        sys.exit(1)
