"""
Train and persist a match outcome predictor.

Predicts P(home win), P(draw), P(away win) for a given matchup.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from src.features import FEATURE_COLS

MODEL_PATH = Path(__file__).parent.parent / "data" / "model.joblib"


def train(feature_df: pd.DataFrame) -> XGBClassifier:
    X = feature_df[FEATURE_COLS].values
    y = feature_df["outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    print(f"  Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"  Log-loss : {log_loss(y_test, proba):.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")
    return clf


def load() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found — run `python main.py train` first.")
    return joblib.load(MODEL_PATH)


def predict_proba(clf, team_a_features: dict, team_b_features: dict) -> np.ndarray:
    """
    Return [P(team_a wins), P(draw), P(team_b wins)].
    team_a is treated as 'home' (neutral=1 for WC knockouts).
    """
    row = {
        "elo_diff": team_a_features["elo"] - team_b_features["elo"],
        "elo_home": team_a_features["elo"],
        "elo_away": team_b_features["elo"],
        "neutral": 1,
        "home_goals_scored": team_a_features.get("goals_scored", 1.2),
        "home_goals_conceded": team_a_features.get("goals_conceded", 1.2),
        "home_win_rate": team_a_features.get("win_rate", 0.33),
        "away_goals_scored": team_b_features.get("goals_scored", 1.2),
        "away_goals_conceded": team_b_features.get("goals_conceded", 1.2),
        "away_win_rate": team_b_features.get("win_rate", 0.33),
    }
    X = np.array([[row[c] for c in FEATURE_COLS]])
    return clf.predict_proba(X)[0]  # [P(A wins), P(draw), P(B wins)]
