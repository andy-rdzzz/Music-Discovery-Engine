from __future__ import annotations

import pandas as pd
from pathlib import Path

REQUIRED_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
TRACK_ID_COLS = ["artists", "track_name"]


def load_kaggle_tracks(csv_path: str | Path) -> pd.DataFrame:
    """Load and Return Normalized Data"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_FEATURES + TRACK_ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Kaggle CSV: {missing}")

    df["artist_norm"] = df["artists"].str.lower().str.strip()
    df["track_norm"] = df["track_name"].str.lower().str.strip()

    before = len(df)
    df = df.dropna(subset=REQUIRED_FEATURES + TRACK_ID_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"[kaggle] Dropped {dropped} rows with missing values ({len(df)} remain)")

    df = df.drop_duplicates(subset=["artist_norm", "track_norm"], keep="first")
    print(f"[kaggle] Loaded {len(df):,} unique tracks")
    return df.reset_index(drop=True)
