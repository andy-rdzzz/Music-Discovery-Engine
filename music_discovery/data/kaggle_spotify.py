from __future__ import annotations

import re
import pandas as pd
from pathlib import Path

REQUIRED_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
TRACK_ID_COLS = ["artists", "track_name"]


def _normalize_name(s: object) -> str:
    """Expanded normalization: lowercase, strip parentheticals and featured-artist credits."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)", "", s)          # strip (Remastered), (feat. X), etc.
    s = re.sub(r"\s+(?:feat|ft)\.?\s+.*", "", s)  # strip feat. X at end
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_kaggle_tracks(csv_path: str | Path) -> pd.DataFrame:
    """Load and return normalized Kaggle Spotify tracks with mean-aggregated features for duplicates."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_FEATURES + TRACK_ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Kaggle CSV: {missing}")

    # Fix 1: expanded normalization
    df["artist_norm"] = df["artists"].apply(_normalize_name)
    df["track_norm"] = df["track_name"].apply(_normalize_name)
    df = df[(df["artist_norm"] != "") & (df["track_norm"] != "")]

    before = len(df)
    df = df.dropna(subset=REQUIRED_FEATURES + TRACK_ID_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"[kaggle] Dropped {dropped} rows with missing values ({len(df)} remain)")

    # Fix 6: mean-aggregate audio features across duplicate (artist_norm, track_norm) groups
    # instead of keep='first', which arbitrarily picks one conflicting feature vector.
    mean_feats = (
        df.groupby(["artist_norm", "track_norm"], sort=False)[REQUIRED_FEATURES]
        .mean()
        .reset_index()
    )
    first_strings = (
        df.drop_duplicates(subset=["artist_norm", "track_norm"], keep="first")[
            ["artist_norm", "track_norm"] + TRACK_ID_COLS
        ]
    )
    df = mean_feats.merge(first_strings, on=["artist_norm", "track_norm"], how="left")

    print(f"[kaggle] Loaded {len(df):,} unique tracks")
    return df.reset_index(drop=True)
