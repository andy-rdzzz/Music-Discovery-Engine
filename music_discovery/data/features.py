from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

LOUDNESS_MIN = -60.0
LOUDNESS_MAX = 0.0
TEMPO_MIN = 40.0
TEMPO_MAX = 220.0


def engineer_features(
    df: pd.DataFrame,
    loudness_min: float = LOUDNESS_MIN,
    loudness_max: float = LOUDNESS_MAX,
    tempo_min: float = TEMPO_MIN,
    tempo_max: float = TEMPO_MAX,
) -> pd.DataFrame:
    """Add loudness_norm, tempo_norm, and 3 circumplex features. Returns df with new cols."""
    if loudness_max == loudness_min:
        raise ValueError("loudness_max must differ from loudness_min")
    if tempo_max == tempo_min:
        raise ValueError("tempo_max must differ from tempo_min")

    out = df.copy()

    out["loudness_norm"] = (out["loudness"] - loudness_min) / (loudness_max - loudness_min)
    out["loudness_norm"] = out["loudness_norm"].clip(0.0, 1.0)

    out["tempo_norm"] = (out["tempo"] - tempo_min) / (tempo_max - tempo_min)
    out["tempo_norm"] = out["tempo_norm"].clip(0.0, 1.0)

    # Russell's Valence-Arousal Circumplex features
    out["arousal"] = (out["energy"] + out["danceability"]) / 2.0
    out["chill_factor"] = out["acousticness"] * (1.0 - out["energy"])
    out["vocal_presence"] = (1.0 - out["instrumentalness"]) * (1.0 - out["speechiness"])

    return out


FEATURE_VECTOR_COLS = [
    "danceability", "energy", "loudness_norm", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo_norm", "speechiness",
    "arousal", "chill_factor", "vocal_presence",
]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract 12D feature matrix from df. Rows are L2-normalised."""
    mat = df[FEATURE_VECTOR_COLS].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms
