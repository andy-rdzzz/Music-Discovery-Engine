from __future__ import annotations

import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from music_discovery.data.lastfm import iter_user_sessions
from music_discovery.data.features import engineer_features, build_feature_matrix, FEATURE_VECTOR_COLS


def build_track_features(
    kaggle_df: pd.DataFrame,
    cfg_features: dict | None = None,
) -> pd.DataFrame:
    """Engineer features for all Kaggle tracks. Returns df with (artist_norm, track_norm, 12 features)."""
    kw = cfg_features or {}
    feats = engineer_features(kaggle_df, **kw)
    keep = ["artist_norm", "track_norm"] + FEATURE_VECTOR_COLS
    return feats[keep].reset_index(drop=True)


def build_triplets(
    lastfm_df: pd.DataFrame,
    track_features_df: pd.DataFrame,
    gap_minutes: int = 30,
    min_session_length: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (anchor, positive, negative) triplet index and per-user history.

    Returns
    -------
    triplets_df : columns [user_id, anchor_idx, pos_idx, neg_idx]
    user_history_df : columns [user_id, track_idx, split]  (split = train/eval)
    """
    rng = random.Random(seed)

    # Build fast lookup: (artist_norm, track_norm) -> row index in track_features_df
    lookup: dict[tuple[str, str], int] = {
        (row.artist_norm, row.track_norm): i
        for i, row in track_features_df.iterrows()
    }

    # All known track indices for negative sampling
    all_track_ids = list(range(len(track_features_df)))

    triplet_rows: list[tuple[str, int, int, int]] = []
    history_rows: list[tuple[str, int, str]] = []

    users = lastfm_df["user_id"].unique()
    for user_id, sessions_df in tqdm(
        iter_user_sessions(lastfm_df, gap_minutes, min_session_length),
        total=len(users),
        desc="Building triplets",
    ):
        # Resolve tracks to feature indices
        sessions_df = sessions_df.copy()
        sessions_df["track_idx"] = sessions_df.apply(
            lambda r: lookup.get((r["artist_norm"], r["track_norm"]), -1), axis=1
        )
        sessions_df = sessions_df[sessions_df["track_idx"] >= 0]

        if sessions_df.empty:
            continue

        # Temporal 80/20 split per user
        unique_sessions = sessions_df["session_id"].unique()
        split_at = int(len(unique_sessions) * 0.8)
        train_sessions = set(unique_sessions[:split_at])

        train_df = sessions_df[sessions_df["session_id"].isin(train_sessions)]
        eval_df = sessions_df[~sessions_df["session_id"].isin(train_sessions)]

        # Record user history
        for _, row in train_df.iterrows():
            history_rows.append((user_id, int(row["track_idx"]), "train"))
        for _, row in eval_df.iterrows():
            history_rows.append((user_id, int(row["track_idx"]), "eval"))

        # Build triplets from train sessions only
        for session_id, session in train_df.groupby("session_id"):
            track_indices = session["track_idx"].tolist()
            if len(track_indices) < 2:
                continue

            for i, anchor_idx in enumerate(track_indices):
                # Positive: another track from same session
                pos_candidates = [t for j, t in enumerate(track_indices) if j != i]
                pos_idx = rng.choice(pos_candidates)

                # Negative: random track from all tracks (not in this session)
                session_set = set(track_indices)
                if len(session_set) >= len(all_track_ids):
                    continue  # no valid negatives exist — skip this anchor

                neg_idx = rng.choice(all_track_ids)
                while neg_idx in session_set:
                    neg_idx = rng.choice(all_track_ids)

                triplet_rows.append((user_id, anchor_idx, pos_idx, neg_idx))

    triplets_df = pd.DataFrame(triplet_rows, columns=["user_id", "anchor_idx", "pos_idx", "neg_idx"])
    user_history_df = pd.DataFrame(history_rows, columns=["user_id", "track_idx", "split"])

    print(f"[triplets] {len(triplets_df):,} triplets | {user_history_df['user_id'].nunique():,} users")
    return triplets_df, user_history_df


def save_processed(
    track_features_df: pd.DataFrame,
    triplets_df: pd.DataFrame,
    user_history_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    track_features_df.to_parquet(out / "track_features.parquet", index=False)
    triplets_df.to_parquet(out / "triplets.parquet", index=False)
    user_history_df.to_parquet(out / "user_history.parquet", index=False)
    print(f"[triplets] Saved processed data to {out}")
