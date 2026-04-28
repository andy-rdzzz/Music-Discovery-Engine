from __future__ import annotations

import random
import warnings
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
    min_neg_pool: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (anchor, positive, negative) triplet index and per-user history.

    Returns
    -------
    triplets_df      : columns [user_id, anchor_idx, pos_idx, neg_idx]
    user_history_df  : columns [user_id, track_idx, split, is_discovery]
                       is_discovery=True for eval rows not seen in that user's train history.
    """
    rng = random.Random(seed)

    # Build fast lookup: (artist_norm, track_norm) -> row index in track_features_df
    lookup: dict[tuple[str, str], int] = {
        (row.artist_norm, row.track_norm): i
        for i, row in track_features_df.iterrows()
    }

    all_track_ids = list(range(len(track_features_df)))

    # Fix 2: build in-distribution negative pool — tracks seen in any user's history
    key_pairs = zip(lastfm_df["artist_norm"], lastfm_df["track_norm"])
    seen_track_ids: set[int] = {lookup[k] for k in key_pairs if k in lookup}
    print(f"[triplets] In-distribution pool: {len(seen_track_ids):,} / {len(all_track_ids):,} tracks")

    triplet_rows: list[tuple[str, int, int, int]] = []
    history_rows: list[tuple[str, int, str, bool]] = []
    skipped_users: list[str] = []

    users = lastfm_df["user_id"].unique()
    for user_id, sessions_df in tqdm(
        iter_user_sessions(lastfm_df, gap_minutes, min_session_length),
        total=len(users),
        desc="Building triplets",
    ):
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

        # Fix 2: exclude this user's train history from negatives
        user_train_set = set(train_df["track_idx"].tolist())
        neg_pool = list(seen_track_ids - user_train_set)

        if len(neg_pool) < min_neg_pool:
            skipped_users.append(str(user_id))
            continue

        # Record user history
        for _, row in train_df.iterrows():
            history_rows.append((user_id, int(row["track_idx"]), "train", False))
        for _, row in eval_df.iterrows():
            track_idx = int(row["track_idx"])
            # Fix 3a: mark whether this eval track is truly unseen in train
            is_discovery = track_idx not in user_train_set
            history_rows.append((user_id, track_idx, "eval", is_discovery))

        # Build triplets from train sessions only
        for _session_id, session in train_df.groupby("session_id"):
            track_indices = session["track_idx"].tolist()
            if len(track_indices) < 2:
                continue

            for i, anchor_idx in enumerate(track_indices):
                pos_candidates = [t for j, t in enumerate(track_indices) if j != i]
                pos_idx = rng.choice(pos_candidates)

                # Fix 4: skip trivial same-track triplets
                if anchor_idx == pos_idx:
                    continue

                neg_idx = rng.choice(neg_pool)
                triplet_rows.append((user_id, anchor_idx, pos_idx, neg_idx))

    if skipped_users:
        warnings.warn(
            f"[triplets] Skipped {len(skipped_users)} user(s) with neg pool < {min_neg_pool}: "
            f"{skipped_users}",
            UserWarning,
            stacklevel=2,
        )

    triplets_df = pd.DataFrame(triplet_rows, columns=["user_id", "anchor_idx", "pos_idx", "neg_idx"])
    user_history_df = pd.DataFrame(
        history_rows, columns=["user_id", "track_idx", "split", "is_discovery"]
    )

    # Fix 4: drop duplicate (user, anchor, pos) triplets
    before = len(triplets_df)
    triplets_df = triplets_df.drop_duplicates(subset=["user_id", "anchor_idx", "pos_idx"])
    dropped_dups = before - len(triplets_df)

    eval_mask = user_history_df["split"] == "eval"
    discovery_rate = user_history_df.loc[eval_mask, "is_discovery"].mean() if eval_mask.any() else 0.0
    neg_in_seen = triplets_df["neg_idx"].isin(seen_track_ids).mean() if len(triplets_df) else 0.0

    print(
        f"[triplets] {len(triplets_df):,} triplets | dropped {dropped_dups:,} duplicates | "
        f"{user_history_df['user_id'].nunique():,} users"
    )
    print(f"[triplets] Neg from seen pool: {neg_in_seen * 100:.1f}%  (target >80%)")
    print(f"[triplets] Eval discovery rate: {discovery_rate * 100:.1f}%  (target >80%)")
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
