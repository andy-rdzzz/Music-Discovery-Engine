from __future__ import annotations

import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from music_discovery.data.lastfm import iter_user_sessions
from music_discovery.data.features import engineer_features, build_feature_matrix, FEATURE_VECTOR_COLS
from music_discovery.data.matcher import TrackMatcher, TRAIN_MATCH_TYPES


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
    rules_path: str | Path | None = None,
    miss_log_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (anchor, positive, negative) triplet index and per-user history.

    Returns
    -------
    triplets_df      : columns [user_id, anchor_idx, pos_idx, neg_idx]
    user_history_df  : columns [user_id, track_idx, split, is_discovery, match_type, match_score, is_ambiguous]
                       is_discovery=True for eval rows not seen in that user's train history.
    """
    rng = random.Random(seed)

    # Build finite-automaton matcher (S0→S1→S2→S3→S4)
    matcher_kwargs = {} if rules_path is None else {"rules_path": rules_path}
    matcher = TrackMatcher(track_features_df, **matcher_kwargs)

    all_track_ids = list(range(len(track_features_df)))

    # Match unique (artist, track) pairs once — cache results for all 18M row lookups
    unique_pairs = (
        lastfm_df[["artist_name", "track_name", "artist_norm", "track_norm"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"[triplets] Matching {len(unique_pairs):,} unique pairs (from {len(lastfm_df):,} events)...")

    match_cache: dict[tuple[str, str], object] = {}
    miss_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in tqdm(unique_pairs.itertuples(index=False), total=len(unique_pairs), desc="Matching pairs"):
        key = (row.artist_name, row.track_name)
        match_cache[key] = matcher.match_normalized(row.artist_norm, row.track_norm)

    # Count raw event misses per unique pair (vectorized merge)
    miss_key_df = pd.DataFrame(
        [{"artist_name": a, "track_name": t} for (a, t), v in match_cache.items() if v is None]
    )
    if not miss_key_df.empty:
        miss_counts_df = (
            lastfm_df[["artist_name", "track_name"]]
            .merge(miss_key_df, on=["artist_name", "track_name"], how="inner")
            .groupby(["artist_name", "track_name"])
            .size()
            .reset_index(name="count")
        )
        miss_counts = {
            (row["artist_name"], row["track_name"]): int(row["count"])
            for _, row in miss_counts_df.iterrows()
        }

    seen_track_ids: set[int] = {
        r.track_idx
        for r in match_cache.values()
        if r is not None and r.match_type in TRAIN_MATCH_TYPES
    }
    print(f"[triplets] In-distribution pool: {len(seen_track_ids):,} / {len(all_track_ids):,} tracks")

    # Persist miss log for coverage_analysis
    if miss_log_path is not None and miss_counts:
        miss_rows = [
            {"artist_raw": a, "track_raw": t, "count": c}
            for (a, t), c in miss_counts.items()
        ]
        miss_df = pd.DataFrame(miss_rows)
        Path(miss_log_path).parent.mkdir(parents=True, exist_ok=True)
        miss_df.to_parquet(miss_log_path, index=False)
        print(f"[triplets] Logged {len(miss_df):,} unique misses → {miss_log_path}")

    triplet_rows: list[tuple[str, int, int, int]] = []
    history_rows: list[tuple] = []
    skipped_users: list[str] = []

    users = lastfm_df["user_id"].unique()
    for user_id, sessions_df in tqdm(
        iter_user_sessions(lastfm_df, gap_minutes, min_session_length),
        total=len(users),
        desc="Building triplets",
    ):
        sessions_df = sessions_df.copy()

        sessions_df["_match"] = [
            match_cache.get((artist_name, track_name))
            for artist_name, track_name in zip(sessions_df["artist_name"], sessions_df["track_name"])
        ]
        sessions_df["track_idx"] = sessions_df["_match"].apply(
            lambda m: m.track_idx if m is not None and m.match_type in TRAIN_MATCH_TYPES else -1
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

        # Record user history from train_df (preserve repeat listens)
        for track_idx, m in train_df[["track_idx", "_match"]].itertuples(index=False, name=None):
            history_rows.append((
                user_id, int(track_idx), "train", False,
                m.match_type if m else "exact",
                m.match_score if m else 100.0,
                m.is_ambiguous if m else False,
            ))
        for track_idx, m in eval_df[["track_idx", "_match"]].itertuples(index=False, name=None):
            track_idx = int(track_idx)
            is_discovery = track_idx not in user_train_set
            history_rows.append((
                user_id, track_idx, "eval", is_discovery,
                m.match_type if m else "exact",
                m.match_score if m else 100.0,
                m.is_ambiguous if m else False,
            ))
        
        train_df_for_triplets = train_df.drop_duplicates(subset=["session_id","track_idx"])

        # Build triplets from train sessions only
        for _session_id, session in train_df_for_triplets.groupby("session_id"):
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
        history_rows,
        columns=["user_id", "track_idx", "split", "is_discovery", "match_type", "match_score", "is_ambiguous"],
    )

    #Drop exact duplicates only (preserve neg_idx)
    before = len(triplets_df)
    triplets_df = triplets_df.drop_duplicates(subset=["user_id", "anchor_idx", "pos_idx", "neg_idx"])
    dropped_dups = before - len(triplets_df)

    eval_mask = user_history_df["split"] == "eval"
    eval_hist = user_history_df[eval_mask]
    distinct_discovery_rate = (
      eval_hist.drop_duplicates(subset=["user_id", "track_idx"])["is_discovery"].mean()
      if eval_mask.any() else 0.0
    )
    per_user_discovery_rate = (
      eval_hist.groupby("user_id")["is_discovery"].mean().mean()
      if eval_mask.any() else 0.0
    )
    neg_in_seen = triplets_df["neg_idx"].isin(seen_track_ids).mean() if len(triplets_df) else 0.0

    print(
        f"[triplets] {len(triplets_df):,} triplets | dropped {dropped_dups:,} duplicates | "
        f"{user_history_df['user_id'].nunique():,} users"
    )
    print(f"[triplets] Neg from seen pool: {neg_in_seen * 100:.1f}%  (target >80%)")
    print(f"[triplets] Eval discovery rate: {distinct_discovery_rate * 100:.1f}%  (target >80%)")
    print(f"[triplets] Per-user discovery rate: {per_user_discovery_rate * 100:.1f}%  (target >80%)")
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
