from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from music_discovery.data.item_bridge import load_item_bridge
from music_discovery.data.side_info import load_artist_similars, load_song_similars, load_song_tags
from music_discovery.data.taste_profile import load_taste_profile


def _assign_splits(df: pd.DataFrame, holdout_k: int, seed: int = 42) -> pd.Series:
    """Return a split Series ("train"/"val"/"test") via per-user random holdout."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype="object")

    for _, group in df.groupby("user_id"):
        idx = group.index.to_numpy()
        if len(idx) <= holdout_k * 2:
            continue  # too few interactions to split; keep all as train
        shuffled = rng.permutation(idx)
        split.iloc[shuffled[:holdout_k]] = "val"
        split.iloc[shuffled[holdout_k : holdout_k * 2]] = "test"

    return split


def build(
    taste_profile_path: str,
    track_metadata_db: str,
    sid_mismatches_path: str,
    lastfm_tags_db: str,
    lastfm_similars_db: str,
    artist_similarity_db: str,
    processed_dir: str,
    min_play_count: int = 1,
    sample_n_users: int | None = None,
    holdout_k: int = 10,
    skip_side_info: bool = False,
) -> None:

    os.makedirs(processed_dir, exist_ok=True)

    print("[interactions] Loading Taste Profile...")
    interactions = load_taste_profile(
        taste_profile_path,
        min_play_count=min_play_count,
        sample_n_users=sample_n_users,
    )

    print("[interactions] Loading item bridge...")
    songs = load_item_bridge(track_metadata_db, sid_mismatches_path)

    valid_songs = set(songs["song_id"])
    before = len(interactions)
    interactions = interactions[interactions["song_id"].isin(valid_songs)].copy()
    print(
        f"[interactions] Dropped {before - len(interactions):,} interactions "
        f"with no metadata ({len(interactions):,} remain)"
    )

    print("[interactions] Assigning train/val/test splits...")
    interactions["split"] = _assign_splits(interactions, holdout_k=holdout_k)
    split_counts = interactions["split"].value_counts().to_dict()
    print(f"[interactions] Split counts: {split_counts}")

    print("[interactions] Writing core parquets...")
    interactions.to_parquet(os.path.join(processed_dir, "interactions.parquet"), index=False)
    songs.to_parquet(os.path.join(processed_dir, "songs.parquet"), index=False)
    print(f"  interactions : {len(interactions):,} rows")
    print(f"  songs        : {len(songs):,} rows")

    if skip_side_info:
        print("[interactions] Skipping side-info (skip_side_info=True). Run again without flag to load tags/similars.")
        return

    print("[interactions] Loading side information in parallel (tags, similars, artist graph)...")
    tasks = {
        "song_tags":      lambda: load_song_tags(lastfm_tags_db, songs),
        "song_similars":  lambda: load_song_similars(lastfm_similars_db, songs),
        "artist_similars": lambda: load_artist_similars(artist_similarity_db),
    }
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(f"  [done] {name}: {len(results[name]):,} rows")

    results["song_tags"].to_parquet(os.path.join(processed_dir, "song_tags.parquet"), index=False)
    results["song_similars"].to_parquet(os.path.join(processed_dir, "song_similars.parquet"), index=False)
    results["artist_similars"].to_parquet(os.path.join(processed_dir, "artist_similars.parquet"), index=False)

    print("[interactions] Done.")


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    build(
        taste_profile_path=d["taste_profile"],
        track_metadata_db=d["track_metadata_db"],
        sid_mismatches_path=d["sid_mismatches"],
        lastfm_tags_db=d["lastfm_tags_db"],
        lastfm_similars_db=d["lastfm_similars_db"],
        artist_similarity_db=d["artist_similarity_db"],
        processed_dir=d["processed_dir"],
        min_play_count=d.get("min_play_count", 1),
        sample_n_users=d.get("sample_n_users"),
        holdout_k=d.get("holdout_k", 10),
    )
